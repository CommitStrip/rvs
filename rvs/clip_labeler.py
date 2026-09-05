#!/usr/bin/env python3
"""
clip_labeler.py - T0.5 语义反射弧：CLIP 零样本文本打标（MIT 权重引擎）
====================================================================
引擎选型（竞品调研 + 许可核查定稿）：
  - 默认引擎 openai CLIP ViT-B/32 双塔 ONNX（权重 MIT，商用无阻塞）；
  - MobileCLIP（快 ~4.8×）权重为 apple-amlr 仅限科研，仅作许可解除后的
    换引擎候选；Clip4Retrofit 无开源代码，不接入。

架构（反射弧毫秒响应的关键）：
  - 标签集文本嵌入【离线预计算】并缓存（build_label_embeddings /
    scripts/gen_label_embeddings.py）——运行时只跑视觉塔（复用 vus
    ClipOnnx 的 embed(bgr) 公共接口），帧循环内一次视觉嵌入 + 缓存矩阵
    余弦，文本塔与分词器不进热路径；
  - 负标签（如 "background"）与正标签同场竞技：无目标帧上负标签得最高
    分，上层据此做目标存在性过滤（吸收检测层过滤职能，绕开 AGPL）；
  - 输入支持运动框裁剪（box/scale 契约与 vus motion_crop 事件一致）——
    反射弧吃"运动区域放大图"，小目标有效分辨率提升。

安全：绝不静默降级——嵌入缓存缺失时构造抛 RuntimeError 并给生成指引
（沿 vus clip_onnx 同一哲学）；标签集变更后需重新生成缓存。
标签集语言建议：CLIP ViT-B/32 为英文-图像对齐，标签用英文效果最佳
（中文展示名由上层映射）。
"""

import hashlib
from pathlib import Path

import numpy as np

_VENDOR_DIR = Path(__file__).resolve().parent / "_vendor"
_LOGIT_SCALE = 100.0


def _load_tokenizer():
    """加载 vendor BPE 分词器；词表缺失抛 RuntimeError 并给下载指引。"""
    bpe_path = _VENDOR_DIR / "bpe_simple_vocab_16e6.txt.gz"
    if not bpe_path.is_file():
        raise RuntimeError(
            f"CLIP 分词器词表缺失: {bpe_path}\n"
            "请运行 scripts/download_clip_onnx.sh（约 1.3MB）。")
    from ._vendor.simple_tokenizer import SimpleTokenizer
    return SimpleTokenizer(str(bpe_path))


def tokenize_batch(texts, tokenizer=None, ctx_len: int = 77) -> np.ndarray:
    """CLIP 标准提示编码：[BOS] + tokens + [EOS] pad 0 → int64 [N, ctx_len]。"""
    if tokenizer is None:
        tokenizer = _load_tokenizer()
    out = np.zeros((len(texts), ctx_len), dtype=np.int64)
    for i, text in enumerate(texts):
        tokens = tokenizer.encode(str(text))[: ctx_len - 2]
        out[i, 0] = 49406                     # BOS
        out[i, 1 + len(tokens)] = 49407       # EOS
        out[i, 1:1 + len(tokens)] = tokens
    return out


def embed_texts_onnx(onnx_path, texts, tokenizer=None, batch: int = 8) -> np.ndarray:
    """用双塔合一 ONNX 的文本塔编码文本（仅离线生成缓存时调用）。

    图像输入喂全零占位——CLIP 双塔结构保证 text_embeds 只由 input_ids
    决定。模型输出必须含文本嵌入（text_embeds 类），否则抛 RuntimeError。
    """
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise RuntimeError(
            "生成文本嵌入需要 onnxruntime: pip install onnxruntime") from e
    session = ort.InferenceSession(str(onnx_path),
                                   providers=["CPUExecutionProvider"])
    in_specs = {i.name: i for i in session.get_inputs()}
    out_names = [o.name for o in session.get_outputs()]
    text_out = next((n for n in out_names if "text" in n.lower()), None)
    if text_out is None:
        raise RuntimeError(
            f"模型 {onnx_path} 的输出不含文本嵌入（outputs={out_names}）；"
            "请改用带 text_embeds 输出的导出文件，或在装有 torch 的机器上"
            "直接用 openai-clip 生成标签嵌入。")
    ids = tokenize_batch(texts, tokenizer)

    def _make_feed(chunk):
        feed = {}
        for name, spec in in_specs.items():
            shape = spec.shape
            if name == "input_ids":
                feed[name] = chunk
            elif name == "attention_mask":
                feed[name] = (chunk != 0).astype(np.int64)
            elif len(shape) == 4 and shape[1] == 3:  # 图像输入 NCHW → 全零占位
                h = shape[2] if isinstance(shape[2], int) else 224
                w = shape[3] if isinstance(shape[3], int) else 224
                feed[name] = np.zeros((len(chunk), 3, h, w), dtype=np.float32)
        return feed

    feats = []
    for i in range(0, len(texts), batch):
        chunk = ids[i:i + batch]
        out = session.run([text_out], _make_feed(chunk))[0]
        feats.append(np.asarray(out, dtype=np.float32).reshape(len(chunk), -1))
    mat = np.concatenate(feats, axis=0)
    return mat / np.linalg.norm(mat, axis=1, keepdims=True).clip(1e-8)


def _cache_path(model_dir, labels, negative_labels) -> Path:
    key = hashlib.sha256(
        repr((tuple(labels), tuple(negative_labels))).encode("utf-8")
    ).hexdigest()[:16]
    return Path(model_dir) / f"clip_label_emb_{key}.npz"


def build_label_embeddings(labels, negative_labels=(), model_dir=None,
                           onnx_filename="clip-visual-vitb32.onnx",
                           tokenizer=None) -> Path:
    """离线生成标签集文本嵌入缓存（npz）。标签集变更后需重新生成。"""
    from vus.clip_onnx import resolve_model_dir
    d = Path(resolve_model_dir(model_dir))
    onnx_path = d / onnx_filename
    if not onnx_path.is_file():
        raise RuntimeError(
            f"CLIP ONNX 模型缺失: {onnx_path}\n"
            "请运行 scripts/download_clip_onnx.sh（int8 约 153MB，hf-mirror）。")
    all_labels = [str(x) for x in list(labels) + list(negative_labels)]
    emb = embed_texts_onnx(onnx_path, all_labels, tokenizer)
    out = _cache_path(d, labels, negative_labels)
    # labels 用 unicode 数组（allow_pickle=False 可安全加载）
    np.savez(out, embeddings=emb, labels=np.array(all_labels, dtype=np.str_))
    return out


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


class CLIPTagger:
    """T0.5 语义反射弧：零样本打标（运行时只跑视觉塔，毫秒级热路径）。

    label(frame, motion_ratio, box=None, scale=1.0) -> [{label, score}]
    与 vus.live.tagger 的 Labeler 协议对齐（box/scale 为可选扩展参数）。
    """

    name = "clip"

    def __init__(self, labels, negative_labels=("background",),
                 model_dir=None, visual_engine=None, emb_path=None,
                 top_k: int = 3):
        from vus.clip_onnx import resolve_model_dir
        self.labels = [str(x) for x in labels]
        self.negative_labels = [str(x) for x in negative_labels]
        if not self.labels:
            raise ValueError("正标签集不能为空")
        self.top_k = max(1, int(top_k))
        d = resolve_model_dir(model_dir)
        self._emb_path = (Path(emb_path) if emb_path
                          else _cache_path(d, self.labels, self.negative_labels))
        if not self._emb_path.is_file():
            raise RuntimeError(
                f"标签嵌入缓存缺失: {self._emb_path}\n"
                "运行时只跑视觉塔（文本编码已离线完成）——请先运行 "
                "scripts/gen_label_embeddings.py 生成缓存。")
        cache = np.load(self._emb_path, allow_pickle=False)
        self._emb = cache["embeddings"].astype(np.float32)
        self._all_labels = [str(x) for x in cache["labels"]]
        if (self._all_labels[:len(self.labels)] != self.labels
                or self._all_labels[len(self.labels):] != self.negative_labels):
            raise RuntimeError(
                "嵌入缓存与标签集不一致，请重新生成缓存。")
        if visual_engine is None:
            from vus.clip_onnx import ClipOnnx
            visual_engine = ClipOnnx(d)      # 模型缺失时其内部给出下载指引
        self._visual = visual_engine

    def label(self, frame_bgr, motion_ratio: float = 0.0,
              box=None, scale: float = 1.0):
        img = self._crop(frame_bgr, box, scale) if box is not None else frame_bgr
        emb = np.asarray(self._visual.embed(img), dtype=np.float32)
        probs = _softmax(_LOGIT_SCALE * (self._emb @ emb))
        order = np.argsort(-probs)[: self.top_k]
        return [{"label": self._all_labels[int(i)],
                 "score": round(float(probs[int(i)]), 4)} for i in order]

    @staticmethod
    def _crop(frame, box, scale: float, padding: float = 0.25):
        """运动框裁剪（坐标契约与 vus encode_frame_crop_b64 一致）。"""
        import cv2
        try:
            bx, by, bw, bh = (float(v) for v in list(box)[:4])
        except (TypeError, ValueError):
            return frame
        if bw <= 0 or bh <= 0:
            return frame
        s = float(scale) if scale and scale > 0 else 1.0
        bx, by, bw, bh = bx / s, by / s, bw / s, bh / s
        ex, ey = bw * padding, bh * padding
        h, w = frame.shape[:2]
        x0, y0 = max(0, round(bx - ex)), max(0, round(by - ey))
        x1, y1 = min(w, round(bx + bw + ex)), min(h, round(by + bh + ey))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return frame
        return frame[y0:y1, x0:x1]
