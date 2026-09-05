"""W-A clip_labeler 单测：嵌入缓存加载、余弦排序、负标签、裁剪输入、
分词 pad 逻辑（mock tokenizer）；真模型/词表在场时的集成测试 skipif。"""

import numpy as np
import pytest

from rvs.clip_labeler import (CLIPTagger, _cache_path, build_label_embeddings,
                              tokenize_batch)

POSITIVE = ["a person walking", "a moving car", "a dog running"]
NEGATIVE = ["background, empty scene, nothing"]


class _FakeVisual:
    """可编程视觉嵌入器：按收到的图尺寸返回预设向量，记录调用。"""

    name = "fake"

    def __init__(self, vector, size_map=None):
        self.vector = np.asarray(vector, dtype=np.float32)
        self.size_map = size_map or {}
        self.seen_shapes = []

    def embed(self, frame_bgr):
        self.seen_shapes.append(frame_bgr.shape[:2])
        if frame_bgr.shape[:2] in self.size_map:
            return self.size_map[frame_bgr.shape[:2]]
        return self.vector


def _write_cache(tmp_path, labels, embeddings, name=None):
    p = tmp_path / (name or "cache.npz")
    np.savez(p, embeddings=np.asarray(embeddings, dtype=np.float32),
             labels=np.array([str(x) for x in labels], dtype=np.str_))
    return p


def test_label_ranks_by_cosine_and_clips_topk(tmp_path):
    # 三正一负；fake 视觉向量最贴近"car"
    emb = np.eye(4, dtype=np.float32)  # 每行单位向量
    cache = _write_cache(tmp_path, POSITIVE + NEGATIVE, emb)
    visual = _FakeVisual(vector=emb[1])  # "a moving car"
    tagger = CLIPTagger(POSITIVE, NEGATIVE, model_dir=tmp_path,
                        visual_engine=visual, emb_path=cache, top_k=3)
    out = tagger.label(np.zeros((48, 64, 3), np.uint8))
    assert len(out) == 3
    scores = [x["score"] for x in out]
    assert scores == sorted(scores, reverse=True)
    assert out[0]["label"] == "a moving car"
    assert all(0.0 <= x["score"] <= 1.0 for x in out)


def test_negative_label_wins_on_empty_scene(tmp_path):
    # 视觉向量贴近负标签 → 负标签得最高分（目标存在性过滤的依据）
    emb = np.eye(4, dtype=np.float32)
    cache = _write_cache(tmp_path, POSITIVE + NEGATIVE, emb)
    visual = _FakeVisual(vector=emb[3])  # 负标签方向
    tagger = CLIPTagger(POSITIVE, NEGATIVE, model_dir=tmp_path,
                        visual_engine=visual, emb_path=cache, top_k=4)
    out = tagger.label(np.zeros((48, 64, 3), np.uint8))
    assert out[0]["label"] == NEGATIVE[0]


def test_box_crop_scales_to_fullres(tmp_path):
    # box (2,2,8,6) @scale 0.25 → 原图 (8,8,32,24) 外扩 25% → 48x36 裁剪区
    emb = np.eye(4, dtype=np.float32)
    cache = _write_cache(tmp_path, POSITIVE + NEGATIVE, emb)
    visual = _FakeVisual(vector=emb[0])
    tagger = CLIPTagger(POSITIVE, NEGATIVE, model_dir=tmp_path,
                        visual_engine=visual, emb_path=cache)
    frame = np.zeros((240, 320, 3), np.uint8)
    tagger.label(frame, box=[2, 2, 8, 6], scale=0.25)
    assert visual.seen_shapes[-1] == (36, 48)   # (h, w)


def test_missing_cache_raises_with_guidance(tmp_path):
    with pytest.raises(RuntimeError, match="gen_label_embeddings"):
        CLIPTagger(POSITIVE, NEGATIVE, model_dir=tmp_path,
                   visual_engine=_FakeVisual(1.0), emb_path=tmp_path / "无.npz")


def test_label_count_mismatch_raises(tmp_path):
    cache = _write_cache(tmp_path, ["only", "two", "labels", "here"],
                         np.eye(4, dtype=np.float32))
    with pytest.raises(RuntimeError, match="不一致"):
        CLIPTagger(POSITIVE, NEGATIVE, model_dir=tmp_path,
                   visual_engine=_FakeVisual(1.0), emb_path=cache)


def test_empty_positive_labels_rejected(tmp_path):
    with pytest.raises(ValueError):
        CLIPTagger([], NEGATIVE, model_dir=tmp_path,
                   visual_engine=_FakeVisual(1.0), emb_path=tmp_path / "x.npz")


def test_cache_path_uses_sha256_and_depends_on_labels(tmp_path):
    p1 = _cache_path(tmp_path, ["a"], ["bg"])
    p2 = _cache_path(tmp_path, ["a"], ["bg"])
    p3 = _cache_path(tmp_path, ["a", "b"], ["bg"])
    assert p1 == p2                      # 同标签集同名（确定性）
    assert p1 != p3                      # 标签集变 → 缓存名变
    assert "clip_label_emb_" in p1.name


def test_tokenize_batch_pads_bos_eos_with_fake_tokenizer():
    class _FakeTok:
        def encode(self, text):
            return {"hello": [1, 2, 3], "": []}[text]

    out = tokenize_batch(["hello", ""], tokenizer=_FakeTok())
    assert out.shape == (2, 77)
    assert out.dtype == np.int64
    assert out[0, 0] == 49406 and out[0, 1:4].tolist() == [1, 2, 3]
    assert out[0, 4] == 49407            # EOS 紧跟最后一个 token
    assert out[0, 5:].sum() == 0         # 其余 pad 0
    assert out[1, 0] == 49406 and out[1, 1] == 49407  # 空文本：BOS+EOS


def test_missing_vocab_raises_with_download_hint(tmp_path, monkeypatch):
    import rvs.clip_labeler as mod
    monkeypatch.setattr(mod, "_VENDOR_DIR", tmp_path)  # 无词表
    with pytest.raises(RuntimeError, match="download_clip_onnx"):
        mod._load_tokenizer()


# ---------- 集成测试：真模型 + 词表在场才跑 ----------

def _model_available():
    from pathlib import Path
    import vus.clip_onnx as co
    d = Path(co.resolve_model_dir(None))
    return (d / "clip-visual-vitb32.onnx").is_file() and \
        (Path(__file__).resolve().parents[1] / "rvs" / "_vendor"
         / "bpe_simple_vocab_16e6.txt.gz").is_file()


@pytest.mark.skipif(not _model_available(), reason="CLIP ONNX/词表未下载")
def test_integration_clip_tagger_end_to_end(tmp_path):
    labels = ["a person", "a moving car", "an empty corridor"]
    out = build_label_embeddings(labels, NEGATIVE, model_dir=tmp_path)
    assert out.is_file()
    tagger = CLIPTagger(labels, NEGATIVE, model_dir=tmp_path)
    res = tagger.label(np.full((240, 320, 3), 40, np.uint8))
    assert res and all(set(x) == {"label", "score"} for x in res)
