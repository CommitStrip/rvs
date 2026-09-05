#!/usr/bin/env python3
"""gen_label_embeddings.py - 离线生成标签集文本嵌入缓存（CLIPTagger 运行时依赖）。

用法:
  python scripts/gen_label_embeddings.py --labels labels.json [--model-dir ./models]

labels.json 格式:
  {"labels": ["a person walking", "a car", ...],
   "negative_labels": ["background, empty scene, nothing"]}

CLIP ViT-B/32 为英文-图像对齐：标签建议英文，中文展示名由上层映射。
标签集变更后需重新运行本脚本（缓存文件名随标签集哈希变化）。
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rvs.clip_labeler import build_label_embeddings  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", required=True, help="标签集 JSON 文件")
    ap.add_argument("--model-dir", default=None,
                    help="模型目录（默认走 VUS_CLIP_MODELS 或 ./models）")
    args = ap.parse_args()

    cfg = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    labels = cfg.get("labels") or []
    negative = cfg.get("negative_labels") or []
    if not labels:
        raise SystemExit("标签集为空：JSON 需含非空 labels 数组")

    out = build_label_embeddings(labels, negative, model_dir=args.model_dir)
    print(f"标签嵌入缓存已生成: {out}")
    print(f"正标签 {len(labels)} 个 + 负标签 {len(negative)} 个")


if __name__ == "__main__":
    main()
