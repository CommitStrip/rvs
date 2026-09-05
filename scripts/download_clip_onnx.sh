#!/bin/bash
# download_clip_onnx.sh - 下载 CLIP ViT-B/32 双塔合一 ONNX（MIT 权重）与 BPE 词表
# 用法: bash scripts/download_clip_onnx.sh [model_dir]   (默认 ./models)
set -e
DIR=${1:-./models}
mkdir -p "$DIR" rvs/_vendor
MIRROR=${HF_ENDPOINT:-https://hf-mirror.com}
REPO="onnx-community/clip-vit-base-patch32-ONNX"

# 模型：优先 int8 量化（约 153MB，边缘内存友好），失败回退 fp32（约 605MB）
MODEL="$DIR/clip-visual-vitb32.onnx"   # 沿 vus clip_onnx 的文件名约定
if [ ! -f "$MODEL" ]; then
  for F in model_quantized.onnx model_int8.onnx model.onnx; do
    echo "尝试下载 $MIRROR/$REPO/resolve/main/onnx/$F ..."
    if curl -L --fail -o "$MODEL.part" "$MIRROR/$REPO/resolve/main/onnx/$F"; then
      mv "$MODEL.part" "$MODEL"
      break
    fi
    rm -f "$MODEL.part"
  done
fi
[ -f "$MODEL" ] || { echo "模型下载失败，请检查网络或手动放置到 $MODEL" >&2; exit 1; }
echo "模型就绪: $MODEL ($(du -h "$MODEL" | cut -f1))"

# BPE 词表（约 1.3MB，MIT）
VOCAB="rvs/_vendor/bpe_simple_vocab_16e6.txt.gz"
if [ ! -f "$VOCAB" ]; then
  curl -L --fail -o "$VOCAB" \
    "https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz"
fi
echo "词表就绪: $VOCAB ($(du -h "$VOCAB" | cut -f1))"
echo "下一步: python scripts/gen_label_embeddings.py --labels 标签集.json"
