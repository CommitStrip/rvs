"""CLIP BPE 分词器（vendor 自 openai/CLIP，MIT 许可）。

来源：https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py
适配（保持算法原样）：
  - ftfy 改为可选依赖：缺失时 basic_clean 退化为不做断字修复
    （仅影响极少数英文连字/断字规范化，不影响常规英文与中文标签）；
  - default_bpe() 指向本目录下的词表文件（scripts/download_clip_onnx.sh 下载）；
  - 新增 tokenize_cliptext()：CLIP 标准提示编码（[BOS]+tokens+[EOS] pad 0）。

SPDX-License-Identifier: MIT
Copyright (c) 2021 OpenAI
"""

import gzip
import html
import os
from functools import lru_cache

import regex as re

_BOS_TOKEN_ID = 49406
_EOS_TOKEN_ID = 49407


@lru_cache()
def default_bpe():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    """ reversible BPE 编码表：utf-8 字节 ↔ 可打印 unicode 字符。"""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word (word = tuple of symbols)."""
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


try:
    import ftfy

    def basic_clean(text):
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        return text.strip()
except ImportError:  # ftfy 可选：缺失时跳过断字修复
    def basic_clean(text):
        text = html.unescape(html.unescape(text))
        return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = default_bpe()):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split("\n")
        merges = merges[1:49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges:
            vocab.append("".join(merge))
        vocab.extend(["<|startoftext|>", "<|endoftext|>"])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {"<|startoftext|>": "<|startoftext|>",
                      "<|endoftext|>": "<|endoftext|>"}
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d"""
            r"""|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(self.encoder[bpe_token]
                              for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def decode(self, tokens):
        text = "".join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode(
            "utf-8", errors="replace").replace("</w>", " ")
        return text


def tokenize_cliptext(texts, ctx_len: int = 77):
    """CLIP 标准提示编码：[BOS] + tokens + [EOS]，pad 0 至 ctx_len。

    返回 int64 数组 [N, ctx_len]（onnxruntime 输入契约）。
    """
    import numpy as np

    tok = SimpleTokenizer()
    out = np.zeros((len(texts), ctx_len), dtype=np.int64)
    for i, text in enumerate(texts):
        tokens = tok.encode(str(text))[: ctx_len - 2]
        out[i, 0] = _BOS_TOKEN_ID
        out[i, 1 + len(tokens)] = _EOS_TOKEN_ID
        out[i, 1:1 + len(tokens)] = tokens
    return out
