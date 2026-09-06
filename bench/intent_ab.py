#!/usr/bin/env python3
"""intent_ab.py - AR 自我意图叠加的 VLM 注意力 A/B（原创方向验证）。

对比同一运动帧的「原帧 vs 意图叠加帧」在 VLM 输出中的归因差异：
叠加帧的回复应正确提及机器人自身意图（如"机器人正要左转"），
原帧则只能从画面内容猜测。需要真实 VLM 端点（凭据全走环境变量，
源码无凭据字面量）。

环境变量：VLM_API_BASE / VLM_API_KEY / VLM_MODEL
用法:
  python bench/intent_ab.py --video 场景视频.avi --turn left
输出：两组回复原文 + 简单关键词归因比对（人工判读为主）。
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from rvs.intent_overlay import IntentOverlay  # noqa: E402
from rvs.proprio import CommandState  # noqa: E402
from vus.live.vlm_client import create_vlm, encode_frame_b64  # noqa: E402


def _grab_frame(video: str, at_s: float):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * at_s))  # 离线取帧，允许 seek
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"取帧失败 @{at_s}s")
    return frame


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--at", type=float, default=1.0, help="取帧时刻（秒）")
    ap.add_argument("--turn", choices=["left", "right", "forward"], default="left")
    args = ap.parse_args()

    frame = _grab_frame(args.video, args.at)
    cmd = CommandState(t=0.0,
                       angular_v=0.8 if args.turn == "left" else (-0.8 if args.turn == "right" else 0.0),
                       linear_v=0.0 if args.turn in ("left", "right") else 0.8,
                       turning=args.turn in ("left", "right"))
    overlay = IntentOverlay()
    frames = {"original": frame, "with_intent": overlay.render(frame, cmd)}
    note = IntentOverlay.prompt_note()

    vlm = create_vlm("openai")  # 端点/密钥/模型全走环境变量
    for name, f in frames.items():
        prompt = (f"你在一台移动机器人上。{note}\n"
                  "只输出 JSON：{\"what\": \"画面在发生什么\", "
                  "\"robot_intent\": \"机器人自己要做什么\"}")
        text = vlm.understand(prompt, (encode_frame_b64_frame(f),), timeout=60)
        print(f"===== {name} =====")
        print(text.strip() or "（空回复）")
        print()


def encode_frame_b64_frame(frame):
    """内存帧 → b64（bench 内联小工具，不走落盘）。"""
    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    import base64
    return base64.b64encode(jpg.tobytes()).decode("ascii") if ok else None


if __name__ == "__main__":
    main()
