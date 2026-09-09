#!/usr/bin/env python3
"""intent_ab.py - AR 自我意图叠加的 VLM 注意力 A/B（原创方向验证）。

对比同一运动帧在不同视图变体下 VLM 的归因差异：
  original     无叠加（基线——世界归因只能靠猜）
  arrow_overlay 洋红意图箭头（瞬时意图）
  plan_overlay  运动规划图（轨迹+走廊+时间标记）

可控场景矩阵（逐项跑 --video 单场景素材，记录 path_risk 判读）：
  路径穿过障碍 / 路径擦边 / 计划方向与目标不符 /
  动态物体进入计划走廊 / 原地转向 / 视觉遮挡

需要真实 VLM 端点（凭据全走环境变量，源码无凭据字面量）：
  VLM_API_BASE / VLM_API_KEY / VLM_MODEL
用法:
  python bench/intent_ab.py --video 场景视频.avi --turn left --mode both
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
    ap.add_argument("--mode", choices=["arrow", "plan", "both"], default="plan",
                    help="叠加变体：arrow=意图箭头 / plan=运动规划图 / "
                         "both=两者都对比（A/B 矩阵）")
    args = ap.parse_args()

    frame = _grab_frame(args.video, args.at)
    cmd = CommandState(t=0.0,
                       angular_v=0.8 if args.turn == "left" else (-0.8 if args.turn == "right" else 0.0),
                       linear_v=0.0 if args.turn in ("left", "right") else 0.8,
                       turning=args.turn in ("left", "right"))
    overlay = IntentOverlay()
    frames = {"original": frame}                 # 基线：无叠加
    if args.mode in ("arrow", "both"):
        frames["arrow_overlay"] = overlay.render(frame, cmd)
    if args.mode in ("plan", "both"):
        frames["plan_overlay"] = overlay.render_plan(frame, cmd)
    note = IntentOverlay.prompt_note()

    vlm = create_vlm("openai")  # 端点/密钥/模型全走环境变量
    for name, f in frames.items():
        prompt = (f"你在一台移动机器人上。{note}\n"
                  "只输出 JSON：{\"what\": \"画面在发生什么\", "
                  "\"robot_intent\": \"机器人自己要做什么\", "
                  "\"path_risk\": \"计划路径与画面物体的空间关系\"}")
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
