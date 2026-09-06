#!/usr/bin/env python3
"""baseline_experiment.py - W-E 三组基线对比实验（mock VLM，调度层实测）。

三组基线（同一素材、同一 mock 后端，严格可复现）：
  A  vus 原版   ：事件触发 + 整帧×2 编码（motion_crop 关）
  B  固定周期   ：每 interval 秒无条件喂最新整帧（竞品 unblink 模式）
  C  rvs 全链   ：事件触发 + ego 嫌疑降权 + motion_crop 末槽位裁剪
                  + 本体感受注入（静止/直行/转弯任务段）

指标（mock 下可严格测量的调度层口径）：
  calls          VLM 调用次数（成本代理）
  understandings 结论条数
  first_understanding_s  首个结论的素材窗末时间（首响代理）
  frames_fed     送入 VLM 的总帧数（token 代理）
  wall_s / realtime_x    处理耗时与实时倍速
  idle_calls     静止段内的调用次数（固定周期的浪费；触发式应为 0）
  ego_suspect_events     C 组自我运动嫌疑事件数

用法:
  python bench/baseline_experiment.py --gen            # 先生成合成场景
  python bench/baseline_experiment.py --run 视频路径 [--label 合成场景]
凭据零依赖：全程 MockVLM，语义质量 A/B 另见 bench/intent_ab.py。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rvs.pipeline import RobotPipeline  # noqa: E402
from rvs.proprio import CommandState  # noqa: E402
from vus.live.state import SessionState  # noqa: E402
from vus.live.tagger import create_labeler  # noqa: E402
from vus.live.vlm_client import MockVLM, encode_frame_b64  # noqa: E402
from vus.source import FileSource  # noqa: E402

# ---------------- 合成场景（确定性，三段式：静止/常态/关键） ----------------

def gen_scene(path, w=640, h=480, fps=30.0, duration_s=60.0):
    """三段式合成场景：0-20s 静止；20-40s 常态匀速目标；40-60s 快速目标。

    每段背景地标布局不同（pHash 可感知的场景切换 → scene_change 触发源）；
    目标高对比大块（运动信号远超编码噪声，避免快系统段碎裂）。
    """
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))
    if not vw.isOpened():
        raise SystemExit(f"VideoWriter 打开失败: {path}")
    n = int(fps * duration_s)
    landmarks = [  # 每段不同的静态地标布局（场景切换信号）
        [(80, 80, 60, 60), (500, 300, 50, 50)],
        [(300, 60, 60, 60), (80, 360, 50, 50)],
        [(520, 100, 60, 60), (300, 380, 50, 50)],
    ]
    for i in range(n):
        frame = np.full((h, w, 3), 40, np.uint8)
        t = i / fps
        seg = 0 if t < 20 else (1 if t < 40 else 2)
        for (lx, ly, lw, lh) in landmarks[seg]:
            frame[ly:ly + lh, lx:lx + lw] = 90
        if 20.0 <= t < 40.0:                          # 常态：中速横移大目标
            x = int(100 * (t - 20.0)) % (w - 80)
            frame[200:300, x:x + 80] = 220
        if 40.0 <= t < 60.0:                          # 关键：快速目标×2
            x = int(320 * (t - 40.0)) % (w - 80)
            frame[180:280, x:x + 80] = 240
            y = int(300 * ((t - 40.0) * 1.5) % (h - 60))
            frame[y:y + 50, w - 90:w - 10] = 200
        vw.write(frame)
    vw.release()
    return path


# ---------------- 指标收集 ----------------

def _metrics(vlm, state, summary, idle_calls, wall_s, ego_events=0):
    und = state.snapshot()["t2"]["timeline"]
    return {
        "calls": len(vlm.calls),
        "understandings": len(und),
        "first_understanding_s": (round(float(und[0]["end"]), 2)
                                  if und else None),
        "frames_fed": sum(c["n_frames"] for c in vlm.calls),
        "wall_s": round(wall_s, 2),
        "idle_calls": idle_calls,
        "ego_suspect_events": ego_events,
        "realtime_x": round(summary["frames"] / wall_s, 2) if wall_s else None,
    }


# ---------------- 三组基线 ----------------

def run_baseline_a(video, workdir):
    """A：vus 原版——事件触发 + 整帧×2（motion_crop 关）。"""
    state, vlm = SessionState(), MockVLM()
    pipe = RobotPipeline(FileSource(video), config={"motion_crop": False},
                         labeler=create_labeler("basic"),
                         out_dir=Path(workdir) / "A")
    und_events = []
    sub = pipe.bus.subscribe()
    worker = pipe.attach_understanding(state=state, vlm=vlm,
                                       config={"min_call_interval": 8.0})
    t0 = time.monotonic()
    try:
        summary = pipe.run()
        worker.wait_idle(timeout=30.0)
    finally:
        worker.stop()
    wall = time.monotonic() - t0
    und_events = [e for e in sub.drain() if e["type"] == "understanding"]
    m = _metrics(vlm, state, summary, 0, wall)
    m["understandings"] = len(und_events)
    m["first_understanding_s"] = (round(float(und_events[0]["t"]), 2)
                                  if und_events else None)
    return m


def run_baseline_b(video, workdir, interval_s=5.0):
    """B：固定周期采样（竞品 unblink 模式）——每 interval 无条件喂最新帧。

    编码口径与 A/C 对齐：帧先落盘再走 encode_frame_b64(path)。
    """
    state, vlm = SessionState(), MockVLM()
    src = FileSource(video)
    if not src.open():
        raise SystemExit("视频打开失败")
    kf_dir = Path(workdir) / "B" / "keyframes"
    kf_dir.mkdir(parents=True, exist_ok=True)
    idle_calls = 0
    t0 = time.monotonic()
    last_call_t = None
    frames = 0
    first_und_t = None
    n = 0
    try:
        while True:
            ok, frame, t = src.read()
            if not ok:
                break
            frames += 1
            state.apply_frame_event({"type": "motion", "t": t, "boxes": []})
            if last_call_t is None or t - last_call_t >= interval_s:
                n += 1
                kf = kf_dir / ("kf_%04d.jpg" % n)
                okj, buf = cv2.imencode(".jpg", frame)
                buf.tofile(str(kf))                       # 中文路径安全落盘
                b64 = encode_frame_b64(kf, max_side=448)
                vlm.understand("固定周期采样", (b64,) if b64 else ())
                last_call_t = t
                if first_und_t is None:
                    first_und_t = t
    finally:
        src.close()
    wall = time.monotonic() - t0
    return {"calls": len(vlm.calls), "understandings": len(vlm.calls),
            "first_understanding_s": (round(first_und_t, 2)
                                      if first_und_t is not None else None),
            "frames_fed": sum(c["n_frames"] for c in vlm.calls),
            "wall_s": round(wall, 2), "idle_calls": idle_calls,
            "ego_suspect_events": 0,
            "realtime_x": round(frames / wall, 2) if wall else None,
            "_frames": frames}


def run_baseline_c(video, workdir, motion_phase=(0.0, 1.0)):
    """C：rvs 全链——触发式 + ego 降权 + motion_crop + 本体感受注入。

    ego 任务包络：前 1/4 静止 → 中段匀速直行 → 转弯 → 末 1/4 静止
    （收尾静止保证段闭合存在非嫌疑触发点——机器人停机即恢复慎思）。
    """
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1
    cap.release()
    dur = total / fps

    def ego(t):
        p = t / max(dur, 1e-6)
        if p < 0.25 or p >= 0.75:
            return CommandState(t=t)                      # 静止（任务首尾）
        if p < 0.5:
            return CommandState(t=t, linear_v=0.6)        # 直行
        return CommandState(t=t, angular_v=0.5)           # 转弯

    state, vlm = SessionState(), MockVLM()
    pipe = RobotPipeline(FileSource(video), ego_provider=ego,
                         labeler=create_labeler("basic"),
                         out_dir=Path(workdir) / "C")
    worker = pipe.attach_understanding(state=state, vlm=vlm,
                                       config={"min_call_interval": 8.0})
    t0 = time.monotonic()
    try:
        summary = pipe.run()
        worker.wait_idle(timeout=30.0)
    finally:
        worker.stop()
    wall = time.monotonic() - t0
    return _metrics(vlm, state, summary, 0, wall,
                    ego_events=summary.get("ego_suspect_events", 0))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gen", action="store_true", help="生成合成场景视频")
    ap.add_argument("--run", help="对指定视频跑三组基线")
    ap.add_argument("--label", default="video", help="素材标签（输出用）")
    ap.add_argument("--out", default=None, help="结果 JSON 落盘路径")
    args = ap.parse_args()

    if args.gen:
        p = gen_scene(Path("scene_synth.avi"))
        print(f"合成场景已生成: {p}")
        return

    video = args.run
    if not video:
        raise SystemExit("需要 --run 视频路径（或先用 --gen 生成素材）")
    workdir = Path("bench_work")
    workdir.mkdir(exist_ok=True)

    print(f"[{args.label}] 基线 A（vus 原版/整帧）...", flush=True)
    a = run_baseline_a(video, workdir)
    print(f"[{args.label}] 基线 B（固定周期 {5.0}s/unblink 模式）...", flush=True)
    b = run_baseline_b(video, workdir)
    print(f"[{args.label}] 基线 C（rvs 全链/ego+motion_crop）...", flush=True)
    c = run_baseline_c(video, workdir)

    result = {"source": args.label, "A_vus_baseline": a,
              "B_fixed_interval": b, "C_rvs_full": c}
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"已落盘: {args.out}")


if __name__ == "__main__":
    main()
