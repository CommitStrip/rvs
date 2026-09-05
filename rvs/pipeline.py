#!/usr/bin/env python3
"""
pipeline.py - 机器人桥接层：vus 感知栈 + 本体感受门控
rvs 不重造感知轮子，复用清单：
  感知   = vus.smart_pipeline.SmartPipeline（快慢双系统、帧差门控、运动框）
  帧源   = vus.source（FrameSource / FileSource / CameraSource / RTSPSource）
  事件总线 = vus.live.events.EventBus（全量扇出、drop-oldest 背压）
  VLM 空槽 = vus.live.vlm_client.create_vlm("mock" | "openai")
  T0.5 标签 = vus.live.tagger.create_labeler
  落盘   = vus.io_utils
触发式慎思（UnderstandingWorker）的深度接入留待下一迭代（需 vus 侧
加 ego 钩子）；本层先把"自我运动嫌疑"标注打进事件流。

事件契约：vus 原生事件 dict 原样透传；motion_start / motion / motion_end
额外携带 ego_suspect / ego_reason 两个字段——机器人下游据此降权
（转弯/加速窗口内的画面变化大概率是自我运动，不是世界事件）。
"""

import time
from pathlib import Path
from typing import Callable, Optional

import cv2
from vus.live.events import EventBus
from vus.smart_pipeline import SmartPipeline

from .proprio import CommandState, ProprioGate

EgoProvider = Callable[[float], Optional[CommandState]]

#: ego 标注注入的事件类型（快系统运动事件族）
_MOTION_TYPES = ("motion_start", "motion", "motion_end")

#: 可透传给 SmartPipeline 的感知参数键（其余键归本体系/上层）
_PERCEPTION_KEYS = (
    "fast_scale", "motion_thresh", "min_area_ratio", "semantic_gate_ratio",
    "motion_confirm_frames", "motion_window", "keyframe_interval_hz",
    "keyframe_diff", "phash_threshold", "hist_threshold", "dedup_threshold",
)


class RobotPipeline:
    """机器人感知桥（同步逐帧驱动；实时部署由外层按节拍调用 run）。"""

    def __init__(self, source, ego_provider: Optional[EgoProvider] = None,
                 config: Optional[dict] = None,
                 labeler=None, out_dir=None):
        """labeler: T0.5 打标器实例（None=不打标；如 CLIPTagger / BasicLabeler），
        在关键帧时刻调用 label(frame, motion_ratio) 并发布 tag 事件；
        out_dir: 关键帧落盘目录（None=只打标不落盘，打标与落盘解耦）。"""
        cfg = dict(config or {})
        self.source = source
        self.ego_provider = ego_provider
        self.labeler = labeler
        self.out_dir = out_dir
        self.gate = ProprioGate(
            window_s=cfg.get("proprio_window_s", 0.6),
            angular_thresh=cfg.get("proprio_angular_thresh", 0.15),
            linear_thresh=cfg.get("proprio_linear_thresh", 0.1))
        self.perception = SmartPipeline(
            {k: cfg[k] for k in _PERCEPTION_KEYS if k in cfg})
        self.bus = EventBus()

    def run(self, max_frames: Optional[int] = None) -> dict:
        """同步驱动直至源结束或 max_frames；返回摘要 dict。

        回放源（vus FileSource）每次 run 都从头完整回放（open 重置时间轴）。
        """
        started = time.monotonic()
        if not self.source.open():
            return {"ok": False, "error": self._source_error()}
        frames = 0
        ego_suspect_events = 0
        keyframes = 0
        tags = 0
        counts: dict = {}
        last_motion_ratio = 0.0
        try:
            while max_frames is None or frames < max_frames:
                ok, frame, t = self.source.read()
                if not ok:
                    break
                frames += 1
                # 本体感受：控制端接入点（先更新，再对本帧判定）
                if self.ego_provider is not None:
                    cmd = self.ego_provider(t)
                    if cmd is not None:
                        self.gate.update(cmd)
                verdict = self.gate.verdict(t)
                for ev in self.perception.process_frame(frame, t):
                    etype = ev.get("type", "")
                    counts[etype] = counts.get(etype, 0) + 1
                    if "motion_ratio" in ev:
                        last_motion_ratio = ev["motion_ratio"]
                    if etype in _MOTION_TYPES:
                        ev["ego_suspect"] = verdict.ego_suspect
                        ev["ego_reason"] = verdict.reason
                        if verdict.ego_suspect:
                            ego_suspect_events += 1
                    if etype == "keyframe":
                        keyframes += 1
                        kf_path = (self._save_keyframe(frame, keyframes, t)
                                   if self.out_dir is not None else "")
                        # 打标与落盘解耦：不落盘也打标（反射弧热路径）
                        if self.labeler is not None:
                            t0 = time.perf_counter()
                            label_out = self.labeler.label(frame, last_motion_ratio)
                            ms = round((time.perf_counter() - t0) * 1000.0, 1)
                            tags += 1
                            self.bus.publish({
                                "type": "tag", "t": t, "labels": label_out,
                                "source": getattr(self.labeler, "name", "custom"),
                                "ms": ms})
                        self.bus.publish(
                            dict(ev, path=kf_path) if kf_path else ev)
                        continue
                    self.bus.publish(ev)
        finally:
            self.source.close()
        return {"ok": True, "frames": frames, "events": counts,
                "ego_suspect_events": ego_suspect_events,
                "keyframes": keyframes, "tags": tags,
                "motion_segments": len(self.perception.motion_segments),
                "elapsed_s": round(time.monotonic() - started, 3)}

    def _save_keyframe(self, frame, index: int, t: float) -> str:
        """关键帧落盘（vus run_live 同款命名），失败返回空串不阻塞帧循环。"""
        p = Path(self.out_dir) / "keyframes" / ("kf_%04d_t%.1fs.jpg" % (index, t))
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            ok, buf = cv2.imencode(p.suffix or ".jpg", frame)
            if not ok:
                return ""
            buf.tofile(str(p))       # 中文路径安全（imencode+tofile）
            return str(p)
        except OSError:
            return ""

    def _source_error(self) -> str:
        """读帧源错误信息（vus stats 为 dict 快照；兼容对象形态）。"""
        stats = getattr(self.source, "stats", None)
        if isinstance(stats, dict):
            return str(stats.get("last_error", "") or "")
        return str(getattr(stats, "last_error", "") or "")

    def attach_understanding(self, state=None, vlm=None, config=None):
        """挂载 vus 触发式慎思（UnderstandingWorker 订阅本桥事件流）。

        vus 侧 ego_gate 钩子已生效：自我运动嫌疑窗口内的触发被降权，
        素材不丢（合并语义）。返回已 start 的 worker（daemon 线程，
        宿主负责 stop()）；understanding 结果既进 worker.results 也以
        {"type": "understanding", ...} 事件回到本桥总线。
        """
        from vus.live.state import SessionState
        from vus.live.understanding import UnderstandingWorker
        worker = UnderstandingWorker(self.bus, state or SessionState(), vlm,
                                     config=config)
        worker.start()
        return worker

    def proprio_line(self) -> str:
        """本体状态文本行（供慎思素材窗的自运动归因上下文）。"""
        cmd = self.gate.last_cmd
        if cmd is None:
            return ""
        if cmd.turning or abs(cmd.angular_v) > 1e-9:
            act = "左转" if cmd.angular_v > 0 else "右转"
            return "%s %.2frad/s %.2fm/s" % (act, cmd.angular_v, cmd.linear_v)
        if abs(cmd.linear_v) > 1e-9:
            return "直行 %.2fm/s" % cmd.linear_v
        return "静止"
