"""pipeline.py：机器人桥（vus 感知栈 + 本体门控）端到端。"""

from pathlib import Path

from vus.source import FileSource

from rvs.pipeline import RobotPipeline
from rvs.proprio import CommandState


def _run(video_path, ego=None, max_frames=None):
    p = RobotPipeline(FileSource(video_path), ego_provider=ego)
    sub = p.bus.subscribe()          # vus 总线：全量扇出
    summary = p.run(max_frames=max_frames)
    return p, summary, sub.drain()


def test_end_to_end_preserves_vus_contract(synthetic_video):
    p, summary, events = _run(synthetic_video)
    assert summary["ok"] is True
    assert summary["frames"] == 90
    assert summary["motion_segments"] >= 1
    types = {e["type"] for e in events}
    assert {"motion_start", "motion_end", "keyframe"} <= types  # vus 原生契约
    motion = [e for e in events if e["type"] in ("motion_start", "motion",
                                                 "motion_end")]
    assert motion
    # ego 契约字段：无本体输入 → 显式非嫌疑
    assert all(e.get("ego_suspect") is False for e in motion)
    assert all(e.get("ego_reason") == "" for e in motion)


def test_turning_provider_marks_suspects(synthetic_video):
    p, summary, events = _run(
        synthetic_video, ego=lambda t: CommandState(t=t, angular_v=0.8))
    motion = [e for e in events if e["type"] in ("motion_start", "motion",
                                                 "motion_end")]
    assert motion
    assert all(e["ego_suspect"] is True for e in motion)
    assert summary["ego_suspect_events"] == len(motion)
    assert p.proprio_line() == "左转 0.80rad/s 0.00m/s"


def test_stationary_provider_no_suspects(synthetic_video):
    p, summary, events = _run(
        synthetic_video, ego=lambda t: CommandState(t=t))
    motion = [e for e in events if e["type"] in ("motion_start", "motion",
                                                 "motion_end")]
    assert motion
    assert not any(e["ego_suspect"] for e in motion)
    assert summary["ego_suspect_events"] == 0


def test_proprio_line_states():
    p = RobotPipeline(FileSource("不存在的视频.avi"), config={})
    assert p.proprio_line() == ""                  # 无本体输入
    p.gate.update(CommandState(t=0.0, angular_v=0.5))
    assert "左转" in p.proprio_line()
    p.gate.update(CommandState(t=0.0, angular_v=-0.5))
    assert "右转" in p.proprio_line()
    p.gate.update(CommandState(t=0.0, linear_v=0.8))
    assert "直行" in p.proprio_line()
    p.gate.update(CommandState(t=0.0))
    assert p.proprio_line() == "静止"


def test_partial_run_and_full_replay(synthetic_video):
    """回放源契约：partial run 停在 max_frames；重跑从头完整回放
    （vus FileSource open 时重置时间轴）。"""
    src = FileSource(synthetic_video)
    p = RobotPipeline(src)
    s1 = p.run(max_frames=10)
    assert s1["ok"] and s1["frames"] == 10
    s2 = p.run()
    assert s2["frames"] == 90


def test_source_open_failure_reported(synthetic_video):
    p = RobotPipeline(FileSource("不存在的视频.avi"))
    summary = p.run()
    assert summary["ok"] is False
    assert summary["error"]


# ---------- W-A：T0.5 打标接线与关键帧捕获 ----------

class _FakeLabeler:
    """协议对齐的可编程打标器（label -> [{label, score}]）。"""

    name = "fake"

    def __init__(self):
        self.calls = []

    def label(self, frame_bgr, motion_ratio=0.0, **kw):
        self.calls.append({"motion_ratio": motion_ratio,
                           "shape": frame_bgr.shape[:2]})
        return [{"label": "a moving car", "score": 0.9}]


def test_labeler_wiring_publishes_tag_events(synthetic_video, tmp_path):
    lb = _FakeLabeler()
    p = RobotPipeline(FileSource(synthetic_video), labeler=lb, out_dir=tmp_path)
    sub = p.bus.subscribe()
    summary = p.run()
    evs = sub.drain()
    tags = [e for e in evs if e["type"] == "tag"]
    kfs = [e for e in evs if e["type"] == "keyframe"]
    assert summary["ok"] and summary["keyframes"] >= 1
    assert summary["tags"] == summary["keyframes"] == len(lb.calls)
    assert len(tags) == summary["tags"]
    assert tags[0]["source"] == "fake" and "ms" in tags[0]
    assert tags[0]["labels"][0]["label"] == "a moving car"
    assert kfs and all(e.get("path") for e in kfs)          # 落盘并附 path
    for e in kfs:
        assert Path(e["path"]).exists()


def test_labeler_without_out_dir_still_tags(synthetic_video):
    p = RobotPipeline(FileSource(synthetic_video), labeler=_FakeLabeler())
    sub = p.bus.subscribe()
    summary = p.run()
    tags = [e for e in sub.drain() if e["type"] == "tag"]
    assert summary["tags"] >= 1 and tags
    assert all("ms" in e for e in tags)                     # 打标耗时遥测


def test_keyframe_without_out_dir_has_no_path(synthetic_video):
    p = RobotPipeline(FileSource(synthetic_video), labeler=_FakeLabeler())
    sub = p.bus.subscribe()
    p.run()
    kfs = [e for e in sub.drain() if e["type"] == "keyframe"]
    assert kfs and all(not e.get("path") for e in kfs)      # 不落盘：无 path


def test_no_labeler_keeps_vus_contract_backwards_compatible(synthetic_video):
    p = RobotPipeline(FileSource(synthetic_video))
    sub = p.bus.subscribe()
    summary = p.run()
    evs = sub.drain()
    assert summary["tags"] == 0
    assert not [e for e in evs if e["type"] == "tag"]       # 无打标器：无 tag（回归）
