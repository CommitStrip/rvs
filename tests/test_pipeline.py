"""pipeline.py：机器人桥（vus 感知栈 + 本体门控）端到端。"""

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
