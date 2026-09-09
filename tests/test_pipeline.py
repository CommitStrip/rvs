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
    suspects = [e for e in events if e.get("ego_suspect")]
    # 嫌疑计数含 motion 族与 keyframe（scene_change 在嫌疑窗口内同样标注）
    assert summary["ego_suspect_events"] == len(suspects) == len(motion) + 2
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


# ---------- W-B：慎思层集成（UnderstandingWorker 挂载） ----------

def test_end_to_end_with_understanding_worker(synthetic_video):
    from vus.live.state import SessionState
    from vus.live.vlm_client import MockVLM as VusMock
    state = SessionState()
    vlm = VusMock()
    pipe = RobotPipeline(FileSource(synthetic_video), labeler=_FakeLabeler())
    worker = pipe.attach_understanding(state=state, vlm=vlm)
    sub = pipe.bus.subscribe()
    try:
        summary = pipe.run()
        assert summary["ok"] and summary["tags"] >= 1
        assert worker.wait_idle(timeout=10.0)
        assert len(vlm.calls) >= 1                          # 慎思调用发生
        assert state.snapshot()["t2"]["now"]                # 结论进滚动状态
        und = [e for e in sub.drain() if e["type"] == "understanding"]
        assert und                                          # 结论回流到桥总线
    finally:
        worker.stop()


def test_ego_suspect_defers_deliberation_trigger(synthetic_video, tmp_path):
    """ego 钩子端到端：嫌疑窗口内的关键帧不触发慎思，但素材（落盘帧）不丢。"""
    from vus.live.state import SessionState
    from vus.live.vlm_client import MockVLM as VusMock

    def ego_always_suspect(t):
        return CommandState(t=t, angular_v=0.8)             # 全程嫌疑

    state = SessionState()
    vlm = VusMock()
    pipe = RobotPipeline(FileSource(synthetic_video),
                         ego_provider=ego_always_suspect, out_dir=tmp_path)
    worker = pipe.attach_understanding(state=state, vlm=vlm)
    try:
        pipe.run()
        worker.wait_idle(timeout=10.0)
        assert not vlm.calls                                # 全程嫌疑：零慎思调用
        assert worker._win.kf                               # 带路径的关键帧在窗（不丢）
        assert all(Path(p).exists() for _t, p in worker._win.kf)
    finally:
        worker.stop()


# ---------- W-F：本体状态行通道（ego_state 事件 → 慎思 prompt） ----------

def test_ego_state_published_only_on_change(synthetic_video):
    """本体状态行变化节流：静止→直行→静止 → 恰 2 条 ego_state 事件。"""
    def ego(t):
        if 1.0 <= t < 2.0:
            return CommandState(t=t, linear_v=0.6)
        return CommandState(t=t)                            # 静止

    p = RobotPipeline(FileSource(synthetic_video), ego_provider=ego)
    sub = p.bus.subscribe()
    summary = p.run()
    states = [e for e in sub.drain() if e["type"] == "ego_state"]
    assert len(states) == 3                                 # ""→静止→直行→静止
    assert states[0]["line"] == "静止"
    assert "直行" in states[1]["line"]
    assert len(states) < summary["frames"]                  # 节流：远小于帧数


def test_no_ego_provider_publishes_no_ego_state(synthetic_video):
    p = RobotPipeline(FileSource(synthetic_video))
    sub = p.bus.subscribe()
    p.run()
    assert not [e for e in sub.drain() if e["type"] == "ego_state"]  # 回归


def test_proprio_line_reaches_deliberation_prompt(synthetic_video, tmp_path):
    """端到端：本体状态行经 ego_state 进素材窗、出现在 VLM prompt。"""
    from vus.live.state import SessionState
    from vus.live.vlm_client import MockVLM as VusMock

    def ego(t):
        if 1.0 <= t < 2.0:
            return CommandState(t=t, linear_v=0.6)
        return CommandState(t=t)

    state = SessionState()
    vlm = VusMock()
    pipe = RobotPipeline(FileSource(synthetic_video), ego_provider=ego,
                         out_dir=tmp_path)
    worker = pipe.attach_understanding(state=state, vlm=vlm)
    try:
        pipe.run()
        worker.wait_idle(timeout=10.0)
        assert vlm.calls
        # 通道打通：prompt 出现本体状态行（值为触发时刻的最新状态——
        # 触发点在收尾静止段时报"静止"，语义正确）
        assert any("【本体状态】" in c["prompt"] for c in vlm.calls)
    finally:
        worker.stop()
