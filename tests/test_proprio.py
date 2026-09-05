"""proprio.py：指令时间窗 + 速率阈值判定。"""

from rvs.proprio import CommandState, ProprioGate


def test_no_proprio_input_never_suspect():
    gate = ProprioGate()
    v = gate.verdict(100.0)
    assert v.ego_suspect is False
    assert v.reason == ""


def test_command_window_suspect():
    gate = ProprioGate(window_s=0.6)
    gate.update(CommandState(t=10.0, angular_v=0.8))
    v = gate.verdict(10.3)
    assert v.ego_suspect is True
    assert v.reason == "command_window"


def test_angular_rate_after_window():
    gate = ProprioGate(window_s=0.6, angular_thresh=0.15)
    gate.update(CommandState(t=10.0, angular_v=0.8))
    v = gate.verdict(10.9)  # 窗外，但角速度仍超阈值
    assert v.ego_suspect is True
    assert v.reason == "angular_rate"


def test_stationary_command_clears_suspect():
    """静止指令不得触发时间窗（每次 update 都会刷新窗口）。"""
    gate = ProprioGate(window_s=0.6)
    gate.update(CommandState(t=10.0, angular_v=0.8))
    gate.update(CommandState(t=11.0))  # 全零指令
    v = gate.verdict(11.3)
    assert v.ego_suspect is False


def test_linear_rate():
    gate = ProprioGate(window_s=0.6, linear_thresh=0.1)
    gate.update(CommandState(t=5.0, linear_v=0.5))
    v = gate.verdict(6.0)
    assert v.ego_suspect is True
    assert v.reason == "linear_rate"


def test_turning_flag_counts_as_angular():
    gate = ProprioGate()
    gate.update(CommandState(t=1.0, turning=True))
    v = gate.verdict(2.0)
    assert v.ego_suspect is True
    assert v.reason == "angular_rate"
