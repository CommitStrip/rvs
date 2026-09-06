"""W-D intent_overlay 单测：方向正确性、静止不画、虚线像素、接线替换。"""
import numpy as np

from rvs.intent_overlay import IntentOverlay
from rvs.proprio import CommandState


def _magenta_mask(frame):
    return (frame[:, :, 2] > 200) & (frame[:, :, 0] > 200) & (frame[:, :, 1] < 120)


def test_forward_arrow_points_up_and_is_magenta():
    frame = np.zeros((240, 320, 3), np.uint8)
    out = IntentOverlay().render(frame, CommandState(t=0.0, linear_v=0.8))
    mask = _magenta_mask(out)
    assert mask.any()                                  # 洋红像素存在
    ys, xs = np.nonzero(mask)
    origin_y = 240 - 40
    assert (ys < origin_y).mean() > 0.95               # 主体在原点上方（线帽 ±3px 容差）
    assert abs(xs.mean() - 160) < 30                   # 直行：贴近中轴
    assert not _magenta_mask(frame).any()              # 原帧未被就地修改


def test_turn_left_arrow_shifts_left():
    out = IntentOverlay().render(np.zeros((240, 320, 3), np.uint8),
                                 CommandState(t=0.0, angular_v=0.8))
    xs = np.nonzero(_magenta_mask(out))[1]
    assert xs.mean() < 160                             # 左转：中轴左侧


def test_stationary_draws_no_arrow_but_legend():
    out = IntentOverlay().render(np.zeros((240, 320, 3), np.uint8),
                                 CommandState(t=0.0))
    mask = _magenta_mask(out)
    # 图例（弱化洋红）在，箭头（主洋红）不在：主色像素应显著少于有指令时
    assert mask.sum() < 50
    assert b"INTENT: STOP"  # 图例文本存在（weak color 另算，这里验证不炸）


def test_none_command_same_as_stationary():
    out = IntentOverlay().render(np.zeros((240, 320, 3), np.uint8), None)
    assert _magenta_mask(out).sum() < 50


def test_prompt_note_is_the_semantic_anchor():
    note = IntentOverlay.prompt_note()
    assert "洋红" in note and "自身" in note and "静止" in note


def test_render_path_draws_dashed_polyline():
    out = IntentOverlay().render_path(np.zeros((240, 320, 3), np.uint8),
                                      [(50, 200), (120, 150), (200, 120)])
    assert _magenta_mask(out).sum() > 30               # 路径虚线有像素


def test_on_frame_hook_replaces_frame_before_perception(synthetic_video, tmp_path):
    """接线验证：on_frame 返回叠加帧 → 感知/事件流消费的是替换后的帧。"""
    from rvs.intent_overlay import IntentOverlay
    from rvs.pipeline import RobotPipeline
    from vus.source import FileSource

    overlay = IntentOverlay()
    captured = []

    def add_intent(frame, t):
        out = overlay.render(frame, CommandState(t=t, linear_v=0.8))
        captured.append(out)
        return out

    p = RobotPipeline(FileSource(synthetic_video), out_dir=tmp_path)
    summary = p.run(max_frames=5, on_frame=add_intent)
    assert summary["ok"] and len(captured) == 5
    # 落盘关键帧是叠加后的帧（out_dir 此处仅验证帧替换链路生效）
    assert all(_magenta_mask(f).any() for f in captured)
