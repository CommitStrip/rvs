"""W-C periphery/全景源/双相机编排单测。

核心验收：旋转免疫——静态世界在匀速 yaw 旋转下零事件
（equirect 列循环移位被绝对 yaw 反卷绕精确抵消，无插值误差）。
"""
import cv2
import numpy as np
import pytest

from rvs.panorama import EquirectFileSource, yaw_to_shift
from rvs.periphery import PeripheralMonitor
from rvs.pipeline import DualCameraRig, RobotPipeline
from vus.source import FileSource

W, H = 640, 160


def _world_frame(obj_col=None):
    """世界坐标全景帧：灰底 + 静态地标（列 200-240）+ 可选动目标。"""
    f = np.full((H, W, 3), 40, np.uint8)
    f[60:100, 200:240] = 200
    if obj_col is not None:
        f[60:100, obj_col:obj_col + 24] = 120
    return f


def _pano_view(world, yaw):
    """相机转过 yaw 后看到的画面（世界内容右移）。"""
    return np.roll(world, yaw_to_shift(yaw, W), axis=1)


def test_rotation_immune_static_world_zero_events():
    """核心验收：静态世界 + 匀速旋转 → 零事件（旋转免疫）。"""
    world = _world_frame()
    m = PeripheralMonitor()
    events = []
    for j in range(30):
        yaw = 0.1 * j                      # 每帧转 0.1 rad，共 ~3 rad
        events.extend(m.process(_pano_view(world, yaw), yaw, j / 30.0))
    assert events == []


def test_moving_object_detected_at_world_azimuth():
    """真实运动检出：col_ratio 是去旋转后的世界方位角比例。"""
    m = PeripheralMonitor()
    events = []
    for j in range(24):
        yaw = 0.1 * j
        obj_col = 300 + j * 6              # 动目标在世界坐标移动
        events.extend(m.process(
            _pano_view(_world_frame(obj_col), yaw), yaw, j / 30.0))
    assert len(events) >= 3
    # 目标世界列 300~430 → col_ratio 稳定在该区间（不随 yaw 漂移）
    ratios = [e["col_ratio"] for e in events]
    assert all(0.3 <= r <= 0.75 for r in ratios)
    assert all(e["type"] == "periphery_motion" for e in events)
    assert "yaw" in events[0] and "bbox" in events[0]


def test_monitor_no_event_before_second_frame():
    m = PeripheralMonitor()
    assert m.process(_pano_view(_world_frame(), 0.0), 0.0, 0.0) == []


def test_equirect_file_source_yaw_provider(tmp_path):
    p = tmp_path / "pano.avi"
    vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"XVID"), 30.0, (W, H))
    assert vw.isOpened()
    for j in range(30):
        vw.write(_pano_view(_world_frame(), 0.1 * j))
    vw.release()

    src = EquirectFileSource(p, yaw_provider=lambda t: 0.1 * t)
    assert src.open() is True
    ok, frame, yaw, t = src.read_panorama()
    assert ok and frame.shape == (H, W, 3)
    assert abs(yaw - t * 0.1) < 1e-6       # yaw 由 provider 注入
    n = 1
    while src.read_panorama()[0]:
        n += 1
    assert n == 30
    src.close()


def test_equirect_file_source_constant_rate(tmp_path):
    p = tmp_path / "pano.avi"
    vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"XVID"), 30.0, (W, H))
    for j in range(10):
        vw.write(_pano_view(_world_frame(), 0.2 * j))
    vw.release()
    src = EquirectFileSource(p, angular_rate_rad_s=0.2)
    src.open()
    _, _, yaw, t = src.read_panorama()
    assert abs(yaw - 0.2 * t) < 1e-6       # yaw = yaw0 + rate*t
    src.close()


def test_dual_camera_rig_end_to_end(synthetic_video, tmp_path):
    """主相机全链 + 全景外围：两类事件共存于同一总线。"""
    # 合成全景视频：慢转 + 世界坐标动目标（0.5~2.5s 移动）
    p = tmp_path / "pano.avi"
    vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"XVID"), 30.0, (W, H))
    for i in range(90):
        t = i / 30.0
        obj_col = (300 + int(60 * (t - 0.5))) if 0.5 <= t <= 2.5 else None
        vw.write(_pano_view(_world_frame(obj_col), 0.1 * t))
    vw.release()

    main = RobotPipeline(FileSource(synthetic_video))
    rig = DualCameraRig(main, EquirectFileSource(p, yaw_provider=lambda t: 0.1 * t))
    sub = main.bus.subscribe()
    summary = rig.run()
    evs = sub.drain()
    periph = [e for e in evs if e["type"] == "periphery_motion"]
    assert summary["ok"] and summary["frames"] == 90
    assert summary["periphery_events"] == len(periph) >= 1   # 外围事件进总线
    assert summary["motion_segments"] >= 1                   # 主相机链照常
    assert periph[0]["col_ratio"] > 0.3                      # 世界方位角
