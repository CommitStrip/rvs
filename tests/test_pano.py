"""panorama.py：equirect 去旋转纯函数。"""

import numpy as np

from rvs.panorama import derotate, horizontal_band, yaw_to_shift


def test_derotate_is_circular_roll():
    img = np.arange(48, dtype=np.uint8).reshape(4, 12, 1).repeat(3, axis=2)
    assert np.array_equal(derotate(img, 3), np.roll(img, -3, axis=1))
    # 循环性：移一整圈回到原样
    assert np.array_equal(derotate(img, 12), img)


def test_yaw_to_shift_mapping():
    assert yaw_to_shift(np.pi, 100) == 50
    assert yaw_to_shift(0.0, 100) == 0
    assert yaw_to_shift(-2.0 * np.pi, 360) == -360
    assert yaw_to_shift(1.0, 0) == 0


def test_horizontal_band_crops_poles():
    img = np.zeros((100, 20, 3), np.uint8)
    band = horizontal_band(img, 0.3, 0.7)
    assert band.shape[0] == 40
    assert band.shape[1] == 20
