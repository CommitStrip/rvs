#!/usr/bin/env python3
"""
panorama.py - 全景辅助（外围视觉）纯函数工具
equirectangular 表示下，水平旋转 = 列循环移位（np.roll）——
去旋转补偿成本约等于零，无需任何插值或光流估计。
真实全景相机接入（PanoramaSource 实现）留待后续；本期提供可测的几何工具。
"""

import math
from abc import ABC, abstractmethod
from typing import Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray]


def derotate(img: ArrayLike, shift_px: int) -> np.ndarray:
    """水平循环移位：yaw 变化对应的去旋转（列循环，零插值）。

    shift_px > 0 表示画面内容左移（机器人右转的补偿方向）。
    """
    return np.roll(img, -int(shift_px), axis=1)


def yaw_to_shift(yaw_rad: float, width: int) -> int:
    """equirect 全景宽 2*pi 弧度：yaw 变化量 → 像素列数。"""
    if width <= 0:
        return 0
    return int(round(yaw_rad / (2.0 * math.pi) * width))


def horizontal_band(img: ArrayLike, top: float = 0.3, bottom: float = 0.7) -> np.ndarray:
    """裁水平带（去除 equirect 极区高畸变行），返回行切片副本视图。"""
    h = img.shape[0]
    r0 = max(0, min(h, int(round(top * h))))
    r1 = max(r0, min(h, int(round(bottom * h))))
    return img[r0:r1]


class PanoramaSource(ABC):
    """全景帧源接口空位（真实相机接入后续实现）。

    约定实现方提供 equirectangular 全帧与当前 yaw（弧度）；
    外围运动检测应在 derotate(帧, yaw_to_shift(dyaw, 宽)) 之后执行，
    使静态世界在补偿帧上保持静止、仅真实目标运动产生残差。
    """

    @abstractmethod
    def read_panorama(self) -> Tuple[bool, np.ndarray, float, float]:
        """返回 (ok, equirect_bgr, yaw_rad, timestamp)。"""
        raise NotImplementedError
