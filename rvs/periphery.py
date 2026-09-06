#!/usr/bin/env python3
"""
periphery.py - 外围视觉监视：全景条带低分辨率帧差（旋转免疫）
=============================================================
与主相机快系统的分工：外围只回答"哪个方位有动静"——粗粒度、零模型、
覆盖主相机 FOV 之外的全部方位；语义理解仍归主相机（中央凹）通道。

核心机制：equirect 表示下 yaw 旋转 = 列循环移位，按绝对 yaw 反卷绕后
静态世界严格静止（roll 无损），仅真实目标运动产生残差——旋转免疫
不需要任何光流估计或 IMU（对比：平面相机必须做单应/IMU warp）。

事件（进总线，供注意力转移决策）：
  {"type": "periphery_motion", "t", "yaw", "bbox", "col_ratio",
   "motion_ratio"}
bbox 为去旋转条带上的 [x, y, w, h]；col_ratio = x / 条带宽，即
目标的世界方位角比例（不随机器转向漂移）。
"""

import cv2
import numpy as np

from .panorama import derotate, horizontal_band, yaw_to_shift


class PeripheralMonitor:
    """有状态逐帧处理器（单线程使用）：全景帧 → 去旋转 → 条带帧差。"""

    def __init__(self, config=None):
        cfg = dict(config or {})
        self.downscale = float(cfg.get("downscale", 0.5))
        self.diff_thresh = int(cfg.get("diff_thresh", 20))
        self.min_area_ratio = float(cfg.get("min_area_ratio", 0.001))
        self.band = (float(cfg.get("band_top", 0.25)),
                     float(cfg.get("band_bottom", 0.75)))
        self._prev = None          # 上一帧去旋转条带灰度图

    def process(self, equirect: np.ndarray, yaw: float, t: float):
        """处理一帧全景，返回注意力转移事件列表（0..n 个）。"""
        events = []
        h, w = equirect.shape[:2]
        # 绝对 yaw 反卷绕：roll 无损循环，静态世界跨帧严格静止
        comp = derotate(equirect, yaw_to_shift(yaw, w))
        band = horizontal_band(comp, *self.band)
        if self.downscale < 1.0:
            bh, bw = band.shape[:2]
            band = cv2.resize(band, (max(1, int(bw * self.downscale)),
                                     max(1, int(bh * self.downscale))))
        gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        if self._prev is not None:
            diff = cv2.absdiff(self._prev, gray)
            _, mask = cv2.threshold(diff, self.diff_thresh, 255,
                                    cv2.THRESH_BINARY)
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            bh, bw = gray.shape
            min_area = self.min_area_ratio * bh * bw
            for c in contours:
                area = cv2.contourArea(c)
                if area < min_area:
                    continue
                x, y, cw, ch = cv2.boundingRect(c)
                events.append({
                    "type": "periphery_motion",
                    "t": round(t, 3),
                    "yaw": round(float(yaw), 4),
                    "bbox": [int(x), int(y), int(cw), int(ch)],
                    "col_ratio": round(x / bw, 4) if bw else 0.0,
                    "motion_ratio": round(
                        float(np.count_nonzero(mask)) / float(bh * bw), 5),
                })
        self._prev = gray
        return events
