#!/usr/bin/env python3
"""测试引导：仓库根入 sys.path（免安装可跑）+ 合成视频 fixture。

感知与帧源直接用已安装的 vus（开发机 pip install -e 相邻 vus 仓库）。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture
def synthetic_video(tmp_path_factory):
    """合成视频（vus.FileSource 回读用）：灰底 + 中段移动方块。

    3s @ 30fps；0.5~2.5s 匀速右移——保证运动段闭合、关键帧产出。
    """
    path = tmp_path_factory.mktemp("video") / "synth.avi"
    fps, w, h, dur = 30.0, 320, 240, 3.0
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))
    assert vw.isOpened()
    for i in range(int(fps * dur)):
        frame = np.full((h, w, 3), 40, np.uint8)
        t = i / fps
        if 0.5 <= t <= 2.5:
            x = int(80 * (t - 0.5)) % (w - 24)
            frame[108:132, x:x + 24] = 200
        vw.write(frame)
    vw.release()
    return path
