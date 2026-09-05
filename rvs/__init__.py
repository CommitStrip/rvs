#!/usr/bin/env python3
"""
rvs - robot-vision-skill
vus（video-understanding-skill）之上的机器人增量层。

vus 已提供：快慢双系统感知、触发式 VLM 慎思（含 mock 空槽）、
T0.5 标签道、帧源族（文件/相机/RTSP）、事件总线、落盘工具。
rvs 只补机器人场景的缺口：

  proprio.py   本体感受——运动指令状态 + 自我运动嫌疑判定
               （指令时间窗 + 速率阈值；事件级标注，不做 warp）
  panorama.py  外围视觉——equirect 全景去旋转纯函数（旋转补偿成本≈0）
  pipeline.py  机器人桥——把 ego 判定接入 vus 感知栈的事件流

接入真实机器人：
  ego_provider(t) -> CommandState   控制端每拍回调（本体感受输入）
  vus.live.vlm_client.create_vlm    慎思后端（"mock" 开箱即跑）
"""

from .panorama import PanoramaSource, derotate, horizontal_band, yaw_to_shift
from .pipeline import RobotPipeline
from .proprio import CommandState, EgoVerdict, ProprioGate

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # 本体系（核心增量）
    "CommandState", "EgoVerdict", "ProprioGate",
    # 机器人桥
    "RobotPipeline",
    # 外围视觉
    "PanoramaSource", "derotate", "yaw_to_shift", "horizontal_band",
]
