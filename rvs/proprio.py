#!/usr/bin/env python3
"""
proprio.py - 本体感受通道
机器人运动指令状态（CommandState）与自我运动判定（ProprioGate）。

设计立场：机器人的运动是它自己发出的——控制指令是已知量，不需要
从画面估计。帧差门控不做运动补偿 warp，而是把"此刻画面变化可能由
自我运动引起"的判定作为事件级标注（ego_suspect）交给下游降权。
已知权衡：匀速运动期间所有帧差都判嫌疑（保守），恢复事件能力需
后续接运动补偿 warp（见 README 路线图）。
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CommandState:
    """控制端在时刻 t 的运动指令状态（本体感受输入）。"""

    t: float
    linear_v: float = 0.0     # m/s，前进为正
    angular_v: float = 0.0    # rad/s，左转（逆时针）为正
    turning: bool = False     # 控制端显式转向标志（可选）


@dataclass
class EgoVerdict:
    """快系统对某帧的自我运动判定（随事件透传给下游）。"""

    ego_suspect: bool = False
    reason: str = ""          # command_window / angular_rate / linear_rate


class ProprioGate:
    """指令时间窗 + 速率阈值双判据。

    - 无本体输入（从未 update）→ 永远非嫌疑（静置相机语义，向后兼容）；
    - 指令发出后 window_s 秒内 → 嫌疑（reason=command_window）；
    - 窗外但角速度超阈值或控制端声明转向 → 嫌疑（reason=angular_rate）；
    - 线速度超阈值 → 嫌疑（reason=linear_rate）。
    """

    def __init__(self, window_s: float = 0.6,
                 angular_thresh: float = 0.15,
                 linear_thresh: float = 0.1):
        self.window_s = float(window_s)
        self.angular_thresh = float(angular_thresh)
        self.linear_thresh = float(linear_thresh)
        self.last_cmd: Optional[CommandState] = None

    def update(self, cmd: CommandState) -> None:
        """控制端每拍调用（指令发布处），指令时间窗从此刷新。"""
        self.last_cmd = cmd

    def verdict(self, t: float) -> EgoVerdict:
        cmd = self.last_cmd
        if cmd is None:
            return EgoVerdict(False, "")
        moving = bool(cmd.turning
                      or abs(cmd.angular_v) > 1e-9
                      or abs(cmd.linear_v) > 1e-9)
        if not moving:
            # 静止指令：画面变化与自我运动无关，不进嫌疑判据
            # （每次 update 都会刷新时间窗，静止指令不得触发窗口）
            return EgoVerdict(False, "")
        if t - cmd.t <= self.window_s:
            return EgoVerdict(True, "command_window")
        if cmd.turning or abs(cmd.angular_v) >= self.angular_thresh:
            return EgoVerdict(True, "angular_rate")
        if abs(cmd.linear_v) >= self.linear_thresh:
            return EgoVerdict(True, "linear_rate")
        return EgoVerdict(False, "")
