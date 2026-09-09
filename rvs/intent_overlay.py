#!/usr/bin/env python3
"""
intent_overlay.py - 自我意图叠加（原创方向，竞品调研未见占位）
==============================================================
把机器人自身的运动意图渲染进喂给 VLM 的画面——"自我"从画面外
（需要 VLM 推理归因）移到画面内（直接可见）。

学术界的邻近工作方向相反（VLM 输出投影进图像做验证，如 Goal2Pixel）；
本模块是"自我意图作为 VLM 输入"的落地。

骨架版渲染运动指令可视化：洋红虚线箭头指向指令方向（画面底部中央
为原点）。真实规划路径投影（世界坐标 → 图像单应）需要相机标定，
留 render_path(points) 接口供后续注入。

使用约定（必须同步进 VLM 提示，见 prompt_note()）：
  洋红 (#FF00FF) 虚线箭头 = 机器人自身的运动意图，不是世界物体；
  静止时不画箭头。颜色刻意选洋红避开自然场景高频色，防误认。

接线：RobotPipeline.run(on_frame=...) 钩子——在感知前替换帧，
VLM 与感知看到的就是带意图标注的画面。
"""

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from .proprio import CommandState

COLOR_PLAN_BGR = (80, 220, 80)        # 绿色：计划轨迹
COLOR_CORRIDOR_BGR = (120, 255, 160)  # 淡绿：占用走廊
COLOR_TIME_BGR = (255, 160, 60)       # 蓝橙：时间标记


def plan_path(cmd: CommandState, horizon_s: float = 2.0,
              dt: float = 0.25) -> list:
    """差速模型路径预测（机器人坐标系：x 右正、y 前进正，单位米；
    theta 左转（angular_v>0）为正）。返回 [(x, y, t), ...] 含起点；
    纯转向（v≈0）画收敛小弧。
    真实规划器的路径点经 project_path/render_path 注入，不走本模型。
    """
    import math
    pts = []
    x = y = 0.0
    theta = 0.0
    t = 0.0
    v = float(cmd.linear_v)
    w = float(cmd.angular_v)
    while t <= horizon_s + 1e-9:
        pts.append((x, y, round(t, 3)))
        x += -v * math.sin(theta) * dt       # 横向（右为正；左转向左偏）
        y += v * math.cos(theta) * dt        # 前向
        theta += w * dt
        t += dt
    return pts


class IntentOverlay:
    """运动意图渲染器（无状态，可安全共享）。"""

    COLOR_BGR = (255, 0, 255)      # 洋红（BGR）
    COLOR_DIM = (180, 60, 180)     # 弱化洋红（图例）

    def __init__(self, config=None):
        cfg = dict(config or {})
        self.thickness = int(cfg.get("thickness", 3))
        self.length = int(cfg.get("length", 80))
        self.dash = int(cfg.get("dash", 10))          # 虚线段长（像素）
        self.legend = bool(cfg.get("legend", True))   # 右上角图例

    # ---------- 渲染 ----------

    def render(self, frame_bgr: np.ndarray, cmd: Optional[CommandState]) -> np.ndarray:
        """按运动指令在画面上画意图箭头（静止时不画）。返回新帧。"""
        out = frame_bgr.copy()
        if cmd is None:
            return out
        h, w = out.shape[:2]
        origin = (w // 2, h - 40)                     # 底部中央为原点
        direction = self._direction(cmd)              # 单位方向向量（画面系）
        if direction is None:                         # 静止：不画
            if self.legend:
                self._draw_legend(out, "INTENT: STOP")
            return out
        end = (int(origin[0] + direction[0] * self.length),
               int(origin[1] + direction[1] * self.length))
        self._dashed_arrow(out, origin, end)
        if self.legend:
            self._draw_legend(out, "INTENT: " + self._label(cmd))
        return out

    def render_path(self, frame_bgr: np.ndarray,
                    image_points: Sequence[Tuple[int, int]]) -> np.ndarray:
        """真实规划器路径投影（接口）：画面坐标点串 → 绿色虚线。"""
        out = frame_bgr.copy()
        pts = [(int(x), int(y)) for x, y in image_points]
        for a, b in zip(pts[:-1], pts[1:]):
            self._dashed_line(out, a, b, COLOR_PLAN_BGR)
        return out

    def render_plan(self, frame_bgr: np.ndarray, cmd: CommandState,
                    horizon_s: float = 2.0, robot_width_m: float = 0.4,
                    px_per_m: Optional[float] = None,
                    time_marks: Sequence[float] = (0.5, 1.0, 2.0)) -> np.ndarray:
        """行动条件化视图主入口：运动规划图——计划轨迹（绿）+ 机器人
        占用走廊（淡绿带）+ 时间标记点。

        运动学骨架版：差速模型积分生成路径；真实规划器的路径经
        render_path(points) 注入（世界→图像标定由上层完成）。
        静止指令（无位移）退化为意图箭头（render）。
        """
        pts = plan_path(cmd, horizon_s)
        moved = any(abs(x) > 1e-6 or abs(y) > 1e-6 for x, y, _t in pts[1:])
        if not moved:
            return self.render(frame_bgr, cmd)
        h, w = frame_bgr.shape[:2]
        span = max(max(abs(x), abs(y)) for x, y, _t in pts) or 1.0
        scale = float(px_per_m) if px_per_m else (h * 0.55) / span
        origin = (w // 2, int(h * 0.8))
        pixel_pts = [(origin[0] + int(x * scale),
                      origin[1] - int(y * scale), t) for x, y, t in pts]

        out = frame_bgr.copy()
        corridor_px = max(3, int(robot_width_m * scale))
        # 占用走廊：粗半透明带（独立图层 blend，避免遮死画面）
        band = frame_bgr.copy()
        for (ax, ay, _ta), (bx, by, _tb) in zip(pixel_pts[:-1], pixel_pts[1:]):
            cv2.line(band, (ax, ay), (bx, by), COLOR_CORRIDOR_BGR,
                     corridor_px, lineType=cv2.LINE_AA)
        out = cv2.addWeighted(band, 0.35, out, 0.65, 0)
        # 中心轨迹：实线绿色
        for (ax, ay, _ta), (bx, by, _tb) in zip(pixel_pts[:-1], pixel_pts[1:]):
            cv2.line(out, (ax, ay), (bx, by), COLOR_PLAN_BGR,
                     max(2, self.thickness), lineType=cv2.LINE_AA)
        # 时间标记：空心圆 + 秒数
        for (px, py, t) in pixel_pts:
            if any(abs(t - tm) < 1e-6 for tm in time_marks):
                cv2.circle(out, (px, py), max(6, corridor_px // 2),
                           COLOR_TIME_BGR, 2, lineType=cv2.LINE_AA)
                cv2.putText(out, f"{t:.1f}s", (px + 6, py - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            COLOR_TIME_BGR, 1, cv2.LINE_AA)
        # 终点实心标记
        ex, ey, _et = pixel_pts[-1]
        cv2.circle(out, (ex, ey), 5, COLOR_PLAN_BGR, -1)
        if self.legend:
            self._draw_legend(out, "PLAN: " + self._label(cmd))
        return out

    @staticmethod
    def prompt_note() -> str:
        """必须拼进 VLM 提示的约定说明（叠加视觉通道的语义锚）。"""
        return ("【画面约定】洋红色(#FF00FF)虚线箭头是机器人自身的运动意图"
                "（本体的行驶方向），不是场景中的物体；静止时无箭头。")

    # ---------- 内部 ----------

    @staticmethod
    def _direction(cmd: CommandState) -> Optional[Tuple[float, float]]:
        """指令 → 画面单位方向向量（x 右正，y 上正转画面 y 负）。"""
        if cmd.turning or abs(cmd.angular_v) > 1e-9:
            side = -1.0 if (cmd.turning and cmd.angular_v >= 0) or cmd.angular_v > 0 else 1.0
            # 左转（angular_v>0）→ 箭头偏左上；右转 → 偏右上
            mag = min(1.0, abs(cmd.angular_v)) if abs(cmd.angular_v) > 1e-9 else 0.7
            return (side * 0.6 * mag, -0.8)
        if abs(cmd.linear_v) > 1e-9:
            return (0.0, -1.0)                        # 直行：正上
        return None                                   # 静止

    @staticmethod
    def _label(cmd: CommandState) -> str:
        if cmd.turning or abs(cmd.angular_v) > 1e-9:
            return "TURN LEFT" if cmd.angular_v > 0 else "TURN RIGHT"
        return "FORWARD"

    def _dashed_arrow(self, img, start, end):
        self._dashed_line(img, start, end)
        cv2.arrowedLine(img, end, end, self.COLOR_BGR,
                        self.thickness + 2, tipLength=0.001)  # 端点收口
        # 箭头两翼
        import math
        dx, dy = end[0] - start[0], end[1] - start[1]
        ang = math.atan2(dy, dx)
        for da in (2.6, -2.6):
            wing = (int(end[0] + 18 * math.cos(ang + da)),
                    int(end[1] + 18 * math.sin(ang + da)))
            cv2.line(img, end, wing, self.COLOR_BGR, self.thickness)

    def _dashed_line(self, img, start, end, color=None):
        import math
        color = color if color is not None else self.COLOR_BGR
        dx, dy = end[0] - start[0], end[1] - start[1]
        dist = max(1.0, math.hypot(dx, dy))
        steps = max(1, int(dist // self.dash))
        ux, uy = dx / dist, dy / dist
        for i in range(0, steps, 2):                  # 隔段画 → 虚线
            a = (int(start[0] + ux * i * self.dash),
                 int(start[1] + uy * i * self.dash))
            b = (int(start[0] + ux * min(i + 1, steps) * self.dash),
                 int(start[1] + uy * min(i + 1, steps) * self.dash))
            cv2.line(img, a, b, color, self.thickness)

    def _draw_legend(self, img, text: str):
        cv2.putText(img, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, self.COLOR_DIM, 2, cv2.LINE_AA)
