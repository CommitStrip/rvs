# rvs — robot-visual-reflex-system

<div align="center">

[English](README.md) · **简体中文**

![Tests](https://img.shields.io/badge/tests-57%20passed-brightgreen)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)

**[vus](https://github.com/CommitStrip/video-understanding-skill) 之上的机器人视觉反射系统：本体感受门控 + 外围视觉 + 语义反射弧 + 行动条件化感知**

</div>

---

## 为什么需要 rvs

[vus](https://github.com/CommitStrip/video-understanding-skill) 已经解决了机器人的实时视频理解：快慢双系统感知（帧差门控约 1.6 ms/帧）、带费用硬上限的触发式 VLM 慎思、关键帧选择、RTSP 接入、SSE 状态服务。

它没有解决、而每个**移动**机器人都需要的是：知道**画面变化里哪部分是自己造成的**。静置相机可以把"没变化"当"没事发生"；移动机器人的每一帧都在变——事件门控、关键帧选择、VLM 触发的经济性全部失效。

rvs 只补这个缺口，不多做。它**不**重新实现感知、帧源、事件总线、VLM 客户端——这些直接消费 vus。

> 定位说明：运动门控 → 目标检测这条链在 NVR 领域已是标准件。本项目的差异化在**本体感受感知语义层**：事件流上的自我运动归因、外围视觉工具、面向机器人消费者的反射级接口。

## 与既有系统的定位对比

| 系统 | 触发模型 | 成本硬上限 | ASR 通道 | 机器人反射 |
|---|---|---|---|---|
| Frigate（NVR） | 运动门控 → 检测器 | 无（无 VLM） | ✗ | ✗ |
| unblink（Go + VLM） | **固定周期采样**（5 秒采 3 帧，见其 `.env.example` 实锤） | ✗ | ✗ | ✗ |
| 流式视频大模型 | 模型侧：VLM 增量吃所有帧 | 取决于模型 | ✗ | ✗ |
| **vus + rvs** | **事件触发**（运动门控 → 段闭合） | **地板间隔 + 单飞合并** | ✓ | ✓ |

最接近的邻居 unblink 以固定节奏采样喂 VLM——无事件门控、无成本硬上限、无 ASR 通道。本组合位（零训练门控 + 事件触发式 VLM + 成本上限工程 + ASR 对齐 + 机器人反射接口）在公开系统与文献中仍然空着。

## rvs 增加什么

### 1. 本体感受门控（`rvs.proprio`）

机器人的运动是**它自己指令产生的**——已知量，不需要从像素估计。`ProprioGate` 融合两个判据：

- **指令时间窗**——运动指令发出后 `window_s` 秒内的帧标记嫌疑（`command_window`）；
- **速率阈值**——角速度/线速度持续超阈保持嫌疑（`angular_rate` / `linear_rate`）；
- **静止指令默认解除嫌疑**——收到停车指令后，机器人不再主动制造画面变化（工程先验）。注意这是先验而非事实：刹车惯性、车体振动、云台回正仍可能带来残余运动，严格场景应叠加里程计/IMU 零速校验（多源接口 `EgoStateProvider`/`EgoState` 已预留，带 confidence 与多源数据字段）。

判定结果以 `ego_suspect` / `ego_reason` 字段随事件流透传——**不做 warp、不做光流**。感知管线原样不动；下游（反射层、VLM 提示组装、导航）自行决定如何降权被标记的事件。

### 2. 外围视觉工具（`rvs.panorama`）

equirectangular 全景纯函数（辅助低分辨率相机，非主相机）：

- `derotate(img, shift_px)`——yaw 补偿就是**列循环移位**（零插值、成本约等于零）：旋转的世界变成静止的世界；
- `yaw_to_shift(yaw_rad, width)`——角度 → 像素换算；
- `horizontal_band(img)`——裁掉极区高畸变行；
- `PanoramaSource`——真实硬件的接口空位（后续实现）。

### 3. 机器人桥（`rvs.pipeline`）

`RobotPipeline` 把 vus `FrameSource` → `ProprioGate` → vus `SmartPipeline` 接成一条链，向每个运动事件注入 `ego_suspect`/`ego_reason` 并发布到 vus `EventBus`。同时捕获关键帧（可选落盘，与打标解耦），接了打标器时在关键帧诞生瞬间按 vus 契约发布 `tag` 事件。

### 4. 语义反射弧——CLIP 零样本打标（`rvs.clip_labeler`）

`CLIPTagger` 给反射层装上真语义：对**可配置词表**做零样本打标（文本即配置，无需训练），并支持**负标签**（"background"）——无目标帧上负标签得最高分，一个模型同时覆盖打标与目标存在性过滤。

**定位边界**：这是低频语义提示层（场景粗标签、注意力提示、VLM 候选语义），**不是安全关键判断**——零样本分数是标签集内的相对归一化（softmax），不是校准过的存在性概率；标签措辞影响结果，类外目标会被强行归入现有标签。避障/行人安全等必须由几何与传感器安全层承担。

- 引擎：**CLIP ViT-B/32 双塔 ONNX（MIT 权重，商用无阻塞）**。MobileCLIP 快约 4.8×，但其 `apple-amlr` 权重仅限科研——保留为换引擎候选，不作默认。
- 延迟设计：标签集文本嵌入**离线预计算**缓存；热路径只跑视觉塔（经 vus `ClipOnnx` 的 `embed()`，每个关键帧一次）+ 一次矩阵向量乘。
- 打标输入支持**运动框裁剪**（与 motion_crop 流水线同一 box/scale 契约）——反射弧吃放大后的运动区域，不吃整帧。

```python
# 一次性准备（见 scripts/）：
bash scripts/download_clip_onnx.sh            # 模型（int8 约 153MB）+ BPE 词表
python scripts/gen_label_embeddings.py --labels labels.json

# 运行时（热路径只有视觉塔）：
tagger = CLIPTagger(labels=["a person walking", "a moving car"],
                    negative_labels=["background, empty scene"])
pipe = RobotPipeline(source, labeler=tagger, out_dir="out")
# 事件流新增：{"type": "tag", "labels": [...], "source": "clip", "ms": ...}
```

## 架构

```mermaid
flowchart LR
    CMD["机器人控制端<br/>运动指令"] -->|ego_provider| G["ProprioGate<br/>指令时间窗 + 速率阈值"]
    SRC["vus FrameSource<br/>文件 / 相机 / RTSP"] --> P["vus SmartPipeline<br/>快系统门控 + 关键帧"]
    G -->|"ego_suspect 标注"| P
    P -->|"事件 + ego 字段"| BUS["vus EventBus"]
    BUS --> R["机器人消费者<br/>反射 / 导航 / VLM 提示"]
    PANO["rvs panorama<br/>去旋转 + 条带"] -.-> P
```

## 安装

```bash
pip install git+https://github.com/CommitStrip/robot-visual-reflex-system.git
# 依赖 vus 核心库（随依赖传递安装）
```

开发模式：

```bash
git clone https://github.com/CommitStrip/video-understanding-skill.git ../video-understanding-skill
pip install -e ../video-understanding-skill
pip install -e . && python -m pytest tests/
```

## 快速开始

```python
from vus.source import FileSource
from rvs import RobotPipeline, CommandState

# 1) 控制端每个节拍喂运动指令
def ego_provider(t):
    return CommandState(t=t, linear_v=0.8)   # 例如来自轮式里程计 / cmd_vel

# 2) 在任意 vus 帧源上跑桥接
pipe = RobotPipeline(FileSource("day_route.mp4"), ego_provider=ego_provider)
summary = pipe.run()

# 3) 消费事件——vus 契约 + 自我运动归因
sub = pipe.bus.subscribe()
for ev in sub.drain():
    if ev["type"] == "motion" and ev["ego_suspect"]:
        ...  # 大概率自我运动：降权或延迟处理

# 慎思（VLM）空槽——不捆绑模型，自带后端：
# from vus.live.vlm_client import create_vlm
# vlm = create_vlm("mock")            # 确定性、零成本
# vlm = create_vlm("openai", ...)     # 端点经环境变量配置
```

### 5. 行动条件化视图——自我意图叠加（`rvs.intent_overlay`）

把机器人自身的行动意图投影进视觉上下文，让慎思模型在行动前看清"我打算走的路会穿过什么"——不是问"画面里有什么"，而是问"考虑到我要做什么，画面中什么最重要"。

- `render(frame, cmd)`：洋红虚线箭头（瞬时意图方向）；
- **`render_plan(frame, cmd)`（主链）**：差速模型路径预测 → **绿色计划轨迹 + 半透明占用走廊 + 蓝色时间标记（0.5/1/2s）+ 终点**——一张第一视角运动规划图；真实规划器的路径经 `render_path(points)` 注入（世界→图像标定由上层完成）；
- `prompt_note()`：计划视图的双语义约定随 `ego_state` 进 VLM prompt——模型必须知道绿线是渲染的计划标注才能正确归因。

**双通道契约（证据与模型视图严格分离）**：感知（运动检测/CLIP）永远吃**原帧**；意图箭头/导航线只渲染进**叠加帧**，叠加帧经 keyframe 事件的 `path` 进 VLM 素材，原帧另存为 `raw_path` 证据。箭头不得污染运动检测——这是架构红线，不是约定。

## 慎思（VLM）空槽

rvs **不捆绑任何模型**。槽位就是 vus 的后端注册表：`create_vlm("mock")` 离线跑通全链（确定性输出）；`create_vlm("openai")` 对接任意 OpenAI 兼容端点。凭据**只从环境变量读取**——源码、配置、示例中一律不出现。

## 路线图

- [x] T0.5 的 CLIP 反射层：有界词表零样本打标、文本嵌入离线缓存、负标签过滤（`rvs.clip_labeler`）
- [x] 慎思集成：`RobotPipeline.attach_understanding()` 把 vus `UnderstandingWorker` 挂上桥事件流；vus 侧 `ego_gate` 钩子在自我运动嫌疑窗口内延迟触发（素材照收不丢）
- [x] 全景外围视觉：`EquirectFileSource`（yaw 注入/角速度合成）+ `PeripheralMonitor`（绝对 yaw 反卷绕条带帧差，静态世界严格静止）+ `DualCameraRig` 双相机编排——外围 `periphery_motion` 事件带世界方位角（col_ratio），供注意力转移决策
- [x] 自我意图渲染：`IntentOverlay` 把机器人运动指令渲染成画面中的洋红虚线箭头（感知前经 `on_frame` 钩子替换帧），prompt 约定文本同步进 VLM 提示；真实路径投影（世界→图像）留 `render_path` 接口，`bench/intent_ab.py` 提供注意力 A/B 验证脚本（需真实 VLM 端点）
- [ ] 平移的前馈运动补偿（指令运动学 → 单应 warp），恢复持续运动期的事件质量
- [ ] MobileCLIP 换引擎（待许可解除——`apple-amlr` 权重仅限科研）

## 基线对比实测（W-E，调度层口径）

三组基线 × **13.4 分钟 1080p 真实片段**（25fps / 20002 帧），MockVLM 调度层实测（语义质量 A/B 需真实端点，另见 `bench/intent_ab.py`）。复现：`python bench/baseline_experiment.py --run <视频> --out 结果.json`。

| 基线 | VLM 调用 | 送入帧数 | 首个结论(s) | 实时倍速* |
|---|---|---|---|---|
| A vus 触发式（整帧×2，motion_crop 关） | **8** | 9 | 37.8 | 97.8× |
| B 固定周期 5s（unblink 模式） | 161 | 161 | **0.0（首帧即计费）** | 116.9× |
| C rvs 全链（ego 降权 + motion_crop） | **8** | 9 | 37.8 | 90.2× |

*实时倍速为本机 mock 口径（无网络往返），与 vus 压测机 4.7× 口径不可比。

**结论**：
1. **vus 的事件触发较固定周期省 95% VLM 调用**（161→8）：8s 地板间隔兜住费用上限（最坏费用=时长/间隔×单价），安静段零调用；固定周期不看内容，第 0 秒就开始计费。**该收益属于 vus 的调度设计**——rvs ego 降权在本样本的额外调用收益为 0（见第 3 条），其价值主张的正确性由合成与端到端测试保证，尚未由真实移动素材量化。
2. **motion_crop 的 token 中性得到实测验证**：A（整帧）与 C（末槽位裁剪替换）送帧数完全一致（9=9）——裁剪替换不增加帧数，收益在小目标有效分辨率。
3. **ego 降权机制生效但该素材敏感度低**：本素材运动包络下 ego 嫌疑事件 7 个，嫌疑窗口内恰无触发点，调用数未变；降权正确性由测试保证（vus ego_gate 单测 + rvs 全程嫌疑零慎思调用端到端测试）。
4. 合成三段场景（确定性素材）补充：固定周期 12 次 vs 触发式 2 次（−83%），且静止段固定周期照常计费而触发式为零。

## 许可

MIT
