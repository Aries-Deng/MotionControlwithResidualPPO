# Differential Drive and Unitree G1 Motion Control Experiments

本项目是在原三部分 homework 框架上扩展出的运动控制实验集合。整体目标是比较两类具身平台上的三种控制思路：传统 PID、端到端 PPO，以及 PID 与残差 PPO 的混合控制。

实验分为两个系统：

- `DifferentialDrivePathTracking-master/`：差速驱动小车路径跟踪实验，在低维运动学模型上实现并比较 PID、PPO、PID + residual PPO。
- `scripts/` 与 `mjlab/`：Unitree G1 人形机器人实验，在 MuJoCo/mjlab 环境中实现单段舞蹈动作跟踪。

## Repository Layout

```text
.
├── DifferentialDrivePathTracking-master/
│   ├── main.py                         # 差速小车 PID 路径跟踪
│   ├── ppo_path_tracking.py            # 差速小车端到端 PPO 路径跟踪
│   ├── ppo_pid_residual_path_tracking.py
│   │                                   # 差速小车 PID + 残差 PPO 路径跟踪
│   ├── Parkour.py                      # 原项目可视化辅助代码
│   ├── outputs/                        # 差速小车实验图像、模型和误差曲线
│   └── README.md                       # 原差速小车项目说明
├── scripts/
│   ├── ppo.py                          # 原 homework 的 NumPy PPO 基础练习
│   ├── dance_single.py                 # Unitree G1 单段舞蹈 PPO 跟踪
│   ├── dance_pid_baseline.py           # Unitree G1 舞蹈 PID/PD baseline 与 hybrid 评估入口
│   ├── dance_pid_residual_ppo.py       # Unitree G1 PID + residual PPO
│   ├── motion/g1_hiphop_tracking.npz   # G1 hiphop 参考动作数据
│   ├── outputs/                        # G1 舞蹈 PID / PPO / hybrid 对比输出
│   ├── model_final.pt                  # G1 舞蹈 PPO 训练结果
│   └── pid_residual_model_final.pt     # G1 舞蹈 residual PPO 训练结果
├── mjlab/
│   ├── dance/                          # Unitree G1 舞蹈跟踪环境配置
│   ├── tracking_rewards.py             # 速度/动作跟踪 reward 工具
│   ├── train.py                        # rsl_rl PPO 训练封装
│   └── evaluate.py                     # rollout、误差图和视频导出工具
├── outputs/                            # homework 根目录输出备份
├── Assignment3.pdf                     # 原作业说明
└── requirements.txt                    # Python 依赖
```

## Experiments

### 1. Differential Drive: PID Path Tracking

入口文件：

```shell
python DifferentialDrivePathTracking-master/main.py
```

该实验使用差速小车运动学模型，将若干带时间戳的关键点采样为参考轨迹。PID 控制器根据 lookahead 目标点计算角速度，并按参考速度前进。

主要输出：

- `DifferentialDrivePathTracking-master/outputs/differential_drive_tracking_error.png`
- `DifferentialDrivePathTracking-master/outputs/differential_drive_tracking_path.png`
- `DifferentialDrivePathTracking-master/outputs/differential_drive_heading_tracking.png`

### 2. Differential Drive: PPO Path Tracking

入口文件：

```shell
python DifferentialDrivePathTracking-master/ppo_path_tracking.py
```

该实验将路径跟踪建模为强化学习任务。策略网络根据当前位置误差、lookahead 误差、航向误差、参考速度和轨迹相位，直接输出速度修正与角速度控制。

主要输出：

- `DifferentialDrivePathTracking-master/outputs/ppo_differential_drive_model.pt`
- `DifferentialDrivePathTracking-master/outputs/ppo_differential_drive_tracking_error.png`
- `DifferentialDrivePathTracking-master/outputs/ppo_differential_drive_tracking_path.png`
- `DifferentialDrivePathTracking-master/outputs/ppo_differential_drive_heading_tracking.png`
- `DifferentialDrivePathTracking-master/outputs/ppo_training_reward.png`

### 3. Differential Drive: PID + Residual PPO

入口文件：

```shell
python DifferentialDrivePathTracking-master/ppo_pid_residual_path_tracking.py
```

该实验使用 PID 作为名义控制器，PPO 只学习叠加在 PID 输出上的有界残差。这样可以保留 PID 的稳定先验，同时让学习策略补偿转弯、速度变化和时序误差带来的偏差。

主要输出：

- `DifferentialDrivePathTracking-master/outputs/ppo_pid_residual_differential_drive_model.pt`
- `DifferentialDrivePathTracking-master/outputs/ppo_pid_residual_tracking_error.png`
- `DifferentialDrivePathTracking-master/outputs/ppo_pid_residual_tracking_path.png`
- `DifferentialDrivePathTracking-master/outputs/ppo_pid_residual_heading_tracking.png`
- `DifferentialDrivePathTracking-master/outputs/ppo_pid_residual_commands.png`
- `DifferentialDrivePathTracking-master/outputs/ppo_pid_residual_training_reward.png`

### 4. Unitree G1: Single Dance Tracking with PPO

入口文件：

```shell
python -m scripts.dance_single
```

该实验使用 `scripts/motion/g1_hiphop_tracking.npz` 作为参考动作。策略观测相位条件下的参考关节状态、根部速度误差和机器人本体状态，通过多项 tracking reward 学习单段舞蹈动作跟踪。

主要输出：

- `scripts/model_final.pt`
- `scripts/dance_single_error.png`
- `scripts/dance_single_video.mp4`

也可以使用已有 checkpoint 直接评估：

```shell
conda run --no-capture-output -n mjlab python - <<'PY'
from scripts.dance_single import play
play("scripts/model_final.pt")
PY
```

### 5. Unitree G1: PID/PD Dance Baseline and Hybrid Evaluation

入口文件：

```shell
conda run --no-capture-output -n mjlab python scripts/dance_pid_baseline.py
```

该文件包含两个用途：

- 用参考关节轨迹直接生成 normalized joint-position action，形成 PID/PD 风格的开环/外环 tracking baseline。
- 在已有 PPO checkpoint 上导出 hybrid/PPO 相关对比结果。

主要输出：

- `scripts/outputs/dance_pid_error.png`
- `scripts/outputs/dance_pid_video.mp4`
- `scripts/outputs/dance_hybrid_pid_error.png`
- `scripts/outputs/dance_hybrid_pid_video.mp4`
- `scripts/outputs/g1_ppo_vs_hybrid_pid_tracking_error.png`

## Installation

基础依赖：

```shell
conda create -n motion_control python=3.10
conda activate motion_control
pip install -r requirements.txt
```

`requirements.txt` 中包含：

```text
numpy
scipy
matplotlib
mediapy
mjlab>=1.1.1
torch
```

差速小车实验可以在 CPU 上运行。Unitree G1 的 PPO 训练建议使用带 GPU 的 `mjlab` 环境。

如果已经配置了课程使用的 `mjlab` 环境，可以直接：

```shell
conda activate mjlab
```

## Notes

- 差速小车部分是低维运动学实验，适合观察 PID、PPO 和 residual learning 在路径跟踪误差上的差异。
- Unitree G1 部分是高自由度全身控制实验，控制难点从二维路径跟踪扩展到动态平衡、关节耦合和参考动作可行性。
- `scripts/` 保留了原 homework 的 NumPy PPO 基础练习，并扩展了 Unitree G1 舞蹈跟踪实验。
- `outputs/` 和各子目录中的 `outputs/` 保存了已生成的误差曲线、路径图和视频，便于直接查看实验结果。
