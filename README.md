# HALO_Experiment_Replication

本项目为对于论文《Hamiltonian Latent Operators for Content and Motion Disentanglement in Image Sequences》中 Sprites 部分进行代码实现和实验复现的一次尝试，代码架构基于论文的一个[官方实现 repo](https://github.com/MdAsifKhan/HALO) 经修改和完善得到。

## 一、本地复现流程

### 1.1 环境准备

#### 1.1.1 安装 Miniconda（如未安装）

从 https://docs.conda.io/en/latest/miniconda.html 下载并安装 Miniconda（Python 3.10+ 版本）。

#### 1.1.2 创建 Conda 环境

```bash
conda create -n HALO python=3.10
conda activate HALO

# 安装 PyTorch
# CUDA 13.0
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
# 或 CPU 版本
pip install torch torchvision

# 安装其他依赖
pip install pyyaml numpy matplotlib scipy einops tensorboard celluloid opencv-python scikit-learn
pip install hydra-core omegaconf
```

### 1.2 数据集准备

Sprites 数据集是论文的主要实验数据集，包含游戏角色精灵的 walk / spellcard / slash 三种动作。本项目采用[此 repo](https://github.com/YingzhenLi/Sprites) 来生成所需的数据集，该项目已 clone 至 Sprites-master 文件夹。

[`configs/sprites.yaml`](configs/sprites.yaml) 中的 `sprites.path` 需被修改指向 `.npy` 文件所在目录。

### 1.3 修改配置文件

在开始训练之前，需要修改 YAML 配置文件中的路径和 GPU 设置：

#### 关键配置项说明

| 配置项 | 说明 |
|--------|------|
| `cuda` | 是否使用 GPU |
| `gpu_ids` | GPU 编号列表 |
| `sprites.path` | 数据路径 |
| `output_model` | 模型保存路径 |
| `output_results` | 结果保存路径 |
| `trainer.batch_size_train` | 训练批次大小 |
| `trainer.num_epochs` | 训练轮数 |

> **单 GPU 配置示例**：如果只有一张 GPU，将 `gpu_ids` 设置为 `[0, 0, 0]`（三个设备 ID 都指向同一张卡）。

### 1.4 训练模型

```bash
# 进入项目目录
cd [项目目录]

# 开始训练
python train.py --config configs/sprites.yaml --dataname sprites
```

训练流程：
1. 读取配置文件 [`sprites.yaml`](configs/sprites.yaml)
2. 加载 Sprites 数据集
3. 已设置 `pretrain: Flase`，默认不执行预训练
4. 开始完整训练
5. 每 20 个 epoch 保存一次模型
6. 训练 800 个 epoch 后完成

欲恢复中断的训练，需修改配置文件中的以下设置（`pretrain` 应当保持为 `True`）：

```yaml
trainer:
  resume: True
  resume_epoch: <上次保存的 epoch 号>
```

然后再次运行训练命令。

### 1.5 测试与评估

#### 1.5.1 运行完整测试

```bash
python test.py --config configs/sprites.yaml --dataname sprites
```

#### 1.5.2 能量可视化

```bash
python visualiseEnergy.py --config configs/sprites.yaml
```

#### 1.5.3 指定测试轮次

在配置文件中设置 `test_epoch` 字段，指定加载哪个 epoch 的 checkpoint：

```yaml
test_epoch: 800   # 加载第 800 个 epoch 的模型
```

### 1.6 输出文件说明

训练完成后，输出目录结构如下：

```
output_results_<dataset>_.../
├── summary/                    # TensorBoard 日志
├── reconstruct_seq_*.png       # 重建序列可视化
├── generate_seq_*.png          # 生成序列可视化
├── image_to_seq_*.png          # 图像到序列
├── motion_composition_*.png    # 运动组合
├── style_transfer_*.png        # 风格迁移
├── seq_rand_variant_*.png      # 随机变体采样
├── seq_rand_invariant_*.png    # 随机不变采样
├── energy_sample_*.png         # 能量可视化（仅 visualiseEnergy.py）
└── pca_phase_space.png         # 相空间 PCA（仅 visualiseEnergy.py）
```

模型保存目录：

```
output_model_<dataset>_.../
├── model_20.pt
├── model_40.pt
├── model_60.pt
├── ...
└── model_800.pt
```

## 二、复现结果

- 模型能够正常学习到人物形象和动作，并以一定的准确度完成序列重建、动作交换等测试任务
- 精细度和还原度上存在瑕疵，部分人物存在模糊、肤色错误（可能是因为 `condnSonU: False` 导致内容-动作解耦不彻底，但如果设置为 `True` 会导致收敛变慢很多）、局部消失、动作还原不到位等问题，应该为训练轮数不足导致。图像模糊也可能是因为上采样的插值模式选择了 bilinear 而非 nearest
- 能量可视化输出中总能量能够正确地维持不变
