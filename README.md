# MyDepth

利用 BIM 深度先验校正单目深度预测的简洁研究代码。当前主要方案是 `global scale + disagreement adapter（3 ResBlocks）+ R36`，同时保留 dense2dense、多级 residual 和 DA3 预训练特征的独立实验入口。

结果更新时间：2026-09-16。下文只汇总本项目 `outputs/` 中当前保存的结果，不将 validation 当成 test，也不将尚未完成的实验当成最终结果。

## 1. 环境配置

### 本机运行环境

当前使用 Conda 环境 `priorbimda`，GPU 为 NVIDIA RTX 4000 Ada Generation，显存约 20 GB。

| 依赖 | 当前版本 |
|---|---|
| Python | 3.10.21 |
| PyTorch / CUDA wheel | 2.6.0+cu124 / CUDA 12.4 |
| torchvision | 0.21.0+cu124 |
| transformers | 4.53.2 |
| NumPy | 1.26.4 |
| OpenCV | 4.11.0.86 |
| xformers | 0.0.29.post3 |
| Open3D / trimesh / IfcOpenShell | 0.19.0 / 5.1.0 / 0.8.5 |
| S3-SAM3D-ToolKit | 0.1.0，本地源码 |

### 新环境参考安装

建议将三个项目放在同一父目录：`MyDepth/`、`S3-SAM3D-ToolKit/`、`Depth-Anything-3/`。后两个分别是自用工具箱和 [DA3 官方项目](https://github.com/ByteDance-Seed/Depth-Anything-3)。以下命令在 `MyDepth/` 下执行：

```bash
conda create -n mydepth python=3.10 -y
conda activate mydepth

python -m pip install torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124
python -m pip install xformers==0.0.29.post3
python -m pip install numpy==1.26.4 transformers==4.53.2 \
    opencv-python==4.11.0.86 tqdm huggingface-hub safetensors trimesh

python -m pip install -e "../S3-SAM3D-ToolKit[exr,ifc]"
python -m pip install -e ../Depth-Anything-3
```

### UniDepthV2 推理依赖

在本机 `priorbimda` 环境中，ViT-L 推理所缺的包可单独安装，保留该环境已有的
PyTorch 2.6.0+cu124、xFormers 和 NumPy 1.26.4：

```bash
conda activate priorbimda
python -m pip install timm==1.0.19 'wandb>=0.19,<0.20'
python -m pip install --no-deps \
    'git+https://github.com/lpiccinelli-eth/UniDepth.git@8d8cfe4c7ee15297099983607febf0d4f32eb3d6'
python -c 'from unidepth.models import UniDepthV2; print(UniDepthV2.__name__)'
```

这里安装的是评测脚本所需的推理路径。UniDepth 官方包还声明了 Gradio、HDF5、
Torchaudio 等训练/演示依赖及 NumPy 2.x；因此这个保留现有版本的环境执行
`pip check` 时会报告这些未安装依赖和 NumPy 版本差异。KNN 扩展仅用于 UniDepth
自带的 3D 评测，当前 `zero_shot_eval.py` 的深度指标不调用它。ViT-L 权重尚未缓存时，
首次运行 `zero_shot_eval.py` 需加 `--allow-network` 下载权重。

PyTorch 安装命令参考 [官方历史版本说明](https://pytorch.org/get-started/previous-versions/)。DA3 安装会带入其自身依赖；本项目不需要 Gradio 或 3D Gaussian rendering，因此无需安装 DA3 的 `app` / `gs` 扩展。上面是参考安装流程，本次核对的是已有环境，未重新建立全新环境验证。

如果已有环境只使用本地源码而未完成 editable 安装，可设置：

```bash
export PYTHONPATH="/home/bgao491/Depth-Anything-3/src:/home/bgao491/S3-SAM3D-ToolKit/src${PYTHONPATH:+:${PYTHONPATH}}"
python -c "import torch, transformers, depth_anything_3, s3dis_sam3d; print(torch.__version__, torch.cuda.is_available())"
```

### 预训练权重

| 用途 | Hugging Face 模型 | 固定 revision |
|---|---|---|
| 原模型的 encoder / DPT | `depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf` | `9560f57a2f07803ba353bb918d6a6e5e005b9277` |
| 缓存的 DA3 深度，以及 DA3 特征实验 | `depth-anything/da3metric-large` | `4010e39f3634a45bc60553321fb49fb760bd594e` |

首次使用需下载相应权重。缓存齐全后可传 `--local-files-only`，禁止联网下载。所有方案的深度 anchor 都是 DA3；“DAv2 / DA3 模型”指用于提取校正特征的预训练网络不同，不代表原模型使用 DAv2 深度作为 anchor。

## 2. 数据与路径

### 训练数据

目前训练入口的默认路径为：

| 内容 | 本机路径 |
|---|---|
| Area1 real BIM 预制数据 | `/mnt/priorbimda-data/area1_priorbimda_504` |
| Area2–5 SyncBIM 预制数据 | `/mnt/priorbimda-data/s23_syncbim_area2_5_504` |
| Stanford 2D-3D-S 原始 RGB / semantic | `/home/bgao491/Stanford2D3DS/no_xyz` |

换机器时使用 `--dataset-root`、`--s23-root` 和 `--extra-dataset-root` 修改路径；训练入口不会自动用工具箱 config 替换这两个默认训练路径。

最小数据格式为：

```text
dataset_root/
├── manifests/
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
└── samples/
    └── .../*.npz
```

manifest 记录样本路径、RGB 路径、room 和 split；NPZ 保存 `intrinsic`、`da3_depth_raw`、`da3_focal_scale`、`bim_depth`、`bim_valid`、`gt_depth`、`gt_valid` 等字段。RGB 不重复存入 NPZ，读取时从原始数据加载并转换到 `[0, 1]`。所有训练输入为 `504×504`，DA3 metric 深度按 `da3_depth_raw × da3_focal_scale` 重建。

Area1 按 room 划分 train / val / test，当前样本数分别为 **7013 / 1673 / 1641**。额外根目录仅用于训练，不改变 Area1 validation / test。`extra_dataset_stride` 对每个额外根目录的 `train.jsonl` 执行间隔取样，主数据不取样；这与制备时的 `--frame-stride` 是两个不同阶段的参数。

| 额外数据 stride | SyncBIM 样本 | 合计训练样本 |
|---:|---:|---:|
| 不使用额外数据 | 0 | 7013 |
| 1 | 45757 | 52770 |
| 3 | 15253 | 22266 |
| 6 | 7627 | 14640 |
| 8 | 5720 | 12733 |

SyncBIM 全量构成为 Area2：14456，Area3：3356，Area4：11970，Area5a：6033，Area5b：9942。

这里“合成”的是 BIM 先验，不是 RGB 或 GT 深度：工具箱从带语义的扫描几何拟合墙、地板、天花板等结构，生成 SyncBIM mesh，再从真实相机位姿渲染 BIM depth。制备入口为 [prepare_s23_syncbim.py](../S3-SAM3D-ToolKit/script/prepare_s23_syncbim.py)，输出可直接交给本项目 dataset。

```bash
python ../S3-SAM3D-ToolKit/script/prepare_s23_syncbim.py \
    --s23-root /home/bgao491/Stanford2D3DS/no_xyz \
    --output-root /mnt/priorbimda-data/s23_syncbim_area2_5_504 \
    --areas 2 3 4 5 --device cuda --local-files-only
```

`Area_5` 自动展开为 `Area_5a` 和 `Area_5b`。当前数据默认制备 stride=1，BIM hit fraction 至少为 0.2；制备信息保存在额外数据根目录的 `syncbim.json`。

### Zero-shot 数据

zero-shot 使用工具箱的 `MP3D_BIMDataset(default_mesh_source="obj_wall_filled")`，路径从工具箱 config 读取，无需给训练命令额外指定 Matterport3D / BIMNet 路径。配置保存在 `~/.s3dis_sam3d.json`，换机器时执行：

```python
from s3dis_sam3d import configure

configure(
    bimnet_root="/your/path/BIMNet_release",
    matterport_root="/your/path/Matterport3D/v1/scans",
)
```

BIMNet 需包含对应的配准结果和 wall-filled OBJ，配置根目录本身不等于已完成几何制备。DA3 预测缓存默认位于 `outputs/zero_shot_raw/da3_cache`，可供不同实验共用。

## 3. 模型与训练

| 模型 | 训练入口 | 主要区别 |
|---|---|---|
| [mymodel.py](model/mymodel.py) | [train.py](train.py) | DAv2 Base，global scale + adapter（3res）+ R36 |
| [dense2dense.py](model/dense2dense.py) | [train_dense2dense.py](train_dense2dense.py) | 无独立 global scale；F144 经空间 head 输出完整 504 分辨率 log-scale，当前不设 tanh 限幅 |
| [mymodel_r36_r72_r144.py](model/mymodel_r36_r72_r144.py) | [train_r36_r72_r144.py](train_r36_r72_r144.py) | global scale + R36/R72/R144，各级独立 adapter（3res），逐级传递校正后的 disagreement |
| [mymodel_da3.py](model/mymodel_da3.py) | [train_dav3.py](train_dav3.py) | 原单级 R36 逻辑，预训练特征改为 DA3Metric-Large；token 1024 维、DPT 256 通道 |

注意命名：DA3 模型文件是 `mymodel_da3.py`，训练文件是 `train_dav3.py`。

### 单级 R36 baseline

BIM 条件包含归一化 log-BIM、valid mask、BIM 与 DA3 的 log-disagreement，经零初始化 patch projection 加入 RGB tokens。最终 CLS 与 patch mean 拼接预测每帧 global log-scale。adapter 使用减去 `log_scale.detach()` 后的 disagreement，在 F36 上预测局部 log-residual。

```text
D_scaled = D_DA3 × exp(s)
r_full   = Upsample(0.1 × tanh(r36_logits))
D_final  = clamp(D_scaled × exp(r_full), 1e-3, 128)
```

BIM patch projection、adapter 输出 projection 和 residual 最后一层均 zero-init。全局 scale head 的最后一层使用小随机权重，并非严格 zero-init。

当前单级 baseline loss 位于 [loss/common.py](loss/common.py)：

```text
L_depth = 0.5 × (weighted pixel log-L1 + weighted frame-macro log-L1)
L_total = L_depth + 0.5 L_scale + 0.5 L_residual
                  + 0.1 L_zero_mean + 0.1 L_equivariance
```

`L_scale` 用最小化每帧 AbsRel 的 oracle log-scale 作为 SmoothL1 target。单级 `priorbim_loss` 当前仍保留 residual target 去均值和 zero-mean 正则。多级模型使用单独的 `priorbim_multiscale_loss`，不能将两者的 loss 配置混为一谈。

训练默认 6 epoch，batch size=4，gradient accumulation=4，有效 batch=16；AdamW weight decay=0.01，backbone lr=5e-6，其余相关模块 lr=5e-5，cosine scheduler，FP16 autocast，梯度裁剪 1.0。训练集按 room 样本数的 `count^-0.5` 加权、有放回采样；每个 epoch 的抽样数等于训练记录数，因此不是逐帧无重复遍历。

训练保留 RGB gain/bias、BIM shift/dropout/log-noise/edge dilation、水平翻转等增强。`--full-deterministic` 启用严格确定性算法并冻结位置编码：DAv2 为 `position_embeddings`，DA3 为 `pos_embed`；它不关闭数据增强。现有主入口未启用 gradient checkpointing。每个 batch 为 equivariance loss 额外执行一次 scale encoder forward，DA3-Large 因而明显慢于 DAv2 Base。

多级模型在 D36/D72/D144 上分别计算 depth loss，当前权重为 0.2/0.3/0.5，再加 scale 和 equivariance loss；depth loss 中 scale 截断梯度，D144 loss 中前两级 residual 也截断梯度。该方案不是三个 head 拟合同一个 residual target。

### 运行示例

以下命令在 `MyDepth/` 下执行；每次复现使用新输出目录，避免覆盖已有结果。

```bash
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Area1-only baseline
python -u train.py --seed 42 --full-deterministic --local-files-only \
    --output outputs/area1_r36_reproduce

# Area1 real BIM + Area2–5 SyncBIM，stride=1
python -u train.py --seed 42 --full-deterministic --local-files-only \
    --extra-dataset-root /mnt/priorbimda-data/s23_syncbim_area2_5_504 \
    --extra-dataset-stride 1 --output outputs/area1_syncbim_stride1_reproduce
```

训练结束后自动加载 validation AbsRel 最优的 `best.pt`，依次评估 Area1 test 和默认 zero-shot。`--no-zero-shot` 可关闭后者。保存文件为 `latest.pt`、`best.pt`、`history.json`、`test_metrics.json`、`zero_shot_metrics.json`；`--resume` 从同一目录的 `latest.pt` 恢复模型、optimizer、scheduler 和 scaler。

现有 shell 的用途：

- `train.sh`：Area1-only DAv2 R36；当前输出到 `outputs/rawassamble1`。
- `train1.sh`：依次运行 frame loss / no weight / only scale / pixel loss 历史消融。
- `train2.sh`：**当前已改为 DA3 + SyncBIM stride=1**，输出到 `outputs/dav3_area1_syncbim_stride1`。
- `train3.sh`：DAv2 + SyncBIM stride=3；脚本带有 **4 小时延迟**。
- `train_dense2dense.sh`、`train_r36_r72_r144.sh`：独立结构实验，均使用扩展数据 stride=1。

这些脚本多数使用固定输出目录，启动前先检查目录名。独立的 `eval.py` CLI 导入 `model.baseline`，`zero_shot_eval.py` CLI 导入 `model.mymodel`；不要直接用它们加载 DA3、dense2dense 或多级 checkpoint。训练结束时的评测使用正确模型实例，多级评测适配也已写在自身训练文件内。

二阶段微调使用 [train_stage2.py](train_stage2.py)，仅加载 checkpoint 模型权重并重建 optimizer；支持 extra root / stride：

```bash
python -u train_stage2.py \
    --checkpoint outputs/area1_syncbim_stride1/best.pt \
    --lr-factor 0.1 --epochs 3 --full-deterministic --local-files-only \
    --output outputs/area1_stage2_reproduce
```

不传 extra root 时仅使用 Area1 train；传 `--iter N`（N>0）时按 optimizer update 数运行并覆盖 epoch 限制，默认 `--iter 0`。当前 stage2 默认 epoch=1，想微调 3 epoch 需像示例一样显式传入。

## 4. 当前结果汇总

### 评测协议

Area1 test 为固定 1641 帧，所有主结果比较使用有效 GT 像素上的 pixel-micro 指标。zero-shot 为 Matterport3D/BIMNet 的 `hxp`、`759`、`1px`，分别选中 624 / 518 / 793 帧，共 1935 帧；选帧条件是 GT valid >10%、BIM hits >20%、相机位于 BIM AABB 内。GT 仅用于选帧和评分，不用于推理时估计 scale；没有 GT median scaling。

zero-shot 同时保存 pixel-micro 和 frame-macro：前者合并所有有效像素，后者先计算每帧指标再等权平均。下面的 macro 是全部选中帧的平均，不是三个场景均值的平均。AbsRel / RMSE 越低越好，δ1 越高越好；RMSE 单位为米。

### 主要实验

| 结果目录 / 方案 | 最优 epoch | Val AbsRel ↓ | Test AbsRel ↓ | Test RMSE ↓ | Test δ1 ↑ | Zero-shot AbsRel ↓ | Zero-shot macro AbsRel ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| DA3 原始深度（无校正） | — | — | 0.08543 | 0.43518 | 0.93802 | 0.12738 | 0.13388 |
| [rawassamble1](outputs/rawassamble1) / Area1-only 参考运行 | 6 | 0.06354 | 0.06268 | 0.41386 | 0.94315 | 0.08766 | 0.09336 |
| [area1_syncbim_stride6](outputs/area1_syncbim_stride6) / 单级 R36 | 4 | 0.06461 | 0.06301 | 0.41390 | 0.94676 | 0.08334 | 0.08886 |
| [area1_syncbim_stride3](outputs/area1_syncbim_stride3) / 单级 R36 | 4 | 0.06484 | 0.06279 | 0.41293 | 0.94691 | 0.08098 | 0.08616 |
| [area1_syncbim_stride1](outputs/area1_syncbim_stride1) / 单级 R36 | 6 | 0.06466 | 0.06102 | 0.41056 | 0.94988 | **0.07765** | **0.08236** |
| [dense2dense_unbounded_syncbim_stride1](outputs/dense2dense_unbounded_syncbim_stride1) | 6 | 0.06513 | 0.06237 | 0.41043 | 0.94606 | 0.07849 | 0.08357 |
| [adapter_r36_r72_r144_syncbim_stride1](outputs/adapter_r36_r72_r144_syncbim_stride1) | 6 | 0.06285 | **0.06047** | **0.40918** | 0.94959 | 0.07923 | 0.08453 |
| [dav3_area1_syncbim_stride1](outputs/dav3_area1_syncbim_stride1) / 进行中，已保存 4 epoch | 3（暂时） | 0.06664（暂时） | — | — | — | — | — |

DA3 特征实验仍在训练，尚无 test / zero-shot 文件；不能依据当前 validation 判断最终性能。上表 checkpoint 由各自 validation 选择，没有按 test / zero-shot 重新选权重。历史 `rawassamble1` 是参考运行，未保存完整源码快照和 CLI，不能把它与新实验的差距全部归因于一个因素。

### Zero-shot 分场景与输出阶段

| 实验 | hxp AbsRel ↓ | 759 AbsRel ↓ | 1px AbsRel ↓ | Overall RMSE ↓ | Overall δ1 ↑ |
|---|---:|---:|---:|---:|---:|
| 单级 R36，stride=6 | 0.09255 | 0.06993 | 0.08511 | 0.34565 | 0.92757 |
| 单级 R36，stride=3 | 0.08819 | 0.06955 | 0.08294 | 0.34278 | 0.92631 |
| 单级 R36，stride=1 | **0.08223** | 0.06657 | **0.08129** | 0.34182 | **0.93147** |
| dense2dense，stride=1 | 0.08444 | 0.06632 | 0.08182 | 0.34230 | 0.92777 |
| R36/R72/R144，stride=1 | 0.08746 | **0.06556** | 0.08187 | **0.34044** | 0.92775 |

| 模型输出 | Area1 test AbsRel ↓ | Zero-shot AbsRel ↓ |
|---|---:|---:|
| 单级 stride=1：global scale | 0.06621 | 0.08500 |
| 单级 stride=1：scale + R36 | 0.06102 | 0.07765 |
| 多级 stride=1：global scale | 0.06594 | 0.08586 |
| 多级 stride=1：scale + R36 | 0.06129 | 0.07954 |
| 多级 stride=1：scale + R36 + R72 | 0.06098 | 0.07944 |
| 多级 stride=1：scale + R36 + R72 + R144 | 0.06047 | 0.07923 |

当前已完成实验中，单级 R36 + 全量 SyncBIM 的 zero-shot AbsRel 最低，相对原始 DA3 下降约 39.04%；相对 `rawassamble1` 参考运行下降约 11.42%。多级 residual 进一步改善 Area1 test，但 zero-shot AbsRel 没有超过单级。dense2dense 也尚未超过单级 stride=1 的 AbsRel。不同指标可能排序不同，不能将 AbsRel 最优等同于所有指标最优。

### 历史复跑与消融

下表保留项目内其他结果文件的数值，方便追踪。实验名不是完整配置记录，不把它们混合求均值或作为严格单因素结论；`train_*loss` 等入口使用的是 `mymodel1_*` 变体，也不等同于当前 `mymodel + SyncBIM`。

| 结果目录 | Area1 test AbsRel ↓ | Zero-shot AbsRel ↓ | Zero-shot macro AbsRel ↓ |
|---|---:|---:|---:|
| [adapter_3resblocks_raw](outputs/adapter_3resblocks_raw) | 未保存 | 0.08971 | 0.09547 |
| [rawassamble](outputs/rawassamble) | 0.06509 | 0.09062 | 0.09648 |
| [raw8_untrainable_emb_s40](outputs/raw8_untrainable_emb_s40) | 0.06234 | 0.08931 | 0.09486 |
| [raw8_untrainable_emb_s42](outputs/raw8_untrainable_emb_s42) | 0.06362 | 0.08729 | 0.09282 |
| [raw11_trainable_emb_s40](outputs/raw11_trainable_emb_s40) | 0.06298 | 0.08991 | 0.09581 |
| [raw11_trainable_emb_s42](outputs/raw11_trainable_emb_s42) | 0.06361 | 0.08805 | 0.09383 |
| [raw_bicubic_s40](outputs/raw_bicubic_s40) | 0.06248 | 0.08820 | 0.09385 |
| [raw_bicubic_s42](outputs/raw_bicubic_s42) | 0.06365 | 0.08959 | 0.09494 |
| [diagnostic_b2a8_matrix_s42](outputs/diagnostic_b2a8_matrix_s42) | 0.06476 | 0.09857 | 0.10489 |
| [train_frameloss](outputs/train_frameloss) | 0.06433 | 0.08697 | 0.09268 |
| [train_noweight](outputs/train_noweight) | 0.06339 | 0.08731 | 0.09295 |
| [train_pixelloss](outputs/train_pixelloss) | 0.06569 | 0.08844 | 0.09392 |
| [train_onlyscale](outputs/train_onlyscale) | 0.06812 | 0.09301 | 0.09844 |

`onlyscale` 变体仅禁用最终输出中的 residual，仍计算 native residual 和相关辅助 loss；它不是删除全部 residual 参数的严格消融。早期 [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md) 还收录了项目外历史实验，但其中部分路径、入口说明和数值已过时；本 README 以当前 `outputs/*/test_metrics.json`、`zero_shot_metrics.json` 和 `history.json` 为准。

## 5. 复现注意事项

- DA3 depth anchor、BIM 制备/配准、RGB resize、有效像素范围、room sampler、物理 batch、loss 和增强都会影响结果。相同有效 batch 不代表训练过程完全相同。
- DAv2 / DA3 checkpoint 的结构与通道不同，不能互相直接加载；dense2dense / 多级 checkpoint 也需匹配对应模型。
- 当前 checkpoint 保存模型与训练状态，但不包含完整 CLI、源码快照或数据版本。复现实验时额外记录运行命令、Git 版本与数据制备方式；结果文件存在不代表其历史配置已完全可追溯。
