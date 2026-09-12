# MyDepth 与 PriorBIMDA adapter-3res 性能差异诊断

## 结论

观察到的“大幅性能差异”不是单一训练随机性造成的，而由两部分构成：

1. **主要部分是 MyDepth evaluator 的 RGB 数值范围错误。** `MP3D_BIMDataset` 返回 `float32 [0,255]` RGB，但当前 evaluator 没有除以 255 就送入模型；模型内部的 `clamp(0,1)` 因而把绝大多数 RGB 值饱和成 1。原 PriorBIMDA evaluator 使用正确的 `[0,1]` RGB。
2. **在统一为原 PriorBIMDA evaluator 后，当前训练的剩余差异来自 micro-batch 配置。** MyDepth 当前默认是 `batch_size=4, accumulation=4`，baseline 是 `batch_size=2, accumulation=8`。两者有效 batch 都是 16，但训练目标并不等价。

此前将差异归因为 BIM/渲染实现不等价是不正确的。源码和三场景逐帧抽查表明，两者使用相同的 wall-filled OBJ、相同的相机几何与 Open3D `t_hit` 光线投射；抽查 BIM depth 的最大绝对差仅为 `4.8e-7` 至 `1.4e-6 m`，不是性能差异来源。

## 受控实验结果

统一采用原 PriorBIMDA Matterport 三场景、相同冻结帧选择、相同 DA3 cache，并按有效像素做三场景 pixel-micro 加权：

| checkpoint | micro-batch / accumulation | evaluator | global-scale AbsRel | final AbsRel |
|---|---:|---|---:|---:|
| Prior 性能最优 adapter-3res | 2 / 8 | Prior 原版 | 0.093128427 | 0.087861108 |
| Prior exact-reproduction | 2 / 8 | Prior 原版 | 0.091688618 | 0.087808945 |
| MyDepth `raw_bicubic_s42` | 4 / 4 | Prior 原版 | 0.094039397 | 0.089588072 |
| MyDepth 单变量实验 | 2 / 8 | Prior 原版 | 0.091521989 | **0.086990434** |
| MyDepth `raw_bicubic_s42` | 4 / 4 | MyDepth 当前版 | 0.098373183 | 0.094988938 |
| Prior checkpoint | 2 / 8 | MyDepth 当前版 | 0.103185167 | 0.100585608 |

### 同一 checkpoint 的 evaluator 单变量实验

使用 MyDepth `raw_bicubic_s42` 的同一个 checkpoint、相同 1935 帧和 2,232,131,543 个 GT 有效像素：

| eval 输入/数值路径 | global-scale AbsRel | final AbsRel |
|---|---:|---:|
| MyDepth 当前 evaluator：RGB `[0,255]` + FP16 | 0.098373183 | 0.094988938 |
| 只修正 RGB `/255`，仍用 FP16 | 0.094119546 | 0.089706683 |
| 再将 autocast 对齐为 Prior 的 BF16 | 0.094032809 | 0.089579270 |
| Prior 原 evaluator：RGB `[0,1]` + BF16 | 0.094039397 | 0.089588072 |

因此，在 `final AbsRel` 的 evaluator 总差值 `0.005400866` 中：

- RGB `/255` 一项解释 `0.005282255`，即 **97.8%**；
- 在正确 RGB 下，FP16 改为 BF16 再解释 `0.000127413`；
- 全部关键项对齐后与 Prior 仅差 `8.8e-6`（约 0.01%），来自 float/uint8 resize 舍入、矩阵位置插值等微小推理路径差异。

DA3 的总体 pixel-micro AbsRel 在两边均为约 `0.1273814223`，所选帧数、帧 ID 哈希和有效像素数也一致。这同时排除了 GT、DA3、帧筛选和指标聚合是主要差异来源的可能。

在可比的原版 evaluator 下：

- 当前 `4/4` 相对 Prior exact-reproduction 差 `0.001779127`，即 2.03%。
- 只把配置改为 `2/8` 后，指标改善 `0.002597638`（相对改善 2.90%）。
- `2/8` 的 MyDepth 最终比 Prior exact-reproduction 还低 `0.000818511`（低 0.93%，AbsRel 越低越好）。

所以训练侧性能落后已由该单变量实验消除。

## 为什么有效 batch=16 仍不等价

深度 loss 在每个 micro-batch 内先计算像素加权比值：

```text
sum(weight * error) / sum(weight)
```

之后才做梯度累积。因此：

- `2×8` 是 8 个 micro-batch 比值的平均；
- `4×4` 是 4 个 micro-batch 比值的平均。

当各样本有效像素数和权重不同，两者的梯度不相同。另外，equivariance 的随机选择/缩放以及 DataLoader worker 的增强分配也会随 micro-batch 分组改变。

## 已排除的原因

- **BIM 渲染/相机几何**：实现相同，三场景抽查 BIM depth 最大绝对差不超过 `1.5e-6 m`。
- **GT、DA3、帧筛选与指标聚合**：raw DA3 总体 AbsRel、1935 个帧 ID 和 2,232,131,543 个有效像素一致。
- **loss 重构**：当前 MyDepth 与原函数的 total 和各分量逐项数值完全一致。
- **模型核心 forward**：对齐 state、输入和 interpolation 后，尺度、F36 residual 与最终深度逐项一致。
- **数据清单/GT/语义 mask**：train/val/test ID 与顺序一致，运行时 GT 和 furniture mask 一致。
- **未开启 gradient checkpointing**：PriorBIMDA 已有的 no-checkpoint 完整实验 final AbsRel 为 0.087034972，没有造成退化。
- **普通随机波动**：本次结论来自固定 seed 下只改变 `batch_size/accumulation` 的完整 6 epoch 因果对照，不是依据相关性猜测。

## 仍存在但未导致本次性能落后的实现差异

以下差异会阻止“逐位相同”的复现，但 `2/8` 实验在保留它们时已经达到/超过 baseline，因此不能把它们称为本次性能下降原因：

- MyDepth RGB resize 使用 CUBIC，Prior 使用 AREA；
- compact BIM artifact 与 Prior artifact 有小量数值差异；
- DataLoader sampler/worker generator 的组织方式不同；
- low2/adapter 构造顺序导致相同 seed 对应不同初始参数；
- MyDepth 使用确定性矩阵实现 position embedding bicubic backward；
- 当前 `full-deterministic` 中关闭 cuDNN TF32 的代码被注释。

## evaluator 修复点

MyDepth `zero_shot_eval.py` 至少应把 RGB 输入改为 `[0,1]`，并与 Prior evaluator 一样在支持时优先使用 BF16。若要求最大程度复刻 Prior，还应保持“uint8 上执行 `INTER_AREA` resize，再转 float32 并除以 255”的顺序。

## 建议训练命令

```bash
PYTHONHASHSEED=42 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python train.py \
  --seed 42 \
  --full-deterministic \
  --batch-size 2 \
  --accumulation 8 \
  --local-files-only \
  --output outputs/adapter_3res_b2a8_s42
```

诊断实验的完整 MyDepth 输出位于 `outputs/diagnostic_b2a8_matrix_s42/`；原版 evaluator 的交叉评估摘要位于 `/tmp/prior_eval_my_s42_*` 和 `/tmp/prior_eval_my_b2a8_*`。
