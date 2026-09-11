# MyDepth 实验改动与结果汇总

更新时间：2026-09-11（Pacific/Auckland）

## 1. 汇总范围与对应方法

- 项目目录：`/home/bgao491/MyDepth`
- 用户给出的 `~/ouputs` 不存在；本报告按实际目录 `/home/bgao491/outputs` 汇总。
- 早期基线另存于项目内的 `/home/bgao491/MyDepth/outputs`，也纳入比较。
- 实验与代码的对应关系由输出时间、`train.py` 当时的模型 import/默认输出目录、本地 VS Code 文件历史，以及 checkpoint 大小共同还原。
- checkpoint 没有保存源码 hash 或完整运行参数，相关源码也是在实验完成后才统一纳入 Git。因此下面的对应关系有充分的本机历史依据，但早期基线与后续复跑之间的所有细微差异无法做到严格的字节级复现。

指标方向：AbsRel、RMSE 越低越好（↓），δ1 越高越好（↑）。表中数值统一保留 5 位小数。

## 2. 所有实验与代码一一对应

| 实验结果目录 | 对应代码/运行态 | 相对上一方案的具体改动 | 状态 |
|---|---|---|---|
| `MyDepth/outputs/adapter_3resblocks_raw` + `MyDepth/outputs/zero_shot_raw` | [`model/baseline.py`](model/baseline.py)，功能上也等同于当前 [`model/mymodel.py`](model/mymodel.py) | 原始基线：CLS 与 patch 均值拼接预测全局尺度；保留原 DAv2 F36 解码路径；BIM disagreement adapter 含 3 个 ResBlock，并预测局部 log-residual | Area1 训练和零样本完成；缺少 `test_metrics.json` |
| `~/outputs/scale_cls` | `mymodel.py` 的历史运行态 | scale descriptor 从 `CLS + mean(patch)` 改成仅 CLS；scale head 输入维度由 1536 改为 768 | 完成 |
| `~/outputs/scale_mean` | `mymodel.py` 的历史运行态 | 在 `scale_cls` 基础上改成仅使用所有 patch token 的均值，不再使用 CLS | 完成 |
| `~/outputs/scale_raw` | [`model/baseline.py`](model/baseline.py) | 恢复 `CLS + mean(patch)` 拼接，scale head 输入恢复为 1536；其余训练设置仍为 batch 2、梯度累积 8 | 完成 |
| `~/outputs/scale_raw1` | [`model/baseline.py`](model/baseline.py) | 模型结构不变；物理 batch 从 2 改为 8，累积从 8 改为 2（有效 batch 都是 16）；增加确定性设置，关闭 gradient checkpointing，并增加 tqdm 显示 | 完成 |
| `~/outputs/Reassemble` | [`model/mymodel1.py`](model/mymodel1.py) | 新增 same-resolution reassemble：取 DINOv2 stage 1/2/3 token，不做空间缩放，各自经 `1×1 + 3×3` 投影到 128 通道后取均值；在 F36 融合处用它替换原 stage-3 分支 | 完成 |
| `~/outputs/Reassemble_zero_init_12plus3` | [`model/mymodel2.py`](model/mymodel2.py) | reassemble 改为只取 stage 1/2；均值融合后经过零初始化 `3×3` adapter，再以 residual 形式加到原 stage-3 分支，避免训练初始时破坏原路径 | 完成 |
| `~/outputs/Reassemble_only_scale` | [`model/mymodel3.py`](model/mymodel3.py) | 保留上一方案的网络计算，但把上采样后的 `log_residual` 强制设为 0，因此最终深度等于仅做全局尺度校正的深度 | 6 epoch、测试和零样本均完成 |

## 3. 共同模型与训练设置

所有实验的主干逻辑相同，区别集中在 scale descriptor、F36 reassemble 和是否使用最终 residual：

1. 使用固定版本的 `Depth-Anything-V2-Metric-Indoor-Base-hf`。
2. BIM 条件由 3 个通道组成：归一化 log-BIM 深度、BIM 有效掩码、裁剪后的 `log(BIM)-log(DA3)` disagreement。
3. BIM 条件经零初始化、kernel/stride 都为 14 的卷积后，加到 DINOv2 patch embedding，属于 early fusion。
4. 全局尺度为 `DA3 × exp(log_scale)`。
5. 局部 refiner 在 native F36 分辨率预测范围受 `0.1 × tanh(...)` 限制的 log-residual，最终深度为 `DA3 × exp(log_scale) × exp(log_residual)`。
6. 训练目标为：`depth + 0.5×scale + 0.5×residual + 0.1×zero_mean + 0.1×equivariance`。
7. Area1 默认训练 6 epoch；训练集按 room 频次的 `count^-0.5` 加权采样；最优 checkpoint 按 validation final AbsRel 选择。

注意：`model/mymodel3.py` 只在最终输出处禁用了 residual，但仍然计算 `log_residual_native`，而且 residual 辅助损失和对应参数仍参与训练。因此 `Reassemble_only_scale` 是“输出仅 scale”，不是“删除 refiner、只训练 scale head”的严格消融。

## 4. Area1 validation/test 结果

| 实验 | 最优 epoch | Val final AbsRel ↓ | Test DA3 AbsRel ↓ | Test scale AbsRel ↓ | Test final AbsRel ↓ | Test final RMSE ↓ | Test final δ1 ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| `adapter_3resblocks_raw` | 6 | 0.06299 | — | — | — | — | — |
| `scale_cls` | 6 | 0.06456 | 0.08543 | 0.06848 | 0.06351 | 0.41377 | 0.94308 |
| `scale_mean` | 6 | 0.06289 | 0.08543 | 0.06759 | 0.06276 | **0.41125** | 0.94231 |
| `scale_raw` | 6 | 0.06407 | 0.08543 | 0.06767 | 0.06354 | 0.41470 | 0.94412 |
| `scale_raw1` | 6 | **0.06270** | 0.08543 | **0.06700** | **0.06235** | 0.41309 | 0.94494 |
| `Reassemble` | 6 | 0.06405 | 0.08543 | 0.06843 | 0.06420 | 0.41480 | 0.94177 |
| `Reassemble_zero_init_12plus3` | 6 | 0.06513 | 0.08543 | 0.06762 | 0.06349 | 0.41573 | 0.94459 |
| `Reassemble_only_scale` | 3 | 0.06837 | 0.08543 | 0.06742 | 0.06742 | 0.41843 | **0.94543** |

说明：

- `scale_raw1` 的 Area1 test final AbsRel 最好，为 0.06235，相对原始 DA3 的 0.08543 下降 27.01%。
- `scale_mean` 的 test RMSE 最好，为 0.41125。
- `Reassemble_only_scale` 在 epoch 3 达到最优 Val AbsRel 0.06837，之后没有继续改善；其 test scale 与 final 完全相同，验证了 residual 已从最终输出中关闭。
- 在结构最接近的一组比较中，`Reassemble_zero_init_12plus3` 的 test final AbsRel 0.06349，比 `Reassemble_only_scale` 的 0.06742 低 5.83%，说明局部 residual 对 Area1 明显有用。不过两者经过独立训练，这不是“同一 checkpoint 开关 residual”的严格配对消融。

## 5. Matterport3D/BIMNet 零样本总体结果

固定评测场景为 `hxp`、`759`、`1px`，共选择 1935 帧。选择规则为 GT 有效率 >10%、BIM hit >20%、相机位于 BIM AABB 内；GT 只用于选帧和打分。下表为所有选中像素的 micro aggregation。

| 实验 | DA3 AbsRel ↓ | Scale AbsRel ↓ | Final AbsRel ↓ | Final RMSE ↓ | Final δ1 ↑ | Final frame-macro AbsRel ↓ |
|---|---:|---:|---:|---:|---:|---:|
| `adapter_3resblocks_raw` | 0.12738 | 0.09388 | **0.08939** | **0.35070** | **0.92434** | **0.09511** |
| `scale_cls` | 0.12738 | 0.11483 | 0.10837 | 0.36607 | 0.89839 | 0.11412 |
| `scale_mean` | 0.12738 | 0.09603 | 0.09529 | 0.36556 | 0.91440 | 0.10078 |
| `scale_raw` | 0.12738 | 0.10244 | 0.09853 | 0.35692 | 0.90795 | 0.10454 |
| `scale_raw1` | 0.12738 | 0.10248 | 0.10190 | 0.36031 | 0.91152 | 0.10781 |
| `Reassemble` | 0.12738 | 0.09659 | 0.09447 | 0.35624 | 0.91527 | 0.10076 |
| `Reassemble_zero_init_12plus3` | 0.12738 | 0.10184 | 0.10017 | 0.36274 | 0.90114 | 0.10606 |
| `Reassemble_only_scale` | 0.12738 | **0.09268** | 0.09268 | 0.35361 | 0.92121 | 0.09829 |

### 分场景 final pixel-micro AbsRel

| 实验 | `hxp` ↓ | `759` ↓ | `1px` ↓ |
|---|---:|---:|---:|
| `adapter_3resblocks_raw` | **0.09589** | **0.07861** | **0.09147** |
| `scale_cls` | 0.10187 | 0.09741 | 0.11988 |
| `scale_mean` | 0.09397 | 0.08600 | 0.10203 |
| `scale_raw` | 0.10386 | 0.08157 | 0.10528 |
| `scale_raw1` | 0.11257 | 0.08308 | 0.10599 |
| `Reassemble` | 0.09705 | 0.08376 | 0.09931 |
| `Reassemble_zero_init_12plus3` | 0.10085 | 0.09041 | 0.10577 |
| `Reassemble_only_scale` | 0.10049 | 0.08043 | 0.09471 |

## 6. 逐项结果解读

### 6.1 `adapter_3resblocks_raw`：原始基线

- 具体结构：CLS 与 patch mean 拼接预测 scale；原生 DAv2 F36 路径负责视觉特征；3-ResBlock disagreement adapter 注入 BIM/DA3 校正信息；low2 head 输出局部 residual。
- Val final AbsRel 为 0.06299。
- 没有保存 Area1 `test_metrics.json`，因此不能把它的 validation 数字直接当作 test 数字与其他实验横向比较。
- 零样本 final AbsRel 为 0.08939，是现有全部完整实验中最好；相对零样本 DA3 0.12738 下降 29.82%。三个场景也全部最好，泛化最稳定。

### 6.2 `scale_cls`：只使用 CLS 预测尺度

- 具体改动：scale head 输入从 1536 维拼接描述符改为 768 维 CLS token。
- Area1 test final AbsRel 为 0.06351，与其他 descriptor 方案接近。
- 零样本 final AbsRel 上升到 0.10837，是完整实验中最差，说明只依赖 CLS 对跨域尺度泛化不够稳健。

### 6.3 `scale_mean`：只使用 patch 均值预测尺度

- 具体改动：移除 CLS，只使用 patch token 的全局均值；scale head 输入仍为 768 维。
- Area1 test final AbsRel 为 0.06276，RMSE 0.41125；两项都优于 `scale_cls`。
- 零样本 final AbsRel 为 0.09529，也显著优于 `scale_cls`。在三个 descriptor 消融中，patch mean 的整体平衡最好。

### 6.4 `scale_raw`：恢复 CLS + patch mean

- 具体改动：scale descriptor 恢复为 1536 维的 `concat(CLS, mean(patch))`。
- Area1 test final AbsRel 为 0.06354；零样本 final AbsRel 为 0.09853。
- 它没有超过 `scale_mean`，说明简单拼接 CLS 并未带来稳定收益。

### 6.5 `scale_raw1`：同结构的训练执行方式复跑

- 具体改动：物理 batch 2→8、累积 8→2，保持有效 batch 16；启用确定性算法设置；关闭 gradient checkpointing。tqdm 只影响显示，不影响模型。
- Area1 test final AbsRel 为 0.06235，是 test 最优；相对 `scale_raw` 改善 1.88%。
- 零样本 final AbsRel 为 0.10190，比 `scale_raw` 的 0.09853 差 3.42%。这说明 in-domain 改善没有转化为跨域改善。

### 6.6 `Reassemble`：stage 1/2/3 同分辨率重组并替换 stage 3

- 具体改动：前三个 DINO stage 的 patch token 全部直接 reshape 到 36×36，各自投影到 128 通道并平均；生成特征替换原 DAv2 stage-3 F36 分支。
- Area1 test final AbsRel 为 0.06420，低于原始 DA3，但没有超过 descriptor 基线。
- 零样本 final AbsRel 为 0.09447，排名第二，仅次于早期 `adapter_3resblocks_raw`；表明直接重组对跨域比对 Area1 更有价值。

### 6.7 `Reassemble_zero_init_12plus3`：stage 1/2 零初始化 residual 加到 stage 3

- 具体改动：移除 stage 3 自定义投影，只聚合 stage 1/2；新增零初始化 3×3 adapter；输出与原始 stage-3 feature 相加，而不是替换原分支。
- Area1 test final AbsRel 为 0.06349，优于直接替换版 `Reassemble` 的 0.06420。
- 零样本 final AbsRel 为 0.10017，反而差于直接替换版 0.09447。零初始化 residual 更稳健地保留了原路径，但没有带来更好的跨域泛化。

### 6.8 `Reassemble_only_scale`：最终输出禁用 residual

- 具体改动：`log_residual` 被固定为 0，最终输出严格等于 `scaled_depth`。
- 最优 validation 出现在 epoch 3（0.06837）；test final AbsRel 为 0.06742，明显弱于启用 residual 的相近方案。
- 零样本 final AbsRel 为 0.09268，排名第二，相对原始 DA3 下降 27.25%；由于 final 与 scale 相同，两列完全一致。
- 与独立训练的 residual-enabled `Reassemble_zero_init_12plus3` 相比，零样本 AbsRel 从 0.10017 降到 0.09268，改善 7.48%。这与 Area1 的结论相反，表明该 residual 分支出现了明显的域内收益/跨域退化。
- 实现仍计算并训练 residual native head；如果目标是严格的参数/训练消融，还应从 forward、loss 和 optimizer parameter groups 中同时删除 residual 分支。

## 7. 结论

1. **Area1 test 最优**：`scale_raw1`，final AbsRel 0.06235。
2. **零样本最优**：`adapter_3resblocks_raw`，final AbsRel 0.08939；相对原 DA3 改善 29.82%。
3. **descriptor 消融结论**：patch mean 比 CLS 更适合尺度预测；CLS+mean 拼接没有稳定胜过 patch mean。
4. **reassemble 结论**：直接替换 stage 3 的版本在零样本上优于零初始化 residual 版本，但两者均未超过早期基线。
5. **refiner 结论**：局部 residual 在 Area1 上通常把 global-scale AbsRel 再降低约 6%–7%；但在 `Reassemble_zero_init_12plus3` 配对中，它使零样本泛化变差。不能据 Area1 收益直接推断跨域收益。
6. Area1 与零样本排序明显不同，模型选择不能只看 Area1 validation/test。若优先跨域部署，当前应保留 `adapter_3resblocks_raw`，`Reassemble_only_scale` 是第二名；若只追求 Area1 test，则选 `scale_raw1`。

## 8. 结果文件索引

| 实验 | History | Area1 test | 零样本 |
|---|---|---|---|
| `adapter_3resblocks_raw` | [`history.json`](outputs/adapter_3resblocks_raw/history.json) | 未保存 | [`best.json`](outputs/zero_shot_raw/best.json) |
| `scale_cls` | [`history.json`](../outputs/scale_cls/history.json) | [`test_metrics.json`](../outputs/scale_cls/test_metrics.json) | [`zeroshot.json`](../outputs/scale_cls/zeroshot.json) |
| `scale_mean` | [`history.json`](../outputs/scale_mean/history.json) | [`test_metrics.json`](../outputs/scale_mean/test_metrics.json) | [`zeroshot.json`](../outputs/scale_mean/zeroshot.json) |
| `scale_raw` | [`history.json`](../outputs/scale_raw/history.json) | [`test_metrics.json`](../outputs/scale_raw/test_metrics.json) | [`zero_shot_metrics.json`](../outputs/scale_raw/zero_shot_metrics.json) |
| `scale_raw1` | [`history.json`](../outputs/scale_raw1/history.json) | [`test_metrics.json`](../outputs/scale_raw1/test_metrics.json) | [`zero_shot_metrics.json`](../outputs/scale_raw1/zero_shot_metrics.json) |
| `Reassemble` | [`history.json`](../outputs/Reassemble/history.json) | [`test_metrics.json`](../outputs/Reassemble/test_metrics.json) | [`zero_shot_metrics.json`](../outputs/Reassemble/zero_shot_metrics.json) |
| `Reassemble_zero_init_12plus3` | [`history.json`](../outputs/Reassemble_zero_init_12plus3/history.json) | [`test_metrics.json`](../outputs/Reassemble_zero_init_12plus3/test_metrics.json) | [`zero_shot_metrics.json`](../outputs/Reassemble_zero_init_12plus3/zero_shot_metrics.json) |
| `Reassemble_only_scale` | [`history.json`](../outputs/Reassemble_only_scale/history.json) | [`test_metrics.json`](../outputs/Reassemble_only_scale/test_metrics.json) | [`zero_shot_metrics.json`](../outputs/Reassemble_only_scale/zero_shot_metrics.json) |

## 9. 当前代码与复现注意事项

- 实验源码已在提交 `44957c1` 中纳入 Git；本次把根目录模型和 loss 重组到 `model/`、`loss/` 的改动仍位于工作树、尚未提交。
- 当前 [`train.py`](train.py) 默认导入 `model.mymodel3.PriorBIMDA`，默认输出到 `outputs/Reassemble_only_scale`。直接再次运行会复用同一路径，存在覆盖/混合结果的风险。
- [`eval.py`](eval.py) 的独立 CLI 固定导入 `model.baseline.PriorBIMDA`，不能直接加载 reassemble checkpoint；训练结束时的内嵌评测没有这个问题，因为它使用内存中的正确模型实例。
- [`zero_shot_eval.py`](zero_shot_eval.py) 的独立 CLI 固定导入 `model.mymodel.PriorBIMDA`；同样不适用于 reassemble checkpoint，但由 `train.py` 调用 `evaluate_zero_shot(model, ...)` 时使用的是正确实例。
- [`model/attention_scale.py`](model/attention_scale.py) 目前没有被上述训练入口引用，因此 `/home/bgao491/outputs` 中没有可明确对应给它的实验结果。
- 后续每次实验建议在输出目录额外保存：模型文件名、完整 CLI args、Git commit/dirty diff、源码 SHA256、随机种子、PyTorch/CUDA 版本。这样无需再依赖编辑器历史推断对应关系。
