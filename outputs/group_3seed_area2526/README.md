# Area2–5 与 Area2–6：三种子 zero-shot 对比

比较两组 adapter + 3 ResBlocks 训练：均使用 Area1 real BIM；第二组在 Area2–5 SyncBIM 之外，额外加入 Area6 SyncBIM。下面的主结果统一采用 **25 个 MP3D/BIMNet 场景、OBJ mesh、训练末轮权重**。指标为 AbsRel，越低越好。

| 训练数据 | seed 40：pixel / frame | seed 41：pixel / frame | seed 42：pixel / frame | 三种子均值：pixel / frame |
| --- | ---: | ---: | ---: | ---: |
| Area1 + Area2–5 | [0.10269 / 0.10735](area1_syncbim_stride1-25-s40/zero_shot_all_metrics_latest_obj.json) | [0.10386 / 0.10865](area1_syncbim_stride1-25-s41/zero_shot_all_metrics_latest_obj.json) | **[0.09819 / 0.10262](area1_syncbim_stride1-25-s42/zero_shot_full_metrics_best_obj.json)** | 0.10158 / 0.10621 |
| Area1 + Area2–6 | **[0.09985 / 0.10435](area1_syncbim_stride1-26-s40/zero_shot_all_metrics_latest_obj.json)** | **[0.10089 / 0.10552](area1_syncbim_stride1-26-s41/zero_shot_all_metrics_latest_obj.json)** | [0.10216 / 0.10664](area1_syncbim_stride1-26-s42/zero_shot_all_metrics_latest_obj.json) | **0.10097 / 0.10550** |

`pixel` 是有效像素汇总的 pixel-micro AbsRel，`frame` 是逐帧平均的 frame-macro AbsRel，并非逐场景平均。Area2–6 相对 Area2–5 的 pixel-micro 差值（相同 seed 配对）依次为 **−0.00284、−0.00297、+0.00396**。加入 Area6 在 seed 40、41 有利，在 seed 42 不利；三种子均值仅改善 0.00062（约 0.6%）。单次最优是 Area2–5 / seed 42 的 0.09819，现有三种子不足以断定 Area6 稳定提升性能。

## 对照条件

- 六份主结果的评测协议相同：504 分辨率、`registered BIMNet obj`、相同的 25 个场景。逐场景的选帧数量及 ID 哈希均一致，共 31,341 帧；DA3 基线 pixel-micro AbsRel 均为 0.16689。不要将本表与目录里的 wall-filled 结果混合比较。
- 六份 checkpoint 均为同一模型结构、训练至第 6 轮。五份使用 `latest.pt`；Area2–5 / seed 42 只保留 `best.pt`，但 checkpoint 的 `epoch=6`，所以也是末轮权重。各组使用相同的 optimizer 参数组配置。
- 训练均按 6 个 epoch、额外数据 stride=1 比较。Area2–5 组共有 52,770 个训练记录（Area1 7,013 + Area2–5 45,757）；Area2–6 组再加入 Area6 的 8,036 个，共 60,806 个。固定 epoch **没有固定训练步数**：计划优化步数分别为 19,788 与 22,806，后者约多 15.3%，且按房间加权采样后的来源比例也会改变。因此结果是“固定 epoch 下增加 Area6”的效果，不能单独归因于 Area6 样本质量。
- 结果 JSON 的 `checkpoint` 字段是移动目录之前的绝对路径，部分文件还保留了更早的输出目录名；这些路径现在可能不存在。应使用本目录各子文件夹中的实际 checkpoint。结果文件名里的 `best` / `latest` 也应结合上述 epoch 信息理解。

## 三场景子集

从上述 **同一批 25 场景 OBJ 结果**中，按有效像素重新汇总 `hxp`、`759`、`1px` 三个场景（共 1,935 帧），得到：

| 训练数据 | seed 40 | seed 41 | seed 42 | 三种子均值 |
| --- | ---: | ---: | ---: | ---: |
| Area1 + Area2–5 | 0.07911 | 0.07950 | **0.07704** | 0.07855 |
| Area1 + Area2–6 | **0.07641** | **0.07888** | 0.07882 | **0.07804** |

此表不是额外的独立实验，只用于检查三场景与全场景结论是否同向；仍以 25 场景主表为准。目录中其他三场景 JSON 可能采用不同 checkpoint 或 wall-filled mesh，不能直接并入主表。
