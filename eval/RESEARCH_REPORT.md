# Wan2.2 缓存方法调研与实验分析

本文记录 `eval` 中新增的 MagCache output 与 D2Cache output，说明当前科研流水线的执行边界，并分析三条 prompt 的服务器实验。文中的质量结论来自当前实验记录。Latency 会受到 GPU 负载、CPU 到 GPU 权重搬运和共享存储状态影响，因此只作为本轮运行记录，不作为方法优劣的主要证据。

## 新增方法

### MagCache output

MagCache 的核心观察是相邻扩散步 residual 的方向较稳定，主要变化来自幅值。官方方法用一次校准得到逐 timestep 的幅值比 `gamma`，从最近一次完整计算开始累乘这些比例，并累计缓存误差

```text
ratio_product[t] = product(gamma[i])
error[t] = error[t - 1] + abs(1 - ratio_product[t])
```

当累计误差低于阈值且连续缓存长度不超过上限时，当前扩散步复用 residual。达到误差阈值或长度上限后执行完整计算并刷新缓存。论文报告的机制和官方实现都作用于 Transformer block feature residual。[MagCache 论文](https://arxiv.org/abs/2506.09045) 以及 [Wan2.2 官方实现](https://github.com/Zehong-Ma/MagCache/blob/main/MagCache4Wan2.2/magcache_generate.py) 给出了原始定义。

当前 [magcache.py](model/magcache.py) 保留相同的逐步幅值比、乘积误差和最大连续缓存长度，将缓存位置改成完整 DiT output residual。配置使用官方 Wan2.2 TI2V-5B T2V conditional 分支的 50 步 ratio，前 20% timestep 保留完整计算，阈值为 `0.06`，最大连续跳过两个 CFG pair。conditional 决策同时控制同一 pair 的 unconditional 分支。

这一版本的作用是检验官方幅值先验能否迁移到现有 output-level 缓存接口。它没有复现 block-level feature cache，因此实验数字不能直接与 MagCache 论文数字对照。

### D2Cache output

D2Cache 使用连续两次完整计算 residual 的差值，对缓存 residual 做二阶修正。官方方法属于 training-free 插件，重点是利用二阶 residual delta 的平滑性缓解连续跳步产生的误差累积。[D2Cache 官方仓库](https://github.com/VG-Huai/D2Cache) 和 [Wan2.1 实现](https://github.com/VG-Huai/D2Cache/blob/main/D2Cache4Wan2.1/delta2cache_generate.py) 给出了原始算法。

当前 [d2cache.py](model/d2cache.py) 沿用 EasyCache 的 output-level 跳步调度，并为 conditional 与 unconditional 分支分别维护最近的 residual delta。跳过完整 DiT 时使用

```text
predicted_residual = cached_residual + correction_scale * residual_delta
```

`correction_scale` 由当前累计误差与上一次刷新前误差的比例得到。这样 `easycache_0.05` 与 `d2cache_output_0.05` 共享调度基础，主要差异集中在缓存输出是否包含 delta correction。官方 D2Cache 在 Transformer block feature 上使用 TeaCache 调度和二阶修正，当前实现仍然属于 output-level 适配。

## Eval 科研流水线

### 方法和任务展开

[sweep.yaml](conf/sweep.yaml) 用 Hydra 描述 Origin、EasyCache、累计 MLP、Temporal GRU、MagCache output 和 D2Cache output。三条 prompt 展开为 36 个视频任务。

```text
Origin              3
EasyCache           9
Model               9
Temporal            9
MagCache output     3
D2Cache output      3
```

每个任务包含 prompt、seed、方法配置、阈值和保护区间。目标实验目录中的 `tasks.jsonl` 保存任务表。启动时若其中的任务记录与当前任务一致，并且对应的 `<prompt>.result.json` 已存在，该任务会被过滤，避免重复生成。

### 多 GPU 视频生成

[pipeline.py](pipeline.py) 启动一个 Python 主进程，再按照可见 GPU 数量创建 worker。每个 worker 只构造一次 `WanTI2V`，随后串行处理分配给该 GPU 的视频。每个视频重新实例化一次 `CacheMethod`，缓存状态不会跨 prompt 泄漏。

[cache_hook.py](cache_hook.py) 在一次 `WanTI2V.generate` 范围内临时替换 DiT forward。方法对象在 forward 前通过 `try_skip` 决定是否复用缓存，完整 DiT 返回后通过 `update` 接收真实 output。生成结束后 hook 恢复原始 forward。Origin 使用只计时、不执行缓存逻辑的包装，基线整体 latency 不包含输入 clone 和缓存控制开销。

每个方法目录直接按 prompt 顺序编号，不再为单个视频建立子目录。

```text
experiment/
  tasks.jsonl
  origin/0.mp4
  origin/0.result.json
  origin/0.raw.log
  easycache_0.05/0.mp4
  model_0.05/0.mp4
  temporal_0.05/0.mp4
  magcache_output_0.06/0.mp4
  d2cache_output_0.05/0.mp4
```

`result.json` 保存生成耗时、完整 DiT 前向与缓存命中路径的 CUDA 时间、调用次数、跳过 pair 数和索引。`raw.log` 保存 Wan 对该视频的原始输出。控制台只保留 Wan 加载、任务开始、任务完成和失败信息。

### 指标和汇总

视频生成和指标计算分成两个作业。`run_generate_8gpu.sbatch` 只负责生成。全部视频完成后，`run_evaluate.sbatch` 只调用一次 `python -m eval.evaluate`。

这个入口先由 [summarize.py](summarize.py) 汇总 latency、每段视频的 DiT 累计时间、skip 和整体 speedup，再由 [evaluate_quality.py](evaluate_quality.py) 计算 PSNR、SSIM、LPIPS 和 FVD，最后在 `metrics/evaluation_summary.md` 生成统一表格。Cache 时间、调用数和单次前向均值继续写入性能 CSV。质量指标按相同 prompt 和 seed 的候选视频与 Origin 配对。当前配置包含 33 个质量配对。

## 当前实验结果

| Method | Latency | Skip | Speedup | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| Origin | 903.5713 | 0.00 | 1.0000 | - | - | - |
| D2Cache output 0.05 | 444.5992 | 27.67 | 2.0895 | 24.1564 | 0.8795 | 0.0835 |
| EasyCache 0.03 | 606.6377 | 21.00 | 1.4879 | 27.3770 | 0.9194 | 0.0478 |
| EasyCache 0.05 | 528.1834 | 28.00 | 1.7111 | 23.3992 | 0.8554 | 0.0997 |
| EasyCache 0.07 | 402.2879 | 31.33 | 2.3396 | 21.8708 | 0.8056 | 0.1303 |
| MagCache output 0.06 | 491.2167 | 24.00 | 1.8489 | 24.4661 | 0.8766 | 0.0808 |
| Model 0.03 | 665.8826 | 11.00 | 1.3823 | 28.4575 | 0.9403 | 0.0352 |
| Model 0.05 | 573.6421 | 18.00 | 1.6066 | 23.6255 | 0.8762 | 0.0843 |
| Model 0.07 | 513.3875 | 22.67 | 1.8005 | 21.3535 | 0.8228 | 0.1283 |
| Temporal 0.03 | 789.0498 | 4.33 | 1.1410 | 28.6225 | 0.9441 | 0.0316 |
| Temporal 0.05 | 737.8308 | 8.33 | 1.2208 | 24.2707 | 0.8844 | 0.0792 |
| Temporal 0.07 | 705.0315 | 11.00 | 1.2771 | 22.0166 | 0.8424 | 0.1119 |

### 同等跳步预算下的结果

D2Cache output 与 EasyCache 0.05 的跳步数几乎相同，分别为 27.67 和 28.00。D2Cache output 的 PSNR 提高 `0.7572`，SSIM 提高 `0.0241`，LPIPS 降低 `0.0162`。这组对照只改变 residual 重建方式，说明二阶 delta correction 在当前 output-level 接口上确实减少了缓存误差。它是本轮最清楚的正向结果。

MagCache output 平均跳过 24 个 pair，位于 EasyCache 0.03 和 0.05 之间。相较 EasyCache 0.05，它少跳过 4 个 pair，同时 PSNR 提高 `1.0669`，SSIM 提高 `0.0212`，LPIPS 降低 `0.0189`。MagCache 的固定幅值曲线给出了可用的中等预算点，但当前数据还不能区分质量提升来自幅值先验还是更保守的跳步数。

D2Cache output 相较 MagCache output 多跳过 `3.67` 个 pair。两者的质量很接近，D2Cache 的 SSIM 高 `0.0029`，PSNR 低 `0.3097`，LPIPS 高 `0.0027`。这两个点形成了较清晰的中高加速区间。

### 低维学习方法的表现

Temporal 0.03 给出当前最高的三项质量指标，PSNR 为 `28.6225`，SSIM 为 `0.9441`，LPIPS 为 `0.0316`，但只跳过 `4.33` 个 pair。Model 0.03 的质量非常接近，额外跳过 `6.67` 个 pair，因此单步累计 MLP 在保守区间具有更好的质量效率平衡。

EasyCache 0.03 是当前更值得关注的 Pareto 点。它平均跳过 21 个 pair，仍达到 `27.3770` PSNR、`0.9194` SSIM 和 `0.0478` LPIPS。它同时优于 Model 0.05 和 Temporal 0.05 的跳步数与三项质量指标。这个结果表明目前学习方法的主要限制来自监督定义、训练分布和在线闭环之间的偏移，继续扩大同一组 8 维特征上的网络容量不一定能解决问题。

阈值从 0.03 调到 0.05 后，Model 多跳过 7 个 pair，PSNR 下降 `4.8320`，LPIPS 上升 `0.0491`。Temporal 多跳过 4 个 pair，PSNR 下降 `4.3518`，LPIPS 上升 `0.0476`。两个模型的风险输出对部署误差增长都不够稳定。旧数据的 target 是未来局部误差的累加代理，而连续跳步会改变 scheduler latent、输入历史和 residual 年龄，这个训练目标无法完整描述在线 rollout。

### Latency 的使用边界

当前 latency 包含一次 `WanTI2V.generate` 内的 prompt encoding、模型搬运、采样和缓存逻辑。单次 DiT 与 Cache 指标由 CUDA Event 测量 GPU 路径，可以隔离这些外围耗时，但不包含缓存策略的纯 Python 主机开销。配置启用了 `offload_model`，每个视频仍有 DiT 与 T5 的设备搬运。共享服务器的 GPU 占用、CPU 内存带宽和存储状态都会改变结果。后续报告应为每个配置重复运行，至少记录中位数和四分位区间，并优先使用匹配 skip budget 的质量指标比较算法。

## 下一步修改方案

### 方案一 重新采集高维原始张量

当前 8 维统计量丢失了 token、空间、时间和通道结构。它适合快速判断缓存风险，无法表达局部运动区域、细粒度残差方向和 conditional 与 unconditional 的结构差异。

[collect_features.py](collect_features.py) 已经提供原始采集入口。它在所有 Transformer block 完成后、`model.head` 之前注册 pre-hook，对每个扩散步分别保存 conditional 与 unconditional 的原始张量。采集不会做归一化、池化、残差、窗口或标签变换。

```text
cache_features/
  dataset.yaml
  manifest.jsonl
  prompt_0000/
    timesteps.pt
    step_00_cond.pt
    step_00_uncond.pt
    ...
```

`manifest.jsonl` 为每条轨迹保存独立的 63 位随机 seed，续跑时复用该 seed。每个 `step_XX_cond.pt` 和 `step_XX_uncond.pt` 保留 Wan 实际 dtype、shape 和分支。`timesteps.pt` 保存对应的扩散 timestep。该 schema 只保存原始表示，监督标签应在训练或反事实 rollout 阶段独立生成。

直接在完整高维张量上预测 refresh 或 residual correction 会带来明显成本。单条轨迹需要保存 50 个 timestep 的两路特征，数据体积和 NFS 读吞吐会快速增长。全分辨率网络还会增加显存、训练时间和部署决策开销。第一轮应先测量真实 tensor shape、单轨迹大小和读取带宽，再选择共享低秩投影、分块池化或轻量时空编码器。原始文件继续保持不变，所有压缩都放在 Dataset 或模型中完成，便于后续更换表示。

这条路线的关键实验是比较 8 维模型、低秩高维编码和完整高维编码在相同 skip budget 下的质量增益及决策开销。高维模型只有在 PSNR、SSIM 和 LPIPS 的收益稳定超过额外推理成本时才值得进入生成管线。

### 推荐顺序

先运行原始高维采集器，测量真实 tensor shape、单轨迹大小和读取带宽。随后在保持原始采集文件不变的前提下，比较共享低秩投影、分块池化和轻量时空编码器。高维张量路线用于回答低维统计损失了多少可预测信息，并通过匹配 skip budget 的实验判断表示收益是否覆盖额外推理成本。

下一轮实验应扩大到至少几十条 prompt，保存独立随机 seed，并在 `11`、`21`、`24` 和 `28` 个平均 skip 附近做匹配预算比较。每个配置报告质量均值与离散程度，latency 使用重复运行的中位数。这样可以把调度收益、residual correction 收益和模型表示收益分别测出来。
