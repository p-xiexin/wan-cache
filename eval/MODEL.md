# 模型与缓存方法接口

`eval/model` 统一管理模型、loss 和在线缓存方法。

```text
eval/model/
  base.py
  origin.py
  easycache.py
  cumulative.py
  temporal.py
  magcache.py
  d2cache.py
```

[base.py](model/base.py) 提供公共 `CacheMethod`。方法对象持有一次视频生成的完整缓存状态。conditional forward 决定整个 CFG pair 是否跳过，unconditional forward 复用同一决定。完整 DiT 返回后，`update` 更新两路 residual cache 和历史。

```python
class CacheMethod:
    def reset(self, sample_steps, warmup_steps, final_full_steps): ...
    def try_skip(self, raw_input, timestep): ...
    def predict_cached_residual(self, raw_input, timestep, is_conditional): ...
    def update(self, raw_input, output): ...
    def summary(self): ...
```

[cache_hook.py](cache_hook.py) 只负责临时绑定和恢复 Wan forward。Origin 不安装 hook，因此基线 latency 不包含缓存控制开销。

## 公共 8 维特征

累计 MLP 和 Temporal 使用同一组在线特征。

```text
0  归一化 timestep
1  当前输入相对变化
2  上一步输入相对变化
3  当前输入平均绝对值
4  上一步输入平均绝对值
5  上一步输出相对变化
6  residual cache 相对幅值
7  采样进度
```

旧的 `*_lazy_data.pt` 保存每个决策点的 `features [N,8]`、局部 cache error `targets [N]` 和 `step_indices [N]`。`TrajectoryDataModule` 先按轨迹文件划分训练集与验证集，再构造连续窗口。

第 `q` 个决策点的 horizon target 为

```text
target[h] = sum(targets[q : q + h + 1])
```

轨迹末端不足 horizon 的位置由 mask 屏蔽。这个 target 是未来局部误差的累计风险代理。

## 累计 MLP

[cumulative.py](model/cumulative.py) 中的 `CacheModel` 接收 `[B,8]`，输出 `[B,H]`。网络预测正风险增量，再通过 `softplus` 和 `cumsum` 生成单调累计风险。

`CacheLoss` 使用分位数回归、低估惩罚、不安全区间惩罚、阈值附近加权和有效 mask。

`CumulativeMethod` 对模型输出加入校准偏移，再取严格低于 cache threshold 的最长连续前缀。计划前缀结束后强制完整计算一个 CFG pair，更新 residual 后再规划。

## Temporal

Temporal 直接复用相同的 `*_lazy_data.pt`，不需要重新采集轨迹。

`TrajectoryDataModule` 在旧轨迹内截取长度为 `K` 的连续特征序列。

```text
features  [B,K+1,8]
targets   [B,H]
masks     [B,H]
```

默认使用 6 个历史 token 和 1 个当前 query，输入形状为 `[B,7,8]`。target 从 query 所在的决策点开始累计未来局部误差。特征均值和标准差只由训练集在 batch 与 time 两个维度上计算，最终仍是 8 维向量。

[temporal.py](model/temporal.py) 中的 `TemporalModel` 使用单层 GRU 编码前 `K` 个历史 token，将 GRU hidden 与当前 query 拼接后预测正风险增量。输出通过 `cumsum` 形成 `[B,H]` 累计风险。

```python
class TemporalModel(torch.nn.Module):
    def __init__(
        self,
        input_dim=8,
        history_length=6,
        hidden_dim=32,
        horizon=4,
    ): ...

    def forward(self, features): ...
```

`TemporalLoss` 复用累计 MLP 的风险 loss。`TemporalMethod` 只把完整计算过的 query 特征加入历史。历史不足 6 个 token 时继续完整计算。历史就绪后，方法把这些 token 与当前 query 组成长度为 7 的序列。模型预测出正的 skip 前缀后会清空历史，前缀结束时强制刷新，并重新积累连续的完整计算 token。

训练仍然使用旧的 full-compute 轨迹。在线出现 skip 后，8 维 query 中的输入和输出历史会随闭环状态变化。这部分分布偏移无法从旧文件还原，需要通过真实视频质量实验判断。多步 target 同样是局部误差的累计代理，不代表同一 residual 连续复用多步的精确 rollout error。现有旧数据与默认评测都使用 `sample_guide_scale=5.0`，修改生成配置后需要重新验证模型。

Temporal 需要重新训练 GRU artifact，但无需重新运行 Wan 数据采集。

训练时在 [conf/train.yaml](conf/train.yaml) 中启用 `eval.model.temporal.TemporalModel`，并运行同一个 `python -m eval.train` 入口。

## MagCache output

[magcache.py](model/magcache.py) 移植自 [MagCache 官方 Wan2.2 实现](https://github.com/Zehong-Ma/MagCache/blob/main/MagCache4Wan2.2/magcache_generate.py)。配置保存 TI2V-5B T2V conditional 分支的逐 pair magnitude ratio。方法累计 ratio 乘积相对 1 的偏差，并在误差达到阈值或连续缓存达到上限时执行完整 DiT。

当前实现缓存完整 DiT 的 output residual。MagCache 原实现缓存 transformer blocks 产生的 feature residual。因此 `magcache_output` 用于检验官方 timestep 先验在 output-level cache 上是否有效，不能视作 block-level 原版复现。

## D2Cache output

[d2cache.py](model/d2cache.py) 的 residual 修正来自 [D2Cache 官方 Wan2.1 实现](https://github.com/VG-Huai/D2Cache/blob/main/D2Cache4Wan2.1/delta2cache_generate.py)。每次完整计算后，conditional 和 unconditional 分支分别保存最新 residual delta。跳过时使用

```text
predicted_residual = cached_residual + scale * residual_delta
```

调度沿用当前 output-level EasyCache 累计误差，使 `easycache` 与 `d2cache_output` 的差异集中在 residual 修正。D2Cache 原版在 transformer block feature 上使用 TeaCache 调度和 delta correction。当前实现是用于方向验证的 output-level adaptation。

## 部署 artifact

累计 MLP 和 Temporal 使用相同的外层格式。

```text
model_config
model_state_dict
feature_mean
feature_std
calibration_offsets
cache_threshold
```

`model_config` 由 Hydra 直接实例化对应模型。两种 artifact 的 feature mean 和 std 都是 8 维。模型输出加上 calibration offsets 后取 `cummax`，避免后续 horizon 风险下降。

## 方法边界

[origin.py](model/origin.py) 始终执行完整 Wan。[easycache.py](model/easycache.py) 实现原始 EasyCache 累计误差与 full-to-full 变化率。[cumulative.py](model/cumulative.py) 和 [temporal.py](model/temporal.py) 分别实现单步 MLP 与时序 GRU 的在线前缀规划。[magcache.py](model/magcache.py) 和 [d2cache.py](model/d2cache.py) 提供两种 output-level 快速消融。

修改网络层和 loss 后需要重新训练对应 artifact。修改 8 维特征、累计 target 或缓存刷新语义时，需要同步修改数据窗口和在线方法。

## 默认实验

[conf/sweep.yaml](conf/sweep.yaml) 包含 Origin、EasyCache、累计 Model、Temporal、MagCache output 和 D2Cache output，共展开 36 个生成任务。[conf/quality.yaml](conf/quality.yaml) 定义 33 个 Origin 与候选视频配对。

```bash
python -m unittest discover -s eval/tests -v
python -m eval.pipeline runtime.dry_run=true
```

CPU 测试验证数据窗口、模型反传、artifact 往返、缓存状态机和任务展开。真实 Wan、GPU latency 和 Temporal 闭环质量尚未验证。
