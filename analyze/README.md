# raw_traj 离线分析

两个脚本读取 `collect_raw_trajectories.py` 保存的 raw 数据，无需 Wan 权重或 GPU，不联网。它们借鉴论文做 **latent/output 层面的诊断**，不复现完整 ICC 或 SVD-Cache，也不分析 Transformer 隐藏 token。图表、数值和路径信息均保存在服务器本地。

整个 `analyze/` 目录可以独立复制到服务器，不依赖仓库其他文件：

```text
analyze/
  analyze_raw_icc.py
  analyze_raw_svd.py
  raw_analysis.py
  requirements.txt
  README.md
  tests/test_raw_analysis.py
```

## 运行

进入复制后的 `analyze/` 目录，在可用环境中准备依赖（PyTorch 2.1+、NumPy、Matplotlib）：

```bash
cd /server/path/analyze
python -m pip install -r requirements.txt
```

若服务器不能联网，提前在允许的环境准备依赖；分析脚本本身不发起网络请求。

分别修改 `analyze_raw_icc.py`、`analyze_raw_svd.py` 顶部常量：

```python
RAW_TRAJ_ROOT = "/server/path/raw_wan_trajectories"
OUTPUT_ROOT = "/server/path/raw_analysis"
TARGET = "residual"      # residual: v-x；x: 输入；v: 模型输出
MAX_TRAJECTORIES = 3     # 0 表示全部
MAX_TOKENS = 4096        # 每条轨迹固定抽样位置；0 表示全部，内存可能很大
```

执行：

```bash
python analyze_raw_icc.py
python analyze_raw_svd.py
```

也支持通过环境变量一次配置两个脚本的数据和输出路径（保留默认常量定义时生效）：

```bash
export RAW_TRAJ_ROOT='/server/path/raw_wan_trajectories'
export RAW_ANALYSIS_OUTPUT='/server/path/raw_analysis'
python analyze_raw_icc.py
python analyze_raw_svd.py
```

路径支持 `~`、`$变量`、`${变量}`，未定义的变量会报错。**相对路径以脚本所在目录为基准，不依赖启动目录。** 输入可为：

- 数据集根目录，下有 `trajectory_*`。
- 数据集根目录，下有 `trajectories/trajectory_*`，或直接传入 `trajectories/`。
- 单条 `trajectory_*` 目录。

只读取带 `_SUCCESS` 的轨迹，并按 `metadata.json` 中的 `shards` 校验文件。缺 shard、重复/缺失步骤、分支不齐、shape 不匹配或 timestep 非单调会报错。数据需要每步两个 CFG 分支和列表/元组形式的 `[C,F,H,W]` 输入输出。

每次加载一条轨迹；shard 使用内存映射，先抽样再转 float32。各步骤与 CFG 分支沿用相同的抽样位置，位置编号保存到 NPZ。后续统计使用 float64。结果是抽样估计，不能把采样位置称作模型内部 token。默认最多读取排序后的前三条轨迹，不能据此作总体代表性结论。

默认输出目录为迁移后 `analyze/outputs/raw_analysis/`。可在 `analyze/` 下运行 `python -m unittest discover -s tests -p 'test_raw_analysis.py' -v` 检查环境与脚本；测试只使用临时生成的合成数据。

## ICC 思路：通道分析

参考 [ICC 官方代码](https://github.com/ccccczzy/icc) 和 [论文 §3.4](https://arxiv.org/html/2505.05829v1)。借鉴 CA/CD 的激活与增量尺度统计；不做论文中依赖内部权重的 CA-SVD/CD-SVD 校准。

输出位于 `OUTPUT_ROOT/icc/TARGET/trajectory_*/分支_item编号/`：

- `channels.png`：每步通道 RMS、增量 RMS、相对增量、按模型时间间隔归一化的增量、增量通道相关矩阵、通道增量能量占比。
- `per_step_channel.csv`：逐步逐通道统计，额外包含平均绝对值、增量平均绝对值、绝对值 p99/RMS 和连续增量方向余弦。
- `channels.csv`：通道汇总。
- `statistics.npz`：全部统计矩阵、步骤、timestep、抽样位置。
- `summary.json`：来源、shape、dtype、抽样信息、变化能量最高的通道。

这里 `delta[t] = value[t] - value[t-1]`；相关性是在所有相邻步和抽样位置上计算的 Pearson 相关，不直接证明通道分组跨阶段稳定。检查绝对变化与相对变化，避免把“大幅值”当成“难预测”。

## SVD-Cache 思路：子空间分析

参考 [SVD-Cache 论文公式 5、6、9、11](https://arxiv.org/html/2601.07396v1)。将 `[C,F,H,W]` 展成 `[FHW,C]` 后抽样，前 `WARMUP_STEPS=3` 步的非中心化 Gram 矩阵确定固定通道基底（等价于对堆叠特征做 SVD），默认保留 90% 能量。`RANK` 可固定维数，否则自动选择。这里的子空间余项与 `TARGET=residual` 中的 `v-x` 不是同一个概念。

输出位于 `OUTPUT_ROOT/svd/TARGET/trajectory_*/分支_item编号/`：

- `subspaces.png`：能量谱、固定基底捕获的特征/增量能量、主成分与余项的变化幅度和方向、预测误差、通道基底。
- `subspaces.csv`、`statistics.npz`：逐步统计、基底、奇异值、timestep、抽样位置。
- `prediction_errors.csv`：每个起点、预测跨度、方法的 MSE 与 NRMSE。
- `prediction_summary.csv`：按预测跨度汇总 NRMSE 和相对复用的 MSE。
- `summary.json`：参数、数据来源和汇总结果。

预测方法包括复用、全空间线性外推、全空间状态 EMA、主子空间状态 EMA + 余项复用、主子空间线性外推 + 余项复用。EMA 按论文公式 11 更新**特征状态**，默认 `beta=0.9`，不冒充导数 EMA。线性外推用记录的真实模型 timestep 间隔缩放，没有假设等间距，也没有擅自将 timestep 转换为 sigma。

每个预测起点使用截至该起点的真实历史；预测跨度默认 `1,2,4` 步，跨度内不读取新真值。基底只使用预热步；误差只评估预热之后的目标。NRMSE 为 `sqrt(sum(MSE) / sum(target_mean_square))`，相对复用 MSE 小于 1 表示优于复用。除数为零的指标输出 JSON null、CSV 空值、NPZ NaN。

基底按轨迹和 CFG 分支分别拟合，尚未验证跨 prompt 基底复用。预测误差属于完整轨迹上的离线检验，不是缓存闭环生成效果或实际加速比。如果自动选择了全部通道，`summary.json` 的 `full_rank=true`，此时没有实质性的子空间划分；不要预设主成分一定更好预测。

两个脚本各自保存 `index.json` 列出本次输出目录。同一配置重复运行会覆盖同名结果；比较不同参数时应更换 `OUTPUT_ROOT`，或先保留上一轮目录。
