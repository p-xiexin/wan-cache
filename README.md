# Wan2.2 缓存方法科研流水线

`eval` 包含模型训练、视频生成、性能汇总和视频质量评测。历史实现保留在 `EasyCache4Wan2.2`，当前实验入口集中在这里。

当前方法、流水线、服务器实验和后续研究路线汇总在 [RESEARCH_REPORT.md](RESEARCH_REPORT.md)。

```text
python -m eval.train
python -m eval.pipeline
python -m eval.evaluate --config eval/conf/quality.yaml
```

视频生成与指标计算分成两个阶段。生成阶段只写视频、结果 JSON 和原始日志。评测阶段读取这些结果，计算 latency、skip、speedup、PSNR、SSIM、LPIPS 和 FVD。

## 目录

```text
eval/
  cache_hook.py
  dataloader.py
  artifacts.py
  train.py
  collect_raw_trajectories.py
  pipeline.py
  evaluate.py
  summarize.py
  evaluate_quality.py
  prompts.txt
  model/
    base.py
    origin.py
    easycache.py
    cumulative.py
    temporal.py
    magcache.py
    d2cache.py
  conf/
    train.yaml
    collect_raw_trajectories.yaml
    sweep.yaml
    quality.yaml
  slurm/
    run_train.sbatch
    run_collect_raw_trajectories.sbatch
    run_generate_8gpu.sbatch
    run_evaluate.sbatch
```

[model/base.py](model/base.py) 管理 CFG pair、conditional 与 unconditional residual cache、保护区间和公共统计。[cache_hook.py](cache_hook.py) 只在一次视频生成期间临时替换 Wan forward，结束后恢复原始 forward。每种方法位于独立模块。

## 原始轨迹采集

[collect_raw_trajectories.py](collect_raw_trajectories.py) 为每个扩散步保存 conditional 和 unconditional 两个 CFG 分支。每条记录只包含 Wan forward 接收的原始 `model_input`、完整计算后返回的原始 `model_output`、timestep、step index 和 branch。张量按 Wan 实际 dtype 与 shape 写盘，不生成 delta、k、均值、范数、cache error 或其他派生特征。

每条轨迹使用独立目录，每 5 个扩散步写一个 shard。所有 shard 和 metadata 完整落盘后才写 `_SUCCESS` 并发布轨迹目录。重启任务时会跳过带 `_SUCCESS` 的轨迹，并重新采集上次中断的单条轨迹。

```text
raw_wan_trajectories/
  dataset.yaml
  manifest.jsonl
  trajectories/
    trajectory_00000000/
      shard_0000.pt
      shard_0001.pt
      ...
      metadata.json
      _SUCCESS
```

单机多 GPU 采集使用下面的入口。

```bash
python -m eval.collect_raw_trajectories \
  paths.checkpoint_dir=/path/to/Wan2.2-TI2V-5B \
  paths.output_root=/path/to/raw_wan_trajectories \
  data.prompt_file=/path/to/prompts.txt \
  parallel.num_processes=auto
```

单机八卡使用下面的 Slurm 入口。8 个进程分别绑定一张 GPU，并按 rank 轮询处理完整轨迹。

```bash
sbatch eval/slurm/run_collect_raw_trajectories.sbatch
```

同一个输出目录应保持 prompt 文件、seed 和生成参数不变。每条轨迹完成后日志会报告实际 GiB，正式扩量前应先采集一条轨迹并据此核算总容量。

## 学习式残差预测器

[PLAN.md](PLAN.md) 的三系数预测器已接入训练与生成入口。配置见 [train_predictor.yaml](conf/train_predictor.yaml) 和 [sweep_predictor.yaml](conf/sweep_predictor.yaml)。

离线训练先按 prompt 划分训练/验证集，同一 prompt 的所有 seed 放在同一侧。每条 raw 轨迹遍历 t=3,…,T−1，为每个 t 直接采样 k<j<i<t，保留完整 latent 时空范围，不执行跳过判断；训练历史组合随 epoch 更新，验证组合固定。需要至少两个不同 prompt 的已完成轨迹和 dataset.yaml。默认使用 Wan2.2-TI2V-5B 的 48 个 latent 通道，其他模型需修改 model.channels。

~~~bash
python -m eval.train --config-name=train_predictor \
  data.data_dir=/path/to/raw_wan_trajectories
~~~

离线预训练默认使用所有可见 GPU，每卡一个 DDP 进程；用 CUDA_VISIBLE_DEVICES 选择显卡，或 parallel.num_processes=1 切回单卡。data.batch_size=auto 按完整 latent 尺寸实测前向、反向和优化器显存，使用 80% 可用显存预算，各卡统一采用最小可用 batch。日志给出每卡 batch 和全局 batch，也可直接指定 data.batch_size=2 等固定值。

首个 batch、每 10 个 batch 和阶段末尾会输出 loss、吞吐、读取/计算耗时、显存峰值与 ETA；耗时操作每 30 秒报告当前阶段。data.num_workers=auto 按分配的 CPU 数设置每卡最多两个读取进程，预取深度为 1；共享内存不足时可设 data.num_workers=0。训练尾部最多补齐 GPU 数减一个样本，保证各卡步数一致；验证样本不补齐、不重复，只有主进程保存权重和训练记录。

训练 Slurm 脚本默认申请 8 张 GPU，可在提交时改为 4 张；普通 Python 入口会自动启动 DDP，无需额外套一层 torchrun：

~~~bash
sbatch --gpus-per-node=4 --cpus-per-task=16 eval/slurm/run_train.sbatch
~~~

offline 和 rollout 均默认使用所有可见 GPU。自动 batch 用于 offline；rollout 每卡独立推进一条完整轨迹，全局同时处理最多 GPU 数条轨迹。独立视频质量验证 validate 仍使用单卡。

输出到 eval/artifacts/polynomial，包含 model.pth、history.csv 和 summary.json；保存 prompt 划分，并记录预测器、残差复用与标准二次外推的节点 MAE，作为预训练诊断。

短段闭环微调从 raw 初始 latent 开始，重新构建文本条件，每卡使用自己的冻结 Wan 和 scheduler 推进。跳过节点的教师输出只作标签。各卡保留独立的刷新判断；本地遇到刷新提前反传并截断梯度，每隔最多 4 个去噪节点在公共边界汇总梯度，按实际预测节点数平均后统一更新参数。没有预测或没有尾部轨迹的卡贡献零梯度；缓存与 solver 数值状态继续保留，不增加刷新或重复轨迹。验证轨迹也按卡分摊，记录完整轨迹漂移并据此选取权重。当前支持 CUDA 上的 T2V：

~~~bash
python -m eval.train --config-name=train_predictor \
  data.data_dir=/path/to/raw_wan_trajectories \
  train.stage=rollout \
  init_artifact=eval/artifacts/polynomial/model.pth \
  output_dir=eval/artifacts/polynomial_rollout \
  paths.checkpoint_dir=/path/to/Wan2.2-TI2V-5B
~~~

同一配置可通过 Slurm 多卡运行：

~~~bash
sbatch --gpus-per-node=4 --cpus-per-task=16 eval/slurm/run_train.sbatch \
  train.stage=rollout init_artifact=eval/artifacts/polynomial/model.pth \
  output_dir=eval/artifacts/polynomial_rollout
~~~

完整验证沿用训练时的 data、seed 和 val_ratio，在保留的 prompt 上生成 Origin/预测器配对视频，计算 PSNR、SSIM、LPIPS、FVD；validation.json 保存每条轨迹的漂移曲线与最终误差。两个质量模型需使用本地权重：

~~~bash
python -m eval.train --config-name=train_predictor \
  data.data_dir=/path/to/raw_wan_trajectories train.stage=validate \
  init_artifact=eval/artifacts/polynomial_rollout/model.pth \
  output_dir=eval/outputs/polynomial_validation \
  paths.checkpoint_dir=/path/to/Wan2.2-TI2V-5B \
  validation.alexnet_path=/path/to/alexnet-owt-7be5be79.pth \
  validation.i3d_path=/path/to/i3d_torchscript.pt
~~~

参考末态由 raw 中的完整 DiT 输出经独立 scheduler 重建；学生轨迹保留自己的 solver 状态。质量评测失败时保留视频、漂移记录与 quality.yaml，可通过 evaluate_quality 重新评测。

生成配置在相同保护区间和阈值下比较 Origin、EasyCache、D2Cache output 与预测器：

~~~bash
python -m eval.pipeline --config-name=sweep_predictor \
  paths.checkpoint_dir=/path/to/Wan2.2-TI2V-5B \
  paths.predictor_artifact=eval/artifacts/polynomial/model.pth

python -m eval.evaluate --config eval/conf/quality_predictor.yaml \
  alexnet_path=/path/to/alexnet-owt-7be5be79.pth \
  i3d_path=/path/to/i3d_torchscript.pt
~~~

[run_train.sbatch](slurm/run_train.sbatch)、[run_generate_8gpu.sbatch](slurm/run_generate_8gpu.sbatch) 和 [run_evaluate.sbatch](slurm/run_evaluate.sbatch) 默认使用预测器配置，并透传命令行参数。提交前设置 EASYCACHE_ROOT、CONDA_SH、WAN_CONDA_ENV、TRAIN_DATA_DIR、WAN_CKPT_DIR；PREDICTOR_ARTIFACT 选择权重，ALEXNET_PATH/I3D_PATH 指定质量模型。生成与评测共用 EVAL_OUTPUT_ROOT。旧流程用 TRAIN_CONFIG=train、GENERATE_CONFIG=sweep、QUALITY_CONFIG=eval/conf/quality.yaml 选择。

新 raw 采集会保存 solver sigma 网格。旧数据按 dataset.yaml 的 solver、步数与 shift 重建，并逐节点核对 timestep；不会把取整后的 timestep 直接除以 1000。分组和逐通道方案仍保留在草稿中。

小张量流程测试可运行 python -m unittest discover -s eval/tests -p test_polynomial.py；覆盖 prompt 隔离、直接时序采样、不等间距外推、刷新判据、权重往返、闭环反传、完整漂移及视频评测接入。

## 原有风险模型训练

累计 MLP 和 Temporal 共用已有的 `*_lazy_data.pt`，无需为 Temporal 重新运行 Wan 采集。两种模型共用 [conf/train.yaml](conf/train.yaml)，通过注释切换下面两个 `model` 块，并同步切换文件顶部的输出目录。

```yaml
# 累计模型
model:
  _target_: eval.model.CacheModel
  input_dim: 8
  horizon: 4

# Temporal 模型
# model:
#   _target_: eval.model.TemporalModel
#   input_dim: 8
#   history_length: 6
#   hidden_dim: 32
#   head_dim: 64
#   horizon: 4
```

累计 MLP 使用单个 8 维特征。Temporal 使用 6 个历史 token 和当前 query，输入形状为 `[B,7,8]`。`data.history_length` 会从当前模型配置读取，数据部分无需手动修改。两者的 target 都从当前位置开始累计旧数据中的未来局部误差。

```bash
python -m eval.train \
  data.data_dir=/path/to/lazy_trajectories
```

训练 Temporal 时切换 target，并把输出目录改为 `eval/artifacts/temporal`。每个训练目录包含 `model.pth`、`history.csv`、`summary.json` 和 Hydra 日志。Temporal 需要单独训练自己的 artifact，累计 MLP 的权重不能直接加载到 GRU。模型与数据合同见 [MODEL.md](MODEL.md)。

对应的 Slurm 入口如下。

```bash
TRAIN_CONFIG=train sbatch eval/slurm/run_train.sbatch
```

脚本通过上述环境变量配置路径。旧模型输出目录在 `train.yaml` 顶部与模型一起切换。

## 视频生成

[conf/sweep.yaml](conf/sweep.yaml) 默认使用三条 prompt，并展开 39 个任务。`prompts.seed` 是所有 prompt 和方法共同使用的单个随机种子，保证每个方法对同一 prompt 使用完全相同的采样初值。

```text
Origin       1 个配置 × 3 条 prompt = 3
EasyCache    3 个阈值 × 3 条 prompt = 9
Model        3 个阈值 × 3 条 prompt = 9
Temporal     3 个阈值 × 3 条 prompt = 9
MagCache output  1 个配置 × 3 条 prompt = 3
D2Cache output   1 个配置 × 3 条 prompt = 3
Piecewise       1 个配置 × 3 条 prompt = 3
```

`magcache_output` 使用 MagCache 官方 Wan2.2 TI2V-5B T2V magnitude ratio、0.06 阈值和最大连续缓存 2 个 pair。`d2cache_output` 使用与 EasyCache 相同的 output-level 调度，并用前两次完整计算得到的 residual delta 修正缓存输出。两者都是完整 DiT output residual 上的快速研究适配。MagCache 和 D2Cache 原实现都作用在 transformer block feature 上，当前结果不能直接与论文数字等同。

`piecewise` 使用已训练的分段多项式先验，通过真实输出节点预测后续 DiT 输出。直接修改该方法的 `use_online_fit` 和 `use_mirror_node` 即可对比，默认 `true/false`；`false/false` 关闭在线校正，`true/true` 加入镜像节点。系数路径为 `paths.piecewise_artifact`，方法说明见 [PIECEWISE_DEPLOYMENT.txt](PIECEWISE_DEPLOYMENT.txt)。

Wan 必须安装在当前 conda 环境中。`paths.checkpoint_dir` 指向 Wan 权重目录，`paths.model_artifact` 和 `paths.temporal_artifact` 指向两套学习方法的权重。

```bash
python -m eval.pipeline runtime.dry_run=true
GENERATE_CONFIG=sweep sbatch eval/slurm/run_generate_8gpu.sbatch
```

生成脚本只启动一个 Python 主进程。`parallel.num_processes=auto` 根据可见 GPU 数量创建 worker。每个 worker 只加载一次 Wan，并连续处理分配到该 GPU 的视频。每个视频重新实例化缓存方法，避免状态跨视频残留。

任务按方法和阈值建目录，prompt 直接按输入顺序编号。

```text
<output_root>/
  tasks.jsonl
  origin/
    0.mp4
    0.result.json
    0.raw.log
  easycache_0.03/
    0.mp4
    1.mp4
    2.mp4
  model_0.05/
    0.mp4
    1.mp4
    2.mp4
  temporal_0.07/
    0.mp4
    1.mp4
    2.mp4
  magcache_output_0.06/
    0.mp4
    1.mp4
    2.mp4
  d2cache_output_0.05/
    0.mp4
    1.mp4
    2.mp4
```

启动时会先读取目标目录中的 `tasks.jsonl`。任务记录存在、对应的 `<prompt>.result.json` 已生成且包含当前前向计时字段时，该任务会直接跳过。旧结果缺少前向计时字段时会自动重新生成。

每个结果的 `timing` 同时保存整段 `generation_seconds`、完整 DiT 前向总时间与调用次数、缓存命中路径总时间与调用次数。前向计时在 GPU 上使用 CUDA Event，不在每次调用后同步。`dit_forward_mean_milliseconds` 只覆盖原始 Wan DiT forward。`cache_forward_mean_milliseconds` 覆盖命中调用中的输入复制、缓存决策和 residual 重建。`dit_to_cache_speedup` 是两种调用平均 GPU 时间之比。

Hydra 主日志位于 `<output_root>/hydra/<slurm_job_id>`，Wan 的逐视频原始输出位于对应的 `<prompt>.raw.log`。

## 指标评测

视频全部生成后运行统一入口。

```bash
python -m eval.evaluate --config eval/conf/quality.yaml
sbatch eval/slurm/run_evaluate.sbatch
```

`evaluate` 和 `evaluate_quality` 接受 YAML 后面的 OmegaConf dotlist override。可以直接切换实验目录、GPU 和任意嵌套配置，无需复制 YAML。

```bash
python -m eval.evaluate \
  --config eval/conf/quality.yaml \
  experiment_root=/path/to/experiment \
  device=cuda:1 \
  fvd_frame_stride=8
```

override 使用 `key=value` 语法。`experiment_root` 为相对路径时相对 `project_root` 解析，绝对路径直接使用。

性能文件写入 `<experiment_root>/metrics`。

```text
performance_per_run.csv
performance_summary.csv
evaluation_summary.md
```

`evaluation_summary.md` 将每个方法和阈值的 latency、整体 speedup、每段视频的 DiT 累计时间、skip、PSNR、SSIM、LPIPS 和 FVD 合并成一张双层表头的结果表。Cache 时间、调用数和单次前向均值保留在性能 CSV 中，不占用主表列。

质量文件写入 `<experiment_root>/metrics/quality`。

```text
quality_per_video.csv
quality_summary.csv
```

[conf/quality.yaml](conf/quality.yaml) 显式列出 3 个 Origin 视频和 11 个候选配置，共 33 个质量配对。`alexnet_path` 必须指向本地 `alexnet-owt-7be5be79.pth`。`i3d_path` 必须指向 StyleGAN V 复现的 Kinetics 400 `i3d_torchscript.pt`。两个模型均从本地加载，计算节点不会联网下载权重。

PSNR、SSIM 和 LPIPS 按配对视频逐帧计算后取均值。FVD 按方法和阈值形成视频分布，使用 16 帧 I3D 特征和 8 帧采样步长，覆盖当前 121 帧视频的第 0 到 120 帧。FVD 只写入 `quality_summary.csv` 和 `evaluation_summary.md`，值越低越好。默认实验每组只有 3 段视频，协方差估计的统计方差很高，正式报告应扩大 prompt 和 seed 数量。

生成目录必须与所选质量配置的 `experiment_root` 一致：预测器流程使用 `quality_predictor.yaml`，原有流程使用 `quality.yaml`。

## 验证边界

```bash
python -m unittest discover -s eval/tests -v
python -m eval.pipeline runtime.dry_run=true
python -m eval.evaluate_quality --config eval/conf/quality.yaml --check-config
```

CPU 测试覆盖数据窗口、训练、artifact 往返、缓存状态机、任务展开和指标汇总。真实 Wan 生成、GPU latency、Temporal 在线质量和多卡 Slurm 执行仍需在服务器验证。
