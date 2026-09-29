# 局部多项式去噪网络

## 1 EasyCache 与 D2Cache

视频生成过程中第 $t$ 个去噪步的 DiT 写为

$$
v_t=u_\theta(x_t,c,t)
$$

$x_t$ 是输入 latent，$v_t$ 是 DiT 预测的 noise 或 velocity。$\lVert\cdot\rVert$ 表示 L1 norm 与平均操作。

EasyCache 用一阶近似描述相邻去噪步的输入输出映射

$$
\frac{\partial u_\theta}{\partial x_t}
\approx
\frac{v_t-v_{t-1}}{x_t-x_{t-1}}
$$

随后将两个高维变化分别取模，得到相对变换率

$$
k_t=
\frac{\lVert v_t-v_{t-1}\rVert}
{\lVert x_t-x_{t-1}\rVert}
$$

这里正是信息被压缩的位置。$x_t-x_{t-1}$ 和 $v_t-v_{t-1}$ 原本包含符号、方向、通道和时空位置，取模后只剩一个标量 $k_t$。

EasyCache 进一步定义 transformation vector

$$
\Delta_t=v_t-x_t
$$

若 $i$ 是最近一次完整计算步，EasyCache 假设稳定区间内 $\Delta_t\approx\Delta_i$，于是缓存步输出为

$$
\hat v_t=x_t+\Delta_i,
\qquad
\Delta_i=v_i-x_i
$$

EasyCache 使用 $k_i$ 估计当前输出变化，并定义 local stability indicator

$$
\varepsilon_t=
\frac{\lVert v_t-v_{t-1}\rVert}
{\lVert v_{t-1}\rVert}
\approx
\frac{k_i\lVert x_t-x_{t-1}\rVert}
{\lVert v_{t-1}\rVert}
$$

论文公式写有百分号因子。实现中可以直接使用比例，只要阈值 $\tau$ 使用相同尺度。缓存段的累计偏差为

$$
E_t=\sum_{n=i+1}^{t}\varepsilon_n
$$

完整更新规则为

$$
v_t=
\begin{cases}
u_\theta(x_t\mid\mathcal T),
&E_t\geq\tau\ \text{or}\ t\in[0,R-1]\cup\{T-1\}
\\
x_t+\Delta_i,
&E_t<\tau
\end{cases}
$$

D2Cache 补充的是相邻 DiT 输出的一阶 delta 之差。它与上式使用同一条 $v_t$ 轨迹，但二阶项定义在 $v_t$ 的时间差分上。为与本文统一，step index 按实际推理顺序递增，定义

$$
d_i^v=v_i-v_{i-1},
\qquad
a_i^v=d_i^v-d_{i-1}^v
=v_i-2v_{i-1}+v_{i-2}
$$

EasyCache一阶 delta cache 将最近一次变化直接延续到下一步

$$
\hat v_{i+1}^{(1)}=v_i+d_i^v
$$

[D2Cache](https://openaccess.thecvf.com/content/CVPR2026/html/Liu_D2Cache_Second-Order_Delta_Caching_for_Higher_Video_Diffusion_Acceleration_CVPR_2026_paper.html) 进一步加入二阶修正

$$
\hat v_{i+1}^{(2)}=v_i+d_i^v+a_i^v
$$

$d_i^v$ 描述当前输出变化，$a_i^v$ 描述该变化自身的方向与幅值如何继续改变。动态缓存的完整 D2Cache 还会累积最近计算区间内的二阶差分，并使用 timestep embedding 得到的误差比例缩放这项，以适配不等长的缓存区间。后面可以将 $a_i^v$ 作为网络输入，同时提供 timestep 与 cache age，让网络学习对应的二阶修正强度。

## 2 局部多项式去噪网络

我会把方法定义为局部多项式去噪网络。它直接学习 DiT 输出轨迹的局部速度和曲率，预测路径不再使用任何 EasyCache 或 D2Cache 更新公式。

设第 $t$ 个实际推理节点上的完整 DiT 轨迹为

$$
v_t=u_\theta(x_t,c,\tau_t)
$$

### Scheduler 与连续噪声坐标

这里的 $u_\theta$ 与第一章 EasyCache 中的 $u_\theta$ 是同一个冻结的 Wan DiT，$v_t$ 也是 EasyCache 和 D2Cache 所分析的同一个模型输出张量。第一章的 $u_\theta(x_t,c,t)$ 沿用了扩散模型的简写习惯，把轨迹下标和模型时间都写成 $t$。从本节开始将两者分开。下标 $t$ 表示实际推理循环中的节点顺序，$\tau_t$ 表示送入 DiT 时间嵌入的模型时间。对 Wan 而言，$v_t$ 的确切含义是 flow velocity。

在 DDPM 中，scheduler 通过预先定义的 $\bar\alpha_\tau$ 将模型时间 $\tau$ 映射到真实噪声强度

$$
x_\tau=
\sqrt{\bar\alpha_\tau}\,x_0+
\sqrt{1-\bar\alpha_\tau}\,\epsilon
$$

DDPM 中常用的整数时间只是这张噪声表的索引。真正决定样本处于哪个噪声阶段的是 $\bar\alpha_\tau$，等价地也可以使用噪声标准差或 logSNR。训练和推理始终采用同一张表时，整数时间可以唯一确定噪声水平，因此直接输入 timestep 通常有效。

Flow matching 不再通过逐步增加高斯噪声定义离散马尔可夫链，而是先定义数据与噪声之间的连续路径。Wan 使用的线性路径可以写为

$$
x(\sigma)=(1-\sigma)x_0+\sigma\epsilon,
\qquad
\frac{\mathrm d x}{\mathrm d\sigma}
=\epsilon-x_0
$$

#### Flow Matching 的基本目标

最直观的做法是使用神经网络拟合生成概率路径的目标速度场 $u_t(x)$。神经网络以连续时间 $t$ 和空间位置 $x$ 作为输入，学习速度场在不同时间与空间位置上的取值。考虑由参数 $\theta$ 表示的速度场

$$
v_\theta(t,x)
\colon
[0,1]\times\mathbb R^d
\rightarrow
\mathbb R^d
$$

Flow Matching 的目标函数定义为

$$
\mathcal L_{\mathrm{FM}}(\theta)
:=
\mathbb E_{\substack{
t\sim\operatorname{Uniform}(0,1)\\
x\sim p_t(x)
}}
\left[
\left\|
v_\theta(t,x)-u_t(x)
\right\|_2^2
\right]
\tag{4}
$$

该式是最原始的 Flow Matching 目标。网络通过时间 $t$ 和空间位置 $x$ 预测速度向量，并回归能够生成概率路径 $p_t$ 的目标速度场 $u_t$。训练完成后，从初始分布采样并沿学习到的速度场求解常微分方程，即可将样本推进到目标分布。

其中 $\sigma=1$ 对应噪声端，$\sigma=0$ 对应数据端。Wan DiT 学习这条路径上的速度场

$$
u_\theta\!\left(x(\sigma),c,\tau(\sigma)\right)
\approx
\frac{\mathrm d x}{\mathrm d\sigma}
$$

scheduler 的工作是选择有限个连续位置 $\sigma_t$，调用同一个 $u_\theta$ 得到 $v_t$，再用 UniPC 或 DPM Solver 将 $x_t$ 推进到 $x_{t+1}$。Wan 将连续噪声位置转换为模型时间

$$
\tau_t=N_{\mathrm{train}}\sigma_t,
\qquad
v_t=u_\theta(x_t,c,\tau_t)
$$

因此第一章从离散节点观察 $u_\theta$ 的输出变化，本节从连续 flow 路径观察同一个输出。两种写法描述的是同一次 DiT forward。

Wan 的[官方 scheduler](https://github.com/Wan-Video/Wan2.1/blob/main/wan/utils/fm_solvers.py)先生成线性基准网格 $\bar\sigma_t$，再使用 shift 系数 $\rho$ 变换

$$
\sigma_t=
\frac{\rho\bar\sigma_t}
{1+(\rho-1)\bar\sigma_t}
$$

当 $\rho\neq1$ 时，均匀的推理序号会映射到非均匀的 $\sigma$ 网格。以 50 步和 $\rho=5$ 为例，根据该公式得到的第一个 $\sigma$ 间隔约为 $0.0041$，最后一个间隔约为 $0.0926$。两者在推理循环中都只前进一个节点，但后者沿连续路径移动的距离约为前者的 23 倍。

这一区别直接影响后面的局部多项式。若只使用推理序号，每次坐标增量恒为 1，网络无法区分输出变化来自 DiT 轨迹本身，还是来自 scheduler 选取了更长的积分区间。改变采样步数、shift 或 solver 后，同一个推理序号还会对应不同的噪声位置。固定一套 scheduler 时，网络可以把 step index 记成一张隐式查找表，但这种表示不具有跨 schedule 的一致含义。

本文用 $\lambda_t$ 统一表示局部多项式采用的连续噪声坐标。Wan 的首版实现直接取

$$
\lambda_t=\sigma_t
$$

因为 $\sigma_t$ 由 FlowUniPC 和 FlowDPM scheduler 直接提供，并且就是上述 flow 路径的参数。DPM Solver 内部还会使用半 logSNR 坐标

$$
\lambda_t^{\mathrm{DPM}}
=
\log(1-\sigma_t)-\log\sigma_t
$$

它等于 logSNR 的一半，端点计算时需要将 $\sigma_t$ 截断到 $[\epsilon,1-\epsilon]$。首轮实验固定使用 scheduler 的实际 $\sigma_t$，后续再将半 logSNR 作为坐标消融，避免同时改变网络结构和轨迹参数化。

连续坐标在后续网络中承担两个不同作用。绝对坐标 $\lambda_t$ 告诉网络当前位于高噪声、中间稳定区还是低噪声阶段。相邻坐标差

$$
\Delta\lambda_t=\lambda_t-\lambda_{t-1}
$$

表示 scheduler 本轮实际推进的有符号坐标步长。使用 $\lambda_t=\sigma_t$ 时，采样过程中相邻的 $\Delta\lambda_t$ 均为负数，因此前后两个步长的比例仍为正数。后文使用的

$$
s_t=
\frac{\Delta\lambda_t}
{\Delta\lambda_{t-1}+\epsilon}
$$

将当前路径长度归一化到上一个已知区间。局部二次式中的 $s_tA_t^{(1)}$ 和 $s_t^2A_t^{(2)}$ 因而分别按照一次与二次幂响应步长变化。输入中的 $d_{t-1}^v$ 和 $a_{t-1}^v$ 仍是完整张量差分，用来描述最近的输出运动。它们本身并非对 $\lambda$ 的严格导数，网络需要结合 $\lambda_t$、$s_t$ 和当前 scheduler 状态 $x_t$，将历史差分解释为当前节点上的速度与曲率系数。

网络输入为

$$
z_t=
\left[
x_t,\;
v_{t-1},\;
d_{t-1}^v,\;
a_{t-1}^v
\right]
$$

其中

$$
d_{t-1}^v=v_{t-1}-v_{t-2}
$$

$$
a_{t-1}^v=d_{t-1}^v-d_{t-2}^v
$$

同时输入前述连续噪声坐标 $\lambda_t$、局部步长比例 $s_t$ 和文本条件 $c$。

这里保留 $x_t$ 很重要。它是 scheduler 实际推进得到的当前状态，可以让网络在前面产生少量预测误差后重新观察真实的当前轨迹位置。历史 $v$ 则提供局部运动趋势。

网络输出两个与 $v_t$ 同形状的多项式系数场

$$
A_t^{(1)},A_t^{(2)}
=
G_\phi(z_t,\lambda_t,s_t,c)
$$

然后按照局部二次轨迹生成当前 DiT 输出

$$
\hat v_t
=
v_{t-1}
+s_tA_t^{(1)}
+s_t^2A_t^{(2)}
$$

$A_t^{(1)}$ 表示局部速度场，$A_t^{(2)}$ 表示局部曲率场。它们都是完整张量，因此保留通道、正负方向以及时空位置。范数只用于监控，不进入主要预测路径。

我会先采用二次多项式。三次项对短期拟合可能更强，但在递归推理中容易放大系数误差。DiT 的全局轨迹也不适合用一个固定多项式描述。更合理的结构是在每个去噪步重新估计局部二次多项式，由 $\lambda_t$ 控制不同噪声阶段的轨迹形状。图中的 U 型变化会自然体现在随时间变化的 $A_t^{(1)}$ 和 $A_t^{(2)}$ 中，因此不需要手工写入 U 型正则。

### 网络结构

网络本身可以非常小。四个张量沿通道拼接后，通过 $1\times1\times1$ 卷积压缩到 32 个通道，随后使用两个 depthwise separable 3D residual block。噪声时间、步长比例和压缩后的文本条件通过 FiLM 调节两个 block，最后使用 $1\times1\times1$ 卷积输出 $2C$ 个通道，分别对应两个系数场。整个网络不进行空间或时间降采样。

## 3 Loss 与闭环训练

训练时应直接监督一段局部轨迹

$$
\mathcal L_{\mathrm{traj}}
=
\sum_{h=0}^{H-1}
w_h
\operatorname{Charbonnier}
\left(
\hat v_{t+h}-v_{t+h}
\right)
$$

同时约束轨迹的一阶变化

$$
\mathcal L_{\mathrm{vel}}
=
\sum_{h=1}^{H-1}
\operatorname{Charbonnier}
\left[
(\hat v_{t+h}-\hat v_{t+h-1})
-
(v_{t+h}-v_{t+h-1})
\right]
$$

还需要在训练中把 $\hat v_t$ 送入真实 scheduler，得到下一步 $\hat x_{t+1}$，再继续预测。这个闭环 rollout 监督决定了网络能否在推理时连续运行。只使用独立单步样本训练，部署后会快速遇到训练集中没有出现过的预测轨迹。

推理开始时需要前三个 DiT 输出作为多项式初始条件，之后可以完全由轻量网络递归运行。如果连这些启动计算也取消，网络必须从文本和初始噪声独立恢复全部语义信息，此时任务已经变成完整的 student DiT 蒸馏，多项式轨迹模型本身无法提供缺失的初始语义条件。
