# 学习式缓存与局部多项式去噪网络

本文记录两条缓存预测路线。第一章介绍 EasyCache 与 D2Cache，第二章独立给出分段多项式轨迹方案：离线拟合输出变化先验，推理时用真实节点在线校正，以累计预测变化选择刷新节点，并外推 DiT 输出。第四至六章保留学习式残差预测器方案：沿用 EasyCache 调度，在实际 sigma 网格上构造残差差商，由轻量网络预测三个外推权重。两条路线分别定义自己的预测与调度规则。

## 1 EasyCache 与 D2Cache

本章保留 EasyCache、D2Cache 的原理笔记，作为调度与轨迹建模的背景。第二章给出我们的分段多项式方案；第四章另行沿用 EasyCache 的残差定义，在实际 sigma 网格上构造学习式外推。

视频生成过程中第 $t$ 个去噪步的 DiT 写为

$$
v_t=u_\theta(x_t,c,\tau_t)
\tag{1.1}
$$

全文用 $t$ 表示推理节点顺序，$\tau_t$ 表示送入 DiT 的模型时间，$x_t$ 是当前 latent。$v_t$ 是完整 DiT 输出，$\hat v_t$ 表示缓存预测；对 Wan，输出是 flow velocity。两路 CFG 用 $b\in\{c,u\}$ 区分，完整输出统一写为 $v_t^b=u_\theta(x_t,c_b,\tau_t)$，不讨论分支差异时省略上标。$\lVert\cdot\rVert$ 统一表示张量元素的平均绝对值，$\delta>0$ 为变化量归一化中的数值稳定常数。

EasyCache 用一阶近似描述相邻去噪步的输入输出映射

$$
\frac{\partial u_\theta}{\partial x_t}
\approx
\frac{v_t-v_{t-1}}{x_t-x_{t-1}}
\tag{1.2}
$$

随后将两个高维变化分别取模，得到相对变换率

$$
k_t=
\frac{\lVert v_t-v_{t-1}\rVert}
{\lVert x_t-x_{t-1}\rVert}
\tag{1.3}
$$

这里正是信息被压缩的位置。$x_t-x_{t-1}$ 和 $v_t-v_{t-1}$ 原本包含符号、方向、通道和时空位置，取模后只剩一个标量 $k_t$。

EasyCache 进一步定义 transformation vector

$$
\Delta_t=v_t-x_t
\tag{1.4}
$$

若 $i$ 是最近一次完整计算步，EasyCache 假设稳定区间内 $\Delta_t\approx\Delta_i$，于是缓存步输出为

$$
\hat v_t=x_t+\Delta_i,
\qquad
\Delta_i=v_i-x_i
\tag{1.5}
$$

EasyCache 使用 $k_i$ 估计当前输出变化，并定义 local stability indicator

$$
\varepsilon_t=
\frac{\lVert v_t-v_{t-1}\rVert}
{\lVert v_{t-1}\rVert+\delta}
\approx
\frac{k_i\lVert x_t-x_{t-1}\rVert}
{\lVert v_{t-1}\rVert+\delta}
\tag{1.6}
$$

论文公式写有百分号因子。实现中可以直接使用比例，只要阈值 $\tau_{\mathrm{cache}}$ 使用相同尺度。记判断前的累计量为 $E_t^{-}$，缓存段中有

$$
E_t^{-}=\sum_{n=i+1}^{t}\varepsilon_n
\tag{1.7}
$$

完整更新规则为

$$
\begin{cases}
v_t=u_\theta(x_t,c,\tau_t),
&E_t^{-}\geq\tau_{\mathrm{cache}}\ \text{or}\ t\in[0,R-1]\cup\{T-1\}
\\
\hat v_t=x_t+\Delta_i,
&\text{其他情况}
\end{cases}
\tag{1.8}
$$

全文用 $m_t=1$ 表示跳过 DiT、$m_t=0$ 表示完整计算；判断后的累计量为 $E_t$，跳过时保留 $E_t^{-}$，完整计算时清零。$i>j>k$ 依次表示预测步 $t$ 之前最近三个完整计算节点。

D2Cache 补充的是相邻 DiT 输出的一阶 delta 之差。它同样分析 DiT 输出 $v_t$，但二阶项定义在 $v_t$ 的时间差分上。为与本文统一，step index 按实际推理顺序递增，定义

$$
d_i^v=v_i-v_{i-1},
\qquad
a_i^v=d_i^v-d_{i-1}^v
=v_i-2v_{i-1}+v_{i-2}
\tag{1.9}
$$

作为输出轨迹外推的对照，一阶 delta 近似将最近一次变化延续到下一步；它与上面的 EasyCache residual 复用公式不同

$$
\hat v_{i+1}^{(1)}=v_i+d_i^v
\tag{1.10}
$$

[D2Cache](https://openaccess.thecvf.com/content/CVPR2026/html/Liu_D2Cache_Second-Order_Delta_Caching_for_Higher_Video_Diffusion_Acceleration_CVPR_2026_paper.html) 进一步加入二阶修正

$$
\hat v_{i+1}^{(2)}=v_i+d_i^v+a_i^v
\tag{1.11}
$$

$d_i^v$ 描述当前输出变化，$a_i^v$ 描述该变化自身的方向与幅值如何继续改变。动态缓存的完整 D2Cache 还会累积最近计算区间内的二阶差分，并使用 timestep embedding 得到的误差比例缩放这项，以适配不等长的缓存区间。这些差分用于理解输出轨迹的变化；第四章使用完整计算节点的残差差商作为基底，由网络学习外推权重。

## 2 分段多项式先验、在线校正与镜像节点

我们的方案把第一章的相对变化量 $\varepsilon_t$ 建模为一条标量轨迹：大量完整去噪轨迹提供共同形状，推理过程中取得的真实 DiT 输出提供当前样本的校正信息。沿用 $t,\tau_t,x_t,v_t^b,\hat v_t^b$、真实节点 $i>j>k$、跳步标记 $m_t$ 与阈值 $\tau_{\mathrm{cache}}$。标量先验使用 conditional 分支训练，$s$ 是求和或曲线求值时的步坐标；标量曲线确定变化进度与刷新时机，真实输出张量提供方向和空间结构。

本章新增的 $\mathcal O_i$ 表示标量校正记录，$h_i(s)$ 表示对数幅值校正，$\zeta_i(s)$ 表示累计变化坐标，分别与后文的张量缓存 $\mathcal H_t^b$、稳定常数 $\delta$、scheduler shift $\rho$ 区分。离线先验记为 $P_{\mathrm{prior}}$，避免与第四章的张量外推基底 $P_t^{(m),b}$ 混淆。

### 离线分段多项式先验

对第 $n$ 条完整轨迹，定义相邻真实输出的相对变化及其对数

$$
\varepsilon_s^{(n)}=
\frac{\lVert v_s^{c,(n)}-v_{s-1}^{c,(n)}\rVert}
{\lVert v_{s-1}^{c,(n)}\rVert+\delta},
\qquad
\ell_s^{(n)}=\log\max(\varepsilon_s^{(n)},\delta).
\tag{2.1}
$$

设分段节点为 $\kappa_0<\cdots<\kappa_J$，在第 $a$ 段使用局部坐标 $z_a(s)=(s-\kappa_a)/(\kappa_{a+1}-\kappa_a)$，定义

$$
P_{\mathrm{prior}}(s)=P_{\mathrm{prior},a}(s)
=\sum_{r=0}^{3}\beta_{a,r}z_a(s)^r,
\qquad
\varepsilon_{\mathrm{prior}}(s)=\exp P_{\mathrm{prior}}(s),
\quad s\in[\kappa_a,\kappa_{a+1}].
\tag{2.2}
$$

在分段连接处约束关于 $s$ 的零至二阶导数连续，得到 $C^2$ 连续的三次样条。令轨迹数为 $N_{\mathrm{traj}}$、$\bar\ell_s=N_{\mathrm{traj}}^{-1}\sum_{n=1}^{N_{\mathrm{traj}}}\ell_s^{(n)}$，离线系数通过线性最小二乘求得

$$
\begin{aligned}
\beta^*={}&\underset{\beta}{\operatorname{argmin}}\;
\sum_s\bigl(P_{\mathrm{prior}}(s)-\bar\ell_s\bigr)^2
+\lambda_{\mathrm{prior}}\sum_s\bigl(\Delta_s^2P_{\mathrm{prior}}(s)\bigr)^2,\\
\text{s.t. }{}&
P_{\mathrm{prior},a}^{(r)}(\kappa_{a+1})
=P_{\mathrm{prior},a+1}^{(r)}(\kappa_{a+1}),
\quad r=0,1,2.
\end{aligned}
\tag{2.3}
$$

这里 $\Delta_s^2$ 是训练步网格上的二阶差分，$\lambda_{\mathrm{prior}}$ 控制平滑惩罚。对均值曲线拟合与对全部样本的对数曲线求平均平方误差，关于 $\beta$ 的最优解相同。当前产物在验证集选择分段和校正参数后，用全部 $500$ 条轨迹重拟合，保存 $15$ 段三次多项式的 $60$ 个系数，选中的 $\lambda_{\mathrm{prior}}=0$。训练对象是 $\log\varepsilon$；高维 DiT 输出不由这些标量系数直接确定。

### 真实 DiT 节点与稀疏区间观测

在线拟合需要知道：当前样本的输出变化，相比离线先验偏大还是偏小。每次 DiT 刷新后，我们利用真实输出构造一条校正记录，供下一节的多项式拟合使用。设最近一个已完成的 DiT 刷新节点为 $i$，截至 $i$ 可用的记录记为

$$
\mathcal O_i=\{(\xi_\ell,e_\ell)\}_{\ell=1}^{K},
\qquad
e_\ell=\text{当前样本相对于 }P_{\mathrm{prior}}\text{ 的对数幅值偏差}.
\tag{2.4}
$$

如果两个相邻 DiT 刷新节点也是相邻去噪步 $s-1,s$，就可以直接计算式（2.1）的单步变化量，并与先验比较：

$$
\xi_\ell=s,
\qquad
e_\ell=\log\max(\varepsilon_s,\delta)-P_{\mathrm{prior}}(s).
\tag{2.5}
$$

如果两个相邻 DiT 刷新节点 $j<i$ 之间跳过了多个去噪步，中间的真实输出没有计算，因此无法获得逐步变化。此时，我们比较以下两个量：

$$
\underbrace{
\varepsilon_{j:i}=
\frac{\lVert v_i^c-v_j^c\rVert}{\lVert v_j^c\rVert+\delta}
}_{\text{真实输出的端点变化}},
\qquad
\underbrace{
M_{j:i}^{\mathrm{prior}}=\sum_{s=j+1}^{i}\varepsilon_{\mathrm{prior}}(s)
}_{\text{先验预计的区间累计变化}}.
\tag{2.6}
$$

两者的比值用于估计该区间相对先验的幅度偏差，并转换到对数空间：

$$
e_\ell=\log\frac{\max(\varepsilon_{j:i},\delta)}{M_{j:i}^{\mathrm{prior}}}.
\tag{2.7}
$$

例如，先验预计累计变化为 $0.04$，真实端点变化为 $0.06$，则将其近似解读为该区间的变化幅度是先验的 $1.5$ 倍。

将这条偏差信息放在先验变化量加权的区间中心

$$
\xi_\ell=
\frac{\sum_{s=j+1}^{i}s\,\varepsilon_{\mathrm{prior}}(s)}
{M_{j:i}^{\mathrm{prior}}},
\tag{2.8}
$$

得到校正记录 $(\xi_\ell,e_\ell)$，供下一节的在线多项式拟合使用。本节只构造观测数据，不求解在线多项式系数。

端点变化可能低估沿途累计变化，因此这是一条**区间代理观测**，不是某个跳过步骤的真实单步变化。它仅在节点 $i$ 完整计算后加入 $\mathcal O_i$；$\xi_\ell<i$ 不代表可以提前使用。近似条件与推导见本章末尾的补充说明。

### 在线多项式校正与镜像软约束

以最新校正记录 $(\xi_K,e_K)$ 为锚点，用低阶多项式描述当前样本对先验的偏离：

$$
h_i(s)=e_K+
\sum_{r=1}^{p_{\mathrm{fit}}}\alpha_{i,r}
\left(\frac{s-\xi_K}{L_{\mathrm{fit}}}\right)^r,
\qquad
\hat\varepsilon(s\mid i)=\exp\bigl(P_{\mathrm{prior}}(s)+h_i(s)\bigr).
\tag{2.9}
$$

该参数化使 $h_i(\xi_K)=e_K$。当最新记录来自相邻真实输出时，校正曲线在该节点匹配实测变化；来自稀疏区间时，锚定的是式（2.7）的代理校正。没有观测时取 $h_i\equiv0$。离线系数 $\beta^*$ 保持冻结，在线仅求少量 $\alpha_{i,r}$。

近似 U 形轨迹提供镜像约束的动机。设离线校准得到的对称轴为 $\mu$，反射校正坐标，而后将源节点的对数偏差作为软目标：

$$
\widetilde\xi_\ell=2\mu-\xi_\ell,
\qquad
\widetilde e_\ell=e_\ell,
\qquad
\mathcal M_i=
\{\ell:\xi_K\ge\mu,\ \xi_K<\widetilde\xi_\ell\le S_{\max}\}.
\tag{2.10}
$$

这里 $S_{\max}$ 是先验支持的最大步坐标。镜像作用于偏离先验的 $e_\ell$，保留 $P_{\mathrm{prior}}$ 已经拟合出的非对称形状。所有源记录都来自截至 $i$ 已完成的真实计算；镜像坐标只参与拟合，不进入真实输出节点集合。

令 $\mathcal W_i$ 为最近 $W$ 条校正记录，以锚点常数项固定的岭回归求在线系数：

$$
\begin{aligned}
\alpha_i^*=\underset{\alpha_i}{\operatorname{argmin}}\;&
\frac{1}{|\mathcal W_i|}
\sum_{\ell\in\mathcal W_i}
\bigl(h_i(\xi_\ell)-e_\ell\bigr)^2\\
&+\frac{\gamma}{|\mathcal M_i|}
\sum_{\ell\in\mathcal M_i}
\bigl(h_i(\widetilde\xi_\ell)-\widetilde e_\ell\bigr)^2
+\lambda_{\mathrm{fit}}(\xi_K)\lVert\alpha_i\rVert_2^2.
\end{aligned}
\tag{2.11}
$$

关闭镜像或 $\mathcal M_i=\varnothing$ 时省略第二项；关闭在线拟合时直接取 $\alpha_i=0$。当前校准参数为 $p_{\mathrm{fit}}=1$、$W=5$、$L_{\mathrm{fit}}=8$、$\mu=26.7$、$\gamma=10^{-3}$，轴前与轴后的正则分别为 $\lambda_{\mathrm{fit}}=10$、$0.1$。镜像目标采用较小权重，因为曲线的近似对称性不足以保证单条轨迹严格对称。

两个开关对应以下数学模型：

| `use_online_fit` | `use_mirror_node` | 校正方式 |
| --- | --- | --- |
| `false` | `false` | $h_i(s)=e_K$，保留先验形状并做幅值对齐 |
| `true` | `false` | 使用真实节点产生的记录拟合 $\alpha_i$，省略镜像项 |
| `true` | `true` | 在相同先验上加入式（2.10）的镜像软约束 |

`false/true` 不构成独立有效组合。若要区分镜像值与额外约束位置的作用，可在相同镜像坐标上改用 $\widetilde e_\ell=e_K$，对应 `mirror_node_mode=anchor` 的几何对照。

### 实际推理流程

推理时，离线先验保持冻结，前文构造的观测记录用于更新当前校正曲线。每个去噪步按以下顺序执行，将变化预测、DiT 刷新和输出外推串成一个流程。

**1. 初始化真实历史与累计量。** 初始化 $E_{-1}=0$；没有校正记录时取 $h=0$。前端保护区间执行真实 DiT，逐步建立观测记录和两个 CFG 分支的输出历史。记总步数为 $T$、前端与末端保护步数为 $R,F$，允许跳过 DiT 的步集合为 $\mathcal I=\{R,\ldots,T-F-1\}$。

**2. 判断当前步是否需要 DiT 刷新。** 设当前步为 $t$，最近一个 DiT 刷新节点为 $i<t$。若当前步处于保护区间，或真实输出节点不足 $p_{\mathrm{out}}+1$ 个，则直接进入真实计算分支；否则，用式（2.9）的当前曲线预测本步变化，并累加自上次刷新以来的变化：

$$
E_t^{-}=E_{t-1}+\hat\varepsilon(t\mid i)
=\sum_{s=i+1}^{t}\hat\varepsilon(s\mid i),
\qquad
\tau_{\mathrm{cache}}=\texttt{cache\_threshold}.
\tag{2.12}
$$

累计量低于阈值时使用预测输出，达到阈值时执行真实 DiT。两个 CFG 分支共享同一次决策：

$$
m_t=
\begin{cases}
1,&t\in\mathcal I,\ \text{真实节点不少于 }p_{\mathrm{out}}+1,
\ E_t^{-}<\tau_{\mathrm{cache}},\\
0,&\text{其他情况}.
\end{cases}
\tag{2.13}
$$

保护区间和历史不足时不计算式（2.12），直接令 $m_t=0$。

**3. 跳过 DiT 时，用真实输出历史外推。** 当 $m_t=1$，取最近 $p_{\mathrm{out}}+1$ 个真实输出节点 $i_0<\cdots<i_{p_{\mathrm{out}}}=i$，使用同一条当前校正曲线建立累计变化坐标：

$$
\zeta_i(i_0)=0,
\qquad
\zeta_i(s)=\sum_{n=i_0+1}^{s}\hat\varepsilon(n\mid i).
\tag{2.14}
$$

在该坐标中，对每个 CFG 分支的真实输出构造 Lagrange 外推：

$$
\begin{aligned}
\mathcal L_r(\zeta)
&=\prod_{\substack{0\le \ell\le p_{\mathrm{out}}\\\ell\ne r}}
\frac{\zeta-\zeta_i(i_\ell)}{\zeta_i(i_r)-\zeta_i(i_\ell)},\\
\widehat v_t^b
&=\sum_{r=0}^{p_{\mathrm{out}}}
\mathcal L_r\bigl(\zeta_i(t)\bigr)v_{i_r}^b,
\qquad b\in\{c,u\}.
\end{aligned}
\tag{2.15}
$$

默认 `tensor_degree=1`，即使用最近两个真实节点 $j<i$，此时

$$
\widehat v_t^b
=v_i^b+
\frac{\zeta_i(t)-\zeta_i(i)}
{\zeta_i(i)-\zeta_i(j)}
\bigl(v_i^b-v_j^b\bigr).
\tag{2.16}
$$

真实张量差提供变化方向，校正曲线确定外推距离；两分支共享坐标，各自使用独立的真实输出历史。取 $p_{\mathrm{out}}=2$ 时使用 $(k,j,i)$ 三个节点。跳过后保留累计量 $E_t=E_t^{-}$，预测输出不加入真实历史。

**4. 执行真实 DiT 后，更新记录并开始下一轮。** 当 $m_t=0$，计算并保存 $v_t^b=u_\theta(x_t,c_b,\tau_t)$，将累计量清零为 $E_t=0$。有可用历史且观测区间位于先验支持范围内时，用 conditional 输出与上一个 DiT 刷新节点比较：相邻去噪步按式（2.5）生成记录，跨步区间按式（2.7）—（2.8）生成代理记录。随后更新校正曲线，令最近真实节点 $i\leftarrow t$，供下一步使用。镜像只参与校正拟合，不加入真实输出历史。

无论本步使用预测还是真实输出，都按原有 CFG 组合交给 Wan scheduler。两次刷新之间校正记录不变，累计量逐步增加；刷新后，后续预测用更新后的同一条曲线重算相关坐标。该流程不预设连续跳过几步，也不要求每次刷新连续完整计算两步。默认 $R=7$、$F=1$，阈值在现有 sweep 的 `cache_thresholds` 中设置。

$E_t^{-}$ 是预测变化预算，不是 $\lVert\widehat v_t-v_t\rVert$ 的误差上界。开关改变校正曲线后也可能改变刷新节点，实验应同时记录质量、真实 DiT 调用数和耗时。当前系数对应固定的 $50$ 步 TI2V-5B 配置，步坐标支持 $1\ldots48$；不同采样网格需要重新校准。已保存的标量离线指标验证相邻 $\varepsilon$ 的预测，稀疏区间校正与张量外推的闭环视频效果仍由服务器实验检验。实现见 [model/piecewise.py](model/piecewise.py)、[polynomial_prior.py](polynomial_prior.py)，训练系数见 [prior.json](artifacts/piecewise_polynomial/prior.json)。

### 补充推导：区间代理近似

当区间内的输出增量方向近似一致、输出范数变化较小时，端点相对变化可近似为累计单步变化，即 $\varepsilon_{j:i}\approx\sum_{s=j+1}^{i}\varepsilon_s$。若 $\varepsilon_s\approx\varepsilon_{\mathrm{prior}}(s)\exp(h(s))$，则区间对数偏差满足

$$
\log\frac{\sum_{s=j+1}^{i}\varepsilon_{\mathrm{prior}}(s)\exp(h(s))}
{M_{j:i}^{\mathrm{prior}}}
\approx
\sum_{s=j+1}^{i}\frac{\varepsilon_{\mathrm{prior}}(s)}
{M_{j:i}^{\mathrm{prior}}}h(s)
\approx h(\xi_\ell).
$$

第一个近似要求区间内校正的变化较小；当 $h$ 在区间内为仿射函数时，最后一步是精确等式，其中 $\xi_\ell$ 即式（2.8）的加权中心。轨迹折返和输出范数变化可能破坏端点变化与累计变化之间的近似关系，因此式（2.7）始终作为代理偏差使用。

## 3 Flow Matching、DiT 与连续噪声坐标

沿用前两章的记号，$t$ 表示推理节点顺序，$\tau_t$ 表示送入 DiT 时间嵌入的模型时间，$v_t=u_\theta(x_t,c,\tau_t)$ 是同一个冻结 Wan DiT 的 flow velocity 输出。本章引入连续噪声坐标 $\sigma_t$，说明它与模型时间 $\tau_t$ 的关系。第二章的 $\zeta_i(s)$ 仍表示累计预测变化，第四章的多项式使用实际 $\sigma_t$ 网格；这两种坐标具有不同定义。

在 DDPM 中，scheduler 通过预先定义的 $\bar\alpha_\tau$ 将模型时间 $\tau$ 映射到真实噪声强度

$$
x_\tau=
\sqrt{\bar\alpha_\tau}\,x_0+
\sqrt{1-\bar\alpha_\tau}\,\epsilon
\tag{3.1}
$$

DDPM 中常用的整数时间只是这张噪声表的索引。真正决定样本处于哪个噪声阶段的是 $\bar\alpha_\tau$，等价地也可以使用噪声标准差或 logSNR。训练和推理始终采用同一张表时，整数时间可以唯一确定噪声水平，因此直接输入 timestep 通常有效。

Flow matching 不再通过逐步增加高斯噪声定义离散马尔可夫链，而是先定义数据与噪声之间的连续路径。Wan 使用的线性路径可以写为

$$
x(\sigma)=(1-\sigma)x_0+\sigma\epsilon,
\qquad
\frac{\mathrm d x}{\mathrm d\sigma}
=\epsilon-x_0
\tag{3.2}
$$

### Flow Matching 的基本目标

最直观的做法是使用神经网络拟合生成概率路径的目标速度场 $u_\sigma(x)$。神经网络以连续路径参数 $\sigma$ 和空间位置 $x$ 作为输入，学习速度场在不同噪声位置与空间位置上的取值。这里沿用 $\sigma$ 表示连续坐标，$t$ 保留为离散推理节点下标。考虑由参数 $\theta$ 表示的速度场

$$
v_\theta(\sigma,x)
\colon
[0,1]\times\mathbb R^d
\rightarrow
\mathbb R^d
\tag{3.3}
$$

Flow Matching 的目标函数定义为

$$
\mathcal L_{\mathrm{FM}}(\theta)
:=
\mathbb E_{\substack{
\sigma\sim\operatorname{Uniform}(0,1)\\
x\sim p_\sigma(x)
}}
\left[
\left\|
v_\theta(\sigma,x)-u_\sigma(x)
\right\|_2^2
\right]
\tag{3.4}
$$

该式是最原始的 Flow Matching 目标。网络通过连续坐标 $\sigma$ 和空间位置 $x$ 预测速度向量，并回归能够生成概率路径 $p_\sigma$ 的目标速度场 $u_\sigma$。训练完成后，从初始分布采样并沿学习到的速度场求解常微分方程，即可将样本推进到目标分布。

其中 $\sigma=1$ 对应噪声端，$\sigma=0$ 对应数据端。Wan DiT 学习这条路径上的速度场

$$
u_\theta\!\left(x(\sigma),c,\tau(\sigma)\right)
\approx
\frac{\mathrm d x}{\mathrm d\sigma}
\tag{3.5}
$$

### Wan DiT 与轻量预测器

以本仓库使用的 Wan2.2-TI2V-5B 文生视频路径为例，DiT 将 latent 分块并映射为 token，经多层 Transformer 后还原为同形状的 velocity。按[官方模型实现](https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/model.py)抽象为下式；省略 batch 维度，$N$ 为视频 token 数，$D$ 为隐藏维度，$L_{\mathrm{DiT}}$ 为块数：

$$
\begin{aligned}
Z^{(0)}
&=\operatorname{PatchEmbed}(x_t)
&&\in\mathbb R^{N\times D},\\
Z^{(\ell+1)}
&=\mathcal B_\ell(Z^{(\ell)};c,\tau_t)
&&\in\mathbb R^{N\times D},\\
v_t
&=\operatorname{Unpatchify}\!\left(
\operatorname{Head}(Z^{(L_{\mathrm{DiT}})};\tau_t)\right).
\end{aligned}
\tag{3.6}
$$

其中 $\ell=0,\ldots,L_{\mathrm{DiT}}-1$。$\operatorname{PatchEmbed}$ 为分块卷积与序列展开；每个块依次执行时空自注意力、文本交叉注意力和前馈网络：

$$
\mathcal B_\ell
=\operatorname{FFN}_{\ell,\tau_t}^{\mathrm{res}}
\circ\operatorname{CrossAttn}_{\ell,c}^{\mathrm{res}}
\circ\operatorname{SelfAttn}_{\ell,\tau_t}^{\mathrm{res}}.
\tag{3.7}
$$

$\circ$ 按从右到左的顺序复合，$\mathrm{res}$ 表示将对应的归一化和残差连接并入算子。自注意力建立全局时空联系，交叉注意力逐 token 引入文本；时间条件控制自注意力与 FFN 的特征缩放、平移及残差门控。上述隐藏特征均为 $N\times D$，输出头投影后经 $\operatorname{Unpatchify}$ 恢复为与 $x_t$ 相同的形状。

[5B 官方配置](https://github.com/Wan-Video/Wan2.2/blob/main/wan/configs/wan_ti2v_5B.py)使用 30 个块、3072 维隐藏特征、24 个注意力头和 $(1,2,2)$ 的 patch。第四章的预测器使用两个 32 通道的卷积残差块，全局池化后拼接时间与缓存间隔等标量，由小型 MLP 生成三个权重；分支条件信息通过各自的缓存残差提供。完整 DiT 每次重新计算全局速度场；轻量网络利用最近三次完整计算的残差基底调整外推强度，高维变化方向由缓存提供。

### Wan scheduler 与噪声坐标

scheduler 的工作是选择有限个连续位置 $\sigma_t$，调用同一个 $u_\theta$ 得到 $v_t$，再用 UniPC 或 DPM Solver 将 $x_t$ 推进到 $x_{t+1}$。Wan 将连续噪声位置转换为模型时间

$$
\tau_t=N_{\mathrm{train}}\sigma_t,
\qquad
v_t=u_\theta(x_t,c,\tau_t)
\tag{3.8}
$$

因此第一章从离散节点观察 $u_\theta$ 的输出变化，本节从连续 flow 路径观察同一个输出。两种写法描述的是同一次 DiT forward。

Wan 的[官方 scheduler](https://github.com/Wan-Video/Wan2.1/blob/main/wan/utils/fm_solvers.py)先生成线性基准网格 $\bar\sigma_t$，再使用 shift 系数 $\rho$ 变换

$$
\sigma_t=
\frac{\rho\bar\sigma_t}
{1+(\rho-1)\bar\sigma_t}
\tag{3.9}
$$

当 $\rho\neq1$ 时，均匀的推理序号会映射到非均匀的 $\sigma$ 网格。以 50 步和 $\rho=5$ 为例，根据该公式得到的第一个 $\sigma$ 间隔约为 $0.0041$，最后一个间隔约为 $0.0926$。两者在推理循环中都只前进一个节点，但后者沿连续路径移动的距离约为前者的 23 倍。

这一区别直接影响后面的局部多项式。若只使用推理序号，每次坐标增量恒为 1，网络无法区分输出变化来自 DiT 轨迹本身，还是来自 scheduler 选取了更长的积分区间。改变采样步数、shift 或 solver 后，同一个推理序号还会对应不同的噪声位置。固定一套 scheduler 时，网络可以把 step index 记成一张隐式查找表，但这种表示不具有跨 schedule 的一致含义。

后续学习式预测器用 $\lambda_t$ 表示局部多项式采用的连续噪声坐标。该路线在 Wan 上的首版实现直接取

$$
\lambda_t=\sigma_t
\tag{3.10}
$$

因为 $\sigma_t$ 由 FlowUniPC 和 FlowDPM scheduler 直接提供，并且就是上述 flow 路径的参数。DPM Solver 内部还会使用半 logSNR 坐标

$$
\lambda_t^{\mathrm{DPM}}
=
\log(1-\sigma_t)-\log\sigma_t
\tag{3.11}
$$

它等于 logSNR 的一半，端点计算时需要将 $\sigma_t$ 截断到 $[\epsilon,1-\epsilon]$。首轮实验固定使用 scheduler 的实际 $\sigma_t$，后续再将半 logSNR 作为坐标消融，避免同时改变网络结构和轨迹参数化。

## 4 学习式缓存预测器设计

### 完整计算节点与外推基底

沿用第三章的节点下标 $t$、噪声坐标 $\sigma_t$ 和模型时间 $\tau_t$。$b\in\{c,u\}$ 分别表示 conditional 与 unconditional 分支，$v_t^b$ 为冻结 DiT 的完整输出，$\hat v_t^b$ 为缓存预测。以下省略 batch 维度；$C$ 为 latent 通道数，$(n_f,n_h,n_w)$ 为时空尺寸：

$$
\begin{gathered}
\mathcal X_r:=\mathbb R^{r\times n_f\times n_h\times n_w},\\
v_t^b=u_\theta(x_t,c_b,\tau_t),
\qquad
x_t,v_t^b,\hat v_t^b\in\mathcal X_C.
\end{gathered}
\tag{4.1}
$$

用 $m_t=1$ 表示跳过 DiT，$m_t=0$ 表示完整计算，判断条件见式（6.4）。预测节点 $t$ 之前最近三次完整计算的节点为

$$
i=\max\{r<t:m_r=0\},\qquad
j=\max\{r<i:m_r=0\},\qquad
k=\max\{r<j:m_r=0\}.
\tag{4.2}
$$

这里 $i>j>k$ 随完整刷新更新，后文省略它们对 $t$ 的依赖。沿用 EasyCache 的残差定义，在实际 sigma 网格上计算差商：

$$
\begin{aligned}
R_r^b&=v_r^b-x_r,
\qquad r\in\{i,j,k\},\\
D_{ij}^b&=\frac{R_i^b-R_j^b}{\sigma_i-\sigma_j},
\\
D_{jk}^b&=\frac{R_j^b-R_k^b}{\sigma_j-\sigma_k},\\
D_{ijk}^b&=\frac{D_{ij}^b-D_{jk}^b}{\sigma_i-\sigma_k}.
\end{aligned}
\tag{4.3}
$$

$R_r^b,D_{ij}^b,D_{jk}^b,D_{ijk}^b\in\mathcal X_C$。sigma 网格严格递减，三个节点无需相邻；差商分母使用真实间隔。缓存刷新时重建这些基底，连续跳过期间保持不变。

### 网络输入与条件

将缓存残差作为零阶基底，并在当前 $\sigma_t$ 上求值一阶、二阶外推项：

$$
\begin{aligned}
P_t^{(0),b}&=R_i^b
&&\in\mathcal X_C,\\
P_t^{(1),b}&=(\sigma_t-\sigma_i)D_{ij}^b
&&\in\mathcal X_C,\\
P_t^{(2),b}&=(\sigma_t-\sigma_i)(\sigma_t-\sigma_j)D_{ijk}^b
&&\in\mathcal X_C.
\end{aligned}
\tag{4.4}
$$

网络沿通道拼接当前 latent 与三个基底；标量条件包含噪声位置、外推距离、两个历史间隔及缓存年龄：

$$
\begin{aligned}
z_t^b&=[x_t,\;P_t^{(0),b},\;P_t^{(1),b},\;P_t^{(2),b}]
&&\in\mathcal X_{4C},\\
q_t&=[\sigma_t,\;\sigma_t-\sigma_i,\;\sigma_i-\sigma_j,\;
\sigma_j-\sigma_k,\;t-i]
&&\in\mathbb R^5.
\end{aligned}
\tag{4.5}
$$

### 网络结构

网络由输入投影、两个卷积残差块、全局池化和小型 MLP 组成，隐藏通道数 $d=32$。两路 CFG 共享全部参数，各自输入本分支的缓存；文本条件 $c_b$ 仅用于完整 DiT。以下卷积特征省略 $t,b$ 下标。

**输入投影。**

$$
U^{(0)}
=\operatorname{Conv}_{1\times1\times1}^{4C\to d}(z_t^b)
\in\mathcal X_d.
\tag{4.6}
$$

**两个卷积残差块。** 对 $\ell=0,1$，依次执行逐通道卷积、激活与通道混合：

$$
U^{(\ell+1)}
=U^{(\ell)}
+\operatorname{Conv}_{1\times1\times1,\ell}^{d\to d}
\left(\operatorname{SiLU}
\left(\operatorname{DWConv}_{3\times3\times3,\ell}(U^{(\ell)})\right)\right)
\in\mathcal X_d.
\tag{4.7}
$$

所有卷积 stride 为 1；逐通道卷积 padding 为 1，其余为 $1\times1\times1$ 卷积。块内各特征均属于 $\mathcal X_d$，时空尺寸保持不变。

**全局池化。** 对时空维取平均：

$$
p_t^b
=\frac{1}{n_fn_hn_w}
\sum_{f=1}^{n_f}\sum_{h=1}^{n_h}\sum_{w=1}^{n_w}
U^{(2)}[:,f,h,w]
\in\mathbb R^d.
\tag{4.8}
$$

**三个标量输出。** 将池化特征与 $q_t$ 直接拼接，经两层 MLP 得到原始输出，按零、一、二阶编号：

$$
\begin{aligned}
s_t^b
&=\operatorname{SiLU}\left(W_1[p_t^b;q_t]+a_1\right)
&&\in\mathbb R^d,\\
o_t^b
&=W_2s_t^b+a_2
&&\in\mathbb R^3.
\end{aligned}
\tag{4.9}
$$

$[p_t^b;q_t]\in\mathbb R^{d+5}$，$W_1\in\mathbb R^{d\times(d+5)}$、$a_1\in\mathbb R^d$，$W_2\in\mathbb R^{3\times d}$、$a_2\in\mathbb R^3$。三个系数使用相同的有界映射：

$$
A_t^{(m),b}
=A_{\max}\tanh(o_t^b[m])
\in(-A_{\max},A_{\max}),
\qquad m=0,1,2,\quad A_{\max}>1.
\tag{4.10}
$$

$A_{\max}$ 是三个系数共享、训练中固定的幅度上限，取大于 $1$ 以覆盖残差复用和标准二次外推所需的 $0,1$。完整计算节点以 DiT 输出为准。三个系数均为无量纲标量，沿全部通道和时空位置广播。式（4.6）—（4.10）构成权重预测器 $G_\phi$。

### 残差多项式外推

三个系数分别调节缓存残差幅度、一阶趋势和二阶修正：

$$
\hat v_t^b
=x_t+\sum_{m=0}^{2}A_t^{(m),b}P_t^{(m),b}
\in\mathcal X_C.
\tag{4.11}
$$

系数 $(A_t^{(0),b},A_t^{(1),b},A_t^{(2),b})=(1,0,0)$ 时，退回 EasyCache 的残差复用 $x_t+R_i^b$；取 $(1,1,1)$ 时，得到经过三个历史残差节点的标准 Newton 二次多项式外推。$A_t^{(0),b}\ne1$ 时，额外沿缓存残差自身的方向进行修正；高维方向和空间结构仍由缓存提供。

分组与逐通道权重的可选改进保存在 [draft.md](draft.md)，未并入主方案。

## 5 Loss 与闭环训练

训练沿时间构造样本，保留完整 latent 空间范围。先按 prompt 划分训练集与验证集，再构造样本。

**阶段一：时序预训练。** 从完整 raw 轨迹直接采样 $k<j<i<t$，覆盖不同噪声阶段、历史间隔与预测距离，不执行跳过判断。以三个真实历史节点 $(i,j,k)$ 构造外推基底，结合当前 $x_t$ 预测 $v_t^b$，输入与标签均取自真实轨迹。逐节点监督，式（5.1）取 $H=1$，学习去噪轨迹的局部时序先验。

**阶段二：短段闭环微调。** 按第六章的跳过判断与状态递推，从实际推理状态开始，每段连续预测最多 $4$ 步，遇到完整刷新提前结束，实际长度为 $1\le H\le4$。段内固定缓存残差与差商，用预测输出经 CFG 和原 scheduler 推进 latent；教师按式（4.1）在当前 latent 上重新计算标签，停止梯度且不写入缓存。梯度通过段内网络与 scheduler 递推传播，跳过判断不反传。段末截断梯度，保留 latent、solver 和缓存状态继续推理，缓存仅在完整刷新时更新。

损失只包含输出误差与一个弱系数正则，对两路 CFG 和预测节点取平均：

$$
\mathcal L
=\frac{1}{2H}\sum_{b\in\{c,u\}}\sum_{h=0}^{H-1}
\left[
\operatorname{mean}\sqrt{(\hat v_{t+h}^b-v_{t+h}^b)^2+\epsilon^2}
+\frac{\lambda}{3}\sum_{m=0}^{2}(A_{t+h}^{(m),b}-1)^2
\right].
\tag{5.1}
$$

平方与开方逐元素计算，$\operatorname{mean}$ 对张量元素取平均，$\epsilon>0$ 为平滑常数。唯一的正则权重 $\lambda>0$ 取较小值，以标准二次外推的 $(1,1,1)$ 为参考抑制过大偏移，允许三个系数相互配合。预测节点掩码与尺度归一化等候选损失保存在 [draft.md](draft.md)，尚未采用。

验证使用完整生成轨迹，检查累计漂移与最终视频质量。

## 6 部署推理

每个节点依次判断计算条件、生成输出、按需刷新缓存并推进 scheduler。两路 CFG 共用同一个 $m_t$。

### 完整计算条件

沿用 EasyCache 的累计变化量与首尾保护区间。设总步数为 $T$，前 $R$ 步、后 $F$ 步完整计算，其中 $R\ge3$、$F\ge0$、$R+F<T$，允许预测的节点集合为

$$
\mathcal I=\{R,\ldots,T-F-1\}.
\tag{6.1}
$$

沿用第一章的平均绝对值范数，$N_z$ 为张量元素数：

$$
\lVert z\rVert:=\frac{1}{N_z}\sum_{p=1}^{N_z}|z_p|.
\tag{6.2}
$$

完整计算节点 $i,j$ 取自式（4.2）。对 $t\in\mathcal I$，用这两个节点的 conditional 输出估计变化率，再累积当前输入变化：

$$
\begin{aligned}
k_t
&=\frac{\lVert v_i^c-v_j^c\rVert}
{\lVert x_i-x_j\rVert+\delta},\\
E_t^{-}
&=E_{t-1}
+k_t\frac{\lVert x_t-x_{t-1}\rVert}
{\lVert v_i^c\rVert+\delta}.
\end{aligned}
\tag{6.3}
$$

其中 $\delta>0$ 为数值稳定常数。给定阈值 $\tau_{\mathrm{cache}}$，判断并更新累计量：

$$
m_t=
\begin{cases}
1, & t\in\mathcal I\ \text{且}\ E_t^{-}<\tau_{\mathrm{cache}},\\
0, & \text{其他情况},
\end{cases}
\qquad
E_t=
\begin{cases}
E_t^{-}, & m_t=1,\\
0, & m_t=0.
\end{cases}
\tag{6.4}
$$

初始化 $E_{-1}=0$。保护区间内直接完整计算并清零累计量，不计算 $E_t^{-}$。当前输入记录每步更新，完整计算记录按下式更新。

### 输出选择与缓存更新

当 $m_t=0$ 时执行式（4.1），当 $m_t=1$ 时使用已有基底，按式（4.4）—（4.11）预测。$\mathcal H_t^b$ 按最新到最旧保存三个完整节点的记录 $(r,\sigma_r,x_r,v_r^b)$；前 $R$ 步完成初始化：

$$
\begin{aligned}
\mathcal H_R^b
&=\bigl((r,\sigma_r,x_r,v_r^b)\bigr)_{r=R-1,R-2,R-3},\\
\mathcal H_{t+1}^b
&=
\begin{cases}
\mathcal H_t^b, & m_t=1,\\
\bigl((t,\sigma_t,x_t,v_t^b),\;\mathcal H_t^b[1],\;\mathcal H_t^b[2]\bigr),
& m_t=0,
\end{cases}
\quad t\ge R.
\end{aligned}
\tag{6.5}
$$

初始化及每次完整刷新后，按更新后的三个节点和式（4.3）重建残差与差商。连续跳过时，三个记录和差商保持不变，仅随当前 $x_t,\sigma_t$ 重新计算外推项与标量权重；预测输出不写入该缓存。
