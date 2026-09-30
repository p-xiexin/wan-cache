# 方案草稿：损失与可选网络改进

状态：待讨论，尚未采用。第 1–4 节保留预测节点掩码、尺度归一化与相邻变化损失提案，第 5 节保存分组与逐通道网络改进。主方案采用输出误差与单一系数正则，以 [PLAN.md](PLAN.md) 为准。

## 1 逐步变化尺度

对固定的实际 sigma 网格，从训练集完整轨迹中分别统计两路 CFG 的典型变化量。记 $\mu(z)=\operatorname{mean}|z|$，则

$$
d_t^b=
\operatorname{median}_{n\in\mathcal D_{\mathrm{train}}}
\mu(v_{n,t}^b-v_{n,t-1}^b),
\qquad
a_t^b=\max(d_t^b,a_{\min}),
$$

其中 $b\in\{c,u\}$，$a_{\min}>0$。统计得到的 $a_t^b$ 在训练中固定。

## 2 归一化误差

沿用 PLAN 的记号，教师标签 $v_t^b$ 取自当前预测轨迹的 latent：

$$
v_t^b=u_\theta(x_t,c_b,\tau_t).
$$

损失中对 $v_t^b$ 停止梯度。输出误差与相邻变化误差分别为

$$
e_t^b=\frac{\hat v_t^b-v_t^b}{a_t^b},
$$

$$
r_t^b=
\frac{
(\hat v_t^b-\hat v_{t-1}^b)
-(v_t^b-v_{t-1}^b)
}{a_t^b}.
$$

## 3 预测节点损失

沿用 PLAN 中的跳过判断 $m_t$，定义实际预测节点和连续预测节点对：

$$
\mathcal P=\{t:m_t=1\},
\qquad
\mathcal Q=\{t:t\in\mathcal P,\ t-1\in\mathcal P\}.
$$

$$
\mathcal L_{\mathrm{traj}}
=
\frac{
\displaystyle\sum_{b\in\{c,u\}}\sum_{t\in\mathcal P}
\operatorname{mean}\sqrt{(e_t^b)^2+\epsilon^2}
}{2\max(1,|\mathcal P|)}.
$$

$$
\mathcal L_{\mathrm{vel}}
=
\frac{
\displaystyle\sum_{b\in\{c,u\}}\sum_{t\in\mathcal Q}
\operatorname{mean}\sqrt{(r_t^b)^2+\epsilon^2}
}{2\max(1,|\mathcal Q|)}.
$$

$$
\mathcal L_{\mathrm{data}}=\mathcal L_{\mathrm{traj}}+\beta\mathcal L_{\mathrm{vel}}.
$$

其中 $\epsilon>0$、$\beta\ge0$，平方与开方逐元素计算，$\operatorname{mean}$ 对张量元素取平均。空预测集合的拟合损失为零。若采用本草案，以 $\mathcal L_{\mathrm{data}}$ 替换 PLAN 式（4.1）的输出误差部分，保留原有系数正则；相邻变化损失仅为本草案中的候选项。

工作区间外及区间内的刷新节点不产生预测损失；损失按实际预测数量平均。较小的 $a_t^b$ 提高对该阶段输出误差的敏感度。

## 4 梯度含义

忽略平均操作的常数因子，单个元素的输出梯度为

$$
\frac{\partial\ell_t}{\partial\hat v_t}
=\frac{e_t}{a_t\sqrt{e_t^2+\epsilon^2}}.
$$

训练时对 $m_t$、教师标签和 $a_t^b$ 停止梯度，段内固定完整节点缓存，通过网络与 scheduler 的 latent 递推反传。教师标签不写入缓存，只有实际完整刷新才更新缓存。尺度统计、下限和损失权重仍需讨论。

## 5 可选网络改进：分组与逐通道外推权重

主方案采用三个全局标量。本节借鉴 [HorizonStream §3.2](https://arxiv.org/html/2605.23889v1#S3.SS2) 按通道学习不同历史保留率的思路，将三个系数扩展为分组控制，不预设通道的运动、结构等语义类别。

**通道分组。** 将 $C$ 个通道按固定索引划分为 $G$ 个非空、不重叠的组 $\mathcal C_g$，覆盖全部通道。$\Pi_g$ 仅保留第 $g$ 组通道；对任意 $Z\in\mathcal X_C$：

$$
(\Pi_g Z)[r,:,:,:]=
\begin{cases}
Z[r,:,:,:], & r\in\mathcal C_g,\\
0, & r\notin\mathcal C_g,
\end{cases}
\qquad
\sum_{g=1}^{G}\Pi_g=I.
\tag{D.1}
$$

**可能的网络形式：共享编码器与分组输出头。** 复用 PLAN 式（3.6）—（3.9）的编码器、池化与 MLP 隐藏层，将作用于 $s_t^b\in\mathbb R^d$ 的末层从 $3$ 维扩展为 $3G$ 维。三个系数及各组均沿用 PLAN 式（3.10）的统一映射：

$$
\begin{aligned}
O_t^b
&=\operatorname{Reshape}_{G\times3}
\left(W_{\mathrm{grp}}s_t^b+a_{\mathrm{grp}}\right)
&&\in\mathbb R^{G\times3},\\
A_t^b
&=A_{\max}\tanh(O_t^b)
&&\in(-A_{\max},A_{\max})^{G\times3}.
\end{aligned}
\tag{D.2}
$$

$W_{\mathrm{grp}}\in\mathbb R^{3G\times d}$，$a_{\mathrm{grp}}\in\mathbb R^{3G}$，$\tanh$ 逐元素作用。$A_t^b$ 第 $g$ 行为 $(A_{t,g}^{(0),b},A_{t,g}^{(1),b},A_{t,g}^{(2),b})$。沿用 PLAN 式（3.4）的基底，重建为

$$
\hat v_t^b
=x_t+\sum_{g=1}^{G}\Pi_g\!\left(
A_{t,g}^{(0),b}R_i^b
+A_{t,g}^{(1),b}P_t^{(1),b}
+A_{t,g}^{(2),b}P_t^{(2),b}
\right)
\in\mathcal X_C.
\tag{D.3}
$$

该方向优先考虑 $G=C$：每个通道独立生成三个权重，共 $3C$ 个数；三列各重排为 $[C,1,1,1]$，沿时空维广播。$G=1$ 退回当前三标量方案；某组系数为 $(1,0,0)$ 时复用该组残差，为 $(1,1,1)$ 时采用该组的标准二次外推。

这里控制的是残差幅度与外推强度，不直接套用 HorizonStream 保留率的 $(0,1)$ 范围。同组内所有时空位置仍共享更新强度。启用时，用式（D.2）替换主方案 MLP 的末层与系数映射、式（D.3）替换 PLAN 式（3.11）；拟合损失、完整计算条件及缓存更新规则不变。PLAN 式（4.1）中每个节点的正则改为 $\frac{\lambda}{3G}\sum_{g=1}^{G}\sum_{m=0}^{2}(A_{t,g}^{(m),b}-1)^2$。
