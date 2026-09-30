分段多项式先验实验包

本地拟合对象：log(q)，q 是相邻完整 DiT 输出的相对变化幅度。
训练是最小二乘求分段三次多项式系数，不训练神经网络，不需要原始张量。
prior.json 是在模型选择结束后用全部 500 条曲线重拟合的部署先验。
summary.json / oof_forecasts.npz 是五折留出模型的评估结果，不是部署先验的独立测试。

推理端只需要 numpy：
    from polynomial_prior import PiecewisePolynomialPrior, OnlinePolynomial
    prior = PiecewisePolynomialPrior.load("prior.json")
    predictor = OnlinePolynomial(prior, use_online_fit=True, use_mirror_node=True)
    predictor.observe(step, measured_q)  # 只接受真实观测
    predicted_q = predictor.predict(future_step)

ablations.json 保存四组开关配置。关闭 online_fit 时仍用最新真实节点作幅值对齐。
mirror_node_mode="anchor" 只增加同位置约束，虚拟节点不读取镜像历史值。
镜像只在真实锚点越过训练得到的对称轴后启用，预测点不会写入真实历史。
全部组共享同一先验、拟合阶数、窗口和正则参数；mirror 仅增加弱镜像约束。

坐标是当前 50 步配置的推理 step，支持 1..48。分段局部 z=(step-left)/(right-left)。
每段 log(q)=c0+c1*z+c2*z^2+c3*z^3，q=exp(log(q))；段间 C2 连续。
不会在训练范围之外自动外推；换步数、模型或 scheduler 应重新校准。
q_s 的真实观测需要完整 v_(s-1) 和 v_s。跨多个步骤的差值不能当作 q_s。

这份包实现系数先验和在线标量校正，不包含刷新调度或完整张量重建。
服务器中的张量预测仍需使用真实 DiT 节点提供方向和幅值，并单独验证。
重新拟合：python analyze/train_piecewise_polynomial.py --data /path/to/lazy_dataset_500 --output /path/to/output
训练依赖 numpy、torch、matplotlib；部署拟合器仅依赖 numpy。
