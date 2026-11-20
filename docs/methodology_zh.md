# Market Risk Engine：数学与假设审查清单

本文按编号列出 Phase 1–4 实际用到的所有公式、估计方法和建模假设，另外附上 Phase 5 的组合优化（第 9 节）。每一条都注明了代码位置，格式是 `文件:行号`。Phase 1–4 以仓库 `main` 分支（commit `b7a9500`）为准；Phase 5 以分支 `phase5-optimization` 为准。

每条末尾的 **[审查点]** 是我认为值得你重点看的地方，包括近似、边界情况，以及我自己发现的不一致。

## 符号约定

| 符号 | 含义 |
|---|---|
| $P_{i,t}$ | 资产 $i$ 在第 $t$ 日的复权收盘价 |
| $r_{i,t}$ | 资产 $i$ 第 $t$ 日的简单收益率 |
| $w \in \mathbb{R}^N$ | 组合权重，$\sum_i w_i = 1$，$N=4$ |
| $V$ | 组合市值，$1{,}000{,}000$ 美元 |
| $\alpha$ | 置信水平（0.95 或 0.99）；$p = 1-\alpha$ 为期望突破概率 |
| $\Phi, \phi$ | 标准正态分布的 CDF 和 PDF |
| $z_q = \Phi^{-1}(q)$ | 标准正态分位数 |
| $\Sigma$ | 资产收益协方差矩阵 |
| $W$ | 估计窗口长度（交易日） |

**符号约定（全项目统一）**：收益和 P&L 都是正数代表盈利；VaR 和 ES 都报告为**正的损失**。

---

## 1. 数据与收益率（Phase 1）

**1.1 价格来源**
使用 Yahoo Finance 的复权收盘价（`auto_adjust=True`），即分红再投资后的总收益价格序列。下载区间为 2006-01-01 至 2025-12-31（含首尾）。yfinance 的 `end` 参数是开区间，所以调用时 `end` 加了 1 天。
代码：`src/mre/data/loader.py:23`（`download_prices`），`:33`（end + 1 天）
**[审查点]** Yahoo 会在发生新分红时追溯修改历史复权价，所以数据存成快照，并用 SHA-256 校验（`loader.py:63`、`:81`）。

**1.2 数据对齐**
只保留四个资产都有正价格的日期（内连接），不做前向填充：

$$\mathcal{T} = \{t : P_{i,t} > 0 \text{ 且非缺失}, \ \forall i\}$$

代码：`src/mre/data/quality.py:144`（`align_prices`）
**[审查点]** 前向填充会制造一个人为的零收益，之后再补一个跳变，扭曲尾部。实际数据中这一步没有删掉任何行。

**1.3 数据质量检查**（只报告，不修改数据），代码 `quality.py:68`：

- 重复日期或未排序 → ERROR
- 价格 $\le 0$ → ERROR
- 连续 $\ge 5$ 个相同价格（陈旧价格）→ WARNING
- $|r_{i,t}| > 15\%$ → WARNING，保留数据
- 相邻两个日期间隔超过 5 个日历日 → WARNING

**1.4 简单收益率**

$$r_{i,t} = \frac{P_{i,t}}{P_{i,t-1}} - 1$$

代码：`src/mre/portfolio.py:20`

**1.5 对数收益率**（已计算，目前没有在模型中使用）

$$\ell_{i,t} = \ln\frac{P_{i,t}}{P_{i,t-1}} = \ln(1 + r_{i,t})$$

代码：`portfolio.py:25`

**1.6 组合收益（固定权重，等价于每日再平衡）**

$$r_{p,t} = \sum_{i=1}^{N} w_i\, r_{i,t} = w^\top r_t$$

只有简单收益率满足这个跨资产的线性加总，对数收益率不满足。
代码：`portfolio.py:40`
**[审查点]** 固定权重假设每天都调回 40/30/20/10，忽略了权重漂移和交易成本。这是"用当前持仓重放历史"的标准 VaR 设定。

**1.7 P&L**

$$\text{PnL}_t = V \cdot r_{p,t}$$

代码：`portfolio.py:55`

---

## 2. VaR 与 Expected Shortfall（Phase 1）

**2.1 定义（损失为正）**
设 $X$ 为收益或 P&L：

$$\text{VaR}_\alpha(X) = -q_{1-\alpha}(X), \qquad \text{ES}_\alpha(X) = -\mathbb{E}\left[X \mid X \le -\text{VaR}_\alpha(X)\right]$$

代码：`src/mre/risk/var.py` 文件头注释

**2.2 Historical VaR**

$$\widehat{\text{VaR}}_\alpha = -\hat q_{1-\alpha}(x_1, \dots, x_n)$$

$\hat q$ 用 numpy 的 `linear` 插值经验分位数：把样本排序为 $x_{(1)} \le \dots \le x_{(n)}$，令 $h = (n-1)(1-\alpha)$，则 $\hat q = x_{(\lfloor h \rfloor + 1)} + (h - \lfloor h \rfloor)\,(x_{(\lfloor h \rfloor + 2)} - x_{(\lfloor h \rfloor + 1)})$。
代码：`var.py:38`
**[审查点]** 插值方法可以在 `config.yaml` 的 `quantile_method` 里修改。在 500 天窗口的 99% 水平下，$h \approx 4.99$，VaR 落在第 5 和第 6 小的观测之间。

**2.3 Historical ES**

$$\widehat{\text{ES}}_\alpha = -\frac{1}{|\mathcal{A}|} \sum_{j \in \mathcal{A}} x_j, \qquad \mathcal{A} = \{j : x_j \le -\widehat{\text{VaR}}_\alpha\}$$

代码：`var.py:45`
**[审查点]** 尾部集合 $\mathcal{A}$ 的大小是 $\lfloor h \rfloor + 1$（非插值部分），所以它不严格等于 $n(1-\alpha)$。另一种做法是 Acerbi–Tasche 的加权估计，它对分位点处的观测给分数权重，在样本小时更精确。

**2.4 Normal VaR**

$$\widehat{\text{VaR}}_\alpha = -(\hat\mu + \hat\sigma\, z_{1-\alpha}) = -\hat\mu + \hat\sigma\, z_\alpha$$

其中 $\hat\mu$ 是样本均值（可选设为 0），$\hat\sigma$ 是样本标准差（`ddof=1`，即除以 $n-1$）。
代码：`var.py:55`（均值和标准差），`:63`

**2.5 Normal ES**

$$\widehat{\text{ES}}_\alpha = -\hat\mu + \hat\sigma\, \frac{\phi(z_\alpha)}{1-\alpha}$$

推导：若 $X \sim N(\mu, \sigma^2)$，则 $\mathbb{E}[X \mid X \le \mu + \sigma z_{1-\alpha}] = \mu - \sigma\,\phi(z_{1-\alpha})/(1-\alpha)$，并利用 $\phi$ 的对称性 $\phi(z_{1-\alpha}) = \phi(z_\alpha)$。
代码：`var.py:70`

**2.6 已用测试验证的性质**：正齐次性 $\text{VaR}(cX) = c\,\text{VaR}(X)$；平移性 $\text{VaR}(X + c) = \text{VaR}(X) - c$；$\text{ES} \ge \text{VaR}$；对正态分布有 $\text{ES}_{97.5\%} / \text{VaR}_{99\%} \approx 1.005$。
代码：`tests/test_var.py`

---

## 3. EWMA（Phase 2）

**3.1 方差递推（没有前视）**
$\sigma^2_t$ 是在第 $t-1$ 日收盘时对第 $t$ 日做的方差预测：

$$\sigma^2_t = \lambda\, \sigma^2_{t-1} + (1-\lambda)\, r^2_{t-1}$$

**初始化**：前 $B = 30$ 个收益只用来给递推一个初值，$\sigma^2_B = \frac{1}{B}\sum_{s=0}^{B-1} r_s^2$，不计入似然函数。
代码：`src/mre/models/ewma.py:42`（初值在 `:53`；递推用 `scipy.signal.lfilter` 向量化，在 `:55`）
**[审查点]** 均值设为 0，这是 RiskMetrics 的惯例。初值和 $B$ 的选择会通过前几项影响 λ 的估计。

**3.2 λ 的最大似然估计（Gaussian QMLE）**

$$\hat\lambda = \arg\max_{\lambda \in [0.80,\, 0.9995]} \sum_{t=B}^{n-1} -\frac{1}{2}\left[\ln(2\pi\sigma^2_t(\lambda)) + \frac{r_t^2}{\sigma^2_t(\lambda)}\right]$$

用有界的 Brent 方法求解（`minimize_scalar`）。只使用估计窗口内的数据。
代码：`ewma.py:61`（负对数似然），`:78`；边界定义在 `:32`
**[审查点]** 即使真实收益是厚尾的，正态拟似然对方差动态的估计仍然是一致的；尾部的厚度交给 GARCH-t 处理。如果 λ 落在边界上，代码会标记出来：独立同分布的数据会使 λ 趋向 1。

**3.3 协方差递推**

$$\Sigma_t = \lambda\, \Sigma_{t-1} + (1-\lambda)\, r_{t-1} r_{t-1}^\top, \qquad \Sigma_B = \frac{1}{B}\sum_{s<B} r_s r_s^\top$$

代码：`ewma.py:93`

**3.4 多元 λ 的 MLE**（所有资产共用一个 λ）

$$\hat\lambda = \arg\max_\lambda \sum_t -\frac{1}{2}\left[N\ln 2\pi + \ln\det\Sigma_t + r_t^\top \Sigma_t^{-1} r_t\right]$$

代码：`ewma.py:114`、`:128`
**[审查点]** Phase 3 的回测用的是组合收益上的**单变量** λ（`backtest/rolling.py:98`），Phase 4 的归因用的是**多变量** λ（`scripts/run_phase4.py:70`）。两者的数值不同（截至 2025 年底分别约为 0.92 和 0.96）。

---

## 4. GARCH(1,1)（Phase 2）

**4.1 模型**

$$r_t = \mu + \varepsilon_t, \qquad \varepsilon_t = \sigma_t z_t, \qquad \sigma^2_t = \omega + \alpha_1 \varepsilon^2_{t-1} + \beta \sigma^2_{t-1}$$

$z_t$ 独立同分布，均值为 0、方差为 1，服从 Normal 或标准化的 Student-t。参数约束为 $\omega > 0$、$\alpha_1, \beta \ge 0$、$\alpha_1 + \beta < 1$（`arch` 包施加的约束）。
代码：`src/mre/models/garch.py:59`（调用 `arch_model(mean="Constant", vol="GARCH", p=1, q=1)`，在 `:67`）

**4.2 单位方差 Student-t 的密度**（$\nu > 2$）

$$f(z;\nu) = \frac{\Gamma\!\left(\frac{\nu+1}{2}\right)}{\Gamma\!\left(\frac{\nu}{2}\right)\sqrt{\pi(\nu-2)}} \left(1 + \frac{z^2}{\nu-2}\right)^{-\frac{\nu+1}{2}}$$

**4.3 对数似然**

$$\ell(\theta) = \sum_t \left[\ln f\!\left(\frac{\varepsilon_t}{\sigma_t};\nu\right) - \ln \sigma_t\right]$$

**数值缩放**：拟合时收益先乘以 100。原始尺度下的对数似然为 $\ell_{\text{raw}} = \ell_{\text{scaled}} + n \ln 100$，这是 Jacobian 修正；参数换算为 $\omega_{\text{raw}} = \omega/100^2$，$\mu_{\text{raw}} = \mu/100$。
代码：`garch.py:26`（缩放系数），`:72`（Jacobian 修正）
**[审查点]** $\sigma^2_0$ 的初始化用的是 `arch` 默认的 backcast（对初始残差平方做指数加权）。

**4.4 信息准则**

$$\text{AIC} = 2k - 2\ell, \qquad \text{BIC} = k\ln n - 2\ell$$

$k = 5$ 对应 t 分布（μ, ω, α, β, ν），$k = 4$ 对应正态分布。
**[审查点]** 不使用 LR 检验比较 t 和 Normal，因为 $\nu = \infty$ 位于参数空间的边界上，LR 统计量不服从标准的 $\chi^2$ 分布。

**4.5 衍生量**

- 持续性 $= \alpha_1 + \beta$
- 半衰期 $= \ln 0.5 / \ln(\alpha_1 + \beta)$
- 长期波动率 $= \sqrt{\omega/(1-\alpha_1-\beta)}$

代码：`garch.py:45–56`

**4.6 一步预测**

$$\sigma^2_{T+1} = \omega + \alpha_1 \varepsilon^2_T + \beta \sigma^2_T$$

取自 `arch.forecast(horizon=1)`，并已用测试和手算递推核对一致。

**4.7 标准化分位数**
若 $T \sim t_\nu$，令 $c = \sqrt{(\nu-2)/\nu}$，则 $z = cT$ 的方差为 1：

$$q_z(p) = c \cdot t_\nu^{-1}(p)$$

代码：`garch.py:90`

**4.8 标准化 ES**（左尾，报告为正数），记 $t_\alpha = t_\nu^{-1}(\alpha)$：

$$\text{ES}_z(\alpha) = c \cdot \frac{f_{t_\nu}(t_\alpha)}{1-\alpha} \cdot \frac{\nu + t_\alpha^2}{\nu - 1}$$

代码：`garch.py:98`
**[审查点]** 测试中已用数值积分验证了单位方差和这个 ES 公式（`tests/test_garch.py`）。

**4.9 GARCH 的 VaR 和 ES**

$$\text{VaR}_\alpha = -(\mu + \sigma_{T+1}\, q_z(1-\alpha)), \qquad \text{ES}_\alpha = -\mu + \sigma_{T+1}\, \text{ES}_z(\alpha)$$

代码：`garch.py:109`
**[审查点]** 这里包含了均值 $\mu$。实际估计中 $\mu \approx 8 \times 10^{-4}$/日，会让 VaR 下降大约 0.08 个百分点。这是 Phase 3 列出的改进选项 (b)。

**4.10 平方序列的 Ljung–Box 检验**（只用于诊断）

$$Q = n(n+2)\sum_{k=1}^{10} \frac{\hat\rho_k^2}{n-k} \sim \chi^2(10)$$

代码：`scripts/run_phase2.py`（`ljung_box`）
**[审查点]** 对标准化残差做检验时，严格来说自由度应扣除估计的参数个数；这里没有扣除。

---

## 5. Ledoit–Wolf 协方差收缩（Phase 2）

记去均值后的数据为 $X \in \mathbb{R}^{T \times N}$，样本协方差 $S = X^\top X / T$（**除以 $T$**，和论文一致），收缩估计为：

$$\hat\Sigma = \delta F + (1-\delta) S$$

**5.1 单位阵目标**（Ledoit & Wolf 2004, JMVA）

$$F = \bar m I, \quad \bar m = \text{tr}(S)/N, \quad d^2 = \|S - F\|_F^2, \quad \bar b^2 = \frac{1}{T^2}\sum_t \|x_t x_t^\top - S\|_F^2, \quad \delta = \frac{\min(\bar b^2, d^2)}{d^2}$$

代码：`src/mre/models/shrinkage.py:41`
**[审查点]** 已用测试与 `sklearn.covariance.ledoit_wolf` 核对一致。

**5.2 常相关目标**（Ledoit & Wolf 2004, "Honey, I shrunk..."）
令 $s_i = \sqrt{S_{ii}}$，平均相关系数 $\bar r = \frac{2}{N(N-1)}\sum_{i<j} \frac{S_{ij}}{s_i s_j}$，目标矩阵 $F_{ii} = S_{ii}$，$F_{ij} = \bar r\, s_i s_j$。

$$\hat\pi = \sum_{i,j} \frac{1}{T}\sum_t (x_{ti}x_{tj} - S_{ij})^2$$

$$\hat\vartheta_{ii,ij} = \frac{1}{T}\sum_t (x_{ti}^2 - S_{ii})(x_{ti}x_{tj} - S_{ij})$$

$$\hat\rho = \sum_i \hat\pi_{ii} + \sum_{i \ne j} \frac{\bar r}{2}\left(\frac{s_j}{s_i}\hat\vartheta_{ii,ij} + \frac{s_i}{s_j}\hat\vartheta_{jj,ij}\right)$$

$$\hat\gamma = \|F - S\|_F^2, \qquad \hat\kappa = \frac{\hat\pi - \hat\rho}{\hat\gamma}, \qquad \delta = \max\!\left(0,\, \min\!\left(1,\, \frac{\hat\kappa}{T}\right)\right)$$

代码：`shrinkage.py:55`（$\hat\pi$ 在 `:67`，$\hat\vartheta$ 在 `:70`，$\hat\rho$ 在 `:75`，$\delta$ 在 `:78`）
**[审查点]** 已用测试与逐元素循环的暴力实现核对一致。在实际数据上 $\delta \approx 0.02$，因为 $N = 4$、$T = 1000$，收缩几乎不改变结果。

**5.3 组合波动率**

$$\sigma_p = \sqrt{w^\top \Sigma w}$$

代码：`shrinkage.py:82`

**[审查点：一个不一致]** Phase 4 的"Sample (500d)"协方差用的是 pandas 的 `.cov()`，除以 $T-1$（`scripts/run_phase4.py:89`）；而 LW 的样本协方差 $S$ 除以 $T$。两者相差 $T/(T-1) \approx 1.002$，对结果影响很小，但口径没有统一。

---

## 6. 滚动回测与覆盖率检验（Phase 3）

**6.1 时间设定**
第 $t$ 日的预测只使用 $r_{t-W}, \dots, r_{t-1}$。样本外从第 1000 个收益开始（2009-12-23），共 4,030 天，所有设定共用这个起点。
代码：`src/mre/backtest/rolling.py:49`

**6.2 各模型的预测方式**

- **Historical 和 Normal**：每天用窗口数据重新计算，公式同第 2.2–2.5 条（`rolling.py:76–91`）。
- **EWMA**：每 21 天用窗口数据重新估计 λ（单变量 MLE）；每天从窗口起点重新递推方差；均值为 0（`rolling.py:94–109`）。
- **GARCH**：每 21 天（敏感性分析为每天）用窗口重拟合。两次重拟合之间参数固定，方差每天更新（`rolling.py:128`）：

$$\sigma^2_t = \omega + \alpha_1 (r_{t-1} - \mu)^2 + \beta\sigma^2_{t-1}$$

  重拟合当天取 $\sigma^2_t = $ 该次拟合给出的一步预测。

**[审查点]** 两次重拟合之间，GARCH 的方差递推会用到窗口之后、但仍早于 $t$ 的收益。这些都是已经观测到的数据，不构成前视；测试 `tests/test_rolling.py::test_no_look_ahead` 验证了这一点。

**6.3 突破指示变量**

$$I_t = \mathbb{1}\{r_t < -\text{VaR}_{\alpha,t}\}$$

代码：`rolling.py:151`

**6.4 Kupiec POF 检验**（$n$ 个观测，$x = \sum_t I_t$，$\hat p = x/n$）

$$\text{LR}_{\text{POF}} = -2\left[(n-x)\ln(1-p) + x\ln p - (n-x)\ln(1-\hat p) - x\ln \hat p\right] \sim \chi^2(1)$$

约定 $0 \cdot \ln 0 = 0$（用 `scipy.special.xlogy` 实现）。
代码：`src/mre/backtest/coverage_tests.py:48`（LR 在 `:53`）

**6.5 Christoffersen 独立性检验**
$n_{ij}$ 为从状态 $i$ 转移到状态 $j$ 的次数，$\pi_{01} = \frac{n_{01}}{n_{00} + n_{01}}$，$\pi_{11} = \frac{n_{11}}{n_{10} + n_{11}}$，$\pi = \frac{n_{01} + n_{11}}{n-1}$。

$$\text{LR}_{\text{ind}} = -2\left[\ln L(\pi) - \ln L(\pi_{01}, \pi_{11})\right] \sim \chi^2(1)$$

$$\ln L(\pi_{01}, \pi_{11}) = n_{00}\ln(1-\pi_{01}) + n_{01}\ln\pi_{01} + n_{10}\ln(1-\pi_{11}) + n_{11}\ln\pi_{11}$$

代码：`coverage_tests.py:58`（转移计数），`:74`
**[审查点]** 当 $n_{10} + n_{11} = 0$（从没出现过突破）时，代码令 $\pi_{11} = 0$，对应的似然项为 0。

**6.6 Christoffersen 条件覆盖检验**（精确形式，基于 $n-1$ 次转移）

$$\text{LR}_{\text{cc}} = -2\left[\ln L(p) - \ln L(\pi_{01}, \pi_{11})\right] \sim \chi^2(2)$$

代码：`coverage_tests.py:83`
**[审查点]** 常见的简写 $\text{LR}_{\text{POF}} + \text{LR}_{\text{ind}}$ 和这个精确形式只差第一个观测，测试中已验证。

**6.7 判定规则**：$p$ 值 $< 5\%$（主设定）或 $< 1\%$（敏感性）时拒绝原假设。测试中已用 Monte Carlo 验证，原假设成立时 5% 水平的实际拒绝率在 [2.5%, 8%] 之间（因为二项分布是离散的，不会恰好等于 5%）。

**6.8 ES 比值**（描述性，不是正式检验）

$$\text{ES ratio} = \frac{\text{mean}\{-r_t : I_t = 1\}}{\text{mean}\{\text{ES}_{\alpha,t} : I_t = 1\}}$$

代码：`rolling.py:157`
**[审查点]** 正式的 ES 回测，比如 Acerbi–Székely Z₂ 检验，还没有实现。


## 6b. 尾部模型扩展与主模型更换（Phase 3b）

**6b.1 GJR-GARCH(1,1,1)**（杠杆效应）

$$\sigma^2_t = \omega + \left(\alpha + \gamma\,\mathbb{1}[\varepsilon_{t-1}<0]\right)\varepsilon^2_{t-1} + \beta\sigma^2_{t-1}$$

$\gamma>0$ 表示负冲击使波动率上升得更多。代码：`garch.py` 中 `GarchFit.next_variance`（重估之间的每日更新），单元测试验证它与 `arch` 自身的滤波逐日一致。

**6b.2 EGARCH(1,1,1)**

$$\ln\sigma^2_t = \omega + \alpha\left(|z_{t-1}| - \sqrt{2/\pi}\right) + \gamma z_{t-1} + \beta\ln\sigma^2_{t-1}$$

$\gamma<0$ 为杠杆效应。$\sqrt{2/\pi}$ 是 `arch` 对所有分布统一使用的中心化常数。缩放换算：$\omega_{\text{raw}} = \omega - (1-\beta)\ln 100^2$。
**[审查点]** 估计时约束 $\alpha \ge 0$、$\beta \le 0.999$，并从 4 个初值出发取似然最高的合理解。不加约束时，部分 500 天窗口会收敛到 $\alpha<0$、$\beta\approx1$ 的解，样本外方差发散。若重估仍不合理（不收敛、预测波动率超过样本标准差的 10 倍、或 $\nu\le2.1$），就沿用上一组参数；全部回测中都没有出现这种情况。

**6b.3 Hansen (1994) 偏 t 分布**（参数 $\eta>2$、$-1<\lambda<1$，单位方差）

$$f(z) = \begin{cases} bc\left(1 + \frac{1}{\eta-2}\left(\frac{bz+a}{1-\lambda}\right)^2\right)^{-\frac{\eta+1}{2}}, & z < -a/b \\ bc\left(1 + \frac{1}{\eta-2}\left(\frac{bz+a}{1+\lambda}\right)^2\right)^{-\frac{\eta+1}{2}}, & z \ge -a/b \end{cases}$$

其中 $c = \frac{\Gamma((\eta+1)/2)}{\sqrt{\pi(\eta-2)}\,\Gamma(\eta/2)}$，$a = 4\lambda c\frac{\eta-2}{\eta-1}$，$b^2 = 1 + 3\lambda^2 - a^2$。$\lambda<0$ 表示左尾更厚。
分位数用 Hansen 的闭式解（`arch` 的 `SkewStudent.ppf`）。由于左右尾不对称，ES 不能用对称性，改用数值积分：

$$\text{ES}_z(\alpha) = -\frac{1}{1-\alpha}\int_0^{1-\alpha} q_z(u)\,du$$

**[审查点]** 单元测试用密度积分独立校验分位数和 ES，并验证 $\lambda=0$ 时退化为标准化 t。

**6b.4 过滤历史模拟（FHS，Barone-Adesi 等 1999）**

$$\text{VaR}_{\alpha,t} = -\left(\mu + \sigma_t\,\hat q_{1-\alpha}(\{z_s\})\right), \qquad \text{ES}_{\alpha,t} = -\mu + \sigma_t\,\widehat{\text{ES}}_\alpha(\{z_s\})$$

$z_s = (r_s - \mu)/\sigma_s$ 是最近 $W$ 天（$s<t$）的标准化残差。$\sigma_s$ 用最近一次重估的参数计算：训练窗口内取拟合结果，之后每日用同一组参数滤波并滚动加入。$\hat q$、$\widehat{\text{ES}}$ 与历史模拟用同一个经验分位数函数。
**[审查点]** $W=1000$ 时，99% 分位数大约由 10 个尾部残差决定；$W=500$ 时只有约 5 个。

**6b.5 主模型更换**

官方 1 日 VaR/ES 主模型从 GARCH-t（$W=500$）改为 **FHS-GJR**（GJR-GARCH-t 滤波，$W=1000$）。GJR-skew-t（$W=500$）作为参数化备选，GARCH-t 保留为对照。配置见 `config.yaml` 的 `var_model`。理由和证据见 `docs/model_validation_report.md` 第 6.5 节。
**[审查点]** $W=1000$ 是在看到样本外结果之后选的（Phase 3 的主窗口是事先定的 500）。这是一个事后选择，应当用之后的新数据继续监控。

---

## 7. 压力测试（Phase 4）

**7.1 历史情景：买入持有（主口径）**
以情景起点 $t_0$（$\ge$ start 的第一个交易日）为基准，在 $t_1$（$\le$ end 的最后一个交易日）结束：

$$R_i = \frac{P_{i,t_1}}{P_{i,t_0}} - 1, \qquad R_p^{\text{BH}} = \sum_i w_i R_i, \qquad \text{各资产贡献} = V w_i R_i \ (\text{加总等于总 P\&L})$$

价值路径为 $V_t = \sum_i w_i P_{i,t}/P_{i,t_0}$。
代码：`src/mre/stress/scenarios.py:62`（价值路径在 `:76`）

**7.2 历史情景：每日再平衡（对比口径）**

$$R_p^{\text{RB}} = \prod_{t=t_0+1}^{t_1}\left(1 + w^\top r_t\right) - 1$$

这种口径下各资产贡献不可加，所以不报告贡献。
代码：`scenarios.py:81–82`

**7.3 最大回撤**

$$\text{MDD} = \min_t \left(\frac{V_t}{\max_{s \le t} V_s} - 1\right)$$

计算包含起点。
代码：`scenarios.py:56`

**7.4 假设情景（瞬时冲击）**

$$\text{PnL} = V \sum_i w_i s_i$$

对简单收益是精确的。没有指定冲击的资产取 0；要求 $s_i > -1$。
代码：`scenarios.py:97`

**7.5 条件压力测试**
假设收益服从零均值多元正态，对一部分资产 $S$ 施加冲击 $s$，其余资产 $U$ 的条件期望为：

$$\mathbb{E}[r_U \mid r_S = s] = \Sigma_{US}\, \Sigma_{SS}^{-1}\, s$$

当只冲击一个资产时，这就是回归 beta 乘以冲击：$\text{Cov}(r_u, r_S)/\text{Var}(r_S) \times s$。
代码：`scenarios.py:113`（公式在 `:125`）
**[审查点]** 这里用的是条件**均值**，没有考虑条件分布的离散程度，比如 $\Sigma_{UU} - \Sigma_{US}\Sigma_{SS}^{-1}\Sigma_{SU}$。另外，零均值假设意味着不计入漂移项。

---

## 8. 风险归因（Phase 4）

**8.1 参数法组合风险**（默认 $\mu = 0$）

$$\text{VaR}(w) = -w^\top\mu + k\,\sigma_p(w), \qquad \sigma_p = \sqrt{w^\top\Sigma w}$$

其中 VaR 取 $k = z_\alpha$，ES 取 $k = \phi(z_\alpha)/(1-\alpha)$。
代码：`src/mre/attribution/euler.py:39`、`:44`

**8.2 Marginal VaR**

$$\text{MVaR}_i = \frac{\partial\,\text{VaR}}{\partial w_i} = -\mu_i + k\,\frac{(\Sigma w)_i}{\sigma_p}$$

代码：`euler.py:60`（已用测试与数值导数核对一致）

**8.3 Component VaR（Euler 分解）**
VaR 是 $w$ 的一次齐次函数，由 Euler 定理：

$$\text{CVaR}_i = w_i \cdot \text{MVaR}_i, \qquad \sum_i \text{CVaR}_i = \text{VaR}, \qquad \text{贡献占比}_i = \frac{\text{CVaR}_i}{\text{VaR}}$$

**8.4 Incremental VaR**（把资产 $i$ 平仓为现金，其他权重不变）

$$\text{IVaR}_i = \text{VaR}(w) - \text{VaR}(w \text{ 中令 } w_i = 0)$$

代码：`euler.py:66`
**[审查点]** 平仓后剩余权重之和小于 1，差额视为持有现金，没有重新归一化。

**8.5 Standalone VaR 与分散化收益**

$$\text{SVaR}_i = -w_i\mu_i + k\,|w_i|\sqrt{\Sigma_{ii}}, \qquad \text{分散化收益} = \sum_i \text{SVaR}_i - \text{VaR}$$

**8.6 历史 Component ES**（不做分布假设）

$$\text{CES}_i = -\mathbb{E}\left[w_i r_{i,t} \mid r_{p,t} \le -\widehat{\text{VaR}}_\alpha\right], \qquad \sum_i \text{CES}_i = \widehat{\text{ES}}_\alpha \ (\text{与第 2.3 条定义一致})$$

代码：`euler.py:83`（公式在 `:95`）
**[审查点]** 500 天窗口在 99% 水平下只有 5 个尾部样本，所以这个分解噪声很大，报告中只作为交叉验证。

**8.7 时变风险贡献**（用固定 λ = 0.94 的 EWMA 协方差路径，属于描述性分析）

$$\text{RC}_{i,t} = \frac{w_i(\Sigma_t w)_i}{w^\top\Sigma_t w}$$

代码：`euler.py:108`
**[审查点]** 这里故意用固定的 λ，而不是用全样本 MLE 估计 λ，以免引入未来信息。

**8.8 Phase 4 的协方差口径**：主口径是 EWMA，λ 用最近 500 天做多元 MLE（第 3.4 条），均值为 0；对比口径是 500 天样本协方差和 LW 常相关收缩。

---

## 9. 组合优化（Phase 5）

9.1–9.4 是满仓的静态优化器（只做多，单资产权重在 $[lb, ub]$ 之间，$ub = 60\%$），在 Phase 5 中用作对比；主策略见 9.7–9.10。

**9.1 最小方差**

$$\min_w\, w^\top\Sigma w \quad \text{s.t.} \quad \mathbf{1}^\top w = 1, \ lb \le w_i \le ub$$

用 SLSQP 求解。
代码：`src/mre/optimization/optimizers.py:53`

**9.2 风险平价（ERC 或指定风险预算 $b$）**
目标是让风险贡献满足 $\frac{w_i(\Sigma w)_i}{w^\top\Sigma w} = b_i$。用 Spinu (2013) 的凸形式求解：

$$\min_{y > 0}\ \frac{1}{2} y^\top\Sigma y - \sum_i b_i \ln y_i, \qquad w = \frac{y}{\mathbf{1}^\top y}$$

一阶条件是 $(\Sigma y)_i = b_i / y_i$，乘以 $y_i$ 后得到 $y_i(\Sigma y)_i = b_i$，所以风险贡献与 $b_i$ 成比例。如果解超出了上限，就改为在约束下用 SLSQP 最小化 $\sum_i (\text{RC}_i - b_i)^2$。
代码：`optimizers.py:68`

**9.3 最小 CVaR（Rockafellar–Uryasev 线性规划）**

$$\min_{w,\zeta,u}\ \zeta + \frac{1}{(1-\alpha)T}\sum_t u_t \quad \text{s.t.} \quad u_t \ge -r_t^\top w - \zeta, \ u_t \ge 0, \ \mathbf{1}^\top w = 1, \ lb \le w \le ub$$

当 $(1-\alpha)T$ 为整数时，最优值等于组合最差的 $(1-\alpha)T$ 个损失的平均。
代码：`optimizers.py:92`
**[审查点]** 500 天窗口在 99% 水平下只有 5 个尾部场景，优化器会去"拟合"这 5 天。所以我建议优化时用 $\alpha = 95\%$，评估时仍然看 99%。

**9.4 最大 Sharpe**（仅作为反面对照）

$$\max_w\ \frac{w^\top\mu - r_f}{\sqrt{w^\top\Sigma w}}$$

用多个初始点的 SLSQP 求解。如果所有资产的 $\mu_i \le r_f$，退回最小方差解。
代码：`optimizers.py:113`

**9.5 滚动调仓回测**
在调仓日 $t$，权重只用 $r_{t-W}, \dots, r_{t-1}$ 计算。两次调仓之间权重随收益漂移：

$$w_{t+1} = \frac{w_t \odot (1 + r_t)}{w_t^\top(1 + r_t)}$$

调仓日扣除成本 $c \sum_i |w^{\text{target}}_i - w^{\text{drift}}_i|$，其中 $c = 5$ bp；第一次建仓的换手率记为 1。
代码：`src/mre/optimization/backtest.py:30`（漂移在 `:65`，换手率在 `:53`）

**9.6 绩效指标**（$r_f = 0$）

- CAGR $= W_T^{252/n} - 1$
- 年化波动率 $= \hat\sigma\sqrt{252}$
- Sharpe $= 252\bar r / (\hat\sigma\sqrt{252})$
- 最大回撤、Calmar $=$ CAGR $/ |\text{MDD}|$
- 日度 99% 历史 VaR 和 ES

代码：`backtest.py:76`


**9.7 每日风险预测**
第 $t$ 日的协方差预测 $\Sigma_t$ 只用 $r_{t-500}, \dots, r_{t-1}$ 计算：在这 500 天上做 EWMA 递推（第 3.3 条），λ 每 21 天用多元 MLE 重新估计一次（第 3.4 条），中间保持不变。
代码：`src/mre/optimization/dynamic.py:31`

**9.8 风险平价 + 波动率目标（主策略）**
先用 $\Sigma_t$ 求风险平价权重 $b_t$（第 9.2 条，上限 60%），再按波动率目标缩放：

$$k_t = \min\!\left(1,\ \frac{\sigma^*}{\sqrt{252\, b_t^\top \Sigma_t b_t}}\right), \qquad w^*_t = k_t\, b_t, \qquad c^*_t = 1 - \mathbf{1}^\top w^*_t \ge 0$$

主设定 $\sigma^* = 10\%$，敏感性分析取 8% 和 12%。$k_t \le 1$ 意味着不加杠杆，剩余部分持有现金。缩放不改变各资产的风险贡献占比。
代码：`dynamic.py:54`、`:63`
**[审查点]** 风险平价组合的事前波动率中位数约为 7.9%，所以 10% 的目标大部分时间不起作用，只在高波动时期才降仓。

**9.9 组合核算（含现金）**
第 $t$ 日的毛收益为

$$R_t = w_t^\top r_t + c_t\, r^f_t, \qquad r^f_t = \frac{\text{IRX}_{t-1}}{100 \times 252}$$

收益实现后，持仓按收益漂移，并保持 $\mathbf{1}^\top w + c = 1$：

$$w_{t+1} = \frac{w_t \odot (1 + r_t)}{1 + R_t}, \qquad c_{t+1} = \frac{c_t (1 + r^f_t)}{1 + R_t}$$

交易成本按比例从净值中扣除，不改变持仓比例。
代码：`dynamic.py:81`（漂移在 `:125–126`）；现金利率在 `scripts/run_phase5.py:55`
**[审查点]** ^IRX 是贴现率口径的报价，这里直接当作货币市场收益率使用，是一个近似。利率缺失的日子沿用前一个报价（利率是水平值而不是价格，所以这样处理可以接受）。

**9.10 阈值调仓规则**
第 $t$ 日只有在 $\max_i |w^*_{t,i} - w_{t,i}| > \tau$ 时才交易（主设定 $\tau = 5$pp，敏感性分析取 2.5pp 和 10pp）。对比规则是固定每 21 天调仓一次。交易成本为 $5\text{bp} \times \sum_i |w^*_{t,i} - w_{t,i}|$，现金交易不收费。
代码：`dynamic.py:81`

**9.11 超额 Sharpe 比率**

$$\text{Sharpe} = \frac{252\, \overline{(R_t - r^f_t)}}{\sqrt{252}\ \text{sd}(R_t - r^f_t)}$$

代码：`scripts/run_phase5.py`（`stats` 函数）
**[审查点]** 这里没有检验不同策略之间 Sharpe 差异的统计显著性，比如 Ledoit–Wolf (2008) 的检验。

---

## 10. 全局假设汇总

| # | 假设 | 位置 | 影响 |
|---|---|---|---|
| A1 | 复权收盘价 = 总收益口径 | `config.yaml` `data.auto_adjust` | 包含分红；Yahoo 的复权方法本身是一种近似 |
| A2 | 固定权重（每日再平衡）用于 VaR | `portfolio.py:40` | 忽略权重漂移和交易成本 |
| A3 | 持有期 1 天，不做 √h 缩放 | 全局 | 10 天 VaR 尚未实现 |
| A4 | 正态模型带样本均值，EWMA 均值为 0 | `config.yaml`；`rolling.py:104` | 不同模型的均值口径不一致 |
| A5 | 估计窗口：Phase 2 用 1,000 天，Phase 3 和 4 用 500 天 | `config.yaml` `models.estimation_window` | Phase 2 报告是用 1,000 天生成的 |
| A6 | GARCH 每 21 天重拟合，方差每日更新 | `config.yaml` `backtest.refit_every` | 敏感性分析显示每日重拟合的结果几乎相同 |
| A7 | 样本外起点统一为第 1000 个收益 | `config.yaml` `backtest.oos_start_index` | 不同窗口在同一批日期上比较 |
| A8 | 条件压力测试假设零均值多元正态 | `scenarios.py:113` | 只用条件均值，不考虑条件方差 |
| A9 | Phase 1–4 中无风险利率 $r_f = 0$；Phase 5 用 ^IRX | `backtest.py:76`；`run_phase5.py:55` | Phase 5 之前的 Sharpe 比率会偏高 |
| A10 | Phase 5 的波动率目标为 10%，不加杠杆 | `config.yaml` `optimization.vol_target` | 平时基本满仓，只在高波动时降仓 |
| A11 | 调仓阈值为 5pp | `config.yaml` `optimization.threshold` | 在换手率和跟踪精度之间取舍 |
