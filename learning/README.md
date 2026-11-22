# 互动学习 notebook（中文）

这 5 本 notebook 是给"看公式头疼"的读者准备的。每一本都按同一个顺序：

1. **先讲直觉**（不用符号）
2. **再用本项目的真实数据算一遍**（SPY/QQQ/TLT/GLD 组合，2006–2025）
3. **动手调参数**（🎛️ 标记的单元格：滑块，或直接改数字）
4. **最后才给公式**，并且每个符号都先用大白话定义

所有数字都来自仓库里的数据（`data/raw/`）、仓库里的函数（`src/mre/`）和已经跑好的回测结果（`reports/`），notebook 里会把关键数字和报告逐一对上（例如 99% VaR \$24,197、GARCH-t 73 次突破 vs 期望 40.3、正式模型 FHS-GJR 47 次）。

| # | notebook | 讲什么 |
|---|---|---|
| 1 | [01_var_vs_es.ipynb](01_var_vs_es.ipynb) | VaR 和 ES：把 5030 天的盈亏"排卡片"；VaR 看不见门后面有多深 |
| 2 | [02_ewma_garch.ipynb](02_ewma_garch.ipynb) | 波动率聚集；EWMA 的 λ（手算 2020 年 3 月）；λ 怎么由最大似然选出；GARCH 的长期均值回归 |
| 3 | [03_fhs_gjr.ipynb](03_fhs_gjr.ipynb) | 跌比涨更吓人（GJR 的 γ）；标准化意外 z；FHS；亲手复现正式模型 2009-12-23 的 VaR |
| 4 | [04_backtesting.ipynb](04_backtesting.ipynb) | Kupiec = 抛 4,030 次硬币；Christoffersen = 突破是否扎堆；所有模型的回测"体检报告" |
| 5 | [05_why_es.ipynb](05_why_es.ipynb) | 两笔贷款的例子：VaR 违反次可加性、ES 不会；组合的分散化；FRTB 的 97.5% ES |

## 怎么运行

```bash
# 在仓库根目录
pip install -e ".[dashboard]"      # 仓库本身 + matplotlib
pip install jupyterlab ipywidgets  # notebook 和滑块
jupyter lab learning/
```

- 不装 `ipywidgets` 也能跑完，只是没有滑块；改单元格里的数字再运行即可。
- 仓库里保存的版本是"静态执行"的结果（`LEARN_STATIC=1`）：滑块的位置换成了几组不同参数各跑一遍的结果，所以在 GitHub 上也能看到"调参数"的效果。在自己的 Jupyter 里运行就会出现真正的滑块。
- 不需要联网：只读本地的 `data/raw/prices.csv` 和 `reports/` 里的文件，不重新跑回测，每本几秒钟就能跑完。

`learn_utils.py` 是几本 notebook 共用的小工具（读数据、读回测结果、显示滑块），不包含任何新的建模。
