"""学习用 notebook 的共用小工具。

这些函数只是把仓库里已有的代码（src/mre）和已经生成好的结果（reports/）
包一层，方便 notebook 直接调用。这里不做任何新的建模。
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # 没有 pip install -e 也能用
    sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mre.config import load_config  # noqa: E402
from mre.data.loader import load_prices  # noqa: E402
from mre.data.quality import align_prices  # noqa: E402
from mre.portfolio import portfolio_returns, simple_returns  # noqa: E402

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.figsize": (9, 4), "axes.grid": True, "grid.alpha": 0.3})

# 已经跑好的滚动回测结果：(文件, 文件里的模型名)
FORECASTS = {
    "FHS-GJR, W=1000 (正式模型)": ("var_model_forecasts.csv.gz", "FHS (GJR-t filter), W=1000"),
    "GJR-skew-t, W=500 (备选)": ("var_model_forecasts.csv.gz", "GJR-skew-t, W=500"),
    "GARCH-t, W=500 (Phase 3 旧主模型)": ("var_model_forecasts.csv.gz", "GARCH-t, W=500"),
    "Historical, W=500": ("phase3_primary_forecasts.csv.gz", "Historical"),
    "Normal, W=500": ("phase3_primary_forecasts.csv.gz", "Normal"),
    "EWMA, W=500": ("phase3_primary_forecasts.csv.gz", "EWMA (MLE lambda)"),
    "GARCH-Normal, W=500": ("phase3_primary_forecasts.csv.gz", "GARCH-Normal"),
}


def load_portfolio():
    """返回 (配置, 四个资产的日收益, 组合日收益)。和 scripts/run_phase*.py 用的是同一份数据。"""
    cfg = load_config(ROOT / "config.yaml")
    prices = align_prices(load_prices(cfg.cache_path)[cfg.tickers])
    asset = simple_returns(prices)
    port = portfolio_returns(asset, cfg.weights)
    return cfg, asset, port


def load_forecasts(name: str) -> pd.DataFrame:
    """读取一个模型 2009-12-23 到 2025-12-31 的逐日 VaR/ES 预测（收益率单位，正数=损失）。"""
    fname, model = FORECASTS[name]
    df = pd.read_csv(ROOT / "reports" / fname, parse_dates=["date"])
    return df[df["model"] == model].set_index("date").sort_index()


def breaches(fc: pd.DataFrame, alpha: float) -> pd.Series:
    """突破 = 当天实际亏损超过了前一天给出的 VaR（和 mre.backtest.rolling.hits 一样）。"""
    k = int(round(alpha * 100))
    return (fc["realized"] < -fc[f"var_{k}"]).astype(int)


def money(x: float) -> str:
    return f"-${-x:,.0f}" if x < 0 else f"${x:,.0f}"


def play(fn, examples=None, **controls):
    """有 ipywidgets 就显示滑块；没有就提示直接改数字。滑块只在运行中的 Jupyter 里出现。

    examples: 几组示例参数。静态版本（GitHub 预览、PDF）里会用它们各跑一遍，代替滑块。
    """
    if os.environ.get("LEARN_STATIC") == "1":  # 批量执行 / 生成静态版本时不建滑块
        print("（在 Jupyter 里这里是滑块。静态版本里换成下面几组参数各跑一遍。）")
        for ex in examples or []:
            print("\n▶ " + "，".join(f"{k} = {v}" for k, v in ex.items()))
            res = fn(**ex)
            if res is not None:  # 返回表格的函数：把表格显示出来
                try:
                    from IPython.display import display
                    display(res)
                except ImportError:
                    print(res)
        return None
    try:
        from ipywidgets import interact
    except ImportError:
        print("没有安装 ipywidgets，所以没有滑块。直接改上面单元格里的数字再运行就行。"
              "（想要滑块：pip install ipywidgets）")
        return None
    interact(fn, **controls)  # interact 自己会把滑块显示出来
