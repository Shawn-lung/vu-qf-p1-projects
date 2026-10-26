from pathlib import Path
from lib.data_preparation import *
from lib.return_models import run_models
from lib.backtest import run_backtest
from lib.export_portfolio import run_final_portfolio

if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent / "data"
    outputs_dir = Path(__file__).resolve().parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    stock_data_prep(data_dir, outputs_dir)
    market_features_prep(data_dir, outputs_dir)
    model_features_prep(outputs_dir)
    rank_features_prep(outputs_dir)
    model_dataset_prep(outputs_dir)
    comparison, _, _ = run_models(outputs_dir)
    print(comparison.round(6).to_string(index=False))

    # portfolio construction and backtest (takes a few minutes: 3 models x 36 monthly optimizations)
    summary, deciles = run_backtest(outputs_dir, data_dir, verbose=False)
    print(summary.round(4).to_string())
    run_final_portfolio(outputs_dir, data_dir)
