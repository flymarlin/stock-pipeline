#!/usr/bin/env python3
"""
Grid → Optuna 二段ハイパラ探索
$ python hyperopt.py --config config/config_20250704.json
"""
import argparse, json, itertools, os, datetime, shutil, uuid
import pandas as pd
import optuna
from tqdm import tqdm
from backtest import run_backtest

# ─────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", default="results")
    return ap.parse_args()

# ─────────────────────────────────────────────────────────
def load_config(path):
    with open(path) as f:
        return json.load(f)

# ─────────────────────────────────────────────────────────
def objective_factory(cfg, mode="optuna"):
    """Optuna 目的関数を生成"""
    def obj(trial):
        params = {}
        # technical
        for k, v in cfg["signals"]["technical"].items():
            params[k] = trial.suggest_categorical(k, v)
        # fundamental
        for k, v in cfg["signals"]["fundamental"].items():
            params[k] = trial.suggest_categorical(k, v)

        stats = run_backtest(
            start_date=cfg["start_date"],
            end_date=cfg["end_date"],
            params=params,
            cache_tag="hyperopt"
        )
        score = stats["CAGR"] - 0.5 * abs(stats["MDD"])
        return score
    return obj

# ─────────────────────────────────────────────────────────
def grid_search(cfg):
    grid_keys = []
    grid_values = []
    # flatten
    for d in (cfg["signals"]["technical"], cfg["signals"]["fundamental"]):
        for k, v in d.items():
            grid_keys.append(k)
            grid_values.append(v)

    results = []
    for combo in tqdm(list(itertools.product(*grid_values)), desc="Grid"):
        params = dict(zip(grid_keys, combo))
        stats = run_backtest(
            start_date=cfg["start_date"],
            end_date=cfg["end_date"],
            params=params,
            cache_tag="hyperopt"
        )
        score = stats["CAGR"] - 0.5 * abs(stats["MDD"])
        results.append({"score": score, **params, **stats})

    return pd.DataFrame(results)

# ─────────────────────────────────────────────────────────
def main():
    args  = parse_args()
    cfg   = load_config(args.config)
    tag   = datetime.datetime.now().strftime("%y%m%d_%H%M") + "_" + uuid.uuid4().hex[:6]
    out   = os.path.join(args.outdir, tag)
    os.makedirs(out, exist_ok=True)
    shutil.copy(args.config, os.path.join(out, "config_used.json"))

    # ① 粗グリッド
    df_grid = grid_search(cfg)
    df_grid.to_csv(os.path.join(out, "grid_results.csv"), index=False)
    top5 = df_grid.nlargest(5, "score")
    top5.to_json(os.path.join(out, "best_params_grid.json"), orient="records", indent=2)

    # ② Optuna（初期点にグリッド上位を渡す）
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cfg["optuna"]["random_seed"])
    )
    study.enqueue_trial(top5.iloc[0].to_dict())  # 代表点だけキュー
    study.optimize(
        objective_factory(cfg),
        n_trials=cfg["optuna"]["trials"],
        show_progress_bar=True
    )

    # 保存
    df_optuna = study.trials_dataframe()
    df_optuna.to_csv(os.path.join(out, "optuna_results.csv"), index=False)
    best = study.best_trial.params
    json.dump(best, open(os.path.join(out, "best_params_optuna.json"), "w"), indent=2)
    print(f"\n✨  Done.  Results → {out}\n")

if __name__ == "__main__":
    main()
