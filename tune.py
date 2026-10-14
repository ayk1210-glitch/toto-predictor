"""설정 조합별로 백테스트를 돌려 Brier 점수를 비교한다. 조정용 시즌과 확인용 시즌을 나눠 과적합을 막는다."""

import argparse
import itertools
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from backtest import run

TRAIN = ["1920", "2021", "2122", "2223"]


def evaluate(args):
    league, test, half_life, goal_weight = args
    res = run(league, TRAIN, test, half_life_days=half_life, goal_weight=goal_weight)
    return {"half_life": half_life, "goal_weight": goal_weight, "n": len(res),
            "brier_model": res.brier_model.mean(), "brier_odds": res.brier_odds.mean(),
            "logloss_model": res.logloss_model.mean()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", default="E0")
    ap.add_argument("--tune", nargs="+", default=["2324", "2425"])
    ap.add_argument("--half-lives", nargs="+", type=float, default=[90, 180, 365, 730])
    ap.add_argument("--goal-weights", nargs="+", type=float, default=[1.0, 0.7, 0.5, 0.3, 0.0])
    args = ap.parse_args()

    grid = [(args.league, args.tune, h, g) for h, g in itertools.product(args.half_lives, args.goal_weights)]
    with ProcessPoolExecutor() as ex:
        rows = list(ex.map(evaluate, grid))
    table = pd.DataFrame(rows).sort_values("brier_model")
    print(table.to_string(index=False, float_format="%.4f"))
