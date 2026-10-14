"""과거 시즌을 한 주씩 걸어가며, 그 주 이전 경기로만 학습해 예측하고 배당과 비교한다."""

import argparse

import numpy as np
import pandas as pd

from toto.data import load
from toto.features import add_features, feature_diff
from toto.model import GOAL_WEIGHT, HALF_LIFE_DAYS, DixonColes

OUTCOMES = ["H", "D", "A"]


def implied_probs(row) -> np.ndarray:
    inv = 1 / np.array([row.odds_h, row.odds_d, row.odds_a])
    return inv / inv.sum()


def brier(p: np.ndarray, y: int) -> float:
    target = np.zeros(3)
    target[y] = 1
    return float(np.sum((p - target) ** 2))


def run(
    league: str, train_seasons: list[str], test_seasons: list[str], edge: float = 0.05,
    half_life_days: float = HALF_LIFE_DAYS, goal_weight: float = GOAL_WEIGHT, use_features: bool = True,
) -> pd.DataFrame:
    df = add_features(load(league, train_seasons + test_seasons))
    test = df[df["season"].isin(test_seasons)].dropna(subset=["odds_h"])
    rows = []
    for week_start, week in test.groupby(test["date"].dt.to_period("W").dt.start_time):
        model = DixonColes.fit(df, as_of=week_start, half_life_days=half_life_days, goal_weight=goal_weight,
                               use_features=use_features)
        diffs = feature_diff(week)
        for r, diff in zip(week.itertuples(), diffs):
            if not (model.knows(r.home) and model.knows(r.away)):
                continue
            pm = model.outcome_probs(r.home, r.away, diff)
            po = implied_probs(r)
            y = 0 if r.hg > r.ag else (1 if r.hg == r.ag else 2)
            odds = np.array([r.odds_h, r.odds_d, r.odds_a])
            ev = pm * odds - 1
            pick = int(np.argmax(ev))
            bet = ev[pick] > edge
            rows.append({
                "date": r.date, "home": r.home, "away": r.away, "result": OUTCOMES[y],
                "p_model": pm.round(3).tolist(), "p_odds": po.round(3).tolist(),
                "brier_model": brier(pm, y), "brier_odds": brier(po, y),
                "logloss_model": -np.log(pm[y]), "logloss_odds": -np.log(po[y]),
                "hit_model": int(np.argmax(pm) == y), "hit_odds": int(np.argmax(po) == y),
                "bet": bet, "profit": (odds[pick] - 1 if pick == y else -1.0) if bet else 0.0,
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", default="E0")
    ap.add_argument("--train", nargs="+", default=["1920", "2021", "2122", "2223", "2324"])
    ap.add_argument("--test", nargs="+", default=["2425", "2526"])
    ap.add_argument("--edge", type=float, default=0.05, help="기대수익이 이 값보다 클 때만 베팅한 것으로 친다")
    ap.add_argument("--half-life", type=float, default=HALF_LIFE_DAYS)
    ap.add_argument("--goal-weight", type=float, default=GOAL_WEIGHT)
    ap.add_argument("--no-features", dest="features", action="store_false", help="휴식일·순위 경쟁 특성을 뺀다")
    args = ap.parse_args()

    res = run(args.league, args.train, args.test, args.edge, args.half_life, args.goal_weight, args.features)
    bets = res[res["bet"]]
    print(f"평가 경기 수: {len(res)}")
    print(f"{'':12}{'모델':>10}{'배당':>10}")
    print(f"{'Brier':12}{res.brier_model.mean():>10.4f}{res.brier_odds.mean():>10.4f}   (낮을수록 좋음)")
    print(f"{'Log loss':12}{res.logloss_model.mean():>10.4f}{res.logloss_odds.mean():>10.4f}   (낮을수록 좋음)")
    print(f"{'적중률':12}{res.hit_model.mean():>10.1%}{res.hit_odds.mean():>10.1%}   (가장 높은 확률 결과가 맞은 비율)")
    if len(bets):
        print(f"가상 베팅: {len(bets)}경기, 1단위씩 걸었을 때 수익률 {bets.profit.sum() / len(bets):+.1%}")
    res.to_csv("backtest_result.csv", index=False)
