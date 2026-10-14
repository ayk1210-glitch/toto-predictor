"""다가오는 경기를 시뮬레이션한다. 일정은 football-data.co.uk fixtures.csv에서 가져오고, --match로 직접 지정할 수도 있다."""

import argparse

import pandas as pd

from toto.confidence import TIERS, classify
from toto.data import load
from toto.model import DixonColes
from toto.simulate import simulate

FIXTURES_URL = "https://football-data.co.uk/fixtures.csv"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", default="E0")
    ap.add_argument("--seasons", nargs="+", default=["2425", "2526", "2627"])
    ap.add_argument("--match", nargs=2, action="append", metavar=("HOME", "AWAY"))
    ap.add_argument("--n", type=int, default=10_000)
    args = ap.parse_args()

    history = load(args.league, args.seasons)
    model = DixonColes.fit(history, as_of=pd.Timestamp.now().normalize())

    if args.match:
        fixtures = [(h, a, None) for h, a in args.match]
    else:
        fx = pd.read_csv(FIXTURES_URL, encoding="utf-8-sig")
        fx = fx[fx["Div"] == args.league]
        fixtures = [(r.HomeTeam, r.AwayTeam, (r.B365H, r.B365D, r.B365A)) for r in fx.itertuples()]
        if not fixtures:
            print("예정된 경기가 없습니다. --match HOME AWAY 로 직접 지정하세요.")

    results = []
    for home, away, odds in fixtures:
        if not (model.knows(home) and model.knows(away)):
            print(f"{home} vs {away}: 학습 데이터에 없는 팀이라 건너뜀")
            continue
        s = simulate(model, home, away, n=args.n)
        probs = [s["home_win"], s["draw"], s["away_win"]]
        pick = max(range(3), key=probs.__getitem__)
        results.append((home, away, odds, s, probs, pick, classify(probs[pick])))

    results.sort(key=lambda r: -r[4][r[5]])
    if results:
        counts = {t.label: sum(r[6] is t for r in results) for t in TIERS}
        print(f"{len(results)}경기 예측 ({args.n:,}회 시뮬레이션): " + ", ".join(f"{k} {v}" for k, v in counts.items()))

    for home, away, odds, s, probs, pick, tier in results:
        pick_label = ["홈 승", "무승부", "원정 승"][pick]
        lam, mu = model.expected_goals(home, away)
        print(f"\n[{tier.label}] {home} vs {away}  ->  {pick_label} {probs[pick]:.0%}  (과거 이 등급 적중률 약 {tier.past_hit_rate:.0%})")
        print(f"  승 {probs[0]:.1%}  무 {probs[1]:.1%}  패 {probs[2]:.1%}   기대 득점 {lam:.2f} : {mu:.2f}")
        print("  많이 나온 스코어: " + ", ".join(f"{sc} ({p:.1%})" for sc, p in s["top_scores"]))
        if odds:
            inv = [1 / o for o in odds]
            print("  배당 기준 확률: " + "  ".join(f"{k} {v / sum(inv):.1%}" for k, v in zip("승무패", inv)))
