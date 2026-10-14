"""다가오는 경기를 시뮬레이션한다. 일정은 football-data.co.uk fixtures.csv에서 가져오고, --match로 직접 지정할 수도 있다."""

import argparse

import pandas as pd

from toto.confidence import TIERS, classify
from toto.data import load
from toto.features import FEATURES, add_features, feature_diff
from toto.model import DixonColes
from toto.simulate import simulate

FIXTURES_URL = "https://football-data.co.uk/fixtures.csv"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", default="E0")
    ap.add_argument("--seasons", nargs="+", default=["2425", "2526", "2627"])
    ap.add_argument("--match", nargs=2, action="append", metavar=("HOME", "AWAY"))
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--date", help="--match 경기의 날짜(YYYY-MM-DD). 없으면 휴식일 특성을 빼고 계산한다")
    args = ap.parse_args()

    history = load(args.league, args.seasons)
    today = pd.Timestamp.now().normalize()
    model = DixonColes.fit(add_features(history), as_of=today)

    if args.match:
        match_date = pd.Timestamp(args.date) if args.date else today
        fixtures = [(h, a, None, match_date) for h, a in args.match]
    else:
        fx = pd.read_csv(FIXTURES_URL, encoding="utf-8-sig")
        fx = fx[fx["Div"] == args.league]
        fixtures = [(r.HomeTeam, r.AwayTeam, (r.B365H, r.B365D, r.B365A), pd.to_datetime(r.Date, dayfirst=True))
                    for r in fx.itertuples()]
        if not fixtures:
            print("예정된 경기가 없습니다. --match HOME AWAY 로 직접 지정하세요.")

    upcoming = pd.DataFrame([{"date": d, "season": args.seasons[-1], "home": h, "away": a, "hg": None, "ag": None,
                              "order": i} for i, (h, a, _, d) in enumerate(fixtures)])
    diffs = []
    if fixtures:
        with_features = add_features(pd.concat([history, upcoming], ignore_index=True))
        diffs = feature_diff(with_features[with_features["hg"].isna()].sort_values("order"))
        if args.match and not args.date:
            diffs[:, FEATURES.index("short_rest")] = 0
    results = []
    for (home, away, odds, _), diff in zip(fixtures, diffs):
        if not (model.knows(home) and model.knows(away)):
            print(f"{home} vs {away}: 학습 데이터에 없는 팀이라 건너뜀")
            continue
        s = simulate(model, home, away, n=args.n, diff=diff)
        probs = [s["home_win"], s["draw"], s["away_win"]]
        pick = max(range(3), key=probs.__getitem__)
        results.append((home, away, odds, s, probs, pick, classify(probs[pick]), diff))

    results.sort(key=lambda r: -r[4][r[5]])
    if results:
        counts = {t.label: sum(r[6] is t for r in results) for t in TIERS}
        print(f"{len(results)}경기 예측 ({args.n:,}회 시뮬레이션): " + ", ".join(f"{k} {v}" for k, v in counts.items()))

    for home, away, odds, s, probs, pick, tier, diff in results:
        pick_label = ["홈 승", "무승부", "원정 승"][pick]
        lam, mu = model.expected_goals(home, away, diff)
        print(f"\n[{tier.label}] {home} vs {away}  ->  {pick_label} {probs[pick]:.0%}  (과거 이 등급 적중률 약 {tier.past_hit_rate:.0%})")
        print(f"  승 {probs[0]:.1%}  무 {probs[1]:.1%}  패 {probs[2]:.1%}   기대 득점 {lam:.2f} : {mu:.2f}")
        print("  많이 나온 스코어: " + ", ".join(f"{sc} ({p:.1%})" for sc, p in s["top_scores"]))
        if odds:
            inv = [1 / o for o in odds]
            print("  배당 기준 확률: " + "  ".join(f"{k} {v / sum(inv):.1%}" for k, v in zip("승무패", inv)))
