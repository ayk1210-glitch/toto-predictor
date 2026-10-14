import pandas as pd

from toto.features import add_features


def _season(n_teams=20):
    """팀 T00이 매 라운드 이기고 나머지는 전부 비기는 38라운드짜리 리그."""
    teams = [f"T{i:02d}" for i in range(n_teams)]
    rows = []
    start = pd.Timestamp("2025-08-01")
    for rnd in range(38):
        order = teams[:1] + teams[1:][rnd % (n_teams - 1):] + teams[1:][: rnd % (n_teams - 1)]
        date = start + pd.Timedelta(days=7 * rnd)
        for k in range(n_teams // 2):
            h, a = order[k], order[-1 - k]
            hg, ag = (1, 0) if h == "T00" else ((0, 1) if a == "T00" else (0, 0))
            rows.append({"date": date, "season": "2526", "home": h, "away": a, "hg": hg, "ag": ag})
    return pd.DataFrame(rows)


def test_runaway_leader_becomes_settled_once_title_is_clinched():
    df = add_features(_season())
    t0 = df[(df.home == "T00") | (df.away == "T00")].reset_index(drop=True)
    settled = [r.h_settled if r.home == "T00" else r.a_settled for r in t0.itertuples()]
    assert settled[0] == 0
    assert settled[-1] == 1


def test_no_flags_on_opening_day():
    df = add_features(_season())
    first = df[df.date == df.date.min()]
    assert first[["h_settled", "a_settled", "h_relegation_fight", "a_relegation_fight"]].to_numpy().sum() == 0


def test_short_rest_counts_days_since_last_league_match():
    df = pd.DataFrame([
        {"date": pd.Timestamp("2025-08-01"), "season": "2526", "home": "A", "away": "B", "hg": 1, "ag": 0},
        {"date": pd.Timestamp("2025-08-04"), "season": "2526", "home": "A", "away": "C", "hg": 1, "ag": 0},
        {"date": pd.Timestamp("2025-08-11"), "season": "2526", "home": "B", "away": "A", "hg": 1, "ag": 0},
    ])
    out = add_features(df)
    assert out.loc[1, "h_short_rest"] == 1
    assert out.loc[1, "a_short_rest"] == 0
    assert out.loc[2, "a_short_rest"] == 0
