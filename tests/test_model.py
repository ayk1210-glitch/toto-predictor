import numpy as np
import pandas as pd
import pytest

from toto.model import DixonColes
from toto.simulate import simulate


@pytest.fixture(scope="module")
def synthetic():
    """강팀 A, 중간 B·C, 약팀 D가 정해진 실력대로 경기한 가상 리그."""
    rng = np.random.default_rng(0)
    strength = {"A": 0.5, "B": 0.0, "C": 0.0, "D": -0.5}
    rows = []
    start = pd.Timestamp("2025-01-01")
    for k in range(60):
        for h in strength:
            for a in strength:
                if h == a:
                    continue
                lam = np.exp(strength[h] - strength[a] + 0.25)
                mu = np.exp(strength[a] - strength[h])
                rows.append({"date": start + pd.Timedelta(days=k), "home": h, "away": a,
                             "hg": rng.poisson(lam), "ag": rng.poisson(mu)})
    df = pd.DataFrame(rows)
    return DixonColes.fit(df, as_of=start + pd.Timedelta(days=61), half_life_days=1e9)


def test_probabilities_sum_to_one(synthetic):
    assert synthetic.score_matrix("A", "D").sum() == pytest.approx(1.0)
    assert synthetic.outcome_probs("B", "C").sum() == pytest.approx(1.0)


def test_strong_team_favoured_both_home_and_away(synthetic):
    assert synthetic.outcome_probs("A", "D")[0] > 0.6
    assert synthetic.outcome_probs("D", "A")[2] > 0.5


def test_home_advantage_recovered(synthetic):
    assert synthetic.home_adv == pytest.approx(0.25, abs=0.1)


def test_monte_carlo_matches_exact_probabilities(synthetic):
    exact = synthetic.outcome_probs("A", "B")
    s = simulate(synthetic, "A", "B", n=200_000, seed=1)
    assert [s["home_win"], s["draw"], s["away_win"]] == pytest.approx(exact.tolist(), abs=0.005)


def test_old_matches_count_less():
    """같은 두 팀이 오래전엔 홈팀이 대승, 최근엔 원정팀이 대승했다면 최근 경기를 따라가야 한다."""
    old = [{"date": pd.Timestamp("2020-01-01"), "home": "X", "away": "Y", "hg": 4, "ag": 0}] * 20
    new = [{"date": pd.Timestamp("2025-01-01"), "home": "X", "away": "Y", "hg": 0, "ag": 4}] * 20
    m = DixonColes.fit(pd.DataFrame(old + new), as_of=pd.Timestamp("2025-01-02"), half_life_days=90)
    p = m.outcome_probs("X", "Y")
    assert p[2] > p[0]
