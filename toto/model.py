"""Dixon-Coles 축구 모델: 팀별 공격력·수비력 + 홈 이점, 저득점 보정, 최근 경기 가중.

goal_weight < 1이면 실제 골 대신 골과 '유효슈팅 x 리그 평균 전환율'을 섞은 값으로 실력을 추정한다.
골은 운이 크게 섞이는데 유효슈팅은 그보다 덜 흔들리기 때문이다."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from .features import FEATURES, feature_diff

MAX_GOALS = 10
# 2023-25 프리미어리그 백테스트로 고르고 2025-26 시즌으로 확인한 값 (tune.py)
HALF_LIFE_DAYS = 180.0
GOAL_WEIGHT = 0.5


def _tau(hg, ag, lam, mu, rho):
    t = np.ones_like(lam)
    t = np.where((hg == 0) & (ag == 0), 1 - lam * mu * rho, t)
    t = np.where((hg == 0) & (ag == 1), 1 + lam * rho, t)
    t = np.where((hg == 1) & (ag == 0), 1 + mu * rho, t)
    t = np.where((hg == 1) & (ag == 1), 1 - rho, t)
    return t


@dataclass
class DixonColes:
    teams: list[str]
    attack: np.ndarray
    defence: np.ndarray
    home_adv: float
    rho: float
    beta: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @classmethod
    def fit(
        cls, matches: pd.DataFrame, as_of: pd.Timestamp, half_life_days: float = HALF_LIFE_DAYS, goal_weight: float = GOAL_WEIGHT,
        use_features: bool = True,
    ) -> "DixonColes":
        """특성은 matches에 features.add_features()를 먼저 적용해 둔 경우에만 쓰인다."""
        past = matches[matches["date"] < as_of]
        teams = sorted(set(past["home"]) | set(past["away"]))
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        h = past["home"].map(idx).to_numpy()
        a = past["away"].map(idx).to_numpy()
        hg = past["hg"].to_numpy()
        ag = past["ag"].to_numpy()
        age = (as_of - past["date"]).dt.days.to_numpy()
        w = 0.5 ** (age / half_life_days)

        if goal_weight < 1 and {"hst", "ast"} <= set(past.columns):
            sot = past[["hst", "ast"]].to_numpy(dtype=float)
            has_shots = ~np.isnan(sot).any(axis=1)
            conv = (hg[has_shots].sum() + ag[has_shots].sum()) / sot[has_shots].sum()
            hy = np.where(has_shots, goal_weight * hg + (1 - goal_weight) * conv * np.nan_to_num(sot[:, 0]), hg)
            ay = np.where(has_shots, goal_weight * ag + (1 - goal_weight) * conv * np.nan_to_num(sot[:, 1]), ag)
        else:
            hy, ay = hg, ag

        has_features = all(f"h_{f}" in past.columns for f in FEATURES)
        x = feature_diff(past) if use_features and has_features else np.zeros((len(past), 0))
        k = x.shape[1]

        def nll(p):
            att, dfc = p[:n], p[n : 2 * n]
            home, rho, beta = p[2 * n], p[2 * n + 1], p[2 * n + 2 :]
            shift = x @ beta
            lam = np.exp(att[h] + dfc[a] + home + shift)
            mu = np.exp(att[a] + dfc[h] - shift)
            tau = np.clip(_tau(hg, ag, lam, mu, rho), 1e-10, None)
            # 목표값이 정수가 아닐 수 있어 포아송 로그우도에서 상수항(log k!)을 뺀 형태를 쓴다.
            ll = np.log(tau) + hy * np.log(lam) - lam + ay * np.log(mu) - mu
            # 공격력 합 0 제약을 벌점으로 걸어 식별 가능하게 만든다.
            return -np.sum(w * ll) + 100 * att.sum() ** 2

        x0 = np.concatenate([np.zeros(n), np.zeros(n), [0.25, -0.05], np.zeros(k)])
        bounds = [(-3, 3)] * (2 * n) + [(-1, 1), (-0.3, 0.3)] + [(-1, 1)] * k
        res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds)
        p = res.x
        return cls(teams, p[:n], p[n : 2 * n], p[2 * n], p[2 * n + 1], p[2 * n + 2 :])

    def feature_effects(self) -> dict[str, float]:
        return dict(zip(FEATURES, self.beta.tolist()))

    def expected_goals(self, home: str, away: str, diff: np.ndarray | None = None) -> tuple[float, float]:
        """diff: 이 경기의 feature_diff 한 행. 특성 없이 학습한 모델이면 무시된다."""
        i, j = self.teams.index(home), self.teams.index(away)
        shift = float(diff @ self.beta) if diff is not None and len(self.beta) else 0.0
        lam = np.exp(self.attack[i] + self.defence[j] + self.home_adv + shift)
        mu = np.exp(self.attack[j] + self.defence[i] - shift)
        return float(lam), float(mu)

    def score_matrix(self, home: str, away: str, diff: np.ndarray | None = None) -> np.ndarray:
        lam, mu = self.expected_goals(home, away, diff)
        g = np.arange(MAX_GOALS + 1)
        m = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
        hg, ag = np.meshgrid(g, g, indexing="ij")
        m *= _tau(hg, ag, np.full_like(m, lam), np.full_like(m, mu), self.rho)
        return m / m.sum()

    def outcome_probs(self, home: str, away: str, diff: np.ndarray | None = None) -> np.ndarray:
        m = self.score_matrix(home, away, diff)
        return np.array([np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()])

    def knows(self, team: str) -> bool:
        return team in self.teams
