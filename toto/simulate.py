import numpy as np

from .model import DixonColes


def simulate(model: DixonColes, home: str, away: str, n: int = 10_000, seed: int | None = None) -> dict:
    """스코어 분포에서 경기를 n번 뽑아 승/무/패 비율과 많이 나온 스코어를 센다."""
    rng = np.random.default_rng(seed)
    m = model.score_matrix(home, away)
    flat = rng.choice(m.size, size=n, p=m.ravel())
    hg, ag = np.divmod(flat, m.shape[1])
    scores, counts = np.unique(np.stack([hg, ag], axis=1), axis=0, return_counts=True)
    top = np.argsort(-counts)[:3]
    return {
        "home_win": float(np.mean(hg > ag)),
        "draw": float(np.mean(hg == ag)),
        "away_win": float(np.mean(hg < ag)),
        "top_scores": [(f"{s[0]}-{s[1]}", c / n) for s, c in zip(scores[top], counts[top])],
    }
