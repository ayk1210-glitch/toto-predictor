"""1순위 결과의 확률로 경기를 등급으로 나눈다.

구간과 과거 적중률은 2025-26 프리미어리그 백테스트(379경기, 휴식일·순위 특성 포함 모델) 결과다.
'확실' 구간은 28경기뿐이라 적중률이 몇 %포인트씩 흔들릴 수 있다."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Tier:
    label: str
    min_prob: float
    past_hit_rate: float


TIERS = [
    Tier("확실", 0.65, 0.71),
    Tier("유력", 0.55, 0.65),
    Tier("애매", 0.45, 0.47),
    Tier("예측 어려움", 0.0, 0.43),
]


def classify(top_prob: float) -> Tier:
    for tier in TIERS:
        if top_prob >= tier.min_prob:
            return tier
    raise ValueError(top_prob)
