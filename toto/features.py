"""경기 직전 시점의 팀 상황을 계산한다: 휴식일, 순위 경쟁 여부.

결과가 없는 행(hg가 비어 있는 예정 경기)은 특성만 계산하고 순위에는 반영하지 않는다.
같은 날 열린 경기 결과는 서로의 순위 계산에 넣지 않는다(킥오프 순서를 몰라 미래 정보가 섞일 수 있다).
리그 경기만 보므로 컵대회와 유럽대항전 일정은 휴식일에 반영되지 않는다."""

import numpy as np
import pandas as pd

SEASON_GAMES = 38
RELEGATION_PLACE = 18
EUROPE_PLACE = 7
FIGHT_WINDOW_GAMES = 10
FIGHT_POINTS = 6
SHORT_REST_DAYS = 4

FEATURES = ["short_rest", "settled", "relegation_fight"]


def _team_flags(team: str, table: pd.DataFrame) -> tuple[int, int]:
    """table: points·played를 가진 팀별 표(index=팀). (settled, relegation_fight)를 돌려준다."""
    ordered = table.sort_values("points", ascending=False)
    pts = table.at[team, "points"]
    remaining = SEASON_GAMES - table.at[team, "played"]
    max_gain = 3 * remaining
    line_safe = ordered["points"].iloc[RELEGATION_PLACE - 1]
    line_drop = ordered["points"].iloc[RELEGATION_PLACE - 2]
    line_europe = ordered["points"].iloc[EUROPE_PLACE - 1]

    safe = pts - line_safe > max_gain
    no_europe = line_europe - pts > max_gain
    relegated = line_drop - pts > max_gain
    runner_up = ordered.iloc[1]
    champion = ordered.index[0] == team and pts - runner_up["points"] > 3 * (SEASON_GAMES - runner_up["played"])
    settled = int((safe and no_europe) or relegated or champion)
    fight = int(remaining <= FIGHT_WINDOW_GAMES and not relegated and abs(pts - line_safe) <= FIGHT_POINTS)
    return settled, fight


def add_features(matches: pd.DataFrame) -> pd.DataFrame:
    df = matches.sort_values("date").reset_index(drop=True)
    out = {f"{side}_{f}": np.zeros(len(df), dtype=int) for side in ("h", "a") for f in FEATURES}
    last_played: dict[str, pd.Timestamp] = {}

    for season, sdf in df.groupby("season", sort=False):
        teams = sorted(set(sdf["home"]) | set(sdf["away"]))
        table = pd.DataFrame({"points": 0, "played": 0}, index=teams)
        for date, day in sdf.groupby("date"):
            for i, r in day.iterrows():
                for side, team in (("h", r.home), ("a", r.away)):
                    prev = last_played.get(team)
                    out[f"{side}_short_rest"][i] = int(prev is not None and (date - prev).days <= SHORT_REST_DAYS)
                    if len(teams) >= RELEGATION_PLACE:
                        settled, fight = _team_flags(team, table)
                        out[f"{side}_settled"][i] = settled
                        out[f"{side}_relegation_fight"][i] = fight
            for r in day.itertuples():
                if pd.isna(r.hg):
                    continue
                hp, ap = (3, 0) if r.hg > r.ag else ((1, 1) if r.hg == r.ag else (0, 3))
                table.loc[r.home, ["points", "played"]] += [hp, 1]
                table.loc[r.away, ["points", "played"]] += [ap, 1]
                last_played[r.home] = last_played[r.away] = date

    return df.assign(**out)


def feature_diff(matches: pd.DataFrame) -> np.ndarray:
    """홈팀 특성 - 원정팀 특성. 양수면 홈팀 쪽에 그 상황이 있다는 뜻."""
    return np.stack([matches[f"h_{f}"] - matches[f"a_{f}"] for f in FEATURES], axis=1).astype(float)
