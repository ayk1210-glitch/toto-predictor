from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BASE_URL = "https://football-data.co.uk/mmz4281/{season}/{league}.csv"

ODDS_PRIORITY = [("AvgCH", "AvgCD", "AvgCA"), ("PSCH", "PSCD", "PSCA"), ("B365H", "B365D", "B365A")]
USE_COLS = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", *(c for cols in ODDS_PRIORITY for c in cols)}


def download(league: str, season: str) -> Path:
    import urllib.request

    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"{league}_{season}.csv"
    urllib.request.urlretrieve(BASE_URL.format(season=season, league=league), path)
    return path


def load(league: str, seasons: list[str]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        path = DATA_DIR / f"{league}_{season}.csv"
        if not path.exists():
            download(league, season)
        frames.append(pd.read_csv(path, encoding="utf-8-sig", usecols=lambda c: c in USE_COLS))
    df = pd.concat(frames, keys=seasons, names=["season", None]).reset_index(level="season")
    df = df.reset_index(drop=True)
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    df["date"] = pd.to_datetime(df["Date"], dayfirst=True)

    # 마감 배당 우선. 마감 배당이 없는 행은 다음 순위로 채운다.
    df["odds_h"] = df["odds_d"] = df["odds_a"] = float("nan")
    for h, d, a in ODDS_PRIORITY:
        if h in df:
            mask = df["odds_h"].isna()
            df.loc[mask, ["odds_h", "odds_d", "odds_a"]] = df.loc[mask, [h, d, a]].to_numpy()

    out = df[["date", "season", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "odds_h", "odds_d", "odds_a"]]
    out = out.rename(columns={"HomeTeam": "home", "AwayTeam": "away", "FTHG": "hg", "FTAG": "ag"})
    out[["hg", "ag"]] = out[["hg", "ag"]].astype(int)
    return out.sort_values("date").reset_index(drop=True)
