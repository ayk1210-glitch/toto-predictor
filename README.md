# toto-predictor

축구 경기 승/무/패 예측. Dixon-Coles 모델로 스코어 분포를 추정하고, 그 분포로 경기를 1만 번 시뮬레이션한다.
데이터는 football-data.co.uk 공개 CSV(결과 + 배당)를 쓴다. API 키가 필요 없다.

## 실행

```bash
python3 -m venv .venv
.venv/bin/pip install numpy scipy pandas pytest

.venv/bin/python backtest.py                              # 지난 두 시즌으로 모델 대 마감 배당 비교
.venv/bin/python predict.py                               # 예정된 경기 예측
.venv/bin/python predict.py --match Arsenal "Man City" --date 2026-10-18   # 경기 직접 지정
.venv/bin/python tune.py                                  # 설정 조합별 백테스트 비교
.venv/bin/python -m pytest -q tests
```

## 현재 성능

2025-26 프리미어리그 379경기, 설정을 고를 때 쓰지 않은 시즌 기준. Brier는 낮을수록 좋다.

| | Brier | 적중률 |
|---|---|---|
| 골만 쓴 모델 | 0.6198 | 46.2% |
| 골 + 유효슈팅 | 0.6135 | 48.5% |
| 골 + 유효슈팅 + 휴식일·순위 상황 (현재 기본값) | 0.6110 | 50.1% |
| 마감 배당 | 0.6073 | 49.6% |

설정값(반감기 180일, 골 비중 0.5)은 2023-24, 2024-25 시즌 백테스트로 골랐다.
휴식일은 리그 경기만 보고 계산한다. 컵대회·유럽대항전 일정은 아직 반영하지 못한다.
아직 배당보다 덜 맞으므로 "모델이 배당보다 유리하다"는 신호는 대부분 모델 오차다.

리그 코드: E0 프리미어리그, SP1 라리가, I1 세리에A, D1 분데스리가, F1 리그1 (`--league`로 지정).
팀 이름은 football-data.co.uk 표기를 따른다(`Man City`, `Nott'm Forest` 등).
