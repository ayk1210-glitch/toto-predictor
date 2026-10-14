# toto-predictor

축구 경기 승/무/패 예측. Dixon-Coles 모델로 스코어 분포를 추정하고, 그 분포로 경기를 1만 번 시뮬레이션한다.
데이터는 football-data.co.uk 공개 CSV(결과 + 배당)를 쓴다. API 키가 필요 없다.

## 실행

```bash
python3 -m venv .venv
.venv/bin/pip install numpy scipy pandas pytest

.venv/bin/python backtest.py                              # 지난 두 시즌으로 모델 대 마감 배당 비교
.venv/bin/python predict.py                               # 예정된 경기 예측
.venv/bin/python predict.py --match Arsenal "Man City"    # 경기 직접 지정
.venv/bin/python -m pytest -q tests
```

리그 코드: E0 프리미어리그, SP1 라리가, I1 세리에A, D1 분데스리가, F1 리그1 (`--league`로 지정).
팀 이름은 football-data.co.uk 표기를 따른다(`Man City`, `Nott'm Forest` 등).
