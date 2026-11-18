# 🍊 전국 공영도매시장 감귤 품종 · 거래단위별 일별 가격 및 거래량 AI 예측 시스템

전국 32개 공영도매시장 감귤 경매 정산 데이터(`2021-01-02 ~ 2026-11-11`, 유효 약 172만 건)와 제주 산지 4개 기상관측소 기상 데이터(`data/forecast.csv`)를 결합하여, **4대 핵심 감귤 품종(조생귤 · 한라봉 · 하우스감귤 · 천혜향) 및 상위 거래단위(16개 주요 규격)**별 **날짜별(Daily) 거래량(톤/일)**과 **가격(원/kg 및 상자당 가격)**을 4종의 AI/머신러닝 모델(**GPU TFT, GPU XGBoost, LightGBM, Random Forest**)로 예측하고 **Streamlit 대시보드** 및 **Docker 컨테이너**로 제공하는 프로젝트입니다.

---

## 📁 프로젝트 구조

```text
├── preprocess.ipynb                # [전처리 & EDA] 품종·연도·월·거래단위·등급별·크기별 가격 및 거래량 통계 분석 (1~9번 섹션)
├── tft.ipynb                       # [모델 1] CUDA GPU Temporal Fusion Transformer (TFT) 학습·예측·해석성 분석
├── xgboost.ipynb                   # [모델 2] CUDA GPU Multi-Quantile XGBoost 학습·예측·해석성 분석
├── lightgbm.ipynb                  # [모델 3] Multi-Quantile LightGBM 학습·예측·해석성 분석
├── random_forest.ipynb             # [모델 4] Quantile Regression Forest (Random Forest) 학습·예측·해석성 분석
├── app.py                          # [대시보드] 선택일 포함 주간(월~토) 모델별 예상 가격·거래량 및 정확도 비교 Streamlit 앱
├── data/
│   ├── forecast.csv                # 제주 4개 관측소(제주·서귀포·성산·고산) 일별 기상 관측 데이터
│   ├── kat*.csv / mkat*.csv        # 공영도매시장 일반감귤 및 만감류 일별 경매 원본 데이터
│   └── streamlit_predictions.pkl   # 4개 튜닝 모델의 전체 기간(2021-01-02 ~ 2026-12-31) 사전 계산 예측 데이터셋
├── Dockerfile                      # Streamlit 시각화 전용 경량 Docker 이미지 빌드 설정 (나눔고딕 한글 폰트 포함)
├── docker-compose.yml              # Docker Compose 배포 설정 (포트 8501)
├── .dockerignore                   # Docker 빌드 시 학습용 노트북·원본 CSV·가상환경 제외 설정
├── requirements.txt                # Streamlit 대시보드 실행용 경량 패키지 목록
└── .gitignore                      # Git 버전 관리 제외 파일 설정
```

---

## 📊 데이터 전처리 및 탐색적 분석 (`preprocess.ipynb`)

1. **데이터 수집 및 병합 (1~3번 섹션)**
   - 일반 감귤(`kat*.csv`) 및 만감류(`mkat*.csv`) 분할 CSV 파일 병합 및 중복·무효 거래 제거 (유효 `1,722,631건`)
   - aT 표준 품종 코드(`07: 한라봉`, `08: 천혜향`, `09: 레드향`, `10: 카라향`, `11: 황금향` 등) 복원 및 1,000건 미만 소수 품종 `'기타'` 통합 (총 **18개 품종**)
   - 파렛트·상자 포장 규격 통합 (`3kg상자`, `5kg상자`, `9kg상자`, `10kg상자` 등) 및 극단 단가 이상치(`100원/kg 미만`, `100,000원/kg 초과`) 정제
2. **품종별 · 연도별 · 월별 시계열 분석 (4~6번 섹션)**
   - 18개 품종별 누적 거래량(톤), 가중평균 단가(원/kg), 평균 상자당 가격, 2021~2026년 연도별·월별 출하 계절성 시각화
3. **품종별 상위 거래단위 선정 (7번 섹션)**
   - 거래건수 **1,000건 이상** & 품종 내 건수 비중 **1.0% 이상**인 품종별 상위 최대 5개 거래단위 추출
4. **품종별 · 상위 거래단위별 등급(`Grade`) 통계 (8번 섹션)**
   - `특`, `상`, `중`(보통), `하`, `기타` 5대 등급별 거래량(톤), 비중(%), 가중평균 단가(원/kg) 및 상자당 가격 분석
5. **품종별 · 상위 거래단위별 크기(`Size`) 통계 (9번 섹션)**
   - 도매시장 68종 세부 크기 표기(`sz_nm`)를 **표준 5대 크기 그룹(`대과`, `중과`, `소과`, `기타규격`, `미지정`)**으로 분류한 피벗 통계표와, 각 `(품종 × 거래단위)`별 **실제 거래 상위 1~5위 세부 크기 규격(`50내, 40내, 10개, 12개, 특대, 대, 중, 소` 등)**의 물량·비중·단가·상자가 상세 통계표 및 시각화 그래프 3종 수록

---

## 🧠 핵심 시계열 피처 엔지니어링 (총 33개 입력 변수 & 8개 시차)

4개 예측 모델 노트북(`tft.ipynb`, `xgboost.ipynb`, `lightgbm.ipynb`, `random_forest.ipynb`)은 100% 동일한 전처리 기준과 파생변수를 공유합니다.

- **도매시장 휴장일 및 품종별 계절적 비수기(`Off-Season`) 자동 제외**:
  - 일요일(`dayofweek == 6`) 및 법정 공휴일(신정, 설·추석 연휴, 성탄절 등) 무거래 휴장일 제외
  - 연중 52주(1주 단위) 중 품종별로 거래가 미미한 주간(피크 주간 대비 1.5% 미만 또는 연간 물량의 0.25% 미만 등)을 **계절적 비수기**로 판별하여 예측에서 제외
- **1주일 전(`t-7`) 가격·거래량 및 예년(2021~2025년) 해당 시점의 가격·거래량 변화량 반영**:
  - `log_vol_lag_7`, `norm_prc_lag_7`: **1주일 전(`t-7`) 거래량 및 단가 수준**
  - `vol_diff_1w`, `prc_diff_1w`: **1주일 전 대비 직전 영업일(`t-1 - t-7`) 단기 변화량(`Δ1w`)**
  - `yoy_same_period_log_vol`, `yoy_same_period_norm_prc`: **예년(전년도 `t-1년` 및 2021~2025년 평년) 해당 시점(동일 주차)의 거래량 및 단가 수준**
  - `yoy_same_period_vol_diff`, `yoy_same_period_prc_diff`: **예년 해당 시점에서의 1주간(`w-1 → w`) 거래량 및 단가 변화량(`Δ예년동시점`)**
  - `vol_yoy_change`, `prc_yoy_change`: **예년 동시점 대비 당해 시점의 연간 변화량(`YoY`)**
- **1주일 전 제주 산지 기상 공변량 (`data/forecast.csv`)**:
  - 수확 $\rightarrow$ 선과장 예조 $\rightarrow$ 해상 운송 $\rightarrow$ 도매시장 상장까지 약 1주일이 소요되는 유통 구조를 반영하여, 직전 1주일간 제주 평균기온, 평년대비 기온편차, 누적 강수 계속시간(우천 수확 지연), 일조시간(착색·당도), 최심신적설(선박 결항) 반영
- **매년 설날·추석 명절 전후 효과 (`Holiday Covariates`)**:
  - 명절 전 대목(`D-18 ~ D-1`) 수요 급증·단가 상승 지수와 명절 직후(`D+1 ~ D+12`) 수요 조정 지수 및 품종별 명절 민감도 반영

---

## ⚙️ 모델별 하이퍼파라미터 튜닝 설정 및 예측 성능 비교

각 모델별로 거래량(`Volume`)과 단가(`Price`) 타깃의 분포 특성에 맞춰 하이퍼파라미터 튜닝을 수행한 최종 결과입니다.

| 모델 (노트북) | 타깃별 최적 튜닝 하이퍼파라미터 | 거래량 MAE / RMSE | 거래량 정확도 (`R²`) | 가격 MAE / RMSE | 가격 정확도 (`R²`) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **GPU TFT**<br>(`tft.ipynb`) | • **거래량**: `d_model=80, lr=0.0055, dropout=0.04, reg=8e-5, bs=3072, epochs=90` + `CosineAnnealingLR` & Best Ckpt<br>• **가격**: `d_model=80, lr=0.0050, dropout=0.04, reg=1e-4, bs=3072, epochs=90` + `CosineAnnealingLR` & Best Ckpt | 5.92 톤/일<br>27.25 톤/일 | **84.57%**<br>(`R² = 0.9582`) | 463 원/kg<br>745 원/kg | **86.42%**<br>(`R² = 0.8631`) |
| **GPU XGBoost**<br>(`xgboost.ipynb`) | • **거래량**: `max_depth=7, lr=0.045, min_child_weight=2, subsample=0.86, colsample=0.86, n_estimators=360` (Best Round)<br>• **가격**: `max_depth=7, lr=0.050, min_child_weight=3, subsample=0.85, colsample=0.82, reg_alpha=0.2, reg_lambda=2.0, n_estimators=380` | 6.05 톤/일<br>31.83 톤/일 | **84.23%**<br>(`R² = 0.9429`) | 431 원/kg<br>726 원/kg | **86.98%**<br>(`R² = 0.8698`) |
| **LightGBM**<br>(`lightgbm.ipynb`) | • **거래량**: `num_leaves=75, max_depth=10, lr=0.048, min_child_samples=20, subsample=0.88, colsample=0.88, n_estimators=400`<br>• **가격**: `num_leaves=70, max_depth=9, lr=0.045, min_child_samples=22, subsample=0.86, colsample=0.85, n_estimators=380` | 4.80 톤/일<br>**26.39 톤/일** | **87.49%**<br>(`R² = 0.9608`) | 405 원/kg<br>708 원/kg | **87.79%**<br>(`R² = 0.8762`) |
| **Random Forest**<br>(`random_forest.ipynb`) | • **거래량**: `n_estimators=220, max_depth=18, min_samples_leaf=3, min_samples_split=6, max_features=0.78`<br>• **가격**: `n_estimators=220, max_depth=18, min_samples_leaf=3, min_samples_split=6, max_features=0.75` | **4.63 톤/일**<br>27.36 톤/일 | **87.93%**<br>(`R² = 0.9579`) | **329 원/kg**<br>**528 원/kg** | **89.92%**<br>(`R² = 0.9311`) |

> **참고**: 가격 정확도(%)는 `100 - MAPE(%)`, 거래량 정확도(%)는 물량 규모 가중 정확도 `100 - WMAPE(%)` 기준입니다. 주력 규격인 `조생귤 5kg상자` 단일 규격 기준 가격 정확도는 **TFT 95.74%, XGBoost 96.34%, LightGBM 97.02%, Random Forest 96.18%**를 기록합니다.

---

## 🖥️ Streamlit 대시보드 (`app.py`) 주요 기능

- **품종 · 거래단위 및 기준 날짜(`2021-01-02 ~ 2026-12-31`) 선택**:
  - 기준 날짜를 선택하면, **선택한 날짜가 포함된 주간의 월요일부터 토요일까지 6일간(도매시장 주간 개장일)**의 실제 및 예측 결과를 자동으로 구성하여 출력합니다.
  - 일요일(전국 공영도매시장 정기 휴장일)을 예측 표시에서 제외하고 실제 경매가 이루어지는 월~토요일을 온전히 표시하며, 선택한 날짜는 `★ 선택일`로 강조됩니다.
- **계절적 비수기(`Off-Season`) 자동 감지 및 예측 Graph 숨김 처리**:
  - 선택한 기준 날짜가 해당 감귤 품종의 계절적 비수기(출하·경매 물량이 미미하여 AI 예측 모델 대상에서 제외된 기간)에 해당하는 경우, **예측 Graph(메인 비교 그래프 및 4개 모델별 상세 그래프) 출력을 자동으로 숨김(비활성화) 처리**합니다.
  - 대신 상단에 **대형 비수기 안내 배너(Off-Season Notice)**를 눈에 띄게 표시하여, 해당 품종의 **정상 출하기 시작~종료 시점**, 비수기 제외 사유, 출하기 날짜 선택 가이드를 상세히 제공합니다.
  - 비수기에도 모델별 성능 평가 지표(Accuracy)와 해당 주간(월~토)의 일정 및 비수기 상태표는 접이식(`Expander`) 메뉴로 깔끔하게 확인할 수 있습니다.
- **1번 메인 그래프 및 상단 요약: 최고 가격 정확도 모델(Random Forest) 중심 직관적 출력**:
  - 상단 KPI 요약 카드와 **1번 메인 비교 그래프**에서는 4개 모델 중 가격 예측 정확도가 가장 높은 **Random Forest(종합 정확도 90.81%)**를 단일 대표 모델로 선정하여, 실제 경매치 및 예년(2021~2025년) 동시점 평균과 함께 해당 주간(월~토) 가격 및 거래량을 깔끔하고 명료하게 비교 출력합니다.
- **2, 3, 4번 섹션: 4개 모델(TFT · XGBoost · LightGBM · Random Forest) 전체 비교 분석 유지**:
  - **2번 개별 모델 상세 그래프(2x2)**, **3번 모델별 정확도(Accuracy · R² · MAE · MAPE) 비교 분석**, **4번 주간(월~토) 상세 비교표**에서는 4개 모델 전체의 예측치와 성능 지표를 기존과 동일하게 다각도로 비교 분석할 수 있습니다.
- **모델별 가격 · 거래량 예측 정확도 실시간 산출 및 시각화**:
  - ① **선택한 `[품종 × 거래단위]` 규격 정확도**, ② **해당 주간(월~토) 구간 정확도**, ③ **전체 16개 규격 종합 정확도**(`Accuracy %`, `R²`, `MAE`, `MAPE`)를 막대 그래프와 비교표로 함께 출력합니다.
- **kg당 단가(원/kg) 표준 고정 및 CSV 다운로드**:
  - 가격 표시 단위를 **`kg당 단가(원/kg)`**로 고정하여 서로 다른 거래단위 간에도 객관적인 가격 수준 비교를 지원하며, 조회된 해당 주간(월~토) 모델별 예측 비교표를 CSV로 다운로드할 수 있습니다.

---

## 🚀 실행 및 Docker 배포 방법

### 1. 로컬 환경에서 Streamlit 대시보드 실행
```powershell
# 가상환경 기준 Streamlit 앱 실행
.\.venv\Scripts\streamlit.exe run app.py
```

### 2. Docker를 이용한 경량 컨테이너 배포 (모델 학습 제외, Streamlit 출력 전용)
모델 학습용 노트북(`*.ipynb`), 대용량 원본 CSV(`data/*.csv`), 무거운 학습 패키지(`PyTorch`, `CUDA` 등)를 제외하고 사전 계산된 예측 결과(`data/streamlit_predictions.pkl`)와 `app.py`만 포함하여 가볍게 배포합니다.

#### Docker Compose로 실행 (권장)
```powershell
# 이미지 빌드 및 백그라운드 컨테이너 실행
docker compose up -d --build

# 컨테이너 중지
docker compose down
```

#### Docker CLI로 직접 빌드 및 실행
```powershell
# 1) Docker 이미지 빌드
docker build -t citrus-forecast-streamlit:latest .

# 2) 컨테이너 실행 (호스트 8501 포트 연결)
docker run -d --name citrus-forecast-app -p 8501:8501 citrus-forecast-streamlit:latest
```

실행 후 웹 브라우저에서 **`http://localhost:8501`**로 접속하면 대시보드를 바로 사용할 수 있습니다.
