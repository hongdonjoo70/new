# 가벼운 Python 3.12 Slim 이미지 사용 (모델 학습 패키지 제외, 시각화 전용)
FROM python:3.12-slim

# 컨테이너 내 환경변수 설정
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Linux 컨테이너 내 Matplotlib 한글 깨짐 방지를 위한 나눔고딕(fonts-nanum) 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-nanum \
    fontconfig \
    && fc-cache -fv \
    && rm -rf /var/lib/apt/lists/*

# Streamlit 출력에 필요한 경량 패키지 설치 (PyTorch/XGBoost 등 무거운 학습 라이브러리 제외)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 및 사전 계산된 예측 결과 데이터(11MB)만 복사
COPY app.py .
COPY data/streamlit_predictions.pkl ./data/streamlit_predictions.pkl

# Streamlit 기본 포트 노출
EXPOSE 8501

# 헬스체크 설정
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Streamlit 실행
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
