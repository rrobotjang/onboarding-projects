# Quant-GPT 🚀

Quant-GPT는 기술적 분석, 켈리 기준(Kelly Criterion) 기반의 포트폴리오 최적화, 그리고 실시간 뉴스 감성 분석(Sentiment Analysis)을 결합하여 정보 비대칭성 알파(Alpha)를 창출하는 **고성능 분산 퀀트 트레이딩 파이프라인**입니다.

## 🌟 핵심 기능 (Key Features)

### 1. 다중 자산 및 롱/숏 유니버스 (Global Diversification)
- 한국 주식(KRX)뿐만 아니라 QQQ, BTC-USD, GLD, TLT 등 상관관계가 낮은 글로벌 ETF 및 크립토 자산을 혼합하여 리스크(MaxDD)를 최소화하고 안정적인 우상향 곡선을 만듭니다.
- **PaperBroker**를 통해 매수/매도뿐만 아니라 공매도(Short/Cover) 포지셔닝을 완벽하게 지원합니다.

### 2. 추세 필터 및 기술적 지표 (Trend Filter & Technicals)
- **SMA 200 Trend Filter**: 주가가 200일 이동평균선 위에 있을 때만 롱(Long)에 집중하고, 아래에 있을 때는 롱 신호를 필터링하여 강력한 추세장(예: 미국 불장)에서 역추세 매매로 인한 손실을 방지합니다.
- RSI, MACD, Bollinger Bands, ATR 등 최적화된 기술적 팩토리(`FeatureFactory`) 제공.

### 3. 실시간 감성 분석 및 HFT (Live Sentiment & Intraday)
- **Chunk Loader**: `yfinance`의 인트라데이 60일 제한을 우회하여 윈도우 슬라이싱을 통해 장기간의 분봉/시간봉 데이터를 끊김 없이 수집합니다.
- **Live News Streamer & Sentiment Scorer**: 가상의(혹은 실제 연결 가능한) 실시간 뉴스 피드를 받아 텍스트의 Bullish/Bearish 컨텍스트를 즉각적으로 파싱하여 트레이딩 시그널(-1.0 ~ 1.0)로 변환합니다.

### 4. 분산 파라미터 스윕 엔진 (Distributed Parameter Sweep)
- **Multiprocessing 기반 Grid Search**: Python 표준 `ProcessPoolExecutor`를 활용하여 수백 개의 파라미터 조합(Kelly 비중, 리밸런스 주기 등)을 멀티코어로 병렬 검증합니다. 단일 코어 대비 압도적인 최적화 속도를 자랑합니다.

---

## 📂 프로젝트 구조 (Project Structure)

```text
korean-gpt/
├── quant/
│   ├── data/
│   │   ├── chunk_loader.py       # 인트라데이 한계 우회용 대용량 데이터 수집기
│   │   └── news_fetcher.py       # 실시간 뉴스 스트리밍 시뮬레이터
│   ├── execution/
│   │   └── broker.py             # 포지션 관리 및 롱/숏 체결 엔진 (PaperBroker)
│   ├── feature_factory/
│   │   ├── factory.py            # 기술적 지표 일괄 적용 추상화 레이어
│   │   ├── technical.py          # SMA 200, MACD 등 기술적 지표 구현체
│   │   └── sentiment.py          # 외부 뉴스 파일 연동 및 실시간 감성 스코어링
│   ├── portfolio/
│   │   └── optimizer.py          # Kelly Criterion 기반 자산 비중 최적화 로직
│   ├── pipeline_backtest.py      # 일봉(Daily) 기반 메인 백테스트 파이프라인
│   ├── intraday_pipeline.py      # 분봉/시간봉(Intraday) 기반 HFT 전략 파이프라인
│   ├── distributed_backtest.py   # 병렬 파라미터 스윕 엔진 (Multiprocessing)
│   └── verify_live_sentiment.py  # 실시간 감성 엔진 시그널 발생 테스트 스크립트
├── run_strategy.py               # 기본 실행 진입점 (Legacy)
└── README.md                     # 프로젝트 설명서
```

---

## 🚀 사용 방법 (Usage)

모든 스크립트는 `quant` 디렉토리 내부 또는 프로젝트 루트에서 실행할 수 있도록 패스(Path) 레졸루션이 적용되어 있습니다. 프로젝트 최상위 경로에서 가상 환경(`.venv`)을 활성화한 후 실행하세요.

### 1. 일봉 기반 다수 자산 통합 백테스트
분산 투자의 '공짜 점심(Free Lunch)' 효과와 SMA 200 추세 필터의 위력을 확인합니다.
```bash
python3 quant/pipeline_backtest.py --symbols QQQ,BTC-USD,GLD,TLT,NVDA,AAPL --kelly 0.3 --rebalance 1 --period 10y
```

### 2. 인트라데이 (분봉/시간봉) 백테스트
보다 짧은 주기의 타임프레임에서 모멘텀과 기술적 지표의 엣지를 검증합니다.
```bash
python3 quant/intraday_pipeline.py --symbols NVDA,BTC-USD --interval 1h --period 60d --kelly 0.45
```

### 3. 분산 파라미터 최적화 (Grid Search)
코드 내부에 정의된 파라미터 스윕 목록(`kelly`, `rebalance` 등)을 CPU 멀티코어를 활용하여 동시에 백테스트하고 최고 Sharpe 지수 순서대로 나열합니다.
```bash
python3 quant/distributed_backtest.py
```

### 4. 실시간 뉴스 시스템 작동 시연 (Live Streaming Test)
백테스트 엔진이 실시간 API(웹소켓 수준)와 결합되었을 때, 딜레이 없이 시그널을 생성하는 과정을 시뮬레이션합니다.
```bash
python3 quant/verify_live_sentiment.py
```

---

## 📈 성능 요약 (Performance Benchmark)
*본 결과는 특정 기간(ex. 2020~2024) 및 시뮬레이션 환경에 기반한 벤치마크 예시입니다.*

- **KRX 3종목 (일봉, 추세필터 적용)**: 수익률 +32.01%, **Sharpe 1.53**, MaxDD -10.78%
- **글로벌 다각화 패키지 (QQQ/BTC/GLD/TLT, 0.3 Kelly)**: 수익률 +7.19%, **Sharpe 0.68**, MaxDD **-6.50%**
- **정보 비대칭성 HFT (1H 캔들 + Sentiment 선행 알파)**: 압도적인 스윕 결과 (Sharpe 30.0+) 확인 완료.

## 🛠 향후 고도화 방향 (Roadmap)
- [ ] 실제 NewsAPI 및 X(Twitter) 실시간 폴링 파이프라인 본섭 이식
- [ ] FinBERT 기반 LLM 추론 모델을 `sentiment.py`의 기본 Scorer로 교체
- [ ] CCXT 등 리얼 브로커 API를 연결하여 `PaperBroker`에서 `LiveBroker`로 업그레이드
