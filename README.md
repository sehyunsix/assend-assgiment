# Binance Unified High-Fidelity Trading Framework

[![Binance](https://img.shields.io/badge/Binance-Futures-F0B90B?style=flat-square&logo=binance)](https://www.binance.com)
[![Python 3.14](https://img.shields.io/badge/Python-3.14-3776AB?style=flat-square&logo=python)](https://www.python.org)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?style=flat-square&logo=docker)](https://www.docker.com)

본 프로젝트는 단순한 데이터 분석을 넘어, 리서치(Research)와 실서비스(Production) 간의 괴리를 극복하기 위한 시스템적 솔루션을 제안합니다. 동일한 바이너리 로직이 과거 데이터 백테스팅과 실시간 웹소켓 환경에서 오차 없이 작동하도록 설계된 Unified Single-Engine Framework입니다.

---

## 1. Mathematical Foundation: Defining Market Stability

본 시스템은 고빈도 트레이딩 환경에서의 시장 안정성을 수학적으로 정의하고, 이를 실시간 스트림 처리를 통해 정량화합니다.

### 1.1 가격 및 비용 지표 (Price & Cost)
- **Mid-Price ($P_m$):**
  $$P_m = \frac{P_{ask} + P_{bid}}{2}$$
- **Spread in Bps ($S_{bps}$):**
  $$S_{bps} = \frac{P_{ask} - P_{bid}}{P_m} \times 10,000$$

### 1.2 유동성 지표 (Liquidity & Inertia)
특정 가격 범위($\Delta$) 내의 총 잔량을 통해 시장의 충격 흡수 능력을 측정합니다.
- **$\Delta$ bps Depth ($D_{\Delta}$):**
  $$D_{\Delta} = \sum_{i} Q_{bid,i} \text{ for } P_{bid,i} \ge P_m(1-\Delta) + \sum_{j} Q_{ask,j} \text{ for } P_{ask,j} \le P_m(1+\Delta)$$
  (기본값: $\Delta=50$bps)

### 1.3 불균형 및 방향성 (Imbalance & Direction)
- **Order Imbalance ($\alpha$):**
  $$\alpha = \frac{\sum Q_{bid} - \sum Q_{ask}}{\sum Q_{bid} + \sum Q_{ask}}$$
  ($\alpha \in [-1, 1]$, 0에 가까울수록 균형 상태)

### 1.4 복원력 정의 (Resilience & Recovery)
충격 발생 후 시스템이 정상 상태로 판단하는 회귀 조건을 정의합니다.
- **Recovery Point ($t_{rec}$):**
  $$S_{bps}(t_{rec}) \le \bar{S}_{bps, baseline} \times \tau$$
  ($\tau$: Recovery Threshold, 기본값 1.5)

---

## 2. Problem Statement: Why This System?

위의 수학적 지표들을 실서비스 환경에서 오차 없이 산출하기 위해 해결해야 하는 세 가지 핵심 시스템적 난제입니다:

1.  **Research-Production Gap**: 연구 환경(Historical)과 실서비스(Real-time)의 코드 베이스가 분리되어 발생하는 로직의 불일치(Deviation).
2.  **Cross-Stream Time Skew**: Trade, Orderbook, Liquidation 스트림이 비동기적으로 유입될 때 발생하는 지표 산출의 왜곡.
3.  **Data Integrity in Chaos**: 변동성이 극심한 구간에서 발생하는 Dirty Data가 의사결정 엔진에 미치는 치명적 오염.

---

## 3. Research and Design Insights

### 3.1 가장 위험한 불확실성 (Critical Uncertainty)
가장 큰 위험은 **데이터 스트림 간의 시간축 불일치(Time Skew)**였습니다. 물리적으로 다른 시점에 도착하는 데이터들을 이벤트 시간 기준으로 정렬하지 못하면, 정의된 수학적 지표($P_m, S_{bps}$ 등)는 실제 시장 상황과 동떨어진 값을 가지게 됩니다.

### 3.2 Dirty Data 판단 조건 (Detection Logic)
`DirtyDataDetector`는 다음 세 가지를 핵심 오염 데이터로 정의합니다:
1. **Out-of-order Data**: 과거 데이터가 현재의 Watermark보다 늦게 도착하는 경우.
2. **Crossed Market**: 매수 호가가 매도 호가보다 높아지는 비정상 상황.
3. **Z-score 기반 Price Spike**: 통계적 범위를 벗어난 비현실적인 가격 변동.

### 3.3 가설 반영 및 의사결정 (Hypothesis Integration)
위의 조건들은 `DataTrustLevel`에 영향을 미치며, 엔진은 데이터 신뢰도가 확보되지 않으면 모든 액션을 즉시 차단(`HALTED`)합니다.

### 3.4 판단 중단 설계 (Stopping Criteria)
1. **데이터 품질 하락**: `DirtyDataDetector`가 신뢰 상실 상태를 감지할 때.
2. **복원력 한계 초과**: 대규모 청산 클러스터가 지속적으로 발생하여 `liquidation_cascade_threshold`를 초과할 때.

---

## 4. Core Systemic Solutions

### 4.1 Unified Execution Engine (Single Binary Path)
하부의 `DecisionEngine`이 데이터 소스(CSV/WebSocket)를 인지하지 못하게 추상화하여, **100% 동일한 수학적 연산 결과**를 보장합니다.

### 4.2 Time Alignment Policy (Watermark Mechanism)
이벤트 시간 기반의 **Watermark** 로직을 도입하여, 서로 다른 스트림 간의 정렬을 강제하고 산출된 지표의 신뢰성을 확보합니다.

### 4.3 Layered Data Trust Model
`DirtyDataDetector`가 감지한 이상 신호를 기반으로 `DecisionEngine`의 상태를 즉각적으로 전이시켜 리스크를 관리합니다.

---

## 5. Hypothesis Validation through System Observation

### 5.1 Aggregated Impact Analysis (통합 시각화)
![Aggregated Impact](output/phase1/liquidation_impact_aggregated.png)
- 수백 개의 청산 사건을 Overlay하여 평균적인 **시스템적 안정성 임계(Stability Threshold)**를 도출합니다.
- 결과: 청산 발생 후 평균 30초 내에 오더북 지표가 Baseline($t_{rec}$)으로 회귀하는 복원력을 확인했습니다.

---

## 6. Technical Stack

- **Concurrency**: `AsyncIO` 기반 고성능 비동기 스트림 처리
- **Data Pipeline**: `Dask`를 활용한 대용량 데이터의 병렬 Sanitization
- **Analytics**: Pandas/NumPy 기반 실시간 수학적 지표 산출
- **Reporting**: JSONL 기반의 상태 전이 이력 추적 및 Matplotlib 시각화 리포트

---

## 7. Output Data Schema

### 7.1 Decisions (`decisions.jsonl`)
엔진이 매 스냅샷마다 내린 개별 결정을 기록합니다. (`ts`, `action`, `reason`, `duration_ms`)

### 7.2 State Transitions (`state_transitions.jsonl`)
시스템의 주요 상태 변화 시점에만 기록되는 고수준 로그입니다. (`ts`, `data_trust`, `hypothesis`, `decision`, `trigger`)

---

## 8. How to Run

### Development Environment
```bash
docker build -t ascend-trading .
```

### Modes
- **Historical Analysis**: `/data` 마운트 후 과거 데이터 기반 전략 가설 검증
- **Real-time Engine**: 실시간 웹소켓 기반의 의사결정 상태 머신 실행

---
**© 2025 Ascend Portfolio Assignment - Trading System Specialist Development**
