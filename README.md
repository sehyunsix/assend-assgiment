# Binance Unified High-Fidelity Trading Framework

[![Binance](https://img.shields.io/badge/Binance-Futures-F0B90B?style=flat-square&logo=binance)](https://www.binance.com)
[![Python 3.14](https://img.shields.io/badge/Python-3.14-3776AB?style=flat-square&logo=python)](https://www.python.org)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?style=flat-square&logo=docker)](https://www.docker.com)

본 프로젝트는 단순한 데이터 분석을 넘어, 리서치(Research)와 실서비스(Production) 간의 괴리를 극복하기 위한 시스템적 솔루션을 제안합니다. 동일한 바이너리 로직이 과거 데이터 백테스팅과 실시간 웹소켓 환경에서 오차 없이 작동하도록 설계된 Unified Single-Engine Framework입니다.

---

## 1. Problem Statement: Why This System?

트레이딩 시스템 구축 시 직면하는 세 가지 핵심 시스템적 난제를 해결하는 데 집중했습니다:

1.  **Research-Production Gap**: 연구 환경(Historical)과 실서비스(Real-time)의 코드 베이스가 분리되어 발생하는 로직의 불일치(Deviation).
2.  **Cross-Stream Time Skew**: 거래(Trade), 호가(Orderbook), 청산(Liquidation) 등 여러 스트림이 비동기적으로 유입될 때 발생하는 시장 상태의 "유령 현상".
3.  **Data Integrity in Chaos**: 변동성이 극심한 구간에서 발생하는 Dirty Data(지연, 누락, 중복)가 의사결정 엔진에 미치는 치명적 오염.

---

## 2. Core Systemic Solutions

### 2.1 Unified Execution Engine (Single Binary Path)
- **개념**: `EngineRunner`는 Historical 모드와 Real-time 모드를 추상화하여, 하부의 `DecisionEngine`이 데이터 소스의 성격(CSV vs WebSocket)을 인지하지 못하게 설계되었습니다.
- **가치**: 연구소에서 검증된 가설이 실서비스에 배포될 때 **100% 동일한 동작**을 보장합니다.

### 2.2 Time Alignment Policy (Watermark Mechanism)
- **해결책**: 이벤트 시간 기반의 **Watermark** 로직을 도입하여, 서로 다른 스트림 간의 정렬을 강제합니다.
- **동작**: 특정 스트림이 늦게 도착하더라도 `Allowed Lateness` 범위 내에서 데이터를 기다리거나, 범위를 벗어날 경우 데이터를 정제(Sanitize)하여 엔진의 판단 오차를 최소화합니다.

### 2.3 Layered Data Trust Model
- **동작 방식**: 
    1. `DirtyDataDetector`가 유입되는 모든 패킷을 실시간 감시.
    2. 이상 신호 발견 시 즉시 해당 구간을 `UNTRUSTED`로 마킹.
    3. `DecisionEngine`은 데이터 신뢰도가 확보되지 않으면 로직의 실행을 즉시 중단(`HALTED`).

---

## 3. Hypothesis Validation through System Observation

본 시스템은 대규모 청산 충격이 클러스터링될 때 시장의 Self-Healing(자폭/복구) 능력을 정량적으로 측정합니다.

### 3.1 Aggregated Impact Analysis (통합 시각화)
![Aggregated Impact](output/phase1/liquidation_impact_aggregated.png)
- **Systemic Goal**: 특정 청산 사건 하나를 분석하는 것이 아니라, 수백 개의 사건을 겹쳐서(Overlay) 평균적인 시장의 안정성 임계(Stability Threshold)를 시스템적으로 도출합니다.
- **결과**: 청산 발생 후 평균 30초 내에 오더북 지표가 Baseline으로 회귀하는 시스템적 복원력을 증명했습니다.

---

## 4. How to Run

### Development Environment
```bash
docker build -t ascend-trading .
```

### Modes
- **Historical Analysis**: `/data` 마운트 후 과거 데이터 기반 전략 가설 검증
- **Real-time Engine**: 실시간 웹소켓 기반의 의사결정 상태 머신 실행

---

## 5. Technical Stack

- **Concurrency**: `AsyncIO` 기반 고성능 비동기 스트림 처리
- **Data Pipeline**: `Dask`를 활용한 대용량 Historical 데이터의 병렬 Sanitization
- **Analytics**: Pandas/NumPy 기반 실시간 오더북 지표(Spread, Depth, Imbalance) 산출
- **Reporting**: JSONL 기반의 상태 전이 이력 추적 및 Matplotlib 시각화 리포트
- **Stability**: `Single Decision State-Machine` 구조를 통한 상태 전이 안정화

---

## 6. Metric Definitions: Mathematical Formulation

시스템에서 측정되는 시장 안정성 지표의 공식 수식입니다:

### 6.1 Mid-Price 및 Spread (Cost)
가중치 없는 중간 가격과 호가 간격을 측정합니다.
- **Mid Price ($P_m$):**
  $$P_m = \frac{P_{ask} + P_{bid}}{2}$$
- **Spread in Bps ($S_{bps}$):**
  $$S_{bps} = \frac{P_{ask} - P_{bid}}{P_m} \times 10,000$$

### 6.2 Market Depth (Inertia)
특정 범위 내의 유동성을 측정합니다.
- **$\Delta$ bps Depth ($D_{\Delta}$):**
  $$D_{\Delta} = \sum_{i} Q_{bid,i} \text{ for } P_{bid,i} \ge P_m(1-\Delta) + \sum_{j} Q_{ask,j} \text{ for } P_{ask,j} \le P_m(1+\Delta)$$
  (본 시스템에서는 $\Delta=50$bps를 기본 유동성 지표로 사용)

### 6.3 Order Imbalance (Direction)
매수/매도 압력의 비대칭성을 측정합니다.
- **Order Imbalance ($\alpha$):**
  $$\alpha = \frac{\sum Q_{bid} - \sum Q_{ask}}{\sum Q_{bid} + \sum Q_{ask}}$$
  ($\alpha \in [-1, 1]$, 0에 가까울수록 균형 상태)

### 6.4 Recovery Condition (Resilience)
충격 발생 후 정상 상태로의 회귀 조건을 정의합니다.
- **Recovery Point ($t_{rec}$):**
  $$S_{bps}(t_{rec}) \le \bar{S}_{bps, baseline} \times \tau$$
  ($\tau$: Recovery Threshold, 기본값 1.5)

---
**© 2025 Ascend Portfolio Assignment - Trading System Specialist Development**
