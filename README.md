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
- **Analytics**: `Pandas/NumPy`를 이용한 마이크로초 단위 오더북 지표 산출
- **Stability**: `Single Decision State-Machine` 구조를 통한 상태 전이 안정화

---
**© 2025 Ascend Portfolio Assignment - Trading System Specialist Development**
