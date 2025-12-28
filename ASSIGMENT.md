---

## **0. 서문**

이 과제는 코딩 테스트가 아니며 정해진 답이 없습니다. 이는 다음과 같은 실제 트레이딩 환경의 특성을 반영합니다.

- 정답이 없습니다.
- 실제로 관측되는 데이터는 불완전하며, 지연되거나 왜곡된 형태로 주어집니다.
- 대부분의 의사결정은 확률적인 추정치에 기반합니다.
- 누구에게도 확실한 의견을 듣기 어렵고 ‘왜?’에 대한 설명은 대부분 사후적으로 덧붙여집니다.

따라서 본 과제는

> 실제 시장 환경에서 어떻게 문제를 정의했는가, 그리고 그 문제를 어떤 방식으로 해결했는가
>

를 중점적으로 평가합니다. 결과 자체보다는 해당 결과를 도출해내는 과정이 중요하며, 과제 설명 및 데이터의 모호함도 의도된 조건입니다. 시장은 항상 변화하고 alpha decay 속도도 계속해서 빨라지고 있습니다. 그래서 기존 데이터로 학습할 수 없는 새롭고 불확실한 시장에서도 유능함을 발휘할 수 있는 인재를 찾기 위해 이러한 과제를 출제하게 되었습니다.

---

## **1. 과제 목표**

다음 질문에 대해 하나의 분석 **시스템을 설계한다는 관점에서 답하십시오**

> "과거의 불완전한 데이터로 검증된 의사결정 구조는 실시간 웹소켓 환경에서도 유효한 판단 근거로 작동할 수 있는가?"
>

위 질문에 대한 답을 도출하기 위해 지원자는 **단 하나의 의사결정 엔진(Single Decision Engine)** 을 설계해야 하며, 이 엔진은 아래 두 환경에서 코드 및 로직 **수정 없이** 동일하게 동작해야 합니다.

이렇게 설계한 Single Decision Engine을 이용해 아래 리서치를 수행하십시오.

| Phase | 환경 | 데이터 소스 |
| --- | --- | --- |
| Phase 1 | Historical Validation | CSV (불완전한 과거 데이터) |
| Phase 2 | Realtime Validation | Binance Futures WebSocket |

> 대규모 청산(Liquidation) 이후, Orderbook 상태는 판단을 허용할 만큼 안정적인가, 아니면 판단을 중단해야 하는가?
>
- **리서치 산출물**
    - **(A) 판단 허용 조건**
    - **(B) 판단 중단 조건**
    - **(C) 붕괴 신호 (Dirty Data와 연관된)**

---

## **2. 핵심 구조**

```
관측 Data / 패턴
        ↓
데이터 신뢰 조건 정의
        ↓
Research 가설 유효성 평가
        ↓
시스템 상태 전이 및 판단 허용 / 제한 / 중단
```

위 흐름은 사고 구조에 꼭 반영되어야 하나 코드 실행 순서로 그대로 반영할 필요는 없습니다.

---

## **3. 입력 데이터 개요 (4 Streams)**

시스템은 아래 4개 스트림을 동시에 처리합니다.

1. **Trades** — 체결 이벤트
2. **Orderbook** — 호가창 상태
3. **Liquidations** — 청산 이벤트
4. **Ticker** — 시장 요약 정보

> 각 스트림의 timestamp가 같더라도 실제로는 완전히 동일한 시간을 의미하지 않을 수 있습니다. 이를 동일한 시간축으로 가정한 설계는 실패로 간주될 수 있습니다.
>

---

## **4. Phase 1 — Historical Validation**

### **4.1 목적**

아래 시스템 행동을 검증:

- 어떤 조건에서 판단을 중단했는가
- 어떤 조건에서 상태가 전이되었는가
- 데이터 품질 이상(Dirty Data) 신호가 시스템의 판단 또는 행동을 변경했는가

### **4.2 Dirty Data**

제공 데이터에는 의도적으로 다음 문제가 포함됩니다.

- out-of-order timestamp
- 중복 이벤트
- fat-finger 가격
- crossed market
- 이벤트 지연 및 누락

### **4.3 데이터 구조**

https://drive.google.com/drive/folders/1c7fG-kAgimlgQ6jWJwMe28wWxrgdQ0wk?usp=sharing (해당 데이터를 사용해 주세요)

```
challenge_data/
├── research/           # Clean 데이터 (연구 및 가설 수립용)
├── validation/         # Dirty 데이터 (검증 시스템 테스트용)
```

---

## **5. Phase 2 — Realtime Validation**

### **5.1 환경**

- 대상: **Binance Futures BTC/USDT Perpetual**
- 입력: WebSocket
- 실제 주문은 하지 않으며, 판단 결과는 시뮬레이션 또는 로그로 평가함

### **5.2 요구사항**

다음 상황에서도 시스템은 종료되지 않아야 합니다.

- 네트워크 단절 및 재연결
- 중복 메시지
- out-of-order 도착
- burst
- 특정 스트림 장시간 정지

### **5.3 제출 요구사항**

지원자는 **직접 Realtime 환경에서 시스템을 실행**하고, 그 결과를 제출해야 합니다. Phase 1에서 발견한 Dirty Data 패턴과 Research에서 도출한 판단 조건 및 제한 로직 등이 Realtime 시스템에 **실제로 반영되어 동작**해야 합니다.

---

## **6. 아키텍처 요구사항**

### **6.1 Single Decision Engine**

- Historical / Realtime 두 환경에서 동일한 판단 기준이 유지되어야 합니다.

### **6.2 Time Alignment Policy**

- event-time / processing-time 정의
- allowed lateness
- buffer / window / watermark

등 파라미터 변경 시 **출력 또는 상태 전이가 실제로 달라져야 합니다.**

### **6.3 Sanitization Policy**

모든 이벤트는 다음으로 분류되어야 합니다.

| 분류 | 의미 |
| --- | --- |
| ACCEPT | 정상 데이터 |
| REPAIR | 수정 가능한 데이터 |
| QUARANTINE | 신뢰 불가 데이터 |

이 중 QUARANTINE 발생은 반드시 **Trust State 전이 및 판단 제한**으로 연결되어야 합니다.

---

## **7. System Behavior Contract**

이 과제는 **구현 방법을 제한하지 않습니다. 아래 내용은 이해를 돕기 위한 예시입니다.**

### **7.1 State (예시)**

| 상태 | 값 |
| --- | --- |
| Data Trust State | TRUSTED / DEGRADED / UNTRUSTED |
| Hypothesis Validity State | VALID / WEAKENING / INVALID |
| Decision Permission State | ALLOWED / RESTRICTED / HALTED |

**Decision Permission**은 위 두 상태의 **조합 결과**여야 합니다.

### **7.2 동작 시나리오 (예시)**

**시나리오 A — 판단 허용**

```
Data Trust: TRUSTED
Hypothesis: VALID
Decision:   ALLOWED
```

**시나리오 B — 판단 제한**

```
Data Trust: DEGRADED
Hypothesis: WEAKENING
Decision:   RESTRICTED
```

**시나리오 C — 판단 중단 (No-Decision Mode)**

```
Data Trust: UNTRUSTED
Hypothesis: INVALID
Decision:   HALTED (**데이터 수집 및 상태 갱신은 지속)**
```

---

## **8. 로그 및 추적성 요구**

시스템은 README 설명이 아닌 실행 로그 / 상태 기록을 통해 다음 내용을 확인할 수 있어야 합니다.

- 왜 판단을 중단했는가?
- 어떤 Dirty Data 신호가 영향을 주었는가?
- 어떤 가설 조건이 붕괴되었는가?
- 그 결과 어떤 상태 전이가 발생했는가?

---

## **9. 제출 형식**

### **9.1 Docker 실행**

```bash
# Phase 1: Historical Validation
docker run -v /path/to/data:/data your-image historical

# Phase 2: Realtime Validation
docker run your-image realtime
```

### **9.2 입력**

| Phase | 입력 방식 |
| --- | --- |
| Historical | `/data/` 경로에 CSV 파일 마운트 |
| Realtime | 컨테이너 내부에서 WebSocket 연결 |

### **9.3 출력**

모든 출력은 `/output/` 디렉토리에 생성

```
/output/
├── state_transitions.jsonl   # 상태 전이 로그
├── decisions.jsonl           # 판단 허용/제한/중단 기록
└── summary.json              # 실행 요약
```

### **9.4 출력 형식**

**state_transitions.jsonl**

```json
{"ts": 1760054400000000, "data_trust": "DEGRADED", "hypothesis": "WEAKENING", "decision": "RESTRICTED", "trigger": "..."}
```

**decisions.jsonl**

```json
{"ts": 1760054400000000, "action": "HALT", "reason": "...", "duration_ms": 5000}
```

### **9.5 필수 제출물**

1. **소스코드 전체**
2. Dockerfile
3. `/output/` 결과물 (Historical + Realtime 각각)
4. README.md
    - README 파일에는 다음 질문에 대한 답을 간단히 서술해주세요.
        - 가장 위험한 불확실성은 무엇이었는가?
        - Dirty Data로부터 어떤 판단 조건을 정의했는가?
        - 그 조건이 가설(의사결정 구조)에 어떻게 반영되는가?
        - 가설 변화가 시스템 동작을 어떻게 바꾸는가?
        - 언제 판단을 중단하도록 설계했는가?
        - 지금 다시 설계한다면, 가장 먼저 제거하거나 단순화할 요소는 무엇인가?

### **9.6 제출 방법**

**GitHub Repository**로 제출

1. Repository 생성 (Public)
2. Repository URL을 [**general@ascendllc.one](mailto:general@ascendllc.one)** 이메일로 전송

```
repository/
├── src/                    # 소스코드
├── Dockerfile
├── README.md
└── output/
    ├── historical/         # Phase 1 결과
    │   ├── state_transitions.jsonl
    │   ├── decisions.jsonl
    │   └── summary.json
    └── realtime/           # Phase 2 결과
        ├── state_transitions.jsonl
        ├── decisions.jsonl
        └── summary.json
```

---

## **10. 기한 및 평가 방식**

- 기한: 2025년 12월 31일(수) 23:59까지
- 평가방식
    - 제출된 Docker 이미지를 **직접 Realtime 환경에서 실행**
    - 소스코드 리뷰를 통해 **Historical/Realtime 간 로직 분기 여부**를 검증
    - 제출된 결과와 평가자 실행 결과의 **일관성**을 확인

---

**Good luck** 🍀

[과제 정리](https://www.notion.so/2d40b456633d808e9f50e450b17c902a?pvs=21)