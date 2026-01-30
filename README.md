# Edge AI 가속기: 이기종 가속기 환경에서의 전력 및 SLO 인식 추론 스케줄링

이기종 AI 가속기를 갖춘 엣지 컴퓨팅 환경에서 실시간 추론 요청을 스케줄링하는 종합 시스템이다. 전력 효율성, SLO 준수, 디바이스 모빌리티 정보 반영에 중점을 두고 있다.

## 개요

이 프로젝트는 이기종 엣지 컴퓨팅 클러스터에 딥러닝 추론 작업을 분배하기 위한 동적 스케줄링 프레임워크를 구현한다. 다양한 가속기 유형(Google Coral TPU, NVIDIA Jetson, Hailo)에서 여러 CNN 모델(MobileNet, ResNet50, EfficientNet, VGG16)의 추론 요청을 처리하며 전력 소비와 데드라인 준수를 동시에 최적화한다.

## 클러스터 아키텍처

### 하드웨어 구성

```
마스터 노드 (Raspberry Pi 5)
├─ Kubernetes (k3s) control plane
├─ Prometheus 모니터링 시스템 (포트 30090)
├─ Grafana 대시보드 (포트 30300)
└─ 스케줄링 알고리즘 실행 환경

워커 노드 (k3s)
├─ Coral 노드 (Coral1, Coral2)
│  ├─ Raspberry Pi 5 기반
│  ├─ Google Coral TPU 가속기
│  └─ FastAPI 추론 서버 (포트 8080)
│
├─ Jetson 노드 (Jetson1, Jetson2)
│  ├─ NVIDIA Jetson Nano
│  ├─ NVIDIA GPU 가속기
│  └─ FastAPI 추론 서버 (포트 8080)
│
├─ Hailo 노드 (Hailo1, Hailo2)
│  ├─ Raspberry Pi 5 기반
│  ├─ Hailo-8L 가속기
│  └─ FastAPI 추론 서버 (포트 8080)
│
└─ 모니터링 에이전트
   ├─ node-exporter (하드웨어 메트릭, 포트 9100)
   └─ 전력 모니터링 (블루투스 기반 전력 측정기)

모니터링 및 데이터 수집 스택
├─ Prometheus (시계열 메트릭 저장소)
├─ node-exporter (CPU, 메모리, 네트워크 메트릭)
├─ 전력 모니터링 에이전트 (각 노드의 전력 소비)
└─ kube-state-metrics (Kubernetes 상태 메트릭)
```

### 네트워킹 구성

- **클러스터 내 통신**: Kubernetes 서비스 DNS 및 Pod IP 기반 통신
- **추론 요청**: 각 워커 노드의 FastAPI 서버로 REST API 호출
- **메트릭 수집**: Prometheus HTTP 쿼리 인터페이스 (포트 30090)
- **모니터링 대시보드**: Grafana Web UI (포트 30300)

## 프로젝트 파일 구조

```
Edge_AI_Accel/
├── 📋 코어 실험 코드
│   ├── main_pregenerated.py           # 메인 실험 오케스트레이터
│   ├── inference_request.py           # 워커 스레드 매니저
│   ├── task_generator.py              # 작업 및 디바이스 생성
│   ├── deadline_adapter.py            # 모빌리티 기반 데드라인 적응
│   ├── performance_profile.py         # 추론 성능 프로파일 저장소
│   ├── prometheus_collector.py        # Prometheus 메트릭 수집
│   ├── slo_monitor.py                 # SLO 준수 모니터링
│   ├── metric_collector.py            # 결과 로깅 및 집계
│   └── utils.py                       # 유틸리티 함수
│
├── 📁 scheduler/ (스케줄링 알고리즘 모듈)
│   ├── base_scheduler.py              # 추상 기본 클래스
│   ├── topsis_scheduler.py            # TOPSIS 다중 기준 스케줄러
│   └── power_aware_topsis_scheduler.py # 전력 인식 TOPSIS
│
├── 📁 mobility/ (모빌리티 시뮬레이션)
│   └── mobility_manager.py            # 디바이스 이동 관리
│
├── 🐳 Docker & 추론 서버 코드
│   ├── docker/
│   │   ├── coral_inference_server/
│   │   │   ├── Dockerfile
│   │   │   ├── coral_inference_server.py
│   │   │   └── requirements.txt
│   │   ├── jetson_inference_server/
│   │   │   ├── Dockerfile
│   │   │   ├── jetson_inference_server.py
│   │   │   └── requirements.txt
│   │   └── hailo_inference_server/
│   │       ├── Dockerfile
│   │       ├── hailo_inference_server.py
│   │       └── requirements.txt
│   │
│   └── yaml/ (Kubernetes 배포 설정)
│       ├── coral_daemonset.yaml
│       ├── jetson_daemonset.yaml
│       └── hailo_daemonset.yaml
│
├── 📊 모니터링 스택 설정
│   └── monitoring/
│       ├── prometheus-configmap.yaml
│       ├── prometheus-deployment.yaml
│       ├── grafana-deployment.yaml
│       └── node-exporter-daemonset.yaml
│
└── 📖 문서
    ├── README.md
    └── README_KO.md (한글)
```

## 주요 코드 파일 설명

### 실험 오케스트레이션

#### `main_pregenerated.py`
메인 실험 오케스트레이터로, 전체 시뮬레이션을 제어한다.

**주요 역할**:
- 모든 시스템 컴포넌트 초기화 (Prometheus, SLO 모니터링, 작업 생성)
- 마스터 노드와 워커 노드 생성 (가속기 설정)
- 포아송 분포를 따르는 사전 생성 작업을 스케줄러에 주입
- 시뮬레이션 중 디바이스 모빌리티 실시간 업데이트
- 최종 메트릭 수집 및 결과 출력

**실행 흐름**:
1. 성능 프로파일 및 작업 설정 로드
2. 지정된 시뮬레이션 기간 동안 작업 생성 (기본값: 600초)
3. 각 작업마다: 모빌리티 인식 데드라인 적응 → 스케줄링 → 워커에 전달
4. 실시간 SLO 준수 및 전력 소비 모니터링
5. CSV 및 JSON 형식의 실험 결과 생성

#### `inference_request.py`
각 엣지 노드의 추론 서버로 HTTP 요청을 보내는 워커 스레드 매니저다.

**주요 역할**:
- 노드별 워커 스레드 풀 관리 (비동기 요청 처리)
- 작업 큐 관리 (노드당 하나의 큐)
- 추론 요청 전송 및 응답 수집
- 결과 로깅 (`master_inference_results.csv`)

**⚠️ 필수 설정**:
실험 실행 전 `WORKERS` 딕셔너리의 Pod IP 주소를 업데이트해야 한다.
```python
WORKERS = {
    "coral1": "http://<coral1_pod_ip>:8080/infer",
    "coral2": "http://<coral2_pod_ip>:8080/infer",
    "jetson1": "http://<jetson1_pod_ip>:8080/infer",
    "jetson2": "http://<jetson2_pod_ip>:8080/infer",
    "hailo1": "http://<hailo1_pod_ip>:8080/infer",
    "hailo2": "http://<hailo2_pod_ip>:8080/infer",
}
```

### 작업 생성 및 관리

#### `task_generator.py`
IoT 디바이스와 추론 작업을 생성한다.

**주요 기능**:
- 커버리지 영역 내 100개 디바이스 생성
- 포아송 분포 기반 작업 도착 시뮬레이션
- 모델 선택: MobileNet, ResNet50, EfficientNet, VGG16
- SLO 클래스 분류: Hard(15%), Normal(35%), Soft(50%)

### 스케줄링 알고리즘

#### `scheduler/base_scheduler.py`
모든 스케줄러가 상속하는 추상 기본 클래스다.

**정의 내용**:
- `schedule()` 메서드 인터페이스
- Node, Task 데이터 구조
- 성능 및 전력 속성

#### `scheduler/topsis_scheduler.py`
TOPSIS(순서 선호도 유사성 기법) 기반 다중 기준 스케줄링을 구현한다.

**평가 기준** (4가지):
1. **데드라인 여유**: (deadline - predicted_completion_time) / deadline
2. **처리 성능**: 1 / average_processing_time
3. **전력 효율성**: 1 / (power_consumption / processing_capability)
4. **로드 공정성**: 1 / (queue_depth + 1)

**동작**:
- 정규화된 결정 행렬 생성
- 가중치 기반 점수 계산
- 이상적 해에 가장 가까운 노드 선택

#### `scheduler/power_aware_topsis_scheduler.py`
실시간 전력 메트릭을 통합한 TOPSIS 변형이다.

**특징**:
- Prometheus에서 현재 노드 전력 소비 조회
- 전력 데이터를 결정 행렬에 통합
- 전력 효율적 노드 우선순위 설정
- 클러스터 상태에 따른 가중치 동적 조정

### 모빌리티 및 데드라인

#### `mobility/mobility_manager.py`
2D 커버리지 영역에서 IoT 디바이스 이동을 시뮬레이션한다.

**시뮬레이션 시나리오**:
- **보행자**: 1-2 m/s 속도, 10% 방향 변화 확률
- **차량**: 5-15 m/s 속도, 5% 방향 변화 확률
- **드론**: 5-15 m/s 속도, 8% 방향 변화 확률

**주요 기능**:
- 각 디바이스의 위치, 속도 추적
- 커버리지 이탈 예상 시간 계산
- 디바이스 상태 업데이트

#### `deadline_adapter.py`
디바이스 모빌리티에 따라 작업 데드라인을 동적으로 조정한다.

**적응 로직**:
- 디바이스가 커버리지를 벗어날 예상 시간 계산
- 예상 이탈 시간과 기본 데드라인 비교
- 필요시 데드라인 확장 (적응 계수 적용)
- 디바이스가 연결을 잃기 전 작업 완료 보장

### 모니터링 및 프로파일링

#### `prometheus_collector.py`
Prometheus에서 실시간 메트릭을 수집한다.

**수집 메트릭**:
- 노드 상태 (up/down)
- 각 노드의 전력 소비
- 옵션: CPU 이용률, 메모리, 네트워크 메트릭

#### `performance_profile.py`
추론 성능 데이터를 저장하는 조회 테이블다.

**저장 구조**:
- 모델별, 가속기별 평균 추론 시간
- 데드라인 안전 마진 계산용
- TOPSIS 스코어링용 성능 지표

#### `slo_monitor.py`
SLO 준수 상황을 모니터링한다.

**추적 항목**:
- 위반 발생 시 로깅
- 위반 유형별 분류 (hard/normal/soft)
- SLO 준수율 계산

#### `metric_collector.py`
실험 결과를 수집하고 저장한다.

**출력 파일**:
- `master_inference_results.csv`: 모든 작업 상세 정보
- `slo_violations_<timestamp>.csv`: 데드라인 미스 기록
- `scheduling_decisions_<timestamp>.csv`: 스케줄링 선택
- `experiment_summary_<timestamp>.json`: 집계 통계

## 도커 및 Kubernetes 설정

### 추론 서버 구성

#### `docker/coral_inference_server/`
Google Coral TPU용 FastAPI 기반 추론 서버다.

**Dockerfile**:
- 베이스 이미지: `arm64v8/ubuntu:22.04`
- Python 3.10, pycoral, tflite-runtime 설치
- FastAPI 서버 구성

**coral_inference_server.py**:
- FastAPI 애플리케이션
- POST `/infer` 엔드포인트
- 모델 로드 및 추론 실행
- 추론 시간, 노드명 포함한 응답 반환

#### `docker/jetson_inference_server/`
NVIDIA Jetson용 추론 서버다.

**특징**:
- NVIDIA CUDA 기반 이미지
- TensorRT 최적화 지원
- Jetson SDK 통합
- NVIDIA GPU 메모리 관리

#### `docker/hailo_inference_server/`
Hailo AI 가속기용 추론 서버다.

**구성**:
- Hailo SDK 포함
- HEF(Hailo Executable Format) 모델 지원
- 특화된 전력 프로파일링

### Kubernetes 배포 YAML

#### `yaml/coral_daemonset.yaml`
Coral TPU 노드에 DaemonSet으로 서버를 배포한다.

**주요 설정**:
- DaemonSet으로 각 Coral 노드에 Pod 배포
- 노드 선택자 (node.hardware = coral)
- Edge TPU USB 디바이스 마운트
- 권한 설정 (privileged mode)
- Pod IP를 통한 서비스 노출

#### `yaml/jetson_daemonset.yaml`
Jetson Nano 노드에 DaemonSet으로 서버를 배포한다.

**주요 설정**:
- Jetson 노드 선택
- GPU 리소스 요청
- NVIDIA 런타임 설정
- /dev/nvhost* 디바이스 마운트

#### `yaml/hailo_daemonset.yaml`
Hailo 가속기 노드에 DaemonSet으로 서버를 배포한다.

**주요 설정**:
- Hailo 노드 식별
- Hailo 드라이버 및 라이브러리 마운트
- 전력 모니터링 에이전트 연동

## 모니터링 스택 설정

### `monitoring/prometheus-configmap.yaml`
Prometheus 설정을 ConfigMap으로 정의한다.

**구성**:
- node-exporter 타겟 정의
- 각 노드의 메트릭 스크래핑 규칙
- 전력 메트릭 수집 설정
- 리테인션 정책

### `monitoring/prometheus-deployment.yaml`
Prometheus 서버를 Deployment로 배포한다.

**주요 설정**:
- 마스터 노드에서 실행
- PersistentVolume으로 데이터 저장
- NodePort 서비스 (포트 30090)
- 메트릭 파기 및 저장 정책

### `monitoring/grafana-deployment.yaml`
Grafana 대시보드를 배포한다.

**기능**:
- Prometheus 데이터소스 자동 등록
- 성능 시각화 대시보드
- 경고(Alert) 설정
- NodePort 서비스 (포트 30300)

### `monitoring/node-exporter-daemonset.yaml`
각 워커 노드에 node-exporter를 배포한다.

**수집 항목**:
- CPU, 메모리, 네트워크 메트릭
- 디스크 I/O 정보
- 시스템 시간 및 부팅 정보
- 수정 가능한 커스텀 메트릭 (전력 데이터)

## 시작 방법

### 필수 조건
1. Kubernetes (k3s) 클러스터 구축 완료
2. 각 워커 노드의 추론 서버 배포 (Docker 이미지)
3. Prometheus 모니터링 스택 배포
4. Python 환경: fastapi, requests, numpy, pycoral 등

### 실행 단계

1. **Pod IP 주소 확인**
```bash
kubectl get pods -o wide
```

2. **inference_request.py의 WORKERS 딕셔너리 업데이트**
```python
WORKERS = {
    "coral1": "http://<실제pod_ip>:8080/infer",
    ...
}
```

3. **메인 실험 실행**
```bash
python3 main_pregenerated.py
```

4. **결과 확인**
- CSV 파일: 작업별 상세 정보
- JSON 파일: 최종 통계 및 메트릭

## 생성되는 출력 파일

| 파일명 | 설명 |
|-------|------|
| `master_inference_results.csv` | 모든 작업 실행 결과 |
| `slo_violations_<timestamp>.csv` | 데드라인 위반 기록 |
| `scheduling_decisions_<timestamp>.csv` | 스케줄링 선택 이력 |
| `devices_<timestamp>.csv` | 디바이스 상태 정보 |
| `experiment_summary_<timestamp>.json` | 집계 통계 및 메트릭 |

## 주요 메트릭

| 메트릭 | 설명 |
|-------|------|
| SLO 준수율 | 데드라인 이전 완료 작업의 % |
| 평균 응답 시간 | 도착부터 완료까지 평균 지연 |
| 에너지 소비 | 모든 노드의 전력-시간 적분 |
| 디바이스 공정성 | 가속기 간 로드 분산 균형 |
| 커버리지 스킵율 | 이탈 디바이스의 스킵된 작업 % |

## 관련 기술

- **스케줄링**: TOPSIS 다중 기준 의사결정
- **모빌리티**: 무작위 보행(random walk) 모델
- **모니터링**: Prometheus + Grafana
- **컨테이너화**: Docker + Kubernetes
- **프레임워크**: FastAPI, Python asyncio
- **가속기**: Coral TPU, NVIDIA Jetson, Hailo

## 저자 및 라이선스

석사 학위 논문 연구 프로젝트 (이기종 AI 가속기 및 엣지 클라우드 인프라)

마지막 업데이트: 2026년 1월
