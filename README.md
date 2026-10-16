# Facial-age-and-Gender-estimation (Face Lab)

FastAPI와 HTML/CSS/JavaScript로 만든 로컬 얼굴 나이·성별 추정 실험 앱입니다. ONNX Model Zoo의 Adience GoogleNet age/gender 모델을 `CPUExecutionProvider`로 실행합니다.

## 기능

- JPG, PNG, WEBP 업로드 및 얼굴 한 명 검출
- 나이 구간·성별 분류·클래스별 확률 표시
- 실제 정답 입력과 수정, 변경 이력 저장
- 예측·피드백·모델 버전·전처리 버전 JSON 저장
- 이력 조회와 JSON 내보내기

## 모델과 전처리

모델 바이너리는 저장소에 넣지 않습니다. 공식 모델 저장소와 다운로드 스크립트를 사용합니다.

- [age_googlenet](https://huggingface.co/onnxmodelzoo/age_googlenet)
- [gender_googlenet](https://huggingface.co/onnxmodelzoo/gender_googlenet)
- 원 프로젝트: [ONNX Model Zoo age_gender](https://github.com/onnx/models/tree/main/validated/vision/body_analysis/age_gender)

`download_models.py`는 모델을 `models/age.onnx`, `models/gender.onnx`로 저장하고 다운로드 리비전과 SHA256을 `models/manifest.json`에 기록합니다. 모델 파일 자체는 `.gitignore`로 제외됩니다.

```powershell
py download_models.py
```

추론 전처리는 Model Zoo 규격에 맞춰 BGR, `224×224`, scale `1`, 평균 `[104, 117, 123]`을 사용합니다. age 클래스 순서는 `0–2, 4–6, 8–12, 15–20, 25–32, 38–43, 48–53, 60–100`이며 gender 순서는 `male, female`입니다. 모델 출력은 이미 softmax 확률이므로 다시 softmax를 적용하지 않습니다.

## 설치 및 실행

저장소 루트에서 실행합니다. 가상환경 활성화나 PowerShell 실행 정책 변경은 필요하지 않습니다.

```powershell
py start.py
```

Windows Python 런처가 없거나 기본 Python이 깨져 있으면 다음을 사용합니다.

```powershell
.\run.cmd
```

두 실행 방식 모두 실행 파일 위치를 기준으로 프로젝트를 찾고, 필요한 패키지가 실제로 import되는 Python을 선택합니다. 기본 포트가 사용 중이면 빈 포트를 자동 선택해 `Open: http://127.0.0.1:<port>`로 출력합니다.

브라우저에서 출력된 주소를 엽니다. API 문서는 `<출력된 주소>/docs`입니다.

새 환경을 직접 준비해야 하는 경우:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_models.py
py start.py
```

## 프로젝트 구조

```text
main.py                 # FastAPI API와 JSON 저장
model.py                # 얼굴 검출·CPU ONNX 추론
start.py                # Python 선택·빈 포트 실행
run.cmd                 # Windows 실행 진입점
download_models.py      # 공식 모델 다운로드와 manifest 생성
evaluate.py             # 피드백 기반 평가
templates/index.html    # 화면
static/style.css        # 스타일
static/app.js           # 업로드·피드백·이력 UI
test_app.py             # API·저장 테스트
tests/                  # 통합 검증 자료
models/manifest.json    # 모델 리비전·SHA256만 Git에 포함
requirements.txt        # 실행 의존성
requirements-dev.txt    # 테스트 의존성
```

## 데이터 보호

실행 중 `data/`에 예측·피드백 JSON과 동의한 이미지가 생성됩니다. 모델 바이너리, 가상환경, 비밀정보, 얼굴 이미지와 생성 데이터는 Git에 포함하지 않습니다. 본인 또는 사용 동의를 받은 이미지만 사용하세요.

## 테스트

```powershell
py -m pip install -r requirements-dev.txt
py -m pytest -q test_app.py
```

실제 모델 통합 테스트는 로컬 모델과 테스트 얼굴 이미지가 있을 때만 실행됩니다. 정확도나 성별 정체성을 확정하는 용도의 모델이 아닙니다.

## GitHub 업로드 전 확인

```powershell
git rev-parse --show-toplevel
git remote -v
git status --short
git ls-files -ci --exclude-standard
```

위 마지막 명령은 비밀정보·환경·얼굴 이미지·예측 데이터·모델 바이너리가 추적되는지 확인하는 용도입니다.
