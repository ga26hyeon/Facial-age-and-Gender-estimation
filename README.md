# Face Lab

FastAPI와 HTML/CSS/JavaScript로 만든 로컬 얼굴 나이·성별 추정 앱입니다. ONNX Model Zoo의 Adience GoogLeNet age/gender 모델을 ONNX Runtime CPU provider로 실행합니다.

## 설치 및 실행

PowerShell에서 저장소 루트로 이동합니다. Python 활성화나 실행 정책 변경은 필요하지 않습니다.

```powershell
py -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.\\.venv\\Scripts\\python.exe download_models.py
py start.py
```

`py`가 없거나 기본 Python을 찾지 못하면 `run.cmd`가 프로젝트의 `.venv`, 상위 `.python-runtime`, PATH의 Python을 확인합니다.

```powershell
.\\run.cmd
```

로그의 `Open: http://127.0.0.1:<port>` 주소를 브라우저에서 엽니다. 기본 포트가 사용 중이면 빈 포트를 자동 선택합니다.

## 모델 다운로드

모델 바이너리는 저장소에 포함하지 않습니다. 아래 스크립트가 Hugging Face의 공식 `onnxmodelzoo/age_googlenet` 및 `onnxmodelzoo/gender_googlenet` 저장소에서 모델을 내려받고 SHA256을 `models/manifest.json`에 기록합니다.

```powershell
.\\.venv\\Scripts\\python.exe download_models.py
```

생성 파일은 `models/age.onnx`, `models/gender.onnx`, `models/manifest.json`입니다. 입력은 BGR 224×224, scale 1, 평균 `[104, 117, 123]`을 사용합니다. age class 순서는 `0–2, 4–6, 8–12, 15–20, 25–32, 38–43, 48–53, 60–100`, gender 순서는 `male, female`입니다. 모델 출력은 이미 확률값이므로 softmax를 다시 적용하지 않습니다.

## 검증

```powershell
.\\.venv\\Scripts\\python.exe -m pip install -r requirements-dev.txt
.\\.venv\\Scripts\\python.exe -m pytest -q test_app.py
```

실제 확인: 필수 import, API 테스트 8개, 서버의 `/`, `/api/health`, 정적 파일, 브라우저의 업로드·이력·JSON 내보내기·피드백 UI를 확인했습니다. 실제 ONNX 추론은 모델 바이너리와 얼굴 이미지가 있는 환경에서 실행됩니다.

미확인: 현재 환경의 원격 GitHub fetch는 네트워크 차단으로 실패했습니다. 개인정보 보호를 위해 얼굴 이미지는 저장소에 포함하지 않았으며, 실제 얼굴 업로드 검증은 별도 이미지로 수행해야 합니다.

## 제외 항목

`.gitignore`는 가상환경, 캐시, 환경변수·비밀정보, 로그, `data/`의 예측·피드백·얼굴 이미지, ONNX 바이너리를 제외하고 `models/manifest.json`만 추적합니다.
