from pathlib import Path
import hashlib
import threading
import cv2
import numpy as np
import onnxruntime as ort

AGE_BINS = [(0, 2), (4, 6), (8, 12), (15, 20), (25, 32), (38, 43), (48, 53), (60, 100)]

class ModelUnavailable(Exception):
    pass

class Estimator:
    def __init__(self, root: Path):
        self.root = root
        self.lock = threading.Lock()
        self.sessions = None
        # Read through Python: OpenCV's Windows file loader can reject Korean paths.
        xml = Path(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml').read_text(encoding='utf-8')
        storage = cv2.FileStorage(xml, cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
        self.detector = cv2.CascadeClassifier()
        self.detector.read(storage.getFirstTopLevelNode())
        storage.release()

    def load(self):
        if self.sessions is not None:
            return
        paths = [self.root / f'{k}.onnx' for k in ('age', 'gender')]
        if not all(p.exists() for p in paths):
            raise ModelUnavailable('모델이 없습니다. py download_models.py를 먼저 실행하세요.')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        try:
            self.sessions = [ort.InferenceSession(str(p), sess_options=options, providers=['CPUExecutionProvider']) for p in paths]
        except Exception as exc:
            raise ModelUnavailable('모델을 읽을 수 없습니다. 모델을 다시 다운로드하세요.') from exc
        self.version = 'googlenet-adience-onnx-v1:' + ':'.join(hashlib.sha256(p.read_bytes()).hexdigest() for p in paths)

    def predict(self, rgb):
        with self.lock:
            self.load()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
            if len(faces) != 1:
                raise ValueError('정면 얼굴이 한 명만 보이는 사진을 사용하세요. 감지된 얼굴: ' + str(len(faces)))
            x, y, w, h = map(int, faces[0])
            pad = int(max(w, h) * .15)
            crop = bgr[max(0,y-pad):min(bgr.shape[0],y+h+pad),max(0,x-pad):min(bgr.shape[1],x+w+pad)]
            blob = cv2.dnn.blobFromImage(crop, 1, (224,224), (104,117,123), swapRB=False)
            outputs = [s.run(None, {s.get_inputs()[0].name: blob})[0].reshape(-1) for s in self.sessions]
            # Model Zoo outputs are softmax probabilities, class order follows Adience.
            age, gender = outputs
            ai, gi = int(np.argmax(age)), int(np.argmax(gender))
            return {'age_range': list(AGE_BINS[ai]), 'age_class': ai,
                    'gender': ['male','female'][gi], 'age_scores': age.tolist(),
                    'gender_scores': gender.tolist(), 'face_box': [x,y,w,h],
                    'model_version': self.version, 'preprocessing_version': 'haar-pad15-bgr224-mean104117123-v1'}
