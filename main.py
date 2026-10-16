from pathlib import Path
from datetime import datetime, timezone
from typing import Literal
import hashlib
import io
import json
import os
import sys
import uuid
import warnings
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from filelock import FileLock

# Embedded Python may omit the script directory from sys.path.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from model import Estimator, ModelUnavailable

DATA = Path(os.environ.get('FACE_DATA_DIR', str(ROOT / 'data')))
DATA.mkdir(parents=True, exist_ok=True)
(DATA / 'images').mkdir(exist_ok=True)
DB = DATA / 'records.json'
LOCK = FileLock(str(DATA / 'records.lock'))
Image.MAX_IMAGE_PIXELS = 16_000_000
app = FastAPI(title='Face Lab', version='1.0.0')
app.mount('/static', StaticFiles(directory=ROOT / 'static'), name='static')
estimator = Estimator(ROOT / 'models')

def now():
    return datetime.now(timezone.utc).isoformat()

def read_records():
    if not DB.exists():
        return []
    try:
        doc = json.loads(DB.read_text(encoding='utf-8'))
        if not isinstance(doc, dict) or doc.get('schema_version') != 1 or not isinstance(doc.get('records'), list):
            raise ValueError()
        for record in doc['records']:
            if (not isinstance(record, dict) or
                    not all(key in record for key in ('id', 'created_at', 'prediction', 'feedback', 'feedback_history', 'image_path', 'retention_consent')) or
                    not isinstance(record['prediction'], dict) or
                    not isinstance(record['feedback_history'], list)):
                raise ValueError()
        return doc['records']
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(500, '이력 파일이 손상되었습니다. 원본을 보존한 후 복구하세요.') from exc

def write_records(records):
    temp = DATA / f'{uuid.uuid4().hex}.tmp'
    try:
        with temp.open('w', encoding='utf-8') as f:
            json.dump({'schema_version': 1, 'records': records}, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, DB)
    finally:
        temp.unlink(missing_ok=True)

@app.get('/')
def index():
    return FileResponse(ROOT / 'templates' / 'index.html')

@app.get('/api/health')
def health():
    return {'status': 'ok', 'models_present': all((ROOT / 'models' / f'{k}.onnx').exists() for k in ('age','gender'))}

@app.post('/api/predict')
def predict(image: UploadFile = File(...), retain_image: bool = Form(False)):
    raw = image.file.read(8 * 1024 * 1024 + 1)
    if len(raw) > 8 * 1024 * 1024:
        raise HTTPException(413, '이미지는 8MB 이하로 업로드하세요.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as im:
                if im.format not in ('JPEG', 'PNG', 'WEBP'):
                    raise ValueError()
                im = ImageOps.exif_transpose(im).convert('RGB')
                if im.width * im.height > Image.MAX_IMAGE_PIXELS:
                    raise ValueError()
                im.thumbnail((1600,1600))
                rgb = np.array(im)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise HTTPException(400, '유효한 JPG, PNG, WEBP 이미지(최대 1,600만 화소)가 필요합니다.')
    try:
        prediction = estimator.predict(rgb)
    except ModelUnavailable as exc:
        raise HTTPException(503, str(exc))
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    record_id = uuid.uuid4().hex
    relative = f'images/{record_id}.jpg' if retain_image else None
    record = {'id': record_id, 'created_at': now(), 'prediction': prediction,
              'processed_image_size': [int(rgb.shape[1]), int(rgb.shape[0])],
              'image_sha256': hashlib.sha256(raw).hexdigest(), 'image_path': relative,
              'retention_consent': retain_image, 'feedback': None, 'feedback_history': []}
    try:
        with LOCK:
            records = read_records()
            if relative:
                Image.fromarray(rgb).save(DATA / relative, quality=92)
            records.append(record)
            write_records(records)
    except Exception:
        if relative:
            (DATA / relative).unlink(missing_ok=True)
        raise
    return record

class Feedback(BaseModel):
    age: int = Field(ge=0, le=120, strict=True)
    gender: Literal['male', 'female', 'other', 'undisclosed']

@app.put('/api/records/{record_id}/feedback')
def feedback(record_id: str, body: Feedback):
    with LOCK:
        records = read_records()
        record = next((r for r in records if r['id'] == record_id), None)
        if record is None:
            raise HTTPException(404, '이력이 없습니다.')
        value = {**body.model_dump(), 'updated_at': now()}
        record['feedback'] = value
        record['feedback_history'].append(value)
        write_records(records)
        return record

@app.get('/api/records')
def history():
    with LOCK:
        return list(reversed(read_records()))

@app.get('/api/export')
def export():
    with LOCK:
        return {'schema_version': 1, 'exported_at': now(), 'records': read_records()}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
