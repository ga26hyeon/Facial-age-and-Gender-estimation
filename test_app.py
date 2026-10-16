import io
import importlib
import sys
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import pytest
from fastapi.testclient import TestClient

@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.setenv('FACE_DATA_DIR', str(tmp_path))
    sys.modules.pop('main', None)
    module = importlib.import_module('main')
    return module, TestClient(module.app)

def picture():
    f=io.BytesIO();Image.new('RGB',(80,80)).save(f,format='PNG');return f.getvalue()

def fake(_):
    return {'age_range':[25,32],'gender':'female','model_version':'test-only','preprocessing_version':'test'}

def test_roundtrip(service, monkeypatch):
    m,c=service;monkeypatch.setattr(m.estimator,'predict',fake)
    assert c.get('/').status_code==200
    assert 'Face Lab' in c.get('/').text
    assert c.get('/static/style.css').status_code==200
    assert c.get('/static/app.js').status_code==200
    assert c.get('/static/index.html').status_code==404
    r=c.post('/api/predict',files={'image':('x.png',picture())},data={'retain_image':'true'})
    assert r.status_code==200
    row=r.json();assert (m.DATA/row['image_path']).exists()
    url=f'/api/records/{row["id"]}/feedback'
    assert c.put(url,json={'age':29,'gender':'female'}).status_code==200
    assert c.put(url,json={'age':30,'gender':'other'}).status_code==200
    assert c.put(url,json={'age':-1,'gender':'female'}).status_code==422
    assert c.put(url,json={'age':1.5,'gender':'female'}).status_code==422
    assert c.put('/api/records/missing/feedback',json={'age':20,'gender':'male'}).status_code==404
    m2=importlib.reload(m)
    rows=TestClient(m2.app).get('/api/records').json()
    assert rows[0]['feedback']['age']==30 and len(rows[0]['feedback_history'])==2
    assert c.get('/api/export').json()['schema_version']==1

def test_bad_upload_and_model_errors(service, monkeypatch):
    m,c=service
    assert c.post('/api/predict',files={'image':('x.png',b'not an image')}).status_code==400
    assert c.post('/api/predict',files={'image':('x.jpg',b'x'*(8*1024*1024+1))}).status_code==413
    def no_model(_): raise m.ModelUnavailable('missing')
    monkeypatch.setattr(m.estimator,'predict',no_model)
    assert c.post('/api/predict',files={'image':('x.png',picture())}).status_code==503
    def no_face(_): raise ValueError('no face')
    monkeypatch.setattr(m.estimator,'predict',no_face)
    assert c.post('/api/predict',files={'image':('x.png',picture())}).status_code==422

def test_concurrent_writes_and_corruption(service,monkeypatch):
    m,c=service;monkeypatch.setattr(m.estimator,'predict',fake)
    def upload(_): return c.post('/api/predict',files={'image':('x.png',picture())})
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(r.status_code==200 for r in pool.map(upload,range(8)))
    rows=c.get('/api/records').json();assert len(rows)==8
    assert all(r['image_path'] is None for r in rows)
    m.DB.write_text('{broken',encoding='utf-8')
    assert c.get('/api/records').status_code==500
    assert m.DB.read_text()=='{broken'

def test_evaluation():
    from evaluate import evaluate
    result=evaluate([{'prediction':fake(None),'feedback':{'age':35,'gender':'other'},'image_path':None}])['test-only']
    assert result['mean_distance_to_age_bin']==3 and result['gender_accuracy'] is None


@pytest.mark.parametrize('document', ['null', '[]', '{"schema_version":1,"records":[null]}'])
def test_invalid_database_structure_preserved(service, document):
    m, c = service
    m.DB.write_text(document, encoding='utf-8')
    assert c.get('/api/records').status_code == 500
    assert m.DB.read_text(encoding='utf-8') == document


def test_real_cpu_roundtrip(service):
    """Optional real inference; sample/model assets stay outside version control."""
    from pathlib import Path
    import json
    m, c = service
    sample = Path(__file__).parent / 'tests' / 'sample.jpg'
    if not sample.exists() or not all((m.ROOT / 'models' / f'{k}.onnx').exists() for k in ('age', 'gender')):
        pytest.skip('Local sample image and downloaded ONNX models are required')
    response = c.post('/api/predict', files={'image': ('sample.jpg', sample.read_bytes(), 'image/jpeg')},
                      data={'retain_image': 'true'})
    assert response.status_code == 200, response.text
    row = response.json()
    prediction = row['prediction']
    assert len(prediction['age_scores']) == 8 and len(prediction['gender_scores']) == 2
    assert prediction['model_version'].startswith('googlenet-adience-onnx-v1:')
    assert all(s.get_providers() == ['CPUExecutionProvider'] for s in m.estimator.sessions)
    assert (m.DATA / row['image_path']).exists()
    assert len(row['processed_image_size']) == 2
    assert c.put(f'/api/records/{row["id"]}/feedback', json={'age': 30, 'gender': 'female'}).status_code == 200
    exported = c.get('/api/export').json()['records']
    assert exported == json.loads(m.DB.read_text(encoding='utf-8'))['records']
    assert exported[0]['feedback']['age'] == 30
