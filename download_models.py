"""Download official ONNX Model Zoo assets; run once before starting."""
from pathlib import Path
from urllib.request import urlopen
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parent / 'models'

def main():
    ROOT.mkdir(exist_ok=True)
    manifest = {}
    for kind in ('age', 'gender'):
        repo = f'onnxmodelzoo/{kind}_googlenet'
        with urlopen(f'https://huggingface.co/api/models/{repo}', timeout=60) as r:
            info = json.load(r)
        name = next(x['rfilename'] for x in info['siblings'] if x['rfilename'].endswith('.onnx'))
        url = f'https://huggingface.co/{repo}/resolve/{info["sha"]}/{name}'
        target = ROOT / f'{kind}.onnx'
        temp = target.with_suffix('.part')
        try:
            with urlopen(url, timeout=120) as src, temp.open('wb') as dst:
                while block := src.read(1024 * 1024):
                    dst.write(block)
            if temp.stat().st_size < 1000000:
                raise RuntimeError('Model download is incomplete')
            os.replace(temp, target)
        finally:
            temp.unlink(missing_ok=True)
        manifest[kind] = {'source': url, 'revision': info['sha'], 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
        print(f'{kind}: {target.stat().st_size:,} bytes')
    (ROOT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
