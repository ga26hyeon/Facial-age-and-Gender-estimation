"""Start the local app from any Windows working directory without activation."""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
MODULES = ('fastapi', 'uvicorn', 'multipart', 'numpy', 'PIL', 'cv2', 'onnxruntime', 'filelock')
MARKER = 'FACE_RUNTIME_OK:'

def choose_python():
    candidates = [[str(ROOT / '.venv' / 'Scripts' / 'python.exe')], [sys.executable]]
    if shutil.which('py'):
        candidates.append([shutil.which('py'), '-3'])
    for parent in [ROOT, *ROOT.parents]:
        candidates.append([str(parent / '.python-runtime' / 'python.exe')])
    if shutil.which('python'):
        candidates.append([shutil.which('python')])
    checked = set()
    code = ('import sys,json; assert sys.version_info >= (3,10); import '
            + ','.join(MODULES)
            + '; print(' + repr(MARKER) + '+json.dumps(sys.executable,ensure_ascii=True))')
    for command in candidates:
        key = tuple(command)
        if key in checked or not command[0]:
            continue
        checked.add(key)
        try:
            result = subprocess.run([*command, '-c', code], cwd=ROOT, capture_output=True,
                                    timeout=25, check=False)
            for line in result.stdout.decode('ascii', errors='ignore').splitlines():
                if result.returncode == 0 and line.startswith(MARKER):
                    executable = Path(json.loads(line[len(MARKER):]))
                    if executable.is_file():
                        return str(executable)
        except (OSError, subprocess.SubprocessError, ValueError):
            continue
    raise SystemExit('No working Python with the required packages was found. '
                     'Install requirements.txt using a working Python interpreter.')

def reserve_socket():
    try:
        first = int(os.environ.get('FACE_PORT', '8000'))
    except ValueError:
        first = 8000
    if not 1024 <= first <= 65435:
        first = 8000
    for port in range(first, first + 100):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            listener.bind(('127.0.0.1', port))
            listener.listen(2048)
            return listener
        except OSError:
            listener.close()
    raise SystemExit('No free local port was found in the tested range.')

def serve():
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    import uvicorn
    with reserve_socket() as listener:
        port = listener.getsockname()[1]
        print(f'Open: http://127.0.0.1:{port}', flush=True)
        server = uvicorn.Server(uvicorn.Config('main:app', host='127.0.0.1', port=port,
                                              log_level='info'))
        server.run(sockets=[listener])

if __name__ == '__main__':
    if sys.argv[1:] == ['--serve']:
        serve()
    else:
        python = choose_python()
        print(f'Python: {python}', flush=True)
        try:
            raise SystemExit(subprocess.call([python, str(ROOT / 'start.py'), '--serve'], cwd=ROOT))
        except KeyboardInterrupt:
            raise SystemExit(0)
