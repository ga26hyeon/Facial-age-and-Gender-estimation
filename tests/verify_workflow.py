"""Isolated service/Git workflow experiments; never changes the real Git index."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.request import urlopen

PROJECT = Path(__file__).resolve().parents[1]
RUN = Path(tempfile.mkdtemp(prefix='workflow-', dir=PROJECT / 'tests'))
FILES = ['main.py', 'model.py', 'app.py', 'download_models.py', 'evaluate.py',
         'test_app.py', 'requirements.txt', 'requirements-dev.txt', 'README.md',
         'VERIFICATION.md', '.gitignore', 'templates/index.html',
         'static/style.css', 'static/app.js']
REPORT = {'checks': [], 'github_network_push_tested': False}


def command(cwd, *args, expected=0):
    result = subprocess.run(list(args), cwd=cwd, capture_output=True,
                            text=True, encoding='utf-8', errors='replace', timeout=45)
    if expected is not None and result.returncode != expected:
        raise AssertionError(f'{args}: {result.returncode}\n{result.stdout}\n{result.stderr}')
    return result


def git(cwd, *args, expected=0):
    return command(cwd, 'git', '-c', 'commit.gpgsign=false', *args, expected=expected)


def check(name, **details):
    REPORT['checks'].append({'name': name, 'passed': True, **details})


def initialize(path, bare=False):
    path.mkdir()
    git(path, 'init', '--template=', '-b', 'main', *(['--bare'] if bare else []))
    if not bare:
        git(path, 'config', '--local', 'user.name', 'Local Workflow Test')
        git(path, 'config', '--local', 'user.email', 'workflow@example.invalid')
    return path


def copy_source(destination):
    for filename in FILES:
        target = destination / filename
        target.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy2(PROJECT / filename, target)


def add_and_commit(path, message):
    git(path, 'add', '--', *FILES)
    names = git(path, 'diff', '--cached', '--name-only').stdout.splitlines()
    assert set(names) <= set(FILES), names
    git(path, 'commit', '-m', message)


def main():
    index = PROJECT / '.git' / 'index'
    original_index = hashlib.sha256(index.read_bytes()).hexdigest() if index.exists() else None
    original_head = git(PROJECT, 'rev-parse', '--verify', 'HEAD', expected=None)
    for cwd in (PROJECT, PROJECT.parent):
        # Absolute Set-Location works from inside and outside the project folder.
        ps_path = str(PROJECT).replace("'", "''")
        result = command(cwd, 'powershell.exe', '-NoProfile', '-NonInteractive', '-Command',
                         f"Set-Location -LiteralPath '{ps_path}'; (Get-Item -LiteralPath '.\\main.py').FullName")
        assert result.stdout.strip() == str(PROJECT / 'main.py')
    check('PowerShell absolute path from parent and project folders')

    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    env = {**os.environ, 'FACE_DATA_DIR': str(RUN / 'service-data')}
    with (RUN / 'server.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'main:app', '--host',
                                    '127.0.0.1', '--port', str(port)], cwd=PROJECT,
                                   env=env, stdout=log, stderr=log)
        try:
            base = f'http://127.0.0.1:{port}'
            for _ in range(60):
                if process.poll() is not None:
                    raise AssertionError('Server exited; inspect server.log')
                try:
                    with urlopen(base + '/api/health', timeout=1) as response:
                        health = json.load(response)
                    break
                except (URLError, TimeoutError):
                    time.sleep(.2)
            else:
                raise AssertionError('Server did not start')
            assert health['status'] == 'ok' and health['models_present']
            for route in ('/', '/static/style.css', '/static/app.js', '/docs'):
                with urlopen(base + route, timeout=5) as response:
                    assert response.status == 200
            check('Real Uvicorn startup, health, HTML, CSS, JS and docs', port=port)
        finally:
            process.terminate()
            process.wait(timeout=10)

    local = initialize(RUN / 'local')
    copy_source(local)
    ignored = ['data/records.json', 'uploads/face.jpg', 'models/age.onnx',
               '.venv/test.txt', '.python-runtime/test.txt', '__pycache__/test.pyc',
               '.env', 'server.log', 'unrelated-document.txt']
    for filename in ignored:
        target = local / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('DUMMY ONLY', encoding='utf-8')
    for filename in ignored:
        git(local, 'check-ignore', '--', filename)
    add_and_commit(local, 'Initial commit')
    tracked = set(git(local, 'ls-files').stdout.splitlines())
    assert tracked == set(FILES)
    check('Source-only staging and exclusions', tracked_count=len(tracked), excluded_count=len(ignored))

    git(local, 'add', '-f', '--', 'data/records.json')
    assert 'data/records.json' in git(local, 'ls-files', '-ci', '--exclude-standard').stdout
    git(local, 'rm', '--cached', '--', 'data/records.json')
    assert (local / 'data/records.json').exists()
    assert 'data/records.json' not in git(local, 'ls-files').stdout
    check('Remove excluded file from index while preserving local file')

    empty = initialize(RUN / 'empty.git', bare=True)
    git(local, 'remote', 'add', 'origin', str(empty))
    attempted_pull = git(local, 'pull', 'origin', 'main', '--allow-unrelated-histories',
                         '--no-rebase', '--no-edit', expected=None)
    assert attempted_pull.returncode != 0
    check('Empty remote: unconditional pull fails as expected; omit pull')
    git(local, 'push', '-u', 'origin', 'main')
    assert git(local, 'rev-parse', 'HEAD').stdout == git(empty, 'rev-parse', 'main').stdout
    check('Initial commit and push to empty local remote')

    existing = initialize(RUN / 'existing.git', bare=True)
    seed = initialize(RUN / 'seed')
    (seed / 'README.md').write_text('# Existing remote README\n', encoding='utf-8')
    git(seed, 'add', '--', 'README.md')
    git(seed, 'commit', '-m', 'Remote README')
    git(seed, 'remote', 'add', 'origin', str(existing))
    git(seed, 'push', '-u', 'origin', 'main')
    git(local, 'remote', 'set-url', 'origin', str(existing))
    old_head = git(local, 'rev-parse', 'HEAD').stdout
    pull = git(local, 'pull', 'origin', 'main', '--allow-unrelated-histories',
               '--no-rebase', '--no-edit', expected=None)
    assert pull.returncode != 0
    assert 'README.md' in git(local, 'diff', '--name-only', '--diff-filter=U').stdout
    git(local, 'merge', '--abort')
    assert git(local, 'rev-parse', 'HEAD').stdout == old_head
    check('Existing remote README: screenshot pull creates conflict; abort preserves source')

    clone = RUN / 'clone-first'
    git(RUN, 'clone', '--template=', str(existing), str(clone))
    git(clone, 'config', '--local', 'user.name', 'Local Workflow Test')
    git(clone, 'config', '--local', 'user.email', 'workflow@example.invalid')
    # In this experiment the local README is chosen deliberately. The remote
    # README remains recoverable in the original commit in the shared history.
    copy_source(clone)
    add_and_commit(clone, 'Add reviewed service source')
    git(clone, 'push', 'origin', 'main')
    assert git(existing, 'rev-list', '--count', 'main').stdout.strip() == '2'
    check('Existing remote: clone-first upload preserves remote commit history')

    assert original_head.stdout == git(PROJECT, 'rev-parse', '--verify', 'HEAD', expected=None).stdout
    current_index = hashlib.sha256(index.read_bytes()).hexdigest() if index.exists() else None
    assert original_index == current_index
    check('Real project Git HEAD and index unchanged')
    REPORT['experiment_directory'] = str(RUN)
    (RUN / 'report.json').write_text(json.dumps(REPORT, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(REPORT, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
