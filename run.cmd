@echo off
setlocal
cd /d "%~dp0"
py -3 -c "print('FACE_BOOT_OK')" 2>nul | "%SystemRoot%\System32\findstr.exe" /x /c:"FACE_BOOT_OK" >nul
if not errorlevel 1 (
    py -3 "%~dp0start.py"
    exit /b
)
if exist "%~dp0..\.python-runtime\python.exe" (
    "%~dp0..\.python-runtime\python.exe" -c "print('FACE_BOOT_OK')" 2>nul | "%SystemRoot%\System32\findstr.exe" /x /c:"FACE_BOOT_OK" >nul
    if not errorlevel 1 (
        "%~dp0..\.python-runtime\python.exe" "%~dp0start.py"
        exit /b
    )
)
if exist "%~dp0.python-runtime\python.exe" (
    "%~dp0.python-runtime\python.exe" -c "print('FACE_BOOT_OK')" 2>nul | "%SystemRoot%\System32\findstr.exe" /x /c:"FACE_BOOT_OK" >nul
    if not errorlevel 1 (
        "%~dp0.python-runtime\python.exe" "%~dp0start.py"
        exit /b
    )
)
python -c "print('FACE_BOOT_OK')" 2>nul | "%SystemRoot%\System32\findstr.exe" /x /c:"FACE_BOOT_OK" >nul
if not errorlevel 1 (
    python "%~dp0start.py"
    exit /b
)
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" "%~dp0start.py"
    exit /b
)
echo No working Python was found. Install Python or provide the parent .python-runtime.
exit /b 1
