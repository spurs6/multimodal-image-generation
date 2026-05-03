@echo off
chcp 65001 >nul 2>&1
setlocal EnableExtensions
cd /d "%~dp0"

REM Default HF_HOME: project\huggingface (override with env or .env)
if not defined HF_HOME set "HF_HOME=%~dp0huggingface"

call "%~dp0install_deps.bat"
if errorlevel 1 (
  pause
  exit /b 1
)

if exist "%~dp0..\.venv-gpu\Scripts\python.exe" (
  set "PY=%~dp0..\.venv-gpu\Scripts\python.exe"
) else if exist "%~dp0.venv310\Scripts\python.exe" (
  set "PY=%~dp0.venv310\Scripts\python.exe"
) else (
  set "PY=python"
)

echo.
echo === Hugging Face models (group webapp, two phases) ===
echo Cache: %HF_HOME%
echo Set HF_TOKEN in .env. Large repos may take hours. Re-run to resume. Per-repo timeout 6h.
echo Order: 1^) Core + SD2.1 + SD1.5 style packs  2^) Reference-mode ControlNets + Annotators + IP-Adapter
echo.

"%PY%" scripts\download_models_stepwise.py --python "%PY%" --group webapp --timeout 21600
pause
