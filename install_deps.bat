@echo off
chcp 65001 >nul 2>&1
REM CUDA PyTorch + requirements.txt (no HF model download)
setlocal EnableExtensions
cd /d "%~dp0"

rem pip / PyTorch do not need HF_HOME; download scripts set it themselves.

if exist "%~dp0..\.venv-gpu\Scripts\python.exe" (
  set "PY=%~dp0..\.venv-gpu\Scripts\python.exe"
) else if exist "%~dp0.venv310\Scripts\python.exe" (
  set "PY=%~dp0.venv310\Scripts\python.exe"
) else (
  set "PY=python"
)

echo Using: %PY%

set "TORCH_INDEX=https://download.pytorch.org/whl/cu124"
echo [1/2] PyTorch GPU (cu124): torch torchvision torchaudio
echo     Edit TORCH_INDEX in this bat for other CUDA builds. See https://pytorch.org
"%PY%" -m pip install torch torchvision torchaudio --index-url "%TORCH_INDEX%"
if errorlevel 1 (
  echo PyTorch GPU install failed.
  exit /b 1
)

echo.
echo [2/2] pip install -r requirements.txt
"%PY%" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 (
  echo pip install failed.
  exit /b 1
)
echo OK. Done.
exit /b 0
