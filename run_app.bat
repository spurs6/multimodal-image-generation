@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem Optional: set PYTHON=C:\path\to\python.exe

if not defined PYTHON (
  if exist "%~dp0..\.venv-gpu\Scripts\python.exe" set "PYTHON=%~dp0..\.venv-gpu\Scripts\python.exe"
)
if not defined PYTHON (
  if exist "%~dp0.venv310\Scripts\python.exe" set "PYTHON=%~dp0.venv310\Scripts\python.exe"
)
if not defined PYTHON (
  where python >nul 2>&1 && set "PYTHON=python"
)
if not defined PYTHON (
  echo Python not found. Set PYTHON= in this file to your python.exe path.
  echo Close this window when done.
  goto stayopen
)

rem Only validate full paths; plain "python" is resolved by PATH
echo %PYTHON%| findstr /I "\\" >nul
if not errorlevel 1 (
  if not exist "%PYTHON%" (
    echo PYTHON path not found: "%PYTHON%"
    echo Edit run_app.bat and set PYTHON= to a valid python.exe
    goto stayopen
  )
)

chcp 65001 >nul 2>&1

echo.
echo === Multimodal API ===
echo Python: "%PYTHON%"
echo URL:    printed below after bind ^(may use 8001+ if 8000 busy^)
echo Browser: opens automatically unless OPEN_BROWSER=0 in .env
echo Stop:   Ctrl+C in this window
echo Logs:   unbuffered ^(-u^) for immediate console output
echo.

set PYTHONUNBUFFERED=1
"%PYTHON%" -u "%~dp0api_server.py"
set "EC=%ERRORLEVEL%"
echo.
echo Server process ended ^(exit code %EC%^).
echo Close this window when finished.

:stayopen
cmd /k
goto stayopen
