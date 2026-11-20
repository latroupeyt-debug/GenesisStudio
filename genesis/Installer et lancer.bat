@echo off
REM ============================================================
REM  Genesis Studio - installation puis lancement
REM  Cree un environnement Python local (.venv) et installe les
REM  dependances, puis demarre l'application.
REM ============================================================
setlocal
cd /d "%~dp0"

where py >nul 2>&1
if %errorlevel%==0 (
    set "PY=py -3"
) else (
    set "PY=python"
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creation de l'environnement Python...
    %PY% -m venv .venv || goto :error
)

echo [2/3] Installation des dependances (PySide6, PyTorch, numpy)...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :error

echo [3/3] Lancement de Genesis Studio...
".venv\Scripts\pythonw.exe" run.py
exit /b 0

:error
echo.
echo Une erreur est survenue pendant l'installation.
echo Verifiez que Python 3.10 ou plus recent est installe (python.org).
pause
exit /b 1
