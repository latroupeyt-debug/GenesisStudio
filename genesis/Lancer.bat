@echo off
REM Lance Genesis Studio (l'environnement .venv doit deja exister).
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" run.py
) else (
    start "" pythonw run.py
)
