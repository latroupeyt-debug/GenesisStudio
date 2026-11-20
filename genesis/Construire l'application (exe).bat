@echo off
REM ============================================================
REM  Cree un executable Windows autonome dans dist\Genesis Studio
REM  Prerequis : Python 3.10+ installe sur la machine.
REM ============================================================
setlocal
cd /d "%~dp0"

where py >nul 2>&1
if %errorlevel%==0 ( set "PY=py -3" ) else ( set "PY=python" )

if not exist ".venv\Scripts\python.exe" (
    %PY% -m venv .venv || goto :error
)

echo Installation des dependances et de PyInstaller...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
".venv\Scripts\python.exe" -m pip install pyinstaller || goto :error

echo Compilation (plusieurs minutes, ~2 Go de fichiers)...
".venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean genesis.spec || goto :error

echo.
echo Termine !
echo Executable : dist\Genesis Studio\Genesis Studio.exe
echo Pour un raccourci : clic droit sur l'exe -^> Envoyer vers -^> Bureau.
pause
exit /b 0

:error
echo.
echo La compilation a echoue. Voir les messages ci-dessus.
pause
exit /b 1
