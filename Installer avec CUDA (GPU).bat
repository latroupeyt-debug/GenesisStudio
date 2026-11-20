@echo off
REM ============================================================
REM  Genesis Studio - installation AVEC support CUDA (GPU NVIDIA)
REM  Installe PyTorch compile pour CUDA afin que l'entrainement
REM  utilise votre carte graphique. Comptez ~2,5 Go de telechargement.
REM ============================================================
setlocal
cd /d "%~dp0"

where py >nul 2>&1
if %errorlevel%==0 ( set "PY=py -3" ) else ( set "PY=python" )

if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Creation de l'environnement Python...
    %PY% -m venv .venv || goto :error
)

echo [2/4] Verification du pilote NVIDIA...
where nvidia-smi >nul 2>&1
if %errorlevel%==0 (
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
) else (
    echo    nvidia-smi introuvable : pilote NVIDIA non installe ?
    echo    Installez le pilote sur https://www.nvidia.com/drivers avant de continuer.
)

echo [3/4] Installation de PyTorch avec CUDA (cu126, puis cu124, puis cu121)...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install "PySide6>=6.6" "numpy>=1.24"
".venv\Scripts\python.exe" -m pip uninstall -y torch
".venv\Scripts\python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cu126 || ^
".venv\Scripts\python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cu124 || ^
".venv\Scripts\python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cu121 || goto :error

echo [4/4] Verification...
".venv\Scripts\python.exe" -c "import torch;print('PyTorch', torch.__version__);print('CUDA disponible :', torch.cuda.is_available());print('GPU :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'aucun')"

echo.
echo Termine. Lancez l'application avec Lancer.bat
echo Dans l'application : Reglages pour verifier le GPU, puis Entrainement -^> Appareil = GPU.
pause
exit /b 0

:error
echo.
echo L'installation CUDA a echoue.
echo Verifiez votre version de pilote NVIDIA puis reessayez.
pause
exit /b 1
