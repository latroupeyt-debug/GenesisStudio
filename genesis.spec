# -*- mode: python ; coding: utf-8 -*-
"""Recette PyInstaller pour Genesis Studio.

    pip install pyinstaller
    pyinstaller genesis.spec

Le résultat apparaît dans dist/Genesis Studio/ (mode « dossier », démarrage rapide).
PyTorch est volumineux : comptez 1,5 à 2,5 Go pour l'application empaquetée.
Pour réduire la taille, installez la version CPU de PyTorch avant de compiler :

    pip install torch --index-url https://download.pytorch.org/whl/cpu
"""

from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

a = Analysis(
    ["run.py"],
    pathex=[],
    binaries=[],
    datas=[("assets", "assets")],
    hiddenimports=[
        "torch",
        "numpy",
        "PySide6.QtSvg",
        "torch.testing",
    ] + collect_submodules("torch.distributed") + collect_submodules("torch.testing"),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "IPython",
        "pytest",
        "torch.utils.tensorboard",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Genesis Studio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    icon="assets/genesis.ico",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Genesis Studio",
)
