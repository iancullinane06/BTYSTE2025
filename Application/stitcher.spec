# -*- mode: python ; coding: utf-8 -*-

import pkgutil
import rasterio

# Dynamically include rasterio submodules
additional_packages = [package.name for package in pkgutil.iter_modules(rasterio.__path__, prefix="rasterio.")]

a = Analysis(
    ['Prototypes\\Rhododendron\\DeepLabV3+\\stitcher.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=additional_packages + ['rasterio.sample'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='stitcher',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['C:\\Users\\rough\\OneDrive\\Desktop\\Coding\\BTYSTE2025\\Application\\EcoLytix-Icon.ico'],
)
