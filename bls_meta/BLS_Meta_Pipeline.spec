# -*- mode: python ; coding: utf-8 -*-
# BLS Meta-Analysis Pipeline — PyInstaller spec
# Build: pyinstaller BLS_Meta_Pipeline.spec
# Output: dist/BLS_Meta_Pipeline/  (onedir — copy the entire folder)

a = Analysis(
    ['run_pipeline.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('bls_meta_pipeline.py',  '.'),
        ('config_ui.py',          '.'),
        ('report_generator.py',   '.'),
    ],
    hiddenimports=[
        # sklearn
        'sklearn.utils._cython_blas',
        'sklearn.neighbors._partition_nodes',
        'sklearn.tree._utils',
        'sklearn.tree._criterion',
        'sklearn.tree._splitter',
        'sklearn.ensemble._forest',
        'sklearn.ensemble._gb_losses',
        # matplotlib
        'matplotlib.backends.backend_agg',
        'matplotlib.backends.backend_tkagg',
        'matplotlib.backends._backend_tk',
        # scipy
        'scipy.special.cython_special',
        'scipy._lib.messagestream',
        'scipy.stats._stats',
        # statsmodels
        'statsmodels.stats.multitest',
        'statsmodels.stats.multicomp',
        'statsmodels.stats.power',
        'statsmodels.compat',
        # PIL
        'PIL',
        'PIL.Image',
        'PIL.ImageTk',
        # others
        'openpyxl',
        'yaml',
        'webbrowser',
        'urllib.request',
        'base64',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['jupyter', 'notebook', 'ipywidgets', 'IPython'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,          # onedir: DLLs go in the folder
    name='BLS_Meta_Pipeline',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    console=False,                  # no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='BLS_Meta_Pipeline',       # → dist/BLS_Meta_Pipeline/
)
