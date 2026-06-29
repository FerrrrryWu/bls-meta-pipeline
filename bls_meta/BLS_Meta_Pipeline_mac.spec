# -*- mode: python ; coding: utf-8 -*-
# BLS Meta-Analysis Pipeline — PyInstaller spec (macOS)
# Build: pyinstaller BLS_Meta_Pipeline_mac.spec
# Output: dist/BLS_Meta_Pipeline.app

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
    exclude_binaries=True,
    name='BLS_Meta_Pipeline',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                    # UPX not reliable on macOS
    console=False,                # no terminal window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,             # None = match current machine arch
    codesign_identity=None,       # set to your Apple dev cert if distributing
    entitlements_file=None,
    icon=None,                    # replace with 'icon.icns' if you have one
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='BLS_Meta_Pipeline',
)

app = BUNDLE(
    coll,
    name='BLS_Meta_Pipeline.app',
    icon=None,                    # replace with 'icon.icns' if you have one
    bundle_identifier='com.bls.meta.pipeline',
    info_plist={
        'CFBundleShortVersionString': '1.0.0',
        'CFBundleDisplayName': 'BLS Meta Pipeline',
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '11.0',   # macOS Big Sur+
    },
)
