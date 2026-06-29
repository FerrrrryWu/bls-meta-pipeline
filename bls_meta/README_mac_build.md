# Mac 版构建说明

## 前提

- macOS 11 (Big Sur) 或更新
- Python 3.9+（推荐 3.11/3.12，避免 3.13 的兼容问题）

没有 Python？安装：
```bash
brew install python@3.12
```

---

## 一键构建

把整个 `bls_meta/` 文件夹复制到 Mac 上，然后在终端运行：

```bash
cd /path/to/bls_meta
bash build_mac.sh
```

脚本会自动：
1. 创建虚拟环境 `venv_mac/`
2. 安装所有依赖（pandas / scipy / sklearn / matplotlib 等）
3. 用 PyInstaller 打包成 `dist/BLS_Meta_Pipeline.app`

首次运行约需 5-10 分钟（依赖下载）。

---

## 输出

```
bls_meta/
└── dist/
    └── BLS_Meta_Pipeline.app   ← 双击即可运行
```

打包分发给同事：

```bash
zip -r BLS_Meta_Pipeline_mac.zip dist/BLS_Meta_Pipeline.app
```

---

## 常见问题

**"无法打开，因为它来自身份不明的开发者"**

右键点击 `.app` → 选「打开」，然后在弹出对话框里点「打开」。
或者终端运行：
```bash
xattr -dr com.apple.quarantine dist/BLS_Meta_Pipeline.app
```

**M1/M2/M4 芯片（Apple Silicon）**

`build_mac.sh` 自动检测架构，直接打出 ARM 原生包，无需额外设置。
如果需要同时兼容 Intel + Apple Silicon（Universal Binary）：
```bash
pyinstaller BLS_Meta_Pipeline_mac.spec --noconfirm \
    --target-arch universal2
```
注意：Universal Binary 需要在 Apple Silicon Mac 上构建，且部分依赖需要 universal2 版本。

**Python 版本问题**

推荐 Python 3.11 或 3.12。Python 3.13 与部分科学计算库可能有兼容问题。

```bash
brew install python@3.12
/opt/homebrew/bin/python3.12 -m venv venv_mac
```
然后修改 `build_mac.sh` 第一行的 `PYTHON` 路径。
