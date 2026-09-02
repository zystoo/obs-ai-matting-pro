# AI Matting Pro - OBS Background Removal Plugin

AI-powered background removal for OBS Studio using Robust Video Matting (RVM). No green screen required. Alpha matting for hair-level edge quality with temporal stability.

## Features

- **Alpha Matting** - Soft, natural edges with hair-level detail (not binary segmentation)
- **No Green Screen** - Works with any background
- **Temporal Stability** - RVM recurrent states eliminate frame-to-frame flicker
- **Cross-Platform** - Windows, macOS, and Linux
- **Auto Light Match** - Automatically adjusts foreground lighting to match background
- **Multiple Modes** - Transparent, blurred, or solid color background
- **Local Processing** - 100% offline, no cloud API, no data sent anywhere

## Installation

### Download Pre-built Binary (Recommended)

1. Go to [Releases](../../releases)
2. Download the package for your platform:
   - **Windows**: `.zip` file
   - **macOS**: `.tar.gz` file
   - **Linux**: `.deb` file or `.tar.gz` file
3. Install:
   - **Windows**: Extract the ZIP and run the installer
   - **macOS**: Extract and copy the `.plugin` bundle to `~/Library/Application Support/obs-studio/plugins/`
   - **Linux**: `sudo dpkg -i *.deb` or extract to `~/.config/obs-studio/plugins/`

### Build from Source

#### Prerequisites

- **CMake** 3.28+
- **C++17 compiler** (MSVC 2022 / Xcode / GCC 11+)
- **OBS Studio** 28+

#### Windows
```bash
git clone --recursive https://github.com/zystoo/obs-ai-matting-pro.git
cd obs-ai-matting-pro
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config RelWithDebInfo
cmake --install build --config RelWithDebInfo --prefix build_prefix
```

#### macOS
```bash
git clone --recursive https://github.com/zystoo/obs-ai-matting-pro.git
cd obs-ai-matting-pro
cmake -S . -B build -G Xcode \
  -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64" \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="13.0"
cmake --build build --config RelWithDebInfo
cmake --install build --config RelWithDebInfo --prefix build_prefix
```

#### Linux
```bash
sudo apt install libobs-dev libonnxruntime-dev libopencv-dev cmake ninja-build
git clone --recursive https://github.com/zystoo/obs-ai-matting-pro.git
cd obs-ai-matting-pro
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build
sudo cmake --install build
```

## Usage

1. In OBS Studio, right-click your camera source
2. Add **Filters** -> **AI Matting Pro**
3. The plugin will automatically download the RVM model on first use
4. Choose your mode:
   - **Transparent** - Removes background entirely (works with OBS scene compositing)
   - **Blurred** - Blurs the background (portrait mode effect)
   - **Solid Color** - Replaces background with a solid color

### Settings

| Parameter | Description | Default |
|---|---|---|
| Mode | Transparent / Blur / Solid Color | Transparent |
| Alpha Gamma | Adjust alpha curve (lower = tighter edges) | 1.0 |
| Quality | Inference resolution (384/512/720) | 512 |
| Detail Ratio | Downsample ratio for inference | 75% |
| Temporal Smoothing | EMA coefficient for frame-to-frame stability | 0.85 |
| Edge Feather | Gaussian blur radius for edge softening | 3 |
| Auto Light Match | Match foreground lighting to background | Off |

## Model

The plugin uses [Robust Video Matting](https://github.com/PeterL1n/RobustVideoMatting) (RVM) with MobileNetV3 backbone.

- **Model size**: 14 MB (MobileNetV3 FP32)
- **Memory**: ~300 MB (GPU) / ~500 MB (CPU)
- **Speed**: 30+ FPS on NVIDIA GPU, 8-12 FPS on CPU (720p)

## Platform Support

| Platform | GPU Backend | Status |
|---|---|---|
| Windows + NVIDIA | DirectML/CUDA | Supported |
| Windows + AMD | DirectML | Supported |
| macOS (Apple Silicon) | CoreML | Supported |
| macOS (Intel) | CPU | Supported |
| Linux + NVIDIA | CUDA | Supported |
| Linux + AMD/Intel | CPU | Supported |

## Based On

This plugin is based on:
- [obs-ai-matting](https://github.com/ale200x/obs-ai-matting) by ale200x - RVM integration architecture
- [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) by PeterL1n - AI model
- [obs-plugintemplate](https://github.com/obsproject/obs-plugintemplate) - OBS build system

## License

MIT License - see LICENSE file for details.
