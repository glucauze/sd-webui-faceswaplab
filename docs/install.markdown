---
layout: page
title: Install
permalink: /install/
---

## Requirements/Recommanded configuration

The extension runs mainly on the CPU to avoid the use of VRAM. However, it is recommended to follow the specifications recommended by sd/a1111 with regard to prerequisites. At the time of writing, a version of python lower than 11 is preferable (even if it works with python 3.11, model loading and performance may fall short of expectations).

### Windows-User : Visual Studio ! Don't neglect this !

Before beginning the installation process, if you are using Windows, you need to install this requirement:

1. Install Visual Studio 2022: This step is required to build some of the dependencies. You can use the Community version of Visual Studio 2022, which can be downloaded from the following link: https://visualstudio.microsoft.com/downloads/

2. OR Install only the VS C++ Build Tools: If you don't need the full Visual Studio suite, you can choose to install only the VS C++ Build Tools. During the installation process, select the option for "Desktop Development with C++" found under the "Workloads -> Desktop & Mobile" section. The VS C++ Build Tools can be downloaded from this link: https://visualstudio.microsoft.com/visual-cpp-build-tools/

3. OR if you don't want to install either the full Visual Studio suite or the VS C++ Build Tools: Follow the instructions provided in section VIII of the documentation.

## Manual Install

To install the extension, follow the steps below:

1. Open the `web-ui` application and navigate to the "Extensions" tab.
2. Use the URL `https://github.com/glucauze/sd-webui-faceswaplab` in the "install from URL" section.
3. Close the `web-ui` application and reopen it.

![](/assets/images/install_from_url.png)


**You may need to restart sd once the installation process is complete.**

On first launch, templates are downloaded, which may take some time. All models are located in the `models/faceswaplab` folder.

If you encounter the error `'NoneType' object has no attribute 'get'`, take the following steps:

1. Download the [inswapper_128.onnx](https://huggingface.co/henryruhs/faceswaplab/resolve/main/inswapper_128.onnx) model.
2. Place the downloaded model inside the `<webui_dir>/models/faceswaplab/` directory.

## Usage

To use this extension, follow the steps below:

1. Navigate to the "faceswaplab" drop-down menu and import an image that contains a face.
2. Enable the extension by checking the "Enable" checkbox.
3. After performing the steps above, the generated result will have the face you selected.
