---
comments: true
description: Export any PyTorch model to ONNX, TorchScript, OpenVINO, CoreML, NCNN, PaddlePaddle, MNN, ExecuTorch, and TensorFlow SavedModel using Ultralytics standalone export functions. Works with timm, torchvision, and custom models.
keywords: export PyTorch model, non-YOLO export, timm export, torchvision export, ONNX export, TorchScript export, OpenVINO export, CoreML export, NCNN export, PaddlePaddle export, MNN export, TensorFlow export, TFLite export, ExecuTorch export, Ultralytics export, model conversion, model deployment
---

# How to Export Non-YOLO Models with Ultralytics

Ultralytics provides standalone export functions that convert any `torch.nn.Module` to popular deployment formats. This means you can export [timm](https://github.com/huggingface/pytorch-image-models) models, [torchvision](https://pytorch.org/vision/) models, or your own custom architectures to ONNX, TorchScript, OpenVINO, CoreML, NCNN, PaddlePaddle, MNN, ExecuTorch, and TensorFlow SavedModel without writing format-specific code for each target.

## Supported Export Formats

All functions accept a standard `torch.nn.Module` and an example input tensor. No YOLO-specific attributes are required.

| Format | Function | Install | Output |
| --------------- | ---------------------- | --------------------------------------------------- | ---------------------- |
| ONNX | `torch2onnx()` | `pip install onnx` | `.onnx` file |
| TorchScript | `torch2torchscript()` | included with PyTorch | `.torchscript` file |
| OpenVINO | `torch2openvino()` | `pip install openvino` | `_openvino_model/` directory |
| CoreML | `torch2coreml()` | `pip install coremltools` | `.mlpackage` |
| TF SavedModel | `onnx2saved_model()` | `pip install onnx2tf tensorflow tf_keras sng4onnx` | `_saved_model/` directory |
| TF Frozen Graph | `keras2pb()` | same as TF SavedModel | `.pb` file |
| NCNN | `torch2ncnn()` | `pip install ncnn pnnx` | `_ncnn_model/` directory |
| MNN | `onnx2mnn()` | `pip install MNN` | `.mnn` file |
| PaddlePaddle | `torch2paddle()` | `pip install paddlepaddle x2paddle` | `_paddle_model/` directory |
| ExecuTorch | `torch2executorch()` | `pip install executorch` | `_executorch_model/` directory |

!!! note

    MNN, TF SavedModel, and TF Frozen Graph exports go through ONNX as an intermediate step. Export to ONNX first, then convert.

## Step-by-Step Examples

Every example below uses the same setup:

```python
import torch
import timm

model = timm.create_model("resnet18", pretrained=True).eval()
im = torch.randn(1, 3, 224, 224)
```

### Export to ONNX

```python
from ultralytics.utils.export import torch2onnx

torch2onnx(model, im, output_file="resnet18.onnx")
```

For dynamic batch size, pass a `dynamic` dictionary:

```python
torch2onnx(model, im, output_file="resnet18_dyn.onnx", dynamic={"images": {0: "batch_size"}})
```

The default opset is `14` and the default input name is `"images"`. Override with the `opset`, `input_names`, or `output_names` arguments.

### Export to TorchScript

No extra dependencies needed. Uses `torch.jit.trace` under the hood.

```python
from ultralytics.utils.export import torch2torchscript

torch2torchscript(model, im, output_file="resnet18.torchscript")
```

### Export to OpenVINO

```python
from ultralytics.utils.export import torch2openvino

ov_model = torch2openvino(model, im, output_dir="resnet18_openvino_model")
```

The directory contains a fixed-name `model.xml` and `model.bin` pair. OpenVINO names the inputs after your model's `forward` argument names (typically `x` for generic models). Supports `half=True` for FP16 and `int8=True` for INT8 quantization (INT8 also requires a `calibration_dataset`). Requires `openvino>=2024.0.0` (or `>=2025.2.0` on macOS 15.4+) and `torch>=2.1`.

### Export to CoreML

```python
import coremltools as ct
from ultralytics.utils.export import torch2coreml

inputs = [ct.TensorType("input", shape=(1, 3, 224, 224))]
ct_model = torch2coreml(model, inputs, im, classifier_names=None, output_file="resnet18.mlpackage")
```

For [classification](https://www.ultralytics.com/glossary/image-classification) models, pass a list of class names to `classifier_names` to add a classification head to the CoreML model. Requires `coremltools>=9.0`, `torch>=1.11`, and `numpy<=2.3.5`. Not supported on Windows. A `BlobWriter not loaded` error at import time usually means `coremltools` has no wheel for your Python version — use Python 3.10–3.13.

### Export to TensorFlow SavedModel

TF SavedModel export goes through ONNX as an intermediate step:

```python
from ultralytics.utils.export import torch2onnx, onnx2saved_model

torch2onnx(model, im, output_file="resnet18.onnx")
keras_model = onnx2saved_model("resnet18.onnx", output_dir="resnet18_saved_model")
```

The function returns a Keras model and also generates TFLite files (`.tflite`) inside `resnet18_saved_model/`. Requires `tensorflow>=2.0.0,<=2.19.0`, `onnx2tf>=1.26.3,<1.29.0`, `tf_keras<=2.19.0`, `sng4onnx>=1.0.1`, `onnx_graphsurgeon>=0.3.26` (install with `--extra-index-url https://pypi.ngc.nvidia.com`), `ai-edge-litert>=1.2.0` (`,<1.4.0` on macOS), `onnxslim>=0.1.71`, `onnx>=1.12.0,<2.0.0`, and `protobuf>=5`.

### Export to TensorFlow Frozen Graph

Building on the TF SavedModel export, you can create a frozen graph:

```python
from pathlib import Path
from ultralytics.utils.export import torch2onnx, onnx2saved_model, keras2pb

torch2onnx(model, im, output_file="resnet18.onnx")
keras_model = onnx2saved_model("resnet18.onnx", output_dir="resnet18_saved_model")
keras2pb(keras_model, output_file=Path("resnet18_saved_model/resnet18.pb"))
```

### Export to NCNN

```python
from ultralytics.utils.export import torch2ncnn

torch2ncnn(model, im, output_dir="resnet18_ncnn_model", device=torch.device("cpu"))
```

The directory contains fixed-name `model.ncnn.param` and `model.ncnn.bin` files along with a `model_ncnn.py` wrapper. Dependencies `ncnn` and `pnnx` are installed automatically on first use.

### Export to MNN

MNN export requires an ONNX file as input. Export to ONNX first, then convert:

```python
from ultralytics.utils.export import torch2onnx, onnx2mnn

torch2onnx(model, im, output_file="resnet18.onnx")
onnx2mnn("resnet18.onnx", output_file="resnet18.mnn")
```

Supports `half=True` for FP16 and `int8=True` for INT8 quantization. Requires `MNN>=2.9.6` and `torch>=1.10`.

### Export to PaddlePaddle

```python
from ultralytics.utils.export import torch2paddle

torch2paddle(model, im, output_dir="resnet18_paddle_model")
```

Requires `x2paddle` and the correct PaddlePaddle distribution for your platform: `paddlepaddle-gpu>=3.0.0,<3.3.0` on CUDA, `paddlepaddle==3.0.0` on ARM64 CPU, or `paddlepaddle>=3.0.0,<3.3.0` on other CPUs. Not supported on NVIDIA Jetson.

### Export to ExecuTorch

```python
from ultralytics.utils.export import torch2executorch

torch2executorch(model, im, output_dir="resnet18_executorch_model")
```

The exported `model.pte` file is saved inside `resnet18_executorch_model/`. Requires `torch>=2.9.0` and a matching [ExecuTorch runtime](https://pytorch.org/executorch/) (`pip install executorch`).

## Verify Your Exported Model

After exporting, verify the model produces correct outputs by comparing against the original PyTorch model:

```python
import numpy as np
import onnxruntime as ort
import torch
import timm

model = timm.create_model("resnet18", pretrained=True).eval()
im = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    pytorch_output = model(im).numpy()

session = ort.InferenceSession("resnet18.onnx")
onnx_output = session.run(None, {"images": im.numpy()})[0]

diff = np.abs(pytorch_output - onnx_output).max()
print(f"Max difference: {diff:.6f}")  # should be < 1e-5
```

For other runtimes, the input name may differ. OpenVINO, for example, uses the model's forward-argument name (typically `x` for generic models), while `torch2onnx` defaults to `"images"`.

## Known Limitations

- **Multi-input support is uneven**: `torch2onnx`, `torch2openvino`, and `torch2torchscript` accept a tuple or list of example tensors for models with multiple inputs. `torch2coreml`, `torch2ncnn`, `torch2paddle`, and `torch2executorch` assume a single input tensor.
- **Eval mode required**: Always call `model.eval()` before exporting.
- **CoreML wheel availability**: `coremltools>=9.0` ships wheels for Python 3.10–3.13. On newer Python versions the C extension fails to load with a `BlobWriter not loaded` error. Use Python 3.10–3.13 for CoreML export.
- **ExecuTorch needs `flatc`**: The ExecuTorch runtime requires the FlatBuffers compiler. Install with `brew install flatbuffers` on macOS or `apt install flatbuffers-compiler` on Ubuntu.
- **No inference via Ultralytics**: Exported non-YOLO models cannot be loaded back through `YOLO()` for inference. Use the native runtime for each format ([ONNX Runtime](../integrations/onnx.md), [OpenVINO Runtime](../integrations/openvino.md), etc.).
- **YOLO-only formats**: [Axelera](../integrations/axelera.md) and [Sony IMX500](../integrations/sony-imx500.md) exports require YOLO-specific model attributes and are not available for generic models.
- **Platform-specific formats**: [TensorRT](../integrations/tensorrt.md) requires an NVIDIA GPU. [RKNN](../integrations/rockchip-rknn.md) requires the `rknn-toolkit2` SDK (Linux only). [Edge TPU](../integrations/edge-tpu.md) requires the `edgetpu_compiler` binary (Linux only).

## FAQ

### What models can I export with Ultralytics?

Any `torch.nn.Module`. This includes models from [timm](https://github.com/huggingface/pytorch-image-models), [torchvision](https://pytorch.org/vision/), or any custom PyTorch model. The model must be in evaluation mode (`model.eval()`) before export. ONNX, OpenVINO, and TorchScript additionally accept a tuple of example tensors for multi-input models.

### Which export formats work without a GPU?

All supported formats (TorchScript, ONNX, OpenVINO, CoreML, TF SavedModel, NCNN, PaddlePaddle, MNN, ExecuTorch) can export on CPU. No GPU is required for the export process itself. TensorRT is the only format that requires an NVIDIA GPU.

### Why does CoreML export fail with BlobWriter error?

The error usually means `coremltools` cannot load its native C extension because no wheel is published for your Python version. `coremltools==9.0` ships wheels for Python 3.10–3.13 on macOS and Linux. Create a Python 3.10–3.13 environment to export CoreML models.

### Can I export models with multiple inputs?

Partially. `torch2onnx`, `torch2openvino`, and `torch2torchscript` accept a tuple or list of example tensors and handle multi-input models correctly. `torch2coreml`, `torch2ncnn`, `torch2paddle`, and `torch2executorch` still assume a single input tensor, so DETR-style models with multiple inputs will need a workaround for those formats.

### How is this different from using torch.onnx.export directly?

The Ultralytics export functions wrap `torch.onnx.export` and similar tools with sensible defaults, automatic dependency checking, and consistent logging. The main advantage is a unified API across ten formats rather than learning each tool's API separately.

### What Ultralytics version do I need?

The standalone export functions are available starting from `ultralytics>=8.4.38` following the [exporter refactor](https://github.com/ultralytics/ultralytics/pull/23914) and the [unified-args update](https://github.com/ultralytics/ultralytics/pull/24120) that standardized the `output_file` and `output_dir` parameters.
