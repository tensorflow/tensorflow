---
name: TensorFlow Lite Converter Issue
about: Use this template for reporting issues during model conversion to TFLite
labels: 'TFLiteConverter'

---

### 1. System information

- OS Platform and Distribution (e.g., Linux Ubuntu 16.04):
- TensorFlow installation (pip package or built from source):
- TensorFlow library (version, if pip package or github SHA, if built from source):

### 2. Code

Provide code to help us reproduce your issues using one of the following options:

#### Option A: Reference colab notebooks

1)  Reference [TensorFlow Model Colab](https://colab.research.google.com/gist/ymodak/e96a4270b953201d5362c61c1e8b78aa/tensorflow-datasets.ipynb?authuser=1): Demonstrate how to build your TF model.
2)  Reference [TensorFlow Lite Model Colab](https://colab.research.google.com/gist/ymodak/0dfeb28255e189c5c48d9093f296e9a8/tensorflow-lite-debugger-colab.ipynb): Demonstrate how to convert your TF model to a TF Lite model (with quantization, if used) and run TFLite Inference (if possible).

```
(You can paste links or attach files by dragging & dropping them below)
- Provide links to your updated versions of the above two colab notebooks.
- Provide links to your TensorFlow model and (optionally) TensorFlow Lite Model.
```

#### Option B: Paste your code here or provide a link to a custom end-to-end colab

```
(You can paste links or attach files by dragging & dropping them below)
- Include code to invoke the TFLite Converter Python API and the errors.
- Provide links to your TensorFlow model and (optionally) TensorFlow Lite Model.
```

### 3. Failure after conversion
If the conversion is successful, but the generated model is wrong, then state what is wrong:

- Model produces wrong results and/or has lesser accuracy.
- Model produces correct results, but it is slower than expected.

### 4. (optional) RNN conversion support
If converting TF RNN to TFLite fused RNN ops, please prefix [RNN] in the title.

### 5. (optional) Any other info / logs
Include any logs or source code that would be helpful to diagnose the problem. If including tracebacks, please include the full traceback. Large logs and files should be attached.
