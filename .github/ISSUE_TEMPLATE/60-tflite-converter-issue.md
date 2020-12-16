---
name: TensorFlow Lite New Converter Issue
about: Use this template for reporting issues during model conversion to TFLite
labels: 'TFLiteConverter'

---

### 1. System information

- OS Platform and Distribution (e.g., Linux Ubuntu 16.04):
- TensorFlow installed from (source or binary):
- TensorFlow version (or github SHA if from source):

### 2. Code (that demonstrates how to reproduce your issue)
**Option A: Using reference colab notebooks**

The reference colab notebooks given below demonstrate: *(TensorFlow Model Colab)* Build model in TF Keras --> *(TensorFlow Lite Model Colab)* Convert to TF Lite (performing quantization techniques) and run TFLite Inference. You may use these colab notebooks as a reference point to generate your model behavior and attach links.

1)  Reference [TensorFlow Model Colab](https://colab.research.google.com/gist/ymodak/e96a4270b953201d5362c61c1e8b78aa/tensorflow-datasets.ipynb?authuser=1)
2)  Reference [TensorFlow Lite Model Colab](https://colab.research.google.com/gist/ymodak/0dfeb28255e189c5c48d9093f296e9a8/tensorflow-lite-debugger-colab.ipynb)

```
# Put links here or attach to the issue
```

**Option B: Paste your code here or provide a link to your custom end-to-end colab**

```
# Provide your code here (or put links here or attach to the issue)
# - Include code to invoke the TFLite Converter Python API
# - Include the output (with errors) from the converter invocation 
```

### 3. Models

Provide links to your TensorFlow model and (optionally) TensorFlow Lite Model

```
# Put links here or attach to the issue
```

### 4. Failure details

If the conversion is successful, but the generated model is wrong, then state what is wrong:

- Producing wrong results and/or decrease in accuracy
- Producing correct results, but the model is slower than expected (model generated from old converter)

### 5. (optional) RNN conversion support
If converting TF RNN to TFLite fused RNN ops, please prefix [RNN] in the title.

### 6. (optional) Any other info / logs

Include any logs or source code that would be helpful to diagnose the problem. If including tracebacks, please include the full traceback. Large logs and files should be attached.
