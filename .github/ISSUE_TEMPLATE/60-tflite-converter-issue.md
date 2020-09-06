---
name: TensorFlow Lite New Converter Issue
about: Use this template for reporting issues during model conversion to TFLite
labels: 'TFLiteConverter'

---


**System information**
- OS Platform and Distribution (e.g., Linux Ubuntu 16.04):
- TensorFlow installed from (source or binary):
- TensorFlow version (or github SHA if from source):


**Command used to run the converter or code if you’re using the Python API**
If possible, please share a link to Colab/Jupyter/any notebook.

```
# Copy and paste here the exact command
```

**The output from the converter invocation**

```
# Copy and paste the output here.
```

**Also, please include a link to the saved model or GraphDef**

```
# Put link here or attach to the issue.
```

**Failure details**
If the conversion is successful, but the generated model is wrong,
state what is wrong:
- Producing wrong results and/or decrease in accuracy
- Producing correct results, but the model is slower than expected (model generated from old converter)


**RNN conversion support**
If converting TF RNN to TFLite fused RNN ops, please prefix [RNN] in the title.

**Any other info / logs**

Include any logs or source code that would be helpful to diagnose the problem. If including tracebacks, please include the full traceback. Large logs and files should be attached.
