# What is an External Delegate?
An external delegate is a special Tensorflow Lite delegate that is simply
initialized from loading a dynamic library which encapsulates an actual
Tensorflow Lite delegate implementation. The actual delegate exposes the
following two creation and deletion C APIs:

* __tflite_plugin_create_delegate__ (declaration seen below) creates a delegate
object based on provided key-value options. It may return NULL to indicate an
error with the detailed information reported by calling `report_error` if
provided. Each option key and value should be null-terminated.

```
TfLiteDelegate* tflite_plugin_create_delegate(
  char** options_keys, char** options_values, size_t num_options,
  void (*report_error)(const char *))
```

* __tflite_plugin_destroy_delegate__ (declaration seen below) destroys the
delegate object that is created by the previous API. NULL as an argument value
is allowed.

```
void tflite_plugin_destroy_delegate(TfLiteDelegate* delegate)
```

The external delegate provides an opague and transparent way to utilize a
Tensorflow Lite delegate when performing inference. In other words, one may
replace the actual Tensorflow Lite delegate by simply updating the dynamic
library without changing the application code. We developed this mainly for
delegate evaluation.

Note, this delegate is the corresponding C++ implementation to the one for
Tensorflow Lite Python binding as shown [here](https://github.com/tensorflow/tensorflow/blob/7145fc0e49be01ef6943f4df386ce38567e37797/tensorflow/lite/python/interpreter.py#L42).
