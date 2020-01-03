# Converter Python API guide

This page provides examples on how to use the
[TensorFlow Lite converter](index.md) using the Python API.

Note: This only contains documentation on the Python API in TensorFlow 2.
Documentation on using the Python API in TensorFlow 1 is available on
[GitHub](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/r1/convert/python_api.md).

[TOC]

## Python API

The Python API for converting TensorFlow models to TensorFlow Lite is
`tf.lite.TFLiteConverter`. `TFLiteConverter` provides the following classmethods
to convert a model based on the original model format:

*   `TFLiteConverter.from_saved_model()`: Converts
    [SavedModel directories](https://www.tensorflow.org/guide/saved_model).
*   `TFLiteConverter.from_keras_model()`: Converts
    [`tf.keras` models](https://www.tensorflow.org/guide/keras/overview).
*   `TFLiteConverter.from_concrete_functions()`: Converts
    [concrete functions](https://tensorflow.org/guide/concrete_function).

This document contains [example usages](#examples) of the API, a detailed list
of [changes in the API between Tensorflow 1 and TensorFlow 2](#differences), and
[instructions](#versioning) on running the different versions of TensorFlow.

## Examples <a name="examples"></a>

### Converting a SavedModel <a name="saved_model"></a>

The following example shows how to convert a
[SavedModel](https://www.tensorflow.org/guide/saved_model) into a
TensorFlow Lite [`FlatBuffer`](https://google.github.io/flatbuffers/).

```python
import tensorflow as tf

# Construct a basic model.
root = tf.train.Checkpoint()
root.v1 = tf.Variable(3.)
root.v2 = tf.Variable(2.)
root.f = tf.function(lambda x: root.v1 * root.v2 * x)

# Save the model.
export_dir = "/tmp/test_saved_model"
input_data = tf.constant(1., shape=[1, 1])
to_save = root.f.get_concrete_function(input_data)
tf.saved_model.save(root, export_dir, to_save)

# Convert the model.
converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
tflite_model = converter.convert()
```

This API does not have the option of specifying the input shape of any input
arrays. If your model requires specifying the input shape, use the
[`from_concrete_functions`](#concrete_function) classmethod instead. The code
looks similar to the following:

```python
model = tf.saved_model.load(export_dir)
concrete_func = model.signatures[
  tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
concrete_func.inputs[0].set_shape([1, 256, 256, 3])
converter = TFLiteConverter.from_concrete_functions([concrete_func])
```

### Converting a Keras model <a name="keras"></a>

The following example shows how to convert a
[`tf.keras` model](https://www.tensorflow.org/guide/keras/overview) into a
TensorFlow Lite [`FlatBuffer`](https://google.github.io/flatbuffers/).

```python
import tensorflow as tf

# Create a simple Keras model.
x = [-1, 0, 1, 2, 3, 4]
y = [-3, -1, 1, 3, 5, 7]

model = tf.keras.models.Sequential(
    [tf.keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(x, y, epochs=50)

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
```

### Converting a concrete function <a name="concrete_function"></a>

The following example shows how to convert a TensorFlow
[concrete function](https://tensorflow.org/guide/concrete_function) into a
TensorFlow Lite [`FlatBuffer`](https://google.github.io/flatbuffers/).

```python
import tensorflow as tf

# Construct a basic model.
root = tf.train.Checkpoint()
root.v1 = tf.Variable(3.)
root.v2 = tf.Variable(2.)
root.f = tf.function(lambda x: root.v1 * root.v2 * x)

# Create the concrete function.
input_data = tf.constant(1., shape=[1, 1])
concrete_func = root.f.get_concrete_function(input_data)

# Convert the model.
#
# `from_concrete_function` takes in a list of concrete functions, however,
# currently only supports converting one function at a time. Converting multiple
# functions is under development.
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()
```

### End-to-end MobileNet conversion <a name="mobilenet"></a>

The following example shows how to convert and run inference on a pre-trained
`tf.keras` MobileNet model to TensorFlow Lite. It compares the results of the
TensorFlow and TensorFlow Lite model on random data. In order to load the model
from file, use `model_path` instead of `model_content`.

```python
import numpy as np
import tensorflow as tf

# Load the MobileNet tf.keras model.
model = tf.keras.applications.MobileNetV2(
    weights="imagenet", input_shape=(224, 224, 3))

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the TensorFlow Lite model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
tflite_results = interpreter.get_tensor(output_details[0]['index'])

# Test the TensorFlow model on random input data.
tf_results = model(tf.constant(input_data))

# Compare the result.
for tf_result, tflite_result in zip(tf_results, tflite_results):
  np.testing.assert_almost_equal(tf_result, tflite_result, decimal=5)
```

## Summary of changes in Python API between 1.X and 2.0 <a name="differences"></a>

The following section summarizes the changes in the Python API from 1.X to 2.0.
If any of the changes raise concerns, please file a
[GitHub issue](https://github.com/tensorflow/tensorflow/issues).

### Formats supported by `TFLiteConverter`

`TFLiteConverter` in 2.0 supports SavedModels and Keras model files generated in
both 1.X and 2.0. However, the conversion process no longer supports frozen
`GraphDefs` generated in 1.X. Users who want to convert frozen `GraphDefs` to
TensorFlow Lite should use `tf.compat.v1.lite.TFLiteConverter`.

### Quantization-aware training

The following attributes and methods associated with
[quantization-aware training](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/quantize)
have been removed from `TFLiteConverter` in TensorFlow 2.0:

*   `inference_type`
*   `inference_input_type`
*   `quantized_input_stats`
*   `default_ranges_stats`
*   `reorder_across_fake_quant`
*   `change_concat_input_ranges`
*   `post_training_quantize` - Deprecated in the 1.X API
*   `get_input_arrays()`

The rewriter function that supports quantization-aware training does not support
models generated by TensorFlow 2.0. Additionally, TensorFlow Lite’s quantization
API is being reworked and streamlined in a direction that supports
quantization-aware training through the Keras API. These attributes will be
removed in the 2.0 API until the new quantization API is launched. Users who
want to convert models generated by the rewriter function can use
`tf.compat.v1.lite.TFLiteConverter`.

### Changes to `TFLiteConverter` attributes

The `target_ops` attribute has become an attribute of `TargetSpec` and renamed
to `supported_ops` in line with future additions to the optimization framework.

Additionally, the following attributes have been removed:

*   `drop_control_dependency` (default: `True`) - Control flow is currently not
    supported by TFLite so it is always `True`.
*   _Graph visualization_ - The recommended approach for visualizing a
    TensorFlow Lite graph in TensorFlow 2.0 will be to use
    [visualize.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/visualize.py).
    Unlike GraphViz, it enables users to visualize the graph after post training
    quantization has occurred. The following attributes related to graph
    visualization will be removed:
    *   `output_format`
    *   `dump_graphviz_dir`
    *   `dump_graphviz_video`

### General API changes

#### Conversion methods

The following methods that were previously deprecated in 1.X will no longer be
exported in 2.0:

*   `lite.toco_convert`
*   `lite.TocoConverter`

#### `lite.constants`

The `lite.constants` API was removed in 2.0 in order to decrease duplication
between TensorFlow and TensorFlow Lite. The following list maps the
`lite.constant` type to the TensorFlow type:

*   `lite.constants.FLOAT`: `tf.float32`
*   `lite.constants.INT8`: `tf.int8`
*   `lite.constants.INT32`: `tf.int32`
*   `lite.constants.INT64`: `tf.int64`
*   `lite.constants.STRING`: `tf.string`
*   `lite.constants.QUANTIZED_UINT8`: `tf.uint8`

Additionally, `lite.constants.TFLITE` and `lite.constants.GRAPHVIZ_DOT` were
removed due to the deprecation of the `output_format` flag in `TFLiteConverter`.

#### `lite.OpHint`

The `OpHint` API is currently not available in 2.0 due to an incompatibility
with the 2.0 APIs. This API enables conversion of LSTM based models. Support for
LSTMs in 2.0 is being investigated. All related `lite.experimental` APIs have
been removed due to this issue.

## Installing TensorFlow <a name="versioning"></a>

### Installing the TensorFlow nightly <a name="2.0-nightly"></a>

The TensorFlow nightly can be installed using the following command:

```
pip install tf-nightly
```

### Build from source code <a name="latest_package"></a>

In order to run the latest version of the TensorFlow Lite Converter Python API,
either install the nightly build with
[pip](https://www.tensorflow.org/install/pip) (recommended) or
[Docker](https://www.tensorflow.org/install/docker), or
[build the pip package from source](https://www.tensorflow.org/install/source).
