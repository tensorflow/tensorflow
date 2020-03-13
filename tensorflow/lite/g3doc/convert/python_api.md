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

This document contains [example usages](#examples) of the API and
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

#### TensorFlow Lite Metadata

Note: TensorFlow Lite Metadata is in experimental (beta) phase.

TensorFlow Lite metadata provides a standard for model descriptions. The
metadata is an important source of knowledge about what the model does and its
input / output information. This makes it easier for other developers to
understand the best practices and for code generators to create platform
specific wrapper code. For more infomation, please refer to the
[TensorFlow Lite Metadata](metadata.md) section.

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

### Custom ops in the experimenal new converter

There is a behavior change in how models containing
[custom ops](https://www.tensorflow.org/lite/guide/ops_custom) (those for which
users use to set allow\_custom\_ops before) are handled in the
[new converter](https://github.com/tensorflow/tensorflow/blob/917ebfe5fc1dfacf8eedcc746b7989bafc9588ef/tensorflow/lite/python/lite.py#L81).

**Built-in TensorFlow op**

If you are converting a model with a built-in TensorFlow op that does not exist
in TensorFlow Lite, you should set allow\_custom\_ops attribute (same as
before), explained [here](https://www.tensorflow.org/lite/guide/ops_custom).

**Custom op in TensorFlow**

If you are converting a model with a custom TensorFlow op, it is recommended
that you write a [TensorFlow kernel](https://www.tensorflow.org/guide/create_op)
and [TensorFlow Lite kernel](https://www.tensorflow.org/lite/guide/ops_custom).
This ensures that the model is working end-to-end, from TensorFlow and
TensorFlow Lite. This also requires setting the allow\_custom\_ops attribute.

**Advanced custom op usage (not recommended)**

If the above is not possible, you can still convert a TensorFlow model
containing a custom op without a corresponding kernel. You will need to pass the
[OpDef](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/op_def.proto)
of the custom op in TensorFlow using --custom\_opdefs flag, as long as you have
the corresponding OpDef registered in the TensorFlow global op registry. This
ensures that the TensorFlow model is valid (i.e. loadable by the TensorFlow
runtime).

If the custom op is not part of the global TensorFlow op registry, then the
corresponding OpDef needs to be specified via the --custom\_opdefs flag. This is
a list of an OpDef proto in string that needs to be additionally registered.
Below is an example of an TFLiteAwesomeCustomOp with 2 inputs, 1 output, and 2
attributes:

```
converter.custom\_opdefs="name: 'TFLiteAwesomeCustomOp' input\_arg: { name: 'InputA'
type: DT\_FLOAT } input\_arg: { name: ‘InputB' type: DT\_FLOAT }
output\_arg: { name: 'Output' type: DT\_FLOAT } attr : { name: 'Attr1' type:
'float'} attr : { name: 'Attr2' type: 'list(float)'}"
```
