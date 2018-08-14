# TensorFlow Lite Optimizing Converter & Interpreter Python API reference

This page provides examples on how to use TOCO and the TensorFlow Lite
interpreter via the Python API. It is complemented by the following documents:

*   [README](../README.md)
*   [Command-line examples](cmdline_examples.md)
*   [Command-line glossary](cmdline_reference.md)

Table of contents:

*   [High-level overview](#high-level-overview)
*   [API](#api)
*   [Basic examples](#basic)
    *   [Exporting a GraphDef from tf.Session](#basic-graphdef-sess)
    *   [Exporting a GraphDef from file](#basic-graphdef-file)
    *   [Exporting a SavedModel](#basic-savedmodel)
    *   [Exporting a tf.keras File](#basic-keras-file)
*   [Complex examples](#complex)
    *   [Exporting a quantized GraphDef](#complex-quant)
*   [TensorFlow Lite Python interpreter](#interpreter)
    *   [Using the interpreter from a model file](#interpreter-file)
    *   [Using the interpreter from model data](#interpreter-data)
*   [Additional instructions](#additional-instructions)
    *   [Build from source code](#latest-package)
    *   [Converting models prior to TensorFlow 1.9.](#pre-tensorflow-1.9)

## High-level overview

While the TensorFlow Lite Optimizing Converter can be used from the command
line, it is often convenient to use it as part of a Python model build and
training script. This is so that conversion can be part of your model
development pipeline. This allows you to know early and often that you are
designing a model that can be targeted to devices with mobile.

## API

The API for converting TensorFlow models to TensorFlow Lite as of TensorFlow 1.9
is `tf.contrib.lite.TocoConverter`. The API for calling the Python intepreter is
`tf.contrib.lite.Interpreter`.

`TocoConverter` provides class methods based on the original format of the
model. `TocoConverter.from_session()` is available for GraphDefs.
`TocoConverter.from_saved_model()` is available for SavedModels.
`TocoConverter.from_keras_model_file()` is available for `tf.Keras` files.
Example usages for simple float-point models are shown in [Basic
Examples](#basic). Examples usages for more complex models is shown in [Complex
Examples](#complex).

**NOTE**: Currently, `TocoConverter` will cause a fatal error to the Python
interpreter when the conversion fails. This will be remedied as soon as
possible.

## Basic examples <a name="basic"></a>

The following section shows examples of how to convert a basic float-point model
from each of the supported data formats into a TensorFlow Lite FlatBuffers.

### Exporting a GraphDef from tf.Session <a name="basic-graphdef-sess"></a>

The following example shows how to convert a TensorFlow GraphDef into a
TensorFlow Lite FlatBuffer from a `tf.Session` object.

```python
import tensorflow as tf

img = tf.placeholder(name="img", dtype=tf.float32, shape=(1, 64, 64, 3))
var = tf.get_variable("weights", dtype=tf.float32, shape=(1, 64, 64, 3))
val = img + var
out = tf.identity(val, name="out")

with tf.Session() as sess:
  converter = tf.contrib.lite.TocoConverter.from_session(sess, [img], [out])
  tflite_model = converter.convert()
  open("converted_model.tflite", "wb").write(tflite_model)
```

### Exporting a GraphDef from file <a name="basic-graphdef-file"></a>

The following example shows how to convert a TensorFlow GraphDef into a
TensorFlow Lite FlatBuffer when the GraphDef is stored in a file. Both `.pb` and
`.pbtxt` files are accepted.

The example uses
[Mobilenet_1.0_224](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_1.0_224_frozen.tgz).
The function only supports GraphDefs frozen via
[freeze_graph.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py).

```python
import tensorflow as tf

graph_def_file = "/path/to/Downloads/mobilenet_v1_1.0_224/frozen_graph.pb"
input_arrays = ["input"]
output_arrays = ["MobilenetV1/Predictions/Softmax"]

converter = tf.contrib.lite.TocoConverter.from_frozen_graph(
  graph_def_file, input_arrays, output_arrays)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
```

### Exporting a SavedModel <a name="basic-savedmodel"></a>

The following example shows how to convert a SavedModel into a TensorFlow Lite
FlatBuffer.

```python
import tensorflow as tf

converter = tf.contrib.lite.TocoConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
```

For more complex SavedModels, the optional parameters that can be passed into
`TocoConverter.from_saved_model()` are `input_arrays`, `input_shapes`,
`output_arrays`, `tag_set` and `signature_key`. Details of each parameter are
available by running `help(tf.contrib.lite.TocoConverter)`.

### Exporting a tf.keras File <a name="basic-keras-file"></a>

The following example shows how to convert a `tf.keras` model into a TensorFlow
Lite FlatBuffer.

```python
import tensorflow as tf

converter = tf.contrib.lite.TocoConverter.from_keras_model_file("keras_model.h5")
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
```

The `tf.keras` file must contain both the model and the weights. A comprehensive
example including model construction can be seen below.

```python
import numpy as np
import tensorflow as tf

# Generate tf.keras model.
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(2, input_shape=(3,)))
model.add(tf.keras.layers.RepeatVector(3))
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(3)))
model.compile(loss=tf.keras.losses.MSE,
              optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
              metrics=[tf.keras.metrics.categorical_accuracy],
              sample_weight_mode='temporal')

x = np.random.random((1, 3))
y = np.random.random((1, 3, 3))
model.train_on_batch(x, y)
model.predict(x)

# Save tf.keras model in HDF5 format.
keras_file = "keras_model.h5"
tf.keras.models.save_model(model, keras_file)

# Convert to TensorFlow Lite model.
converter = tf.contrib.lite.TocoConverter.from_keras_model_file(keras_file)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
```

## Complex examples <a name="complex"></a>

For models where the default value of the attributes is not sufficient, the
attribute's values should be set before calling `convert()`. In order to call
any constants use `tf.contrib.lite.constants.<CONSTANT_NAME>` as seen below with
`QUANTIZED_UINT8`. Run `help(tf.contrib.lite.TocoConverter)` in the Python
terminal for detailed documentation on the attributes.

Although the examples are demonstrated on GraphDefs containing only constants.
The same logic can be applied irrespective of the input data format.

### Exporting a quantized GraphDef <a name="complex-quant"></a>

The following example shows how to convert a quantized model into a TensorFlow
Lite FlatBuffer.

```python
import tensorflow as tf

img = tf.placeholder(name="img", dtype=tf.float32, shape=(1, 64, 64, 3))
const = tf.constant([1., 2., 3.]) + tf.constant([1., 4., 4.])
val = img + const
out = tf.fake_quant_with_min_max_args(val, min=0., max=1., name="output")

with tf.Session() as sess:
  converter = tf.contrib.lite.TocoConverter.from_session(sess, [img], [out])
  converter.inference_type = tf.contrib.lite.constants.QUANTIZED_UINT8
  input_arrays = converter.get_input_arrays()
  converter.quantized_input_stats = {input_arrays[0] : (0., 1.)}  # mean, std_dev
  tflite_model = converter.convert()
  open("converted_model.tflite", "wb").write(tflite_model)
```

## TensorFlow Lite Python interpreter <a name="interpreter"></a>

### Using the interpreter from a model file <a name="interpreter-file"></a>

The following example shows how to use the TensorFlow Lite Python interpreter
when provided a TensorFlow Lite FlatBuffer file. The example also demonstrates
how to run inference on random input data. Run
`help(tf.contrib.lite.Interpreter)` in the Python terminal to get detailed
documentation on the interpreter.

```python
import numpy as np
import tensorflow as tf

# Load TFLite model and allocate tensors.
interpreter = tf.contrib.lite.Interpreter(model_path="converted_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
```

### Using the interpreter from model data <a name="interpreter-data"></a>

The following example shows how to use the TensorFlow Lite Python interpreter
when starting with the TensorFlow Lite Flatbuffer model previously loaded. This
example shows an end-to-end use case, starting from building the TensorFlow
model.

```python
import numpy as np
import tensorflow as tf

img = tf.placeholder(name="img", dtype=tf.float32, shape=(1, 64, 64, 3))
const = tf.constant([1., 2., 3.]) + tf.constant([1., 4., 4.])
val = img + const
out = tf.identity(val, name="out")

with tf.Session() as sess:
  converter = tf.contrib.lite.TocoConverter.from_session(sess, [img], [out])
  tflite_model = converter.convert()

# Load TFLite model and allocate tensors.
interpreter = tf.contrib.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
```

## Additional instructions

### Build from source code <a name="latest-package"></a>

In order to run the latest version of the TOCO Python API, clone the TensorFlow
repository, configure the installation, and build and install the pip package.
Detailed instructions are available
[here](https://www.tensorflow.org/install/install_sources).

### Converting models prior to TensorFlow 1.9. <a name="pre-tensorflow-1.9"></a>

To use TOCO in TensorFlow 1.7 and TensorFlow 1.8, use the `toco_convert`
function. Run `help(tf.contrib.lite.toco_convert)` to get details about accepted
parameters.
