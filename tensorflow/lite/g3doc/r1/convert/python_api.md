# Converter Python API guide

This page describes how to convert TensorFlow models into the TensorFlow Lite
format using the TensorFlow Lite Converter Python API.

If you're looking for information about how to run a TensorFlow Lite model,
see [TensorFlow Lite inference](../guide/inference.md).

Note: This page describes the converter in the TensorFlow nightly release,
installed using `pip install tf-nightly`. For docs describing older versions
reference ["Converting models from TensorFlow 1.12"](#pre_tensorflow_1.12).


## High-level overview

While the TensorFlow Lite Converter can be used from the command line, it is
often convenient to use in a Python script as part of the model development
pipeline. This allows you to know early that you are designing a model that can
be targeted to devices with mobile.

## API

The API for converting TensorFlow models to TensorFlow Lite is
`tf.lite.TFLiteConverter`, which provides class methods based on the original
format of the model. For example, `TFLiteConverter.from_session()` is available
for GraphDefs, `TFLiteConverter.from_saved_model()` is available for
SavedModels, and `TFLiteConverter.from_keras_model_file()` is available for
`tf.Keras` files.

Example usages for simple float-point models are shown in
[Basic Examples](#basic). Examples usages for more complex models is shown in
[Complex Examples](#complex).

## Basic examples <a name="basic"></a>

The following section shows examples of how to convert a basic float-point model
from each of the supported data formats into a TensorFlow Lite FlatBuffers.

### Exporting a GraphDef from tf.Session <a name="basic_graphdef_sess"></a>

The following example shows how to convert a TensorFlow GraphDef into a
TensorFlow Lite FlatBuffer from a `tf.Session` object.

```python
import tensorflow as tf

img = tf.placeholder(name="img", dtype=tf.float32, shape=(1, 64, 64, 3))
var = tf.get_variable("weights", dtype=tf.float32, shape=(1, 64, 64, 3))
val = img + var
out = tf.identity(val, name="out")

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  converter = tf.lite.TFLiteConverter.from_session(sess, [img], [out])
  tflite_model = converter.convert()
  open("converted_model.tflite", "wb").write(tflite_model)
```

### Exporting a GraphDef from file <a name="basic_graphdef_file"></a>

The following example shows how to convert a TensorFlow GraphDef into a
TensorFlow Lite FlatBuffer when the GraphDef is stored in a file. Both `.pb` and
`.pbtxt` files are accepted.

The example uses
[Mobilenet_1.0_224](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_1.0_224_frozen.tgz).
The function only supports GraphDefs frozen using
[freeze_graph.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py).

```python
import tensorflow as tf

graph_def_file = "/path/to/Downloads/mobilenet_v1_1.0_224/frozen_graph.pb"
input_arrays = ["input"]
output_arrays = ["MobilenetV1/Predictions/Softmax"]

converter = tf.lite.TFLiteConverter.from_frozen_graph(
  graph_def_file, input_arrays, output_arrays)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
```

### Exporting a SavedModel <a name="basic_savedmodel"></a>

The following example shows how to convert a SavedModel into a TensorFlow Lite
FlatBuffer.

```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
```

For more complex SavedModels, the optional parameters that can be passed into
`TFLiteConverter.from_saved_model()` are `input_arrays`, `input_shapes`,
`output_arrays`, `tag_set` and `signature_key`. Details of each parameter are
available by running `help(tf.lite.TFLiteConverter)`.

### Exporting a tf.keras File <a name="basic_keras_file"></a>

The following example shows how to convert a `tf.keras` model into a TensorFlow
Lite FlatBuffer. This example requires
[`h5py`](http://docs.h5py.org/en/latest/build.html) to be installed.

```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model_file("keras_model.h5")
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
converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_file)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
```

## Complex examples <a name="complex"></a>

For models where the default value of the attributes is not sufficient, the
attribute's values should be set before calling `convert()`. In order to call
any constants use `tf.lite.constants.<CONSTANT_NAME>` as seen below with
`QUANTIZED_UINT8`. Run `help(tf.lite.TFLiteConverter)` in the Python
terminal for detailed documentation on the attributes.

Although the examples are demonstrated on GraphDefs containing only constants.
The same logic can be applied irrespective of the input data format.

### Exporting a quantized GraphDef <a name="complex_quant"></a>

The following example shows how to convert a quantized model into a TensorFlow
Lite FlatBuffer.

```python
import tensorflow as tf

img = tf.placeholder(name="img", dtype=tf.float32, shape=(1, 64, 64, 3))
const = tf.constant([1., 2., 3.]) + tf.constant([1., 4., 4.])
val = img + const
out = tf.fake_quant_with_min_max_args(val, min=0., max=1., name="output")

with tf.Session() as sess:
  converter = tf.lite.TFLiteConverter.from_session(sess, [img], [out])
  converter.inference_type = tf.lite.constants.QUANTIZED_UINT8
  input_arrays = converter.get_input_arrays()
  converter.quantized_input_stats = {input_arrays[0] : (0., 1.)}  # mean, std_dev
  tflite_model = converter.convert()
  open("converted_model.tflite", "wb").write(tflite_model)
```


## Additional instructions

### Build from source code <a name="latest_package"></a>

In order to run the latest version of the TensorFlow Lite Converter Python API,
either install the nightly build with
[pip](https://www.tensorflow.org/install/pip) (recommended) or
[Docker](https://www.tensorflow.org/install/docker), or
[build the pip package from source](https://www.tensorflow.org/install/source).

### Converting models from TensorFlow 1.12 <a name="pre_tensorflow_1.12"></a>

Reference the following table to convert TensorFlow models to TensorFlow Lite in
and before TensorFlow 1.12. Run `help()` to get details of each API.

TensorFlow Version | Python API
------------------ | ---------------------------------
1.12               | `tf.contrib.lite.TFLiteConverter`
1.9-1.11           | `tf.contrib.lite.TocoConverter`
1.7-1.8            | `tf.contrib.lite.toco_convert`
