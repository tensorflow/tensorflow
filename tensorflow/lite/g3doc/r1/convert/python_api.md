# Converter Python API guide

This page describes how to convert TensorFlow models into the TensorFlow Lite
format using the
[`tf.compat.v1.lite.TFLiteConverter`](https://www.tensorflow.org/api_docs/python/tf/compat/v1/lite/TFLiteConverter)
Python API. It provides the following class methods based on the original format
of the model:

*   `tf.compat.v1.lite.TFLiteConverter.from_saved_model()`: Converts a
    [SavedModel](https://www.tensorflow.org/guide/saved_model).
*   `tf.compat.v1.lite.TFLiteConverter.from_keras_model_file()`: Converts a
    [Keras](https://www.tensorflow.org/guide/keras/overview) model file.
*   `tf.compat.v1.lite.TFLiteConverter.from_session()`: Converts a GraphDef from
    a session.
*   `tf.compat.v1.lite.TFLiteConverter.from_frozen_graph()`: Converts a Frozen
    GraphDef from a file. If you have checkpoints, then first convert it to a
    Frozen GraphDef file and then use this API as shown [here](#checkpoints).

In the following sections, we discuss [basic examples](#basic) and
[complex examples](#complex).

## Basic examples <a name="basic"></a>

The following section shows examples of how to convert a basic model from each
of the supported model formats into a TensorFlow Lite model.

### Convert a SavedModel <a name="basic_savedmodel"></a>

The following example shows how to convert a
[SavedModel](https://www.tensorflow.org/guide/saved_model) into a TensorFlow
Lite model.

```python
import tensorflow as tf

# Convert the model.
converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
```

### Convert a Keras model file <a name="basic_keras_file"></a>

The following example shows how to convert a
[Keras](https://www.tensorflow.org/guide/keras/overview) model file into a
TensorFlow Lite model.

```python
import tensorflow as tf

# Convert the model.
converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file('keras_model.h5')
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
```

The Keras file contains both the model and the weights. A comprehensive example
is given below.

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

# Save tf.keras model in H5 format.
keras_file = 'keras_model.h5'
tf.keras.models.save_model(model, keras_file)

# Convert the model.
converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(keras_file)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
```

### Convert a GraphDef from a session <a name="basic_graphdef_sess"></a>

The following example shows how to convert a GraphDef from a `tf.Session` object
into a TensorFlow Lite model .

```python
import tensorflow as tf

img = tf.placeholder(name='img', dtype=tf.float32, shape=(1, 64, 64, 3))
var = tf.get_variable('weights', dtype=tf.float32, shape=(1, 64, 64, 3))
val = img + var
out = tf.identity(val, name='out')

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  # Convert the model.
  converter = tf.compat.v1.lite.TFLiteConverter.from_session(sess, [img], [out])
  tflite_model = converter.convert()

  # Save the model.
  with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

### Convert a Frozen GraphDef from file <a name="basic_graphdef_file"></a>

The following example shows how to convert a Frozen GraphDef (or a frozen
graph), usually generated using the
[freeze_graph.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py)
script, into a TensorFlow Lite model.

The example uses
[Mobilenet_1.0_224](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_1.0_224_frozen.tgz).

```python
import tensorflow as tf

# Convert the model.
converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
    graph_def_file='/path/to/mobilenet_v1_1.0_224/frozen_graph.pb',
                    # both `.pb` and `.pbtxt` files are accepted.
    input_arrays=['input'],
    input_shapes={'input' : [1, 224, 224,3]},
    output_arrays=['MobilenetV1/Predictions/Softmax']
)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
```

#### Convert checkpoints <a name="checkpoints"></a>

1.  Convert checkpoints to a Frozen GraphDef as follows
    (*[reference](https://laid.delanover.com/how-to-freeze-a-graph-in-tensorflow/)*):

    *   Install [bazel](https://docs.bazel.build/versions/master/install.html)
    *   Clone the TensorFlow repository: `git clone
        https://github.com/tensorflow/tensorflow.git`
    *   Build freeze graph tool: `bazel build
        tensorflow/python/tools:freeze_graph`
        *   The directory from which you run this should contain a file named
            'WORKSPACE'.
        *   If you're running on Ubuntu 16.04 OS and face issues, update the
            command to `bazel build -c opt --copt=-msse4.1 --copt=-msse4.2
            tensorflow/python/tools:freeze_graph`
    *   Run freeze graph tool: `bazel run tensorflow/python/tools:freeze_graph
        --input_graph=/path/to/graph.pbtxt --input_binary=false
        --input_checkpoint=/path/to/model.ckpt-00010
        --output_graph=/path/to/frozen_graph.pb
        --output_node_names=name1,name2.....`
        *   If you have an input `*.pb` file instead of `*.pbtxt`, then replace
            `--input_graph=/path/to/graph.pbtxt --input_binary=false` with
            `--input_graph=/path/to/graph.pb`
        *   You can find the output names by exploring the graph using
            [Netron](https://github.com/lutzroeder/netron) or
            [summarize graph tool](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms#inspecting-graphs).

2.  Now [convert the Frozen GraphDef file](#basic_graphdef_file) to a TensorFlow
    Lite model as shown in the example above.

## Complex examples <a name="complex"></a>

For models where the default value of the attributes is not sufficient, the
attribute's values should be set before calling `convert()`. Run
`help(tf.compat.v1.lite.TFLiteConverter)` in the Python terminal for detailed
documentation on the attributes.

### Convert a quantize aware trained model <a name="complex_quant"></a>

The following example shows how to convert a quantize aware trained model into a
TensorFlow Lite model.

The example uses
[Mobilenet_1.0_224](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_1.0_224_frozen.tgz).

```python
import tensorflow as tf

# Convert the model.
converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
    graph_def_file='/path/to/mobilenet_v1_1.0_224/frozen_graph.pb',
    input_arrays=['input'],
    input_shapes={'input' : [1, 224, 224,3]},
    output_arrays=['MobilenetV1/Predictions/Softmax'],
)
converter.quantized_input_stats = {'input' : (0., 1.)}  # mean, std_dev (input range is [-1, 1])
converter.inference_type = tf.int8 # this is the recommended type.
# converter.inference_input_type=tf.uint8 # optional
# converter.inference_output_type=tf.uint8 # optional
tflite_model = converter.convert()

# Save the model.
with open('quantized_model.tflite', 'wb') as f:
  f.write(tflite_model)
```

## Convert models from TensorFlow 1.12 <a name="pre_tensorflow_1.12"></a>

Reference the following table to convert TensorFlow models to TensorFlow Lite in
and before TensorFlow 1.12. Run `help()` to get details of each API.

TensorFlow Version | Python API
------------------ | ---------------------------------
1.12               | `tf.contrib.lite.TFLiteConverter`
1.9-1.11           | `tf.contrib.lite.TocoConverter`
1.7-1.8            | `tf.contrib.lite.toco_convert`
