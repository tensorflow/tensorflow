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
    *   [Exporting a GraphDef with constants](#basic-graphdef-const)
    *   [Exporting a GraphDef with variables](#basic-graphdef-var)
    *   [Exporting a SavedModel](#basic-savedmodel)
*   [Complex examples](#complex)
    *   [Exporting a quantized GraphDef](#complex-quant)
*   [TensorFlow Lite Python interpreter](#interpreter)
    *   [Using the interpreter from a model file](#interpreter-file)
    *   [Using the interpreter from model data](#interpreter-data)

## High-level overview

While the TensorFlow Lite Optimizing Converter can be used from the command
line, it is often convenient to use it as part of a Python model build and
training script. This is so that conversion can be part of your model
development pipeline. This allows you to know early and often that you are
designing a model that can be targeted to devices with mobile.

## API

The API for converting TensorFlow models to TensorFlow Lite is
`tf.contrib.lite.TocoConverter`. The API for calling the Python intepreter is
`tf.contrib.lite.Interpreter`.

`TocoConverter` provides class methods based on the original format of the
model. `TocoConverter.from_session()` is available for GraphDefs.
`TocoConverter.from_saved_model()` is available for SavedModels. Example usages
for simple float-point models are shown in [Basic Examples](#basic). Examples
usages for more complex models is shown in [Complex Examples](#complex).

**NOTE**: Currently, `TocoConverter` will cause a fatal error to the Python
interpreter when the conversion fails. This will be remedied as soon as
possible.

## Basic examples <a name="basic"></a>

The following section shows examples of how to convert a basic float-point model
from each of the supported data formats into a TensorFlow Lite FlatBuffers.

### Exporting a GraphDef with constants <a name="basic-graphdef-const"></a>

The following example shows how to convert a TensorFlow GraphDef with constants
into a TensorFlow Lite FlatBuffer.

```python
import tensorflow as tf

img = tf.placeholder(name="img", dtype=tf.float32, shape=(1, 64, 64, 3))
const = tf.constant([1., 2., 3.]) + tf.constant([1., 4., 4.])
val = img + const
out = tf.identity(val, name="out")

with tf.Session() as sess:
  converter = tf.contrib.lite.TocoConverter.from_session(sess, [img], [out])
  tflite_model = converter.convert()
  open("converted_model.tflite", "wb").write(tflite_model)
```

### Exporting a GraphDef with variables <a name="basic-graphdef-var"></a>

If a model has variables, they need to be turned into constants through a
process known as freezing. It can be accomplished by setting `freeze_variables`
to `True` as shown in the example below.

```python
import tensorflow as tf

img = tf.placeholder(name="img", dtype=tf.float32, shape=(1, 64, 64, 3))
var = tf.get_variable("weights", dtype=tf.float32, shape=(1, 64, 64, 3))
val = img + var
out = tf.identity(val, name="out")

with tf.Session() as sess:
  converter = tf.contrib.lite.TocoConverter.from_session(sess, [img], [out],
                                                        freeze_variables=True)
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

## Complex examples <a name="complex"></a>

For models where the default value of the attributes is not sufficient, the
variables values should be set before calling `convert()`. In order to call any
constants use `tf.contrib.lite.constants.<CONSTANT_NAME>` as seen below with
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
  converter.quantized_input_stats = [(0., 1.)]  # mean, std_dev
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
