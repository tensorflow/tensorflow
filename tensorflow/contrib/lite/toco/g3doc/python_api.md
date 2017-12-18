# TensorFlow Lite Optimizing Converter (TOCO) Python API reference

## High-level overview

While the TensorFlow Lite Optimizing Converter can be used from the command
line, it is often convenient to use it as part of Python model build and
training script. This is so that conversion can be part of your model
development pipeline. This allows you to know early and often that you are
designing a model that can be targeted to devices with mobile.

## API

In Python you can run `help(tf.contrib.lite)` to get documentation on functions.
In particular, `tf.contrib.lite.toco_convert` presents a simple API and
`tf.contrib.lite.toco_from_protos` allows more detailed control of TOCO using
the protobuf interface to TOCO.

## Example

In particular, here we show creating a simple model and converting it to a
TensorFlow Lite Model.

```python
import tensorflow as tf

img = tf.placeholder(name="img", dtype=tf.float32, shape=(1, 64, 64, 3))
val = img + tf.constant([1., 2., 3.]) + tf.constant([1., 4., 4.])
out = tf.identity(val, name="out")
with tf.Session() as sess:
  tflite_model = tf.contrib.lite.toco_convert(sess.graph_def, [img], [out])
  open("test.tflite", "wb").write(tflite_model)
```

**NOTE** Currently, the TOCO command will cause a fatal error to the Python
interpreter when TOCO conversion fails. This will be remedied as soon as
possible.

## Example 2: Export with variables

If a model has variables, they need to be turned into constants. This process is
known as freezing, and it can actually be accomplished with

```python
import tensorflow as tf

img = tf.placeholder(name="img", dtype=tf.float32, shape=(1, 64, 64, 3))
var = tf.get_variable("weights", dtype=tf.float32, shape=(1,64,64,3))
val = img + var

def canonical_name(x):
  return x.name.split(":")[0]

out = tf.identity(val, name="out")
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  out_tensors = [out]
  frozen_graphdef = tf.graph_util.convert_variables_to_constants(
      sess, sess.graph_def, map(canonical_name, out_tensors))
  tflite_model = tf.contrib.lite.toco_convert(
      frozen_graphdef, [img], out_tensors)
  open("converted_model.tflite", "wb").write(tflite_model)
```
