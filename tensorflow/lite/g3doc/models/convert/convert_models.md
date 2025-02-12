# Convert TensorFlow models

This page describes how to convert a TensorFlow model
to a TensorFlow Lite model (an optimized
[FlatBuffer](https://google.github.io/flatbuffers/) format identified by the
`.tflite` file extension) using the TensorFlow Lite converter.

Note: This guide assumes you've both
[installed TensorFlow 2.x](https://www.tensorflow.org/install/pip#tensorflow-2-packages-are-available)
and trained models in TensorFlow 2.x.
If your model is trained in TensorFlow 1.x, considering
[migrating to TensorFlow 2.x](https://www.tensorflow.org/guide/migrate/tflite).
To identify the installed TensorFlow version, run
`print(tf.__version__)`.

## Conversion workflow

The diagram below illustrations the high-level workflow for converting
your model:

![TFLite converter workflow](../../images/convert/convert.png)

**Figure 1.** Converter workflow.

You can convert your model using one of the following options:

1.  [Python API](#python_api) (***recommended***):
    This allows you to integrate the conversion into your development pipeline,
    apply optimizations, add metadata and many other tasks that simplify
    the conversion process.
2.  [Command line](#cmdline): This only supports basic model conversion.

Note: In case you encounter any issues during model conversion, create a
[GitHub issue](https://github.com/tensorflow/tensorflow/issues/new?template=60-tflite-converter-issue.md).

## Python API <a name="python_api"></a>

*Helper code: To learn more about the TensorFlow Lite converter
API, run `print(help(tf.lite.TFLiteConverter))`.*

Convert a TensorFlow model using
[`tf.lite.TFLiteConverter`](https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter).
A TensorFlow model is stored using the SavedModel format and is
generated either using the high-level `tf.keras.*` APIs (a Keras model) or
the low-level `tf.*` APIs (from which you generate concrete functions). As a
result, you have the following three options (examples are in the next few
sections):

  *   `tf.lite.TFLiteConverter.from_saved_model()` (**recommended**): Converts
      a [SavedModel](https://www.tensorflow.org/guide/saved_model).
  *   `tf.lite.TFLiteConverter.from_keras_model()`: Converts a
      [Keras](https://www.tensorflow.org/guide/keras/overview) model.
  *   `tf.lite.TFLiteConverter.from_concrete_functions()`: Converts
      [concrete functions](https://www.tensorflow.org/guide/intro_to_graphs).



### Convert a SavedModel (recommended) <a name="saved_model"></a>

The following example shows how to convert a
[SavedModel](https://www.tensorflow.org/guide/saved_model) into a TensorFlow
Lite model.

```python
import tensorflow as tf

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
```

### Convert a Keras model <a name="keras"></a>

The following example shows how to convert a
[Keras](https://www.tensorflow.org/guide/keras/overview) model into a TensorFlow
Lite model.

```python
import tensorflow as tf

# Create a model using high-level tf.keras.* APIs
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1]),
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dense(units=1)
])
model.compile(optimizer='sgd', loss='mean_squared_error') # compile the model
model.fit(x=[-1, 0, 1], y=[-3, -1, 1], epochs=5) # train the model
# (to generate a SavedModel) tf.saved_model.save(model, "saved_model_keras_dir")

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
```

### Convert concrete functions <a name="concrete_function"></a>

The following example shows how to convert
[concrete functions](https://www.tensorflow.org/guide/intro_to_graphs) into a
TensorFlow Lite model.

```python
import tensorflow as tf

# Create a model using low-level tf.* APIs
class Squared(tf.Module):
  @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
  def __call__(self, x):
    return tf.square(x)
model = Squared()
# (ro run your model) result = Squared(5.0) # This prints "25.0"
# (to generate a SavedModel) tf.saved_model.save(model, "saved_model_tf_dir")
concrete_func = model.__call__.get_concrete_function()

# Convert the model.

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func],
                                                            model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
```

### Other features

*   Apply [optimizations](../../performance/model_optimization.md). A common
    optimization used is
    [post training quantization](../../performance/post_training_quantization.md),
    which can further reduce your model latency and size with minimal loss in
    accuracy.

*   Add [metadata](metadata.md), which makes it easier to create platform
    specific wrapper code when deploying models on devices.

### Conversion errors

The following are common conversion errors and their solutions:

*   Error: `Some ops are not supported by the native TFLite runtime, you can
    enable TF kernels fallback using TF Select. See instructions:
    https://www.tensorflow.org/lite/guide/ops_select. TF Select ops: ..., ..,
    ...`

    Solution: The error occurs as your model has TF ops that don't have a
    corresponding TFLite implementation. You can resolve this by
    [using the TF op in the TFLite model](../../guide/ops_select.md)
    (recommended). If you want to generate a model with TFLite ops only, you can
    either add a request for the missing TFLite op in
    [Github issue #21526](https://github.com/tensorflow/tensorflow/issues/21526)
    (leave a comment if your request hasn’t already been mentioned) or
    [create the TFLite op](../../guide/ops_custom.md#create_and_register_the_operator)
    yourself.

*   Error: `.. is neither a custom op nor a flex op`

    Solution: If this TF op is:

    *   Supported in TF: The error occurs because the TF op is missing from the
        [allowlist](../../guide/op_select_allowlist.md) (an exhaustive list of
        TF ops supported by TFLite). You can resolve this as follows:

        1.  [Add missing ops to the allowlist](../../guide/op_select_allowlist.md#add_tensorflow_core_operators_to_the_allowed_list).
        2.  [Convert the TF model to a TFLite model and run inference](../../guide/ops_select.md).

    *   Unsupported in TF: The error occurs because TFLite is unaware of the
        custom TF operator defined by you. You can resolve this as follows:

        1.  [Create the TF op](https://www.tensorflow.org/guide/create_op).
        2.  [Convert the TF model to a TFLite model](../../guide/op_select_allowlist.md#users_defined_operators).
        3.  [Create the TFLite op](../../guide/ops_custom.md#create_and_register_the_operator)
            and run inference by linking it to the TFLite runtime.

## Command Line Tool <a name="cmdline"></a>

**Note:** It is highly recommended that you use the [Python API](#python_api)
listed above instead, if possible.

If you've
[installed TensorFlow 2.x from pip](https://www.tensorflow.org/install/pip), use
the `tflite_convert` command. To view all the available flags, use the
following command:

```sh
$ tflite_convert --help

`--output_file`. Type: string. Full path of the output file.
`--saved_model_dir`. Type: string. Full path to the SavedModel directory.
`--keras_model_file`. Type: string. Full path to the Keras H5 model file.
`--enable_v1_converter`. Type: bool. (default False) Enables the converter and flags used in TF 1.x instead of TF 2.x.

You are required to provide the `--output_file` flag and either the `--saved_model_dir` or `--keras_model_file` flag.
```

If you have the
[TensorFlow 2.x source](https://www.tensorflow.org/install/source)
downloaded and want to run the converter from that source without building and
installing the package,
you can replace '`tflite_convert`' with
'`bazel run tensorflow/lite/python:tflite_convert --`' in the command.


### Converting a SavedModel <a name="cmdline_saved_model"></a>

```sh
tflite_convert \
  --saved_model_dir=/tmp/mobilenet_saved_model \
  --output_file=/tmp/mobilenet.tflite
```

### Converting a Keras H5 model <a name="cmdline_keras_model"></a>

```sh
tflite_convert \
  --keras_model_file=/tmp/mobilenet_keras_model.h5 \
  --output_file=/tmp/mobilenet.tflite
```

## Next Steps

Use the [TensorFlow Lite interpreter](../../guide/inference.md) to run inference
on a client device (e.g. mobile, embedded).
