# JAX models with TensorFlow Lite

This page provides a path for users who want to train models in JAX and deploy
to mobile for inference ([example colab](https://colab.sandbox.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/examples/jax_conversion/jax_to_tflite.ipynb)).

The methods in this guide produce a `tflite_model` which can be used directly
with the TFLite interpreter code example or saved to a TFLite FlatBuffer file.

## Prerequisite

It's recommended to try this feature with the newest TensorFlow nightly Python
package.

```
pip install tf-nightly --upgrade
```

We will use the [Orbax
Export](https://orbax.readthedocs.io/en/latest/orbax_export_101.html) library to
export JAX models. Make sure your JAX version is at least 0.4.20 or above.

```
pip install jax --upgrade
pip install orbax-export --upgrade
```

## Convert JAX models to TensorFlow Lite

We use the TensorFlow
[SavedModel](https://www.tensorflow.org/guide/saved_model) as the intermediate
format between JAX and TensorFlow Lite. Once you have a SavedModel then
existing TensorFlow Lite APIs can be used to complete the conversion process.

```py
# This code snippet converts a JAX model to TFLite through TF SavedModel.
from orbax.export import ExportManager
from orbax.export import JaxModule
from orbax.export import ServingConfig
import tensorflow as tf
import jax.numpy as jnp

def model_fn(_, x):
  return jnp.sin(jnp.cos(x))

jax_module = JaxModule({}, model_fn, input_polymorphic_shape='b, ...')

# Option 1: Simply save the model via `tf.saved_model.save` if no need for pre/post
# processing.
tf.saved_model.save(
    jax_module,
    '/some/directory',
    signatures=jax_module.methods[JaxModule.DEFAULT_METHOD_KEY].get_concrete_function(
        tf.TensorSpec(shape=(None,), dtype=tf.float32, name="input")
    ),
    options=tf.saved_model.SaveOptions(experimental_custom_gradients=True),
)
converter = tf.lite.TFLiteConverter.from_saved_model('/some/directory')
tflite_model = converter.convert()

# Option 2: Define pre/post processing TF functions (e.g. (de)?tokenize).
serving_config = ServingConfig(
    'Serving_default',
    # Corresponds to the input signature of `tf_preprocessor`
    input_signature=[tf.TensorSpec(shape=(None,), dtype=tf.float32, name='input')],
    tf_preprocessor=lambda x: x,
    tf_postprocessor=lambda out: {'output': out}
)
export_mgr = ExportManager(jax_module, [serving_config])
export_mgr.save('/some/directory')
converter = tf.lite.TFLiteConverter.from_saved_model('/some/directory')
tflite_model = converter.convert()

# Option 3: Convert from TF concrete function directly
converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [
        jax_module.methods[JaxModule.DEFAULT_METHOD_KEY].get_concrete_function(
            tf.TensorSpec(shape=(None,), dtype=tf.float32, name="input")
        )
    ]
)
tflite_model = converter.convert()
```

## Check the converted TFLite model

After the model is converted to TFLite, you can run TFLite interpreter APIs to
check model outputs.

```py
# Run the model with TensorFlow Lite
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors() input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.set_tensor(input_details[0]["index"], input_data)
interpreter.invoke()
result = interpreter.get_tensor(output_details[0]["index"])
```
