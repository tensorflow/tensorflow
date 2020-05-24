# Converting Quantized Models

This page provides information for how to convert quantized TensorFlow Lite
models. For more details, please see the
[model optimization](../performance/model_optimization.md).

# Post-training: Quantizing models for CPU model size

The simplest way to create a small model is to quantize the weights to 8 bits
and quantize the inputs/activations "on-the-fly", during inference. This
has latency benefits, but prioritizes size reduction.

During conversion, set the `optimizations` flag to optimize for size:

```
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()
```

# Full integer quantization of weights and activations

We can get further latency improvements, reductions in peak memory usage, and
access to integer only hardware accelerators by making sure all model math is
quantized. To do this, we need to measure the dynamic range of activations and
inputs with a representative data set. You can simply create an input data
generator and provide it to our converter.

```
import tensorflow as tf

def representative_dataset_gen():
  for _ in range(num_calibration_steps):
    # Get sample input data as a numpy array in a method of your choosing.
    yield [input]

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
```

# During training: Quantizing models for integer-only execution.

Quantizing models for integer-only execution gets a model with even faster
latency, smaller size, and integer-only accelerators compatible model.
Currently, this requires training a model with
["fake-quantization" nodes](https://github.com/tensorflow/tensorflow/tree/r1.13/tensorflow/contrib/quantize).

This is only available in the v1 converter. A longer term solution that's
compatible with 2.0 semantics is in progress.

Convert the graph:

```
converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.inference_type = tf.lite.constants.QUANTIZED_UINT8
input_arrays = converter.get_input_arrays()
converter.quantized_input_stats = {input_arrays[0] : (0., 1.)}  # mean_value, std_dev
tflite_model = converter.convert()
```

For fully integer models, the inputs are uint8. When the `inference_type` is set
to `QUANTIZED_UINT8` as above, the real_input_value is standardised using the
[standard-score](https://en.wikipedia.org/wiki/Standard_score) as follows:

real_input_value = (quantized_input_value - mean_value) / std_dev_value

The `mean_value` and `std_dev values` specify how those uint8 values map to the
float input values used while training the model. For more details, please see
the
[TFLiteConverter](https://www.tensorflow.org/api_docs/python/tf/compat/v1/lite/TFLiteConverter)

`mean` is the integer value from 0 to 255 that maps to floating point 0.0f.
`std_dev` is 255 / (float_max - float_min).

For most users, we recommend using post-training quantization. We are working on
new tools for post-training and during training quantization that we hope will
simplify generating quantized models.
