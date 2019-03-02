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
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
```

# During training: Quantizing models for integer-only execution.

Quantizing models for integer-only execution gets a model with even faster
latency, smaller size, and integer-only accelerators compatible model.
Currently, this requires training a model with
["fake-quantization" nodes](https://github.com/tensorflow/tensorflow/tree/r1.13/tensorflow/contrib/quantize).

Convert the graph:

```
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.inference_type = tf.lite.constants.QUANTIZED_UINT8
input_arrays = converter.get_input_arrays()
converter.quantized_input_stats = {input_arrays[0] : (0., 1.)}  # mean, std_dev
tflite_model = converter.convert()
```

For fully integer models, the inputs are uint8. The `mean` and `std_dev values`
specify how those uint8 values map to the float input values used while training
the model.

`mean` is the integer value from 0 to 255 that maps to floating point 0.0f.
`std_dev` is 255 / (float_max - float_min)

For most users, we recommend using post-training quantization. We are working on
new tools for post-training and during training quantization that we hope will
simplify generating quantized models.
