# Post-training quantization

Post-training quantization is a general technique to reduce model size while also
providing up to 3x lower latency with little degradation in model accuracy. Post-training
quantization quantizes weights from floating point to 8-bits of precision. This technique
is enabled as an option in the [TensorFlow Lite converter](../convert):

```
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
```

At inference, weights are converted from 8-bits of precision to floating point and
computed using floating-point kernels. This conversion is done once and cached to reduce latency.

To further improve latency, hybrid operators dynamically quantize activations to 8-bits and
perform computations with 8-bit weights and activations. This optimization provides latencies
close to fully fixed-point inference. However, the outputs are still stored using
floating point, so that the speedup with hybrid ops is less than a full fixed-point computation.
Hybrid ops are available for the most compute-intensive operators in a network:

*  [tf.contrib.layers.fully_connected](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/fully_connected)
*  [tf.nn.conv2d](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d)
*  [tf.nn.embedding_lookup](https://www.tensorflow.org/api_docs/python/tf/nn/embedding_lookup)
*  [BasicRNN](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/BasicRNNCell)
*  [tf.nn.bidirectional_dynamic_rnn for BasicRNNCell type](https://www.tensorflow.org/api_docs/python/tf/nn/bidirectional_dynamic_rnn)
*  [tf.nn.dynamic_rnn for LSTM and BasicRNN Cell types](https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn)


Since weights are quantized post training, there could be an accuracy loss, particularly for
smaller networks. Pre-trained fully quantized models are provided for specific networks in
the [TensorFlow Lite model repository](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/models.md#image-classification-quantized-models){:.external}. It is important to check the accuracy of the quantized model to verify that any degradation
in accuracy is within acceptable limits. There is a tool to evaluate [TensorFlow Lite model accuracy](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/accuracy/README.md){:.external}.

If the accuracy drop is too high, consider using [quantization aware training](https://github.com/tensorflow/tensorflow/tree/r1.13/tensorflow/contrib/quantize){:.external}.

### Representation for quantized tensors

TensorFlow approaches the conversion of floating-point arrays of numbers into
8-bit representations as a compression problem. Since the weights and activation
tensors in trained neural network models tend to have values that are distributed
across comparatively small ranges (e.g. -15 to +15 for weights or -500 to
1000 for image model activations).

Since neural networks tend to be robust at handling noise, the error introduced
by quantizing to a small set of values maintains the precision of the overall
results within an acceptable threshold. A chosen representation must perform
fast calculations, especially with large matrix multiplications that comprise
the bulk of the computations while running a model.

This is represented with two floats that store the overall minimum and maximum
values corresponding to the lowest and highest quantized value. Each entry in the
quantized array represents a float value in that range, distributed linearly
between the minimum and maximum.

With our post-training quantization tooling, we use symmetric quantization for
our weights, meaning we expand the represented range and force the min and max
to be the negative of each other.

For example, with an overall minimum of -10.0 and a maximum
of 30.0f, we instead represent a minimum of -30.0 and maximum of 30.0f. In an
8-bit array, the quantized values would be represented as follows:

<figure>
  <table>
    <tr><th>Quantized</th><th>Float</th></tr>
    <tr><td>-42</td><td>-10.0</td></tr>
    <tr><td>0</td><td>0</td></tr>
    <tr><td>127</td><td>30.0</td></tr>
    <tr><td>-127</td><td>30.0 (this value does not ever show up)</td></tr>
  </table>
  <figcaption>
    <b>Table 2</b>: Quantized value range example
  </figcaption>
</figure>

The advantages of this representation format are:

* It efficiently represents an arbitrary magnitude of ranges.
* The linear spread makes multiplications straightforward.
* A symmetric range for weights enables downstream hardware optimizations.
