# Model optimization

Tensorflow Lite and the
[Tensorflow Model Optimization Toolkit](https://www.tensorflow.org/model_optimization)
provide tools to minimize the complexity of optimizing inference.

Inference efficiency is particularly important for edge devices, such as mobile
and Internet of Things (IoT). Such devices have many restrictions on processing,
memory, power-consumption, and storage for models. Furthermore, model
optimization unlocks the processing power of fixed-point hardware and next
generation hardware accelerators.

## Model quantization

Quantizing deep neural networks uses techniques that allow for reduced precision
representations of weights and, optionally, activations for both storage and
computation. Quantization provides several benefits:

* Support on existing CPU platforms.
* Quantization of activations reduces memory access costs for reading and storing intermediate activations.
* Many CPU and hardware accelerator implementations provide SIMD instruction capabilities, which are especially beneficial for quantization.

TensorFlow Lite provides several levels of support for quantization.

*   Tensorflow Lite [post-training quantization](post_training_quantization.md)
    quantizes weights and activations post training easily.
*   [Quantization-aware training](https://github.com/tensorflow/tensorflow/tree/r1.13/tensorflow/contrib/quantize){:.external}
    allows for training of networks that can be quantized with minimal accuracy
    drop; this is only available for a subset of convolutional neural network
    architectures.

### Latency and accuracy results

Below are the latency and accuracy results for post-training quantization and
quantization-aware training on a few models. All latency numbers are measured on
Pixel&nbsp;2 devices using a single big core. As the toolkit improves, so will the numbers here:

<figure>
  <table>
    <tr>
      <th>Model</th>
      <th>Top-1 Accuracy (Original) </th>
      <th>Top-1 Accuracy (Post Training Quantized) </th>
      <th>Top-1 Accuracy (Quantization Aware Training) </th>
      <th>Latency (Original) (ms) </th>
      <th>Latency (Post Training Quantized) (ms) </th>
      <th>Latency (Quantization Aware Training) (ms) </th>
      <th> Size (Original) (MB)</th>
      <th> Size (Optimized) (MB)</th>
    </tr> <tr><td>Mobilenet-v1-1-224</td><td>0.709</td><td>0.657</td><td>0.70</td>
      <td>124</td><td>112</td><td>64</td><td>16.9</td><td>4.3</td></tr>
    <tr><td>Mobilenet-v2-1-224</td><td>0.719</td><td>0.637</td><td>0.709</td>
      <td>89</td><td>98</td><td>54</td><td>14</td><td>3.6</td></tr>
   <tr><td>Inception_v3</td><td>0.78</td><td>0.772</td><td>0.775</td>
      <td>1130</td><td>845</td><td>543</td><td>95.7</td><td>23.9</td></tr>
   <tr><td>Resnet_v2_101</td><td>0.770</td><td>0.768</td><td>N/A</td>
      <td>3973</td><td>2868</td><td>N/A</td><td>178.3</td><td>44.9</td></tr>
 </table>
  <figcaption>
    <b>Table 1</b> Benefits of model quantization for select CNN models
  </figcaption>
</figure>

## Choice of tool

As a starting point, check if the models in
[hosted models](../guide/hosted_models.md) can work for your application. If
not, we recommend that users start with the
[post-training quantization tool](post_training_quantization.md) since this is
broadly applicable and does not require training data.

For cases where the accuracy and latency targets are not met, or hardware
accelerator support is important,
[quantization-aware training](https://github.com/tensorflow/tensorflow/tree/r1.13/tensorflow/contrib/quantize){:.external}
is the better option. See additional optimization techniques under the
[Tensorflow Model Optimization Toolkit](https://www.tensorflow.org/model_optimization).

Note: Quantization-aware training supports a subset of convolutional neural network architectures.
