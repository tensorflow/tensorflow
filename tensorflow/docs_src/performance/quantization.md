# Fixed Point Quantization

Quantization techniques store and calculate numbers in more compact formats.
[TensorFlow Lite](/mobile/tflite/) adds quantization that uses an 8-bit fixed
point representation.

Since a challenge for modern neural networks is optimizing for high accuracy, the
priority has been improving accuracy and speed during training. Using floating
point arithmetic is an easy way to preserve accuracy and GPUs are designed to
accelerate these calculations.

However, as more machine learning models are deployed to mobile devices,
inference efficiency has become a critical issue. Where the computational demand
for *training* grows with the amount of models trained on different
architectures, the computational demand for *inference* grows in proportion to
the amount of users.

## Quantization benefits


Using 8-bit calculations help your models run faster and use less power. This is
especially important for mobile devices and embedded applications that can't run
floating point code efficiently, for example, Internet of Things (IoT) and
robotics devices. There are additional opportunities to extend this support to
more backends and research lower precision networks.

### Smaller file sizes {: .hide-from-toc}

Neural network models require a lot of space on disk. For example, the original
AlexNet requires over 200 MB for the float format—almost all of that for the
model's millions of weights. Because the weights are slightly different
floating point numbers, simple compression formats perform poorly (like zip).

Weights fall in large layers of numerical values. For each layer, weights tend to
be normally distributed within a range. Quantization can shrink file sizes by
storing the minimum and maximum weight for each layer, then compress each
weight's float value to an 8-bit integer representing the closest real number in
a linear set of 256 within the range.

### Faster inference {: .hide-from-toc}

Since calculations are run entirely on 8-bit inputs and outputs, quantization
reduces the computational resources needed for inference calculations. This is
more involved, requiring changes to all floating point calculations, but results
in a large speed-up for inference time.

### Memory efficiency {: .hide-from-toc}

Since fetching 8-bit values only requires 25% of the memory bandwidth of floats,
more efficient caches avoid bottlenecks for RAM access. In many cases, the power
consumption for running a neural network is dominated by memory access. The
savings from using fixed-point 8-bit weights and activations are significant. 

Typically, SIMD operations are available that run more operations per clock
cycle. In some cases, a DSP chip is available that accelerates 8-bit calculations
resulting in a massive speedup.

## Fixed point quantization techniques

The goal is to use the same precision for weights and activations during both
training and inference. But an important difference is that training consists of
a forward pass and a backward pass, while inference only uses a forward pass.
When we train the model with quantization in the loop, we ensure that the forward
pass matches precision for both training and inference.

To minimize the loss in accuracy for fully fixed point models (weights and
activations), train the model with quantization in the loop. This simulates
quantization in the forward pass of a model so weights tend towards values that
perform better during quantized inference. The backward pass uses quantized
weights and activations and models quantization as a straight through estimator.
(See Bengio et al., [2013](https://arxiv.org/abs/1308.3432))

Additionally, the minimum and maximum values for activations are determined
during training. This allows a model trained with quantization in the loop to be
converted to a fixed point inference model with little effort, eliminating the
need for a separate calibration step.

## Quantization training with TensorFlow

TensorFlow can train models with quantization in the loop. Because training
requires small gradient adjustments, floating point values are still used. To
keep models as floating point while adding the quantization error in the training
loop, @{$array_ops#Fake_quantization$fake quantization} nodes simulate the
effect of quantization in the forward and backward passes.

Since it's difficult to add these fake quantization operations to all the
required locations in the model, there's a function available that rewrites the
training graph. To create a fake quantized training graph:

```
# Build forward pass of model.
loss = tf.losses.get_total_loss()

# Call the training rewrite which rewrites the graph in-place with
# FakeQuantization nodes and folds batchnorm for training. It is
# often needed to fine tune a floating point model for quantization
# with this training tool. When training from scratch, quant_delay
# can be used to activate quantization after training to converge
# with the float graph, effectively fine-tuning the model.
tf.contrib.quantize.create_training_graph(quant_delay=2000000)

# Call backward pass optimizer as usual.
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
optimizer.minimize(loss)
```

The rewritten *eval graph* is non-trivially different from the *training graph*
since the quantization ops affect the batch normalization step. Because of this,
we've added a separate rewrite for the *eval graph*:

```
# Build eval model
logits = tf.nn.softmax_cross_entropy_with_logits_v2(...)

# Call the eval rewrite which rewrites the graph in-place with
# FakeQuantization nodes and fold batchnorm for eval.
tf.contrib.quantize.create_eval_graph()

# Save the checkpoint and eval graph proto to disk for freezing
# and providing to TFLite.
with open(eval_graph_file, ‘w’) as f:
  f.write(str(g.as_graph_def()))
saver = tf.train.Saver()
saver.save(sess, checkpoint_name)
```

Methods to rewrite the training and eval graphs are an active area of research
and experimentation. Although rewrites and quantized training might not work or
improve performance for all models, we are working to generalize these
techniques.

## Generating fully quantized models

The previously demonstrated after-rewrite eval graph only *simulates*
quantization. To generate real fixed point computations from a trained
quantization model, convert it to a fixed point kernel. Tensorflow Lite supports
this conversion from the graph resulting from `create_eval_graph`.

First, create a frozen graph that will be the input for the TensorFlow Lite
toolchain:

```
bazel build tensorflow/python/tools:freeze_graph && \
  bazel-bin/tensorflow/python/tools/freeze_graph \
  --input_graph=eval_graph_def.pb \
  --input_checkpoint=checkpoint \
  --output_graph=frozen_eval_graph.pb --output_node_names=outputs
```

Provide this to the TensorFlow Lite Optimizing Converter (TOCO) to get a fully
quantized TensorFLow Lite model:

```
bazel build tensorflow/contrib/lite/toco:toco && \
  ./bazel-bin/third_party/tensorflow/contrib/lite/toco/toco \
  --input_file=frozen_eval_graph.pb \
  --output_file=tflite_model.tflite \
  --input_format=TENSORFLOW_GRAPHDEF --output_format=TFLITE \
  --inference_type=QUANTIZED_UINT8 \
  --input_shape="1,224, 224,3" \
  --input_array=input \
  --output_array=outputs \
  --std_value=127.5 --mean_value=127.5
```

See the documentation for `tf.contrib.quantize` and
[TensorFlow Lite](/mobile/tflite/).

## Quantized accuracy

Fixed point [MobileNet](https://arxiv.org/abs/1704.0486) models are released with
8-bit weights and activations. Using the rewriters, these models achieve the
Top-1 accuracies listed in Table 1. For comparison, the floating point accuracies
are listed for the same models. The code used to generate these models
[is available](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md)
along with links to all of the pretrained mobilenet_v1 models.

<figure>
  <table>
    <tr>
      <th>Image Size</th>
      <th>Depth</th>
      <th>Top-1 Accuracy:<br>Floating point</th>
      <th>Top-1 Accuracy:<br>Fixed point: 8 bit weights and activations</th>
    </tr>
    <tr><td>128</td><td>0.25</td><td>0.415</td><td>0.399</td></tr>
    <tr><td>128</td><td>0.5</td><td>0.563</td><td>0.549</td></tr>
    <tr><td>128</td><td>0.75</td><td>0.621</td><td>0.598</td></tr>
    <tr><td>128</td><td>1</td><td>0.652</td><td>0.64</td></tr>
    <tr><td>160</td><td>0.25</td><td>0.455</td><td>0.435</td></tr>
    <tr><td>160</td><td>0.5</td><td>0.591</td><td>0.577</td></tr>
    <tr><td>160</td><td>0.75</td><td>0.653</td><td>0.639</td></tr>
    <tr><td>160</td><td>1</td><td>0.68</td><td>0.673</td></tr>
    <tr><td>192</td><td>0.25</td><td>0.477</td><td>0.458</td></tr>
    <tr><td>192</td><td>0.5</td><td>0.617</td><td>0.604</td></tr>
    <tr><td>192</td><td>0.75</td><td>0.672</td><td>0.662</td></tr>
    <tr><td>192</td><td>1</td><td>0.7</td><td>0.69</td></tr>
    <tr><td>224</td><td>0.25</td><td>0.498</td><td>0.482</td></tr>
    <tr><td>224</td><td>0.5</td><td>0.633</td><td>0.622</td></tr>
    <tr><td>224</td><td>0.75</td><td>0.684</td><td>0.679</td></tr>
    <tr><td>224</td><td>1</td><td>0.709</td><td>0.697</td></tr>
  </table>
  <figcaption>
    <b>Table 1</b>: MobileNet Top-1 accuracy on Imagenet Validation dataset.
  </figcaption>
</figure>

## Representation for quantized tensors

TensorFlow approaches the conversion of floating-point arrays of numbers into
8-bit representations as a compression problem. Since the weights and activation
tensors in trained neural network models tend to have values that are distributed
across comparatively small ranges (for example, -15 to +15 for weights or -500 to
1000 for image model activations). And since neural nets tend to be robust
handling noise, the error introduced by quantizing to a small set of values
maintains the precision of the overall results within an acceptable threshold. A
chosen representation must perform fast calculations, especially the large matrix
multiplications that comprise the bulk of the computations while running a model.

This is represented with two floats that store the overall minimum and maximum
values corresponding to the lowest and highest quantized value. Each entry in the
quantized array represents a float value in that range, distributed linearly
between the minimum and maximum. For example, with a minimum of -10.0 and maximum
of 30.0f, and an 8-bit array, the quantized values represent the following:

<figure>
  <table>
    <tr><th>Quantized</th><th>Float</th></tr>
    <tr><td>0</td><td>-10.0</td></tr>
    <tr><td>128</td><td>10.0</td></tr>
    <tr><td>255</td><td>30.0</td></tr>
  </table>
  <figcaption>
    <b>Table 2</b>: Example quantized value range
  </figcaption>
</figure>

The advantages of this representation format are:

* It efficiently represents an arbitrary magnitude of ranges.
* The values don't have to be symmetrical.
* The format represents both signed and unsigned values.
* The linear spread makes multiplications straightforward.

Alternative techniques use lower bit depths by non-linearly distributing the
float values across the representation, but currently are more expensive in terms
of computation time. (See Han et al.,
[2016](https://arxiv.org/abs/1510.00149).)

The advantage of having a clear definition of the quantized format is that it's
always possible to convert back and forth from fixed-point to floating-point for
operations that aren't quantization-ready, or to inspect the tensors for
debugging.
