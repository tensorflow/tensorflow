# Quantization-aware training

Quantization-aware model training ensures that the forward pass matches precision
for both training and inference. There are two aspects to this:

* Operator fusion at inference time are accurately modeled at training time.
* Quantization effects at inference are modeled at training time.

For efficient inference, TensorFlow combines batch normalization with the preceding
convolutional and fully-connected layers prior to quantization by
[folding batch norm layers](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/quantize/python/fold_batch_norms.py){:.external}. 

The quantization error is modeled using [fake quantization](../api_guides/python/array_ops.md#Fake_quantization)
nodes to simulate the effect of quantization in the forward and backward passes. The
forward-pass models quantization, while the backward-pass models quantization as a
straight-through estimator. Both the forward- and backward-pass simulate the quantization
of weights and activations. Note that during back propagation, the parameters are
updated at high precision as this is needed to ensure sufficient precision in
accumulating tiny adjustments to the parameters.


Additionally, the minimum and maximum values for activations are determined
during training. This allows a model trained with quantization in the loop to be
converted to a fixed point inference model with little effort, eliminating the
need for a separate calibration step.

Since it's difficult to add these fake quantization operations to all the
required locations in the model, there's a function available that rewrites the
training graph. To create a fake quantized training graph:

```python
# Build forward pass of model.
loss = tf.losses.get_total_loss()

# Call the training rewrite which rewrites the graph in-place with
# FakeQuantization nodes and folds batchnorm for training. It is
# often needed to fine tune a floating point model for quantization
# with this training tool. When training from scratch, quant_delay
# can be used to activate quantization after training to converge
# with the float graph, effectively fine-tuning the model.
g = tf.get_default_graph()
tf.contrib.quantize.create_training_graph(input_graph=g,
                                          quant_delay=2000000)

# Call backward pass optimizer as usual.
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
optimizer.minimize(loss)
```

The rewritten *eval graph* is non-trivially different from the *training graph*
since the quantization ops affect the batch normalization step. Because of this,
we've added a separate rewrite for the *eval graph*:

```python
# Build eval model
logits = tf.nn.softmax_cross_entropy_with_logits_v2(...)

# Call the eval rewrite which rewrites the graph in-place with
# FakeQuantization nodes and fold batchnorm for eval.
g = tf.get_default_graph()
tf.contrib.quantize.create_eval_graph(input_graph=g)

# Save the checkpoint and eval graph proto to disk for freezing
# and providing to TFLite.
with open(eval_graph_file, ‘w’) as f:
  f.write(str(g.as_graph_def()))
saver = tf.train.Saver()
saver.save(sess, checkpoint_name)
```

Methods to rewrite the training and eval graphs are an active area of research
and experimentation. Although rewrites and quantized training might not work or
improve performance for all models, we are working to generalize these techniques.


## Generating fully-quantized models

The previously demonstrated after-rewrite eval graph only *simulates*
quantization. To generate real fixed-point computations from a trained
quantization model, convert it to a fixed-point kernel. TensorFlow Lite supports
this conversion from the graph resulting from `create_eval_graph`.

First, create a frozen graph that will be the input for the TensorFlow Lite
toolchain:

```
freeze_graph \
  --input_graph=eval_graph_def.pb \
  --input_checkpoint=checkpoint \
  --output_graph=frozen_eval_graph.pb --output_node_names=outputs
```

Provide this to the TensorFlow Lite Optimizing Converter (TOCO) to get a
fully-quantized TensorFlow Lite model:

```
toco \
  --input_file=frozen_eval_graph.pb \
  --output_file=tflite_model.tflite \
  --input_format=TENSORFLOW_GRAPHDEF --output_format=TFLITE \
  --inference_type=QUANTIZED_UINT8 \
  --input_shape="1,224, 224,3" \
  --input_array=input \
  --output_array=outputs \
  --std_value=127.5 --mean_value=127.5
```

See the documentation for `tf.contrib.quantize` and [TensorFlow Lite](../lite/).


## Quantized accuracy results

The following are results of training some popular CNN models (Mobilenet-v1,
Mobilenet-v2, and Inception-v3) using this tool:

<figure>
  <table>
    <tr>
      <th>Model</th>
      <th>Top-1 Accuracy:<br>Floating point</th>
      <th>Top-1 Accuracy:<br>Fixed point: 8 bit weights and activations</th>
    </tr>
    <tr><td>Mobilenet-v1-128-0.25</td><td>0.415</td><td>0.399</td></tr>
    <tr><td>Mobilenet-v1-128-0.5</td><td>0.563</td><td>0.549</td></tr>
    <tr><td>Mobilenet-v1-128-0.75</td><td>0.621</td><td>0.598</td></tr>
    <tr><td>Mobilenet-v1-128-1</td><td>0.652</td><td>0.64</td></tr>
    <tr><td>Mobilenet-v1-160-0.25</td><td>0.455</td><td>0.435</td></tr>
    <tr><td>Mobilenet-v1-160-0.5</td><td>0.591</td><td>0.577</td></tr>
    <tr><td>Mobilenet-v1-160-0.75</td><td>0.653</td><td>0.639</td></tr>
    <tr><td>Mobilenet-v1-160-1</td><td>0.68</td><td>0.673</td></tr>
    <tr><td>Mobilenet-v1-192-0.25</td><td>0.477</td><td>0.458</td></tr>
    <tr><td>Mobilenet-v1-192-0.5</td><td>0.617</td><td>0.604</td></tr>
    <tr><td>Mobilenet-v1-192-0.75</td><td>0.672</td><td>0.662</td></tr>
    <tr><td>Mobilenet-v1-192-1</td><td>0.7</td><td>0.69</td></tr>
    <tr><td>Mobilenet-v1-224-0.25</td><td>0.498</td><td>0.482</td></tr>
    <tr><td>Mobilenet-v1-224-0.5</td><td>0.633</td><td>0.622</td></tr>
    <tr><td>Mobilenet-v1-224-0.75</td><td>0.684</td><td>0.679</td></tr>
    <tr><td>Mobilenet-v1-224-1</td><td>0.709</td><td>0.697</td></tr>
    <tr><td>Mobilenet-v2-224-1</td><td>0.718</td><td>0.708</td></tr>
   <tr><td>Inception_v3</td><td>0.78</td><td>0.775</td></tr>
  </table>
  <figcaption>
    <b>Table 1</b>: Top-1 accuracy of floating point and fully quantized CNNs on Imagenet Validation dataset.
  </figcaption>
</figure>

Our pre-trained models are available in the
<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/models.md#image-classification-quantized-models" class="external">TensorFlow Lite model repository</a>. The code used to generate
these models <a href="https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1_train.py" class="external">is available</a>.



These rewrites are an active area of research and experimentation, so the
rewrites and quantized training will likely not work across all models, though
we hope to work towards generalizing these techniques.

[1] B.Jacob et al., "Quantization and Training of Neural Networks for Efficient
Integer-Arithmetic-Only Inference", https://arxiv.org/abs/1712.05877

[2] P.Gysel et al., "HARDWARE-ORIENTED APPROXIMATION OF CONVOLUTIONAL
NEURAL NETWORKS", https://arxiv.org/pdf/1604.03168.pdf

[3] Y.Bengio et al., "Estimating or Propagating Gradients Through Stochastic
Neurons for Conditional Computation", https://arxiv.org/abs/1308.3432
