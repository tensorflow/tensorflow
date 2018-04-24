# Quantized Training Rewrites

tf.contrib.quantize provides tools for transforming graphs to include ops to
model quantization of weights, biases and activations during both training and
inference. The details of the transformation implemented in this package is
described here [1].

This is done using the
[fake quantization op](https://www.tensorflow.org/versions/r0.12/api_docs/python/array_ops/fake_quantization).

Literature has shown that fixed point networks provide comparable performance to
floating point networks [2]. This is achieved by modeling the quantization
operation during training in both the forward and backward passes.
The fake quantization operator achieves this by modeling the quantizer as a pass
through estimator [3]. Note that during back propagation, the parameters are
updated at high precision as this is needed to ensure sufficient precision in
accumulating tiny adjustments to the parameters. However, for the forward pass,
the parameters and activations are quantized to the desired lower precision.

## How to use the Rewrites

tf.contrib.quantize provides two rewrites, one to train for quantization and
one to create a [TensorFlow Lite](https://www.tensorflow.org/mobile/tflite/)
compatible eval graph.

```
# Build forward pass of model.
…
loss = tf.losses.get_total_loss()

# Call the training rewrite which rewrites the graph in-place with FakeQuantization nodes
# and folds batchnorm for training.
# It is often needed to finetune a floating point model for quantization with this training tool.
# When training from scratch, quant_delay can be used to activate quantization after
# training to convergence with the float graph, effectively finetuning the model.
tf.contrib.quantize.create_training_graph(quant_delay=2000000)

# Call backward pass optimizer as usual.
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
optimizer.minimize(loss)
```

Additionally, the rewritten eval graph is non-trivially different from the
training graph due the effects of quantization on batch normalization. Thus,
we offer a separate rewrite for the eval_graph.

```
# Build eval model
…
logits = tf.nn.softmax_cross_entropy_with_logits(...)

# Call the eval rewrite which rewrites the graph in-place with FakeQuantization nodes
# and fold batchnorm for eval.
tf.contrib.quantize.create_eval_graph()

# Save the checkpoint and eval graph proto to disk for freezing and providing to TFLite.
with open(eval_graph_file, ‘w’) as f:
  f.write(str(g.as_graph_def()))
saver = tf.train.Saver()
saver.save(sess, checkpoint_name)
```

These rewrites are an active area of research and experimentation, so the
rewrites and quantized training will likely not work across all models, though
we hope to work towards generalizing these techniques.

[1] B.Jacob et al., "Quantization and Training of Neural Networks for Efficient
Integer-Arithmetic-Only Inference", https://arxiv.org/abs/1712.05877

[2] P.Gysel et al., "HARDWARE-ORIENTED APPROXIMATION OF CONVOLUTIONAL
NEURAL NETWORKS", https://arxiv.org/pdf/1604.03168.pdf

[3] Y.Bengio et al., "Estimating or Propagating Gradients Through Stochastic
Neurons for Conditional Computation", https://arxiv.org/abs/1308.3432
