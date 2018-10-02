# TPU support for TensorFlow #

This directory contains code required to re-target a TensorFlow model to run
on TPUs.

## Example usage - TPU Estimator

Below shows example usage of the TPU Estimator for a simple convolutional
network.

```python
import tensorflow as tf

from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer

def model_fn(features, labels, mode, params):
  # Define the model to construct the logits
  logits = # ...
  loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
  optimizer = tpu_optimizer.CrossShardOptimizer(
    tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate))
  train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
  return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

def input_fn(params):
  # ...
  pass

def main():
  run_config = tpu_config.RunConfig(
    master=FLAGS.master,
    # ...
  )
  estimator = tpu_estimator.TpuEstimator(
    model_fn=model_fn,
    use_tpu=FLAGS.use_tpu,
    config=run_config,
    batch_size=FLAGS.batch_size)
  estimator.train(input_fn=input_fn, max_steps=FLAGS.train_steps)
```

For the complete [executable] example, see our open source TPU models.
