"""End-to-end benchmark for batch normalization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow as tf


def batch_norm_op(tensor, mean, variance, beta, gamma, scale):
  """Fused kernel for batch normalization."""
  return tf.nn.batch_norm_with_global_normalization(tensor, mean, variance,
                                                    beta, gamma, 0.001, scale)

# Note that the naive implementation is much much slower:
# batch_norm = (tensor - mean) * tf.rsqrt(variance + 0.001)
# if scale:
#   batch_norm *= gamma
# return batch_norm + beta
def batch_norm_py(tensor, mean, variance, beta, gamma, scale):
  """Python implementation of batch normalization."""
  inv = tf.rsqrt(variance + 0.001)
  if scale:
    inv *= gamma
  return tensor * inv + (beta - mean * inv)


def build_graph(device, input_shape, axes, num_layers, py, scale, train):
  """Build a graph containing a sequence of batch normalizations.

  Args:
    device: string, the device to run on.
    input_shape: shape of the input tensor.
    axes: axes that are to be normalized across.
    num_layers: number of batch normalization layers in the graph.
    py: whether to use the python implementation.
    scale: scale after normalization.
    train: if true, also run backprop.

  Returns:
    An array of tensors to run()
  """
  moment_shape = []
  for axis in range(len(input_shape)):
    if axis not in axes:
      moment_shape.append(input_shape[axis])
  with tf.device(device):
    tensor = tf.Variable(tf.truncated_normal(input_shape))
    for _ in range(num_layers):
      mean, variance = tf.nn.moments(tensor, axes)
      beta = tf.Variable(tf.zeros(moment_shape))
      gamma = tf.Variable(tf.constant(1.0, shape=moment_shape))
      if py:
        tensor = batch_norm_py(tensor, mean, variance, beta, gamma, scale)
      else:
        tensor = batch_norm_op(tensor, mean, variance, beta, gamma, scale)
    if train:
      return tf.gradients([tensor], tf.trainable_variables())
    else:
      return [tensor]


def run_graph(device, input_shape, axes, num_layers, py, scale, train,
              num_iters):
  """Run the graph and print its execution time.

  Args:
    device: string, the device to run on.
    input_shape: shape of the input tensor.
    axes: axes that are to be normalized across.
    num_layers: number of batch normalization layers in the graph.
    py: whether to use the python implementation.
    scale: scale after normalization.
    train: if true, also run backprop.
    num_iters: number of steps to run.
  """
  graph = tf.Graph()
  with graph.as_default():
    outputs = build_graph(device, input_shape, axes, num_layers, py, scale,
                          train)
  with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    _ = session.run([out.op for out in outputs])  # warm up.
    start_time = time.time()
    for _ in range(num_iters):
      _ = session.run([out.op for out in outputs])
    duration = time.time() - start_time
    print("shape:%d/%d #layers:%d python:%r scale:%r train:%r - %f secs" %
          (len(input_shape), len(axes), num_layers, py, scale, train, duration))


def main(unused_argv):
  print("Forward convolution.")
  run_graph("/cpu:0", [8, 128, 128, 32], [0, 1, 2], 5, False, True, False, 5)
  run_graph("/cpu:0", [8, 128, 128, 32], [0, 1, 2], 5, True, True, False, 5)
  print("Forward/backward convolution.")
  run_graph("/cpu:0", [8, 128, 128, 32], [0, 1, 2], 5, False, True, True, 5)
  run_graph("/cpu:0", [8, 128, 128, 32], [0, 1, 2], 5, True, True, True, 5)
  print("Forward fully-connected.")
  # Not implemented yet in TF.
  # run_graph("/cpu:0", [1024, 32], [0], 10, False, True, False, 5)
  run_graph("/cpu:0", [1024, 32], [0], 10, True, True, False, 5)
  print("Forward/backward fully-connected.")
  # Not implemented yet in TF.
  # run_graph("/cpu:0", [1024, 32], [0], 10, False, True, True, 5)
  run_graph("/cpu:0", [1024, 32], [0], 10, True, True, True, 5)


if __name__ == "__main__":
  tf.app.run()
