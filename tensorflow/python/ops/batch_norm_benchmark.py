# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""End-to-end benchmark for batch normalization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow as tf

from tensorflow.python.ops import gen_nn_ops
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean("use_gpu", True, """Run GPU benchmarks.""")


def batch_norm_op(tensor, mean, variance, beta, gamma, scale):
  """Fused kernel for batch normalization."""
  # _batch_norm_with_global_normalization is deprecated in v9
  tf.get_default_graph().graph_def_versions.producer = 8
  # pylint: disable=protected-access
  return gen_nn_ops._batch_norm_with_global_normalization(
      tensor, mean, variance, beta, gamma, 0.001, scale)
  # pylint: enable=protected-access


# Note that the naive implementation is much much slower:
# batch_norm = (tensor - mean) * tf.rsqrt(variance + 0.001)
# if scale:
#   batch_norm *= gamma
# return batch_norm + beta
def batch_norm_py(tensor, mean, variance, beta, gamma, scale):
  """Python implementation of batch normalization."""
  return tf.nn.batch_normalization(
      tensor, mean, variance, beta, gamma if scale else None, 0.001)


def batch_norm_slow(tensor, mean, variance, beta, gamma, scale):
  batch_norm = (tensor - mean) * tf.rsqrt(variance + 0.001)
  if scale:
    batch_norm *= gamma
  return batch_norm + beta


def build_graph(device, input_shape, axes, num_layers, mode, scale, train):
  """Build a graph containing a sequence of batch normalizations.

  Args:
    device: string, the device to run on.
    input_shape: shape of the input tensor.
    axes: axes that are to be normalized across.
    num_layers: number of batch normalization layers in the graph.
    mode: "op", "py" or "slow" depending on the implementation.
    scale: scale after normalization.
    train: if true, also run backprop.

  Returns:
    An array of tensors to run()
  """
  moment_shape = []
  keep_dims = mode == "py" or mode == "slow"
  if keep_dims:
    for axis in range(len(input_shape)):
      if axis in axes:
        moment_shape.append(1)
      else:
        moment_shape.append(input_shape[axis])
  else:
    for axis in range(len(input_shape)):
      if axis not in axes:
        moment_shape.append(input_shape[axis])
  with tf.device("/%s:0" % device):
    tensor = tf.Variable(tf.truncated_normal(input_shape))
    for _ in range(num_layers):
      mean, variance = tf.nn.moments(tensor, axes, keep_dims=keep_dims)
      beta = tf.Variable(tf.zeros(moment_shape))
      gamma = tf.Variable(tf.constant(1.0, shape=moment_shape))
      if mode == "py":
        tensor = batch_norm_py(tensor, mean, variance, beta, gamma, scale)
      elif mode == "op":
        tensor = batch_norm_op(tensor, mean, variance, beta, gamma, scale)
      elif mode == "slow":
        tensor = batch_norm_slow(tensor, mean, variance, beta, gamma, scale)
    if train:
      return tf.gradients([tensor], tf.trainable_variables())
    else:
      return [tensor]


def print_difference(mode, t1, t2):
  """Print the difference in timing between two runs."""
  difference = (t2 - t1) / t1 * 100.0
  print("=== %s: %.1f%% ===" % (mode, difference))


class BatchNormBenchmark(tf.test.Benchmark):
  """Benchmark batch normalization."""

  def _run_graph(
      self, device, input_shape, axes, num_layers, mode, scale, train,
      num_iters):
    """Run the graph and print its execution time.

    Args:
      device: string, the device to run on.
      input_shape: shape of the input tensor.
      axes: axes that are to be normalized across.
      num_layers: number of batch normalization layers in the graph.
      mode: "op", "py" or "slow" depending on the implementation.
      scale: scale after normalization.
      train: if true, also run backprop.
      num_iters: number of steps to run.

    Returns:
      The duration of the run in seconds.
    """
    graph = tf.Graph()
    with graph.as_default():
      outputs = build_graph(device, input_shape, axes, num_layers, mode, scale,
                            train)
    with tf.Session(graph=graph) as session:
      tf.initialize_all_variables().run()
      _ = session.run([out.op for out in outputs])  # warm up.
      start_time = time.time()
      for _ in range(num_iters):
        _ = session.run([out.op for out in outputs])
      duration = time.time() - start_time
    print("%s shape:%d/%d #layers:%d mode:%s scale:%r train:%r - %f secs" %
          (device, len(input_shape), len(axes), num_layers, mode, scale, train,
           duration / num_iters))

    name_template = (
        "batch_norm_{device}_input_shape_{shape}_axes_{axes}_mode_{mode}_"
        "layers_{num_layers}_scale_{scale}_"
        "train_{train}")

    self.report_benchmark(
        name=name_template.format(
            device=device, mode=mode, num_layers=num_layers, scale=scale,
            train=train,
            shape=str(input_shape).replace(" ", ""),
            axes=str(axes)).replace(" ", ""),
        iters=num_iters, wall_time=duration / num_iters)

    return duration

  def benchmark_batch_norm(self):
    print("Forward convolution (lower layers).")
    shape = [8, 128, 128, 32]
    axes = [0, 1, 2]
    t1 = self._run_graph("cpu", shape, axes, 10, "op", True, False, 5)
    t2 = self._run_graph("cpu", shape, axes, 10, "py", True, False, 5)
    t3 = self._run_graph("cpu", shape, axes, 10, "slow", True, False, 5)
    print_difference("op vs py", t1, t2)
    print_difference("py vs slow", t2, t3)
    if FLAGS.use_gpu:
      t1 = self._run_graph("gpu", shape, axes, 10, "op", True, False, 50)
      t2 = self._run_graph("gpu", shape, axes, 10, "py", True, False, 50)
      t3 = self._run_graph("gpu", shape, axes, 10, "slow", True, False, 50)
      print_difference("op vs py", t1, t2)
      print_difference("py vs slow", t2, t3)
    print("Forward/backward convolution (lower layers).")
    t1 = self._run_graph("cpu", shape, axes, 10, "op", True, True, 5)
    t2 = self._run_graph("cpu", shape, axes, 10, "py", True, True, 5)
    t3 = self._run_graph("cpu", shape, axes, 10, "slow", True, True, 5)
    print_difference("op vs py", t1, t2)
    print_difference("py vs slow", t2, t3)
    if FLAGS.use_gpu:
      t1 = self._run_graph("gpu", shape, axes, 10, "op", True, True, 50)
      t2 = self._run_graph("gpu", shape, axes, 10, "py", True, True, 50)
      t2 = self._run_graph("gpu", shape, axes, 10, "slow", True, True, 50)
      print_difference("op vs py", t1, t2)
      print_difference("py vs slow", t2, t3)
    print("Forward convolution (higher layers).")
    shape = [256, 17, 17, 32]
    axes = [0, 1, 2]
    t1 = self._run_graph("cpu", shape, axes, 10, "op", True, False, 5)
    t2 = self._run_graph("cpu", shape, axes, 10, "py", True, False, 5)
    t3 = self._run_graph("cpu", shape, axes, 10, "slow", True, False, 5)
    print_difference("op vs py", t1, t2)
    print_difference("py vs slow", t2, t3)
    if FLAGS.use_gpu:
      t1 = self._run_graph("gpu", shape, axes, 10, "op", True, False, 50)
      t2 = self._run_graph("gpu", shape, axes, 10, "py", True, False, 50)
      t3 = self._run_graph("gpu", shape, axes, 10, "slow", True, False, 50)
      print_difference("op vs py", t1, t2)
      print_difference("py vs slow", t2, t3)
    print("Forward/backward convolution (higher layers).")
    t1 = self._run_graph("cpu", shape, axes, 10, "op", True, True, 5)
    t2 = self._run_graph("cpu", shape, axes, 10, "py", True, True, 5)
    t3 = self._run_graph("cpu", shape, axes, 10, "slow", True, True, 5)
    print_difference("op vs py", t1, t2)
    print_difference("py vs slow", t2, t3)
    if FLAGS.use_gpu:
      t1 = self._run_graph("gpu", shape, axes, 10, "op", True, True, 50)
      t2 = self._run_graph("gpu", shape, axes, 10, "py", True, True, 50)
      t3 = self._run_graph("gpu", shape, axes, 10, "slow", True, True, 50)
      print_difference("op vs py", t1, t2)
      print_difference("py vs slow", t2, t3)
    print("Forward fully-connected.")
    shape = [1024, 32]
    axes = [0]
    t1 = self._run_graph("cpu", shape, axes, 10, "py", True, False, 5)
    t2 = self._run_graph("cpu", shape, axes, 10, "slow", True, False, 5)
    print_difference("py vs slow", t1, t2)
    if FLAGS.use_gpu:
      t1 = self._run_graph("gpu", shape, axes, 10, "py", True, False, 50)
      t2 = self._run_graph("gpu", shape, axes, 10, "slow", True, False, 50)
      print_difference("py vs slow", t1, t2)
    print("Forward/backward fully-connected.")
    t1 = self._run_graph("cpu", shape, axes, 10, "py", True, True, 50)
    t2 = self._run_graph("cpu", shape, axes, 10, "slow", True, True, 50)
    print_difference("py vs slow", t1, t2)
    if FLAGS.use_gpu:
      t1 = self._run_graph("gpu", shape, axes, 10, "py", True, True, 5)
      t2 = self._run_graph("gpu", shape, axes, 10, "slow", True, True, 5)
      print_difference("py vs slow", t1, t2)


if __name__ == "__main__":
  tf.test.main()
