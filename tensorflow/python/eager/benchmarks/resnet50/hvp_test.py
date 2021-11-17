# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests and benchmarks for Hessian-vector products with ResNet50."""

import gc
import time

from absl.testing import parameterized
import tensorflow as tf

from tensorflow.python.eager import forwardprop
from tensorflow.python.eager.benchmarks.resnet50 import resnet50
from tensorflow.python.eager.benchmarks.resnet50 import resnet50_test_util


def _forward_over_back_hvp(model, images, labels, vector):
  with forwardprop.ForwardAccumulator(
      model.trainable_variables, vector) as acc:
    with tf.GradientTape() as grad_tape:
      logits = model(images, training=True)
      loss = tf.compat.v1.losses.softmax_cross_entropy(
          logits=logits, onehot_labels=labels)
    grads = grad_tape.gradient(loss, model.trainable_variables)
  return acc.jvp(grads)


def _back_over_forward_hvp(model, images, labels, vector):
  with tf.GradientTape() as grad_tape:
    grad_tape.watch(model.trainable_variables)
    with forwardprop.ForwardAccumulator(
        model.trainable_variables, vector) as acc:
      logits = model(images, training=True)
      loss = tf.compat.v1.losses.softmax_cross_entropy(
          logits=logits, onehot_labels=labels)
  return grad_tape.gradient(acc.jvp(loss), model.trainable_variables)


def _tf_gradients_forward_over_back_hvp(model, images, labels, vector):
  with tf.GradientTape() as grad_tape:
    logits = model(images, training=True)
    loss = tf.compat.v1.losses.softmax_cross_entropy(
        logits=logits, onehot_labels=labels)
  variables = model.trainable_variables
  grads = grad_tape.gradient(loss, variables)
  helpers = tf.nest.map_structure(tf.ones_like, grads)
  transposing = tf.gradients(grads, variables, helpers)
  return tf.gradients(transposing, helpers, vector)


def _back_over_back_hvp(model, images, labels, vector):
  with tf.GradientTape() as outer_tape:
    with tf.GradientTape() as inner_tape:
      logits = model(images, training=True)
      loss = tf.compat.v1.losses.softmax_cross_entropy(
          logits=logits, onehot_labels=labels)
    grads = inner_tape.gradient(loss, model.trainable_variables)
  return outer_tape.gradient(
      grads, model.trainable_variables, output_gradients=vector)


class HVPTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ("forward_over_back_eager", _forward_over_back_hvp),
      ("forward_over_back_function", tf.function(_forward_over_back_hvp)),
      ("tf_gradients", tf.function(_tf_gradients_forward_over_back_hvp)),
      ("back_over_back_eager", _back_over_back_hvp),
      ("back_over_back_function", tf.function(_back_over_back_hvp)),
      ("back_over_forward_eager", _back_over_forward_hvp),
      ("back_over_forward_function", tf.function(_back_over_forward_hvp)))
  def test_hvp_shapes(self, hvp_function):
    device, data_format = resnet50_test_util.device_and_data_format()
    model = resnet50.ResNet50(data_format)
    with tf.device(device):
      images, labels = resnet50_test_util.random_batch(2, data_format)
      images = tf.constant(images)
      labels = tf.constant(labels)
      model.build(images.shape)
      vector = [tf.ones_like(v) for v in model.trainable_variables]

      # Note that numerical differences build up to quite large differences here
      # in the final hvp. tensorflow/python/eager:forwardprop_test has a
      # smaller-scale test that the computations are close on a much smaller but
      # otherwise similar model.
      hvp = hvp_function(model, images, labels, vector)
      for hvp_component, variable in zip(hvp, model.trainable_variables):
        self.assertEqual(hvp_component.shape, variable.shape)
        self.assertEqual(hvp_component.dtype, variable.dtype)


class HVPBenchmarks(tf.test.Benchmark):

  def _force_device_sync(self):
    # If this function is called in the context of a non-CPU device
    # (e.g., inside a 'with tf.device("/gpu:0")' block)
    # then this will force a copy from CPU->NON_CPU_DEVICE->CPU,
    # which forces a sync. This is a roundabout way, yes.
    tf.constant(1.).cpu()

  def _hvp_benchmark(self, hvp_fn, label, batch_sizes,
                     num_iters=30, num_burn=5):
    device, data_format = resnet50_test_util.device_and_data_format()
    model = resnet50.ResNet50(data_format)
    for batch_size in batch_sizes:
      with tf.device(device):
        images, labels = resnet50_test_util.random_batch(
            batch_size, data_format)
        images = tf.constant(images)
        labels = tf.constant(labels)
        model.build(images.shape)
        vector = [tf.ones_like(v) for v in model.trainable_variables]
        for _ in range(num_burn):
          results = hvp_fn(model, images, labels, vector)
          for result in results:
            result.cpu()
        self._force_device_sync()
        gc.collect()
        start = time.time()
        for _ in range(num_iters):
          results = hvp_fn(model, images, labels, vector)
          for result in results:
            result.cpu()
        self._force_device_sync()
        resnet50_test_util.report(
            self, label, start, num_iters, device, batch_size, data_format)

  def benchmark_forward_over_backward_hvp_eager(self):
    self._hvp_benchmark(_forward_over_back_hvp,
                        "forward_over_backward_hvp_eager",
                        batch_sizes=[8])

  def benchmark_forward_over_backward_hvp_function(self):
    self._hvp_benchmark(tf.function(_forward_over_back_hvp),
                        "forward_over_backward_hvp_function",
                        batch_sizes=[8])

  def benchmark_tf_gradients_forward_over_backward_hvp_function(self):
    self._hvp_benchmark(tf.function(_tf_gradients_forward_over_back_hvp),
                        "tf_gradients_forward_over_backward_hvp_function",
                        batch_sizes=[8])

  def benchmark_backward_over_backward_hvp_eager(self):
    self._hvp_benchmark(_back_over_back_hvp,
                        "backward_over_backward_hvp_eager",
                        batch_sizes=[8])

  def benchmark_backward_over_backward_hvp_function(self):
    self._hvp_benchmark(tf.function(_back_over_back_hvp),
                        "backward_over_backward_hvp_function",
                        batch_sizes=[8])

  def benchmark_backward_over_forward_hvp_eager(self):
    self._hvp_benchmark(_back_over_forward_hvp,
                        "backward_over_forward_hvp_eager",
                        batch_sizes=[8])

  def benchmark_backward_over_forward_hvp_function(self):
    self._hvp_benchmark(tf.function(_back_over_forward_hvp),
                        "backward_over_forward_hvp_function",
                        batch_sizes=[8])


if __name__ == "__main__":
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
