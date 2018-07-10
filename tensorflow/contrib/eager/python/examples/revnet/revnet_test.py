# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for basic building blocks used in eager mode RevNet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gc
import time

import tensorflow as tf
from tensorflow.contrib.eager.python.examples.revnet import blocks_test
from tensorflow.contrib.eager.python.examples.revnet import config as config_
from tensorflow.contrib.eager.python.examples.revnet import revnet
from tensorflow.python.client import device_lib
tfe = tf.contrib.eager


def train_one_iter(model, inputs, labels, optimizer, global_step=None):
  """Train for one iteration."""
  grads, vars_, loss = model.compute_gradients(inputs, labels, training=True)
  optimizer.apply_gradients(zip(grads, vars_), global_step=global_step)

  return loss


class RevNetTest(tf.test.TestCase):

  def setUp(self):
    super(RevNetTest, self).setUp()
    config = config_.get_hparams_cifar_38()
    # Reconstruction could cause numerical error, use double precision for tests
    config.dtype = tf.float64
    config.fused = False  # Fused batch norm does not support tf.float64
    shape = (config.batch_size,) + config.input_shape
    self.model = revnet.RevNet(config=config)
    self.x = tf.random_normal(shape=shape, dtype=tf.float64)
    self.t = tf.random_uniform(
        shape=[config.batch_size],
        minval=0,
        maxval=config.n_classes,
        dtype=tf.int64)
    self.config = config

  def tearDown(self):
    del self.model
    del self.x
    del self.t
    del self.config
    super(RevNetTest, self).tearDown()

  def test_call(self):
    """Test `call` function."""

    y, _ = self.model(self.x, training=False)
    self.assertEqual(y.shape, [self.config.batch_size, self.config.n_classes])

  def _check_grad_angle_combined(self, grads, grads_true):
    """Verify that the reconstructed gradients has correct direction.

    Due to numerical imprecision, the magnitude may be slightly different.
    Yet according to the paper, the angle should be roughly the same.

    Args:
      grads: list of gradients from reconstruction
      grads_true: list of true gradients
    """

    def _combine(gs):
      return [tf.reshape(g, [-1]) for g in gs]

    g1_all = tf.concat(_combine(grads), axis=0)
    g2_all = tf.concat(_combine(grads_true), axis=0)

    self.assertEqual(len(g1_all.shape), 1)
    self.assertEqual(len(g2_all.shape), 1)

    degree = blocks_test.compute_degree(g1_all, g2_all)
    self.assertLessEqual(degree, 1e0)

  def test_compute_gradients(self):
    """Test `compute_gradients` function."""
    self.model(self.x, training=False)  # Initialize model
    grads, vars_, loss = self.model.compute_gradients(
        inputs=self.x, labels=self.t, training=True, l2_reg=True)
    self.assertTrue(isinstance(grads, list))
    self.assertTrue(isinstance(vars_, list))
    self.assertEqual(len(grads), len(vars_))
    for grad, var in zip(grads, vars_):
      self.assertEqual(grad.shape, var.shape)

    # Compare against the true gradient computed by the tape
    with tf.GradientTape() as tape:
      logits, _ = self.model(self.x, training=True)
      loss_true = self.model.compute_loss(logits=logits, labels=self.t)
    grads_true = tape.gradient(loss_true, vars_)
    self.assertAllClose(loss, loss_true)
    self.assertAllClose(grads, grads_true, rtol=1e-4, atol=1e-4)
    self._check_grad_angle_combined(grads, grads_true)

  def test_call_defun(self):
    """Test `call` function with defun."""
    y, _ = tfe.defun(self.model.call)(self.x, training=False)
    self.assertEqual(y.shape, [self.config.batch_size, self.config.n_classes])

  def test_compute_gradients_defun(self):
    """Test `compute_gradients` function with defun."""
    compute_gradients = tfe.defun(self.model.compute_gradients)
    grads, vars_, _ = compute_gradients(self.x, self.t, training=True)
    self.assertTrue(isinstance(grads, list))
    self.assertTrue(isinstance(vars_, list))
    self.assertEqual(len(grads), len(vars_))
    for grad, var in zip(grads, vars_):
      if grad is not None:
        self.assertEqual(grad.shape, var.shape)

  def test_training_graph(self):
    """Test model training in graph mode."""
    with tf.Graph().as_default():
      config = config_.get_hparams_cifar_38()
      x = tf.random_normal(
          shape=(self.config.batch_size,) + self.config.input_shape)
      t = tf.random_uniform(
          shape=(self.config.batch_size,),
          minval=0,
          maxval=self.config.n_classes,
          dtype=tf.int32)
      global_step = tfe.Variable(0., trainable=False)
      model = revnet.RevNet(config=config)
      model(x)
      updates = model.get_updates_for(x)

      x_ = tf.identity(x)
      grads_all, vars_all, _ = model.compute_gradients(x_, t, training=True)
      optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
      with tf.control_dependencies(updates):
        train_op = optimizer.apply_gradients(
            zip(grads_all, vars_all), global_step=global_step)

      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(1):
          sess.run(train_op)


# Benchmark related
def device_and_data_format():
  return ("/gpu:0",
          "channels_first") if tf.test.is_gpu_available() else ("/cpu:0",
                                                                "channels_last")


def random_batch(batch_size, config):
  shape = (batch_size,) + config.input_shape
  images = tf.random_uniform(shape)
  labels = tf.random_uniform(
      [batch_size], minval=0, maxval=config.n_classes, dtype=tf.int32)

  return images, labels


class MockIterator(object):

  def __init__(self, tensors):
    self._tensors = [tf.identity(x) for x in tensors]

  def next(self):
    return self._tensors


class RevNetBenchmark(tf.test.Benchmark):
  """Eager and graph benchmarks for RevNet."""

  def _train_batch_sizes(self):
    """Shamelessly copied from `resnet50_test.py`.

    Note: This is targeted towards ImageNet. CIFAR-10 should allow more
    aggressive batch sizes.

    Returns:
      A tuple of possible batch sizes
    """
    for device in device_lib.list_local_devices():
      if tf.DeviceSpec.from_string(device.name).device_type == "GPU":
        if "K20" in device.physical_device_desc:
          return (16,)
        if "P100" in device.physical_device_desc:
          return (16, 32, 64)
      if tf.DeviceSpec.from_string(device.name).device_type == "TPU":
        return (32,)
    return (16, 32)

  def _force_device_sync(self):
    """Shamelessly copied from `resnet50_test.py`."""
    tf.constant(1.).cpu()

  def _report(self, label, start, num_iters, device, batch_size, data_format):
    avg_time = (time.time() - start) / num_iters
    dev = tf.DeviceSpec.from_string(device).device_type.lower()
    name = "%s_%s_batch_%d_%s" % (label, dev, batch_size, data_format)
    extras = {"examples_per_sec": batch_size / avg_time}
    self.report_benchmark(
        iters=num_iters, wall_time=avg_time, name=name, extras=extras)

  def _benchmark_eager_apply(self,
                             label,
                             device_and_format,
                             defun=False,
                             execution_mode=None,
                             compiled=False):
    config = config_.get_hparams_imagenet_56()
    with tfe.execution_mode(execution_mode):
      device, data_format = device_and_format
      model = revnet.RevNet(config=config)
      if defun:
        model.call = tfe.defun(model.call, compiled=compiled)
      batch_size = 64
      num_burn = 5
      num_iters = 10
      with tf.device(device):
        images, _ = random_batch(batch_size, config)
        for _ in range(num_burn):
          model(images, training=False)
        if execution_mode:
          tfe.async_wait()
        gc.collect()
        start = time.time()
        for _ in range(num_iters):
          model(images, training=False)
        if execution_mode:
          tfe.async_wait()
        self._report(label, start, num_iters, device, batch_size, data_format)

  def benchmark_eager_apply_sync(self):
    self._benchmark_eager_apply(
        "eager_apply_sync", device_and_data_format(), defun=False)

  def benchmark_eager_apply_async(self):
    self._benchmark_eager_apply(
        "eager_apply_async",
        device_and_data_format(),
        defun=False,
        execution_mode=tfe.ASYNC)

  def benchmark_eager_call_defun(self):
    self._benchmark_eager_apply(
        "eager_apply_with_defun", device_and_data_format(), defun=True)

  def _benchmark_eager_train(self,
                             label,
                             make_iterator,
                             device_and_format,
                             defun=False,
                             execution_mode=None,
                             compiled=False):
    config = config_.get_hparams_imagenet_56()
    with tfe.execution_mode(execution_mode):
      device, data_format = device_and_format
      for batch_size in self._train_batch_sizes():
        (images, labels) = random_batch(batch_size, config)
        model = revnet.RevNet(config=config)
        optimizer = tf.train.GradientDescentOptimizer(0.1)
        if defun:
          model.call = tfe.defun(model.call)

        num_burn = 3
        num_iters = 10
        with tf.device(device):
          iterator = make_iterator((images, labels))
          for _ in range(num_burn):
            (images, labels) = iterator.next()
            train_one_iter(model, images, labels, optimizer)
          if execution_mode:
            tfe.async_wait()
          self._force_device_sync()
          gc.collect()

          start = time.time()
          for _ in range(num_iters):
            (images, labels) = iterator.next()
            train_one_iter(model, images, labels, optimizer)
          if execution_mode:
            tfe.async_wait()
          self._force_device_sync()
          self._report(label, start, num_iters, device, batch_size, data_format)

  def benchmark_eager_train_sync(self):
    self._benchmark_eager_train(
        "eager_train_sync", MockIterator, device_and_data_format(), defun=False)

  def benchmark_eager_train_async(self):
    self._benchmark_eager_train(
        "eager_train_async",
        MockIterator,
        device_and_data_format(),
        defun=False,
        execution_mode=tfe.ASYNC)

  def benchmark_eager_train_defun(self):
    self._benchmark_eager_train(
        "eager_train", MockIterator, device_and_data_format(), defun=False)

  def benchmark_eager_train_datasets_with_defun(self):

    def make_iterator(tensors):
      with tf.device("/device:CPU:0"):
        ds = tf.data.Dataset.from_tensors(tensors).repeat()
      return tfe.Iterator(ds)

    self._benchmark_eager_train(
        "eager_train_dataset_with_defun",
        make_iterator,
        device_and_data_format(),
        defun=True)


if __name__ == "__main__":
  tf.enable_eager_execution()
  tf.test.main()
