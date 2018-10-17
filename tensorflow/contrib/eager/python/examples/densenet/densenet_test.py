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
"""Tests and Benchmarks for Densenet model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gc
import time
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from tensorflow.contrib.eager.python.examples.densenet import densenet
from tensorflow.python.client import device_lib


class DensenetTest(tf.test.TestCase):

  def test_bottleneck_true(self):
    depth = 7
    growth_rate = 2
    num_blocks = 3
    output_classes = 10
    num_layers_in_each_block = -1
    batch_size = 1
    data_format = ('channels_first') if tf.test.is_gpu_available() else (
        'channels_last')

    model = densenet.DenseNet(depth, growth_rate, num_blocks,
                              output_classes, num_layers_in_each_block,
                              data_format, bottleneck=True, compression=0.5,
                              weight_decay=1e-4, dropout_rate=0,
                              pool_initial=False, include_top=True)

    if data_format == 'channels_last':
      rand_input = tf.random_uniform((batch_size, 32, 32, 3))
    else:
      rand_input = tf.random_uniform((batch_size, 3, 32, 32))
    output_shape = model(rand_input).shape
    self.assertEqual(output_shape, (batch_size, output_classes))

  def test_bottleneck_false(self):
    depth = 7
    growth_rate = 2
    num_blocks = 3
    output_classes = 10
    num_layers_in_each_block = -1
    batch_size = 1
    data_format = ('channels_first') if tf.test.is_gpu_available() else (
        'channels_last')

    model = densenet.DenseNet(depth, growth_rate, num_blocks,
                              output_classes, num_layers_in_each_block,
                              data_format, bottleneck=False, compression=0.5,
                              weight_decay=1e-4, dropout_rate=0,
                              pool_initial=False, include_top=True)

    if data_format == 'channels_last':
      rand_input = tf.random_uniform((batch_size, 32, 32, 3))
    else:
      rand_input = tf.random_uniform((batch_size, 3, 32, 32))
    output_shape = model(rand_input).shape
    self.assertEqual(output_shape, (batch_size, output_classes))

  def test_pool_initial_true(self):
    depth = 7
    growth_rate = 2
    num_blocks = 4
    output_classes = 10
    num_layers_in_each_block = [1, 2, 2, 1]
    batch_size = 1
    data_format = ('channels_first') if tf.test.is_gpu_available() else (
        'channels_last')

    model = densenet.DenseNet(depth, growth_rate, num_blocks,
                              output_classes, num_layers_in_each_block,
                              data_format, bottleneck=True, compression=0.5,
                              weight_decay=1e-4, dropout_rate=0,
                              pool_initial=True, include_top=True)

    if data_format == 'channels_last':
      rand_input = tf.random_uniform((batch_size, 32, 32, 3))
    else:
      rand_input = tf.random_uniform((batch_size, 3, 32, 32))
    output_shape = model(rand_input).shape
    self.assertEqual(output_shape, (batch_size, output_classes))

  def test_regularization(self):
    if tf.test.is_gpu_available():
      rand_input = tf.random_uniform((10, 3, 32, 32))
      data_format = 'channels_first'
    else:
      rand_input = tf.random_uniform((10, 32, 32, 3))
      data_format = 'channels_last'
    weight_decay = 1e-4

    conv = tf.keras.layers.Conv2D(
        3, (3, 3),
        padding='same',
        use_bias=False,
        data_format=data_format,
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    conv(rand_input)  # Initialize the variables in the layer

    def compute_true_l2(vs, wd):
      return tf.reduce_sum(tf.square(vs)) * wd

    true_l2 = compute_true_l2(conv.variables, weight_decay)
    keras_l2 = tf.add_n(conv.losses)
    self.assertAllClose(true_l2, keras_l2)

    with tf.GradientTape() as tape_true, tf.GradientTape() as tape_keras:
      loss = tf.reduce_sum(conv(rand_input))
      loss_with_true_l2 = loss + compute_true_l2(conv.variables, weight_decay)
      loss_with_keras_l2 = loss + tf.add_n(conv.losses)

    true_grads = tape_true.gradient(loss_with_true_l2, conv.variables)
    keras_grads = tape_keras.gradient(loss_with_keras_l2, conv.variables)
    self.assertAllClose(true_grads, keras_grads)

    optimizer.apply_gradients(zip(keras_grads, conv.variables))
    keras_l2_after_update = tf.add_n(conv.losses)
    self.assertNotAllClose(keras_l2, keras_l2_after_update)


def compute_gradients(model, images, labels):
  with tf.GradientTape() as tape:
    logits = model(images, training=True)
    cross_ent = tf.losses.softmax_cross_entropy(
        logits=logits, onehot_labels=labels)
    regularization = tf.add_n(model.losses)
    loss = cross_ent + regularization
    tf.contrib.summary.scalar(name='loss', tensor=loss)
  return tape.gradient(loss, model.variables)


def apply_gradients(model, optimizer, gradients):
  optimizer.apply_gradients(zip(gradients, model.variables))


def device_and_data_format():
  return ('/gpu:0',
          'channels_first') if tf.test.is_gpu_available() else ('/cpu:0',
                                                                'channels_last')


def random_batch(batch_size, data_format):
  shape = (3, 224, 224) if data_format == 'channels_first' else (224, 224, 3)
  shape = (batch_size,) + shape

  num_classes = 1000
  images = tf.random_uniform(shape)
  labels = tf.random_uniform(
      [batch_size], minval=0, maxval=num_classes, dtype=tf.int32)
  one_hot = tf.one_hot(labels, num_classes)

  return images, one_hot


class MockIterator(object):

  def __init__(self, tensors):
    self._tensors = [tf.identity(x) for x in tensors]

  def next(self):
    return self._tensors


class DensenetBenchmark(tf.test.Benchmark):

  def __init__(self):
    self.depth = 121
    self.growth_rate = 32
    self.num_blocks = 4
    self.output_classes = 1000
    self.num_layers_in_each_block = [6, 12, 24, 16]

  def _train_batch_sizes(self):
    """Choose batch sizes based on GPU capability."""
    for device in device_lib.list_local_devices():
      if tf.DeviceSpec.from_string(device.name).device_type == 'GPU':
        if 'K20' in device.physical_device_desc:
          return (16,)
        if 'P100' in device.physical_device_desc:
          return (16, 32, 64)

      if tf.DeviceSpec.from_string(device.name).device_type == 'TPU':
        return (32,)
    return (16, 32)

  def _report(self, label, start, num_iters, device, batch_size, data_format):
    avg_time = (time.time() - start) / num_iters
    dev = tf.DeviceSpec.from_string(device).device_type.lower()
    name = '%s_%s_batch_%d_%s' % (label, dev, batch_size, data_format)
    extras = {'examples_per_sec': batch_size / avg_time}
    self.report_benchmark(
        iters=num_iters, wall_time=avg_time, name=name, extras=extras)

  def _force_device_sync(self):
    # If this function is called in the context of a non-CPU device
    # (e.g., inside a 'with tf.device("/gpu:0")' block)
    # then this will force a copy from CPU->NON_CPU_DEVICE->CPU,
    # which forces a sync. This is a roundabout way, yes.
    tf.constant(1.).cpu()

  def _benchmark_eager_apply(self, label, device_and_format, defun=False,
                             execution_mode=None):
    with tfe.execution_mode(execution_mode):
      device, data_format = device_and_format
      model = densenet.DenseNet(self.depth, self.growth_rate, self.num_blocks,
                                self.output_classes,
                                self.num_layers_in_each_block, data_format,
                                bottleneck=True, compression=0.5,
                                weight_decay=1e-4, dropout_rate=0,
                                pool_initial=True, include_top=True)
      if defun:
        # TODO(apassos) enable tfe.function here
        model.call = tfe.defun(model.call)
      batch_size = 64
      num_burn = 5
      num_iters = 30
      with tf.device(device):
        images, _ = random_batch(batch_size, data_format)
        for _ in xrange(num_burn):
          model(images, training=False).cpu()
        if execution_mode:
          tfe.async_wait()
        gc.collect()
        start = time.time()
        for _ in xrange(num_iters):
          model(images, training=False).cpu()
        if execution_mode:
          tfe.async_wait()
        self._report(label, start, num_iters, device, batch_size, data_format)

  def benchmark_eager_apply_sync(self):
    self._benchmark_eager_apply('eager_apply', device_and_data_format(),
                                defun=False)

  def benchmark_eager_apply_async(self):
    self._benchmark_eager_apply(
        'eager_apply_async', device_and_data_format(), defun=False,
        execution_mode=tfe.ASYNC)

  def benchmark_eager_apply_with_defun(self):
    self._benchmark_eager_apply('eager_apply_with_defun',
                                device_and_data_format(), defun=True)

  def _benchmark_eager_train(self,
                             label,
                             make_iterator,
                             device_and_format,
                             defun=False,
                             execution_mode=None):
    with tfe.execution_mode(execution_mode):
      device, data_format = device_and_format
      for batch_size in self._train_batch_sizes():
        (images, labels) = random_batch(batch_size, data_format)
        model = densenet.DenseNet(self.depth, self.growth_rate, self.num_blocks,
                                  self.output_classes,
                                  self.num_layers_in_each_block, data_format,
                                  bottleneck=True, compression=0.5,
                                  weight_decay=1e-4, dropout_rate=0,
                                  pool_initial=True, include_top=True)
        optimizer = tf.train.GradientDescentOptimizer(0.1)
        apply_grads = apply_gradients
        if defun:
          model.call = tfe.defun(model.call)
          apply_grads = tfe.defun(apply_gradients)

        num_burn = 3
        num_iters = 10
        with tf.device(device):
          iterator = make_iterator((images, labels))
          for _ in xrange(num_burn):
            (images, labels) = iterator.next()
            apply_grads(model, optimizer,
                        compute_gradients(model, images, labels))
          if execution_mode:
            tfe.async_wait()
          self._force_device_sync()
          gc.collect()

          start = time.time()
          for _ in xrange(num_iters):
            (images, labels) = iterator.next()
            apply_grads(model, optimizer,
                        compute_gradients(model, images, labels))
          if execution_mode:
            tfe.async_wait()
          self._force_device_sync()
          self._report(label, start, num_iters, device, batch_size, data_format)

  def benchmark_eager_train_sync(self):
    self._benchmark_eager_train('eager_train', MockIterator,
                                device_and_data_format(), defun=False)

  def benchmark_eager_train_async(self):
    self._benchmark_eager_train(
        'eager_train_async',
        MockIterator,
        device_and_data_format(),
        defun=False,
        execution_mode=tfe.ASYNC)

  def benchmark_eager_train_with_defun(self):
    self._benchmark_eager_train(
        'eager_train_with_defun', MockIterator,
        device_and_data_format(), defun=True)

  def benchmark_eager_train_datasets(self):

    def make_iterator(tensors):
      with tf.device('/device:CPU:0'):
        ds = tf.data.Dataset.from_tensors(tensors).repeat()
      return tfe.Iterator(ds)

    self._benchmark_eager_train(
        'eager_train_dataset', make_iterator,
        device_and_data_format(), defun=False)

  def benchmark_eager_train_datasets_with_defun(self):

    def make_iterator(tensors):
      with tf.device('/device:CPU:0'):
        ds = tf.data.Dataset.from_tensors(tensors).repeat()
      return tfe.Iterator(ds)

    self._benchmark_eager_train(
        'eager_train_dataset_with_defun', make_iterator,
        device_and_data_format(), defun=True)


if __name__ == '__main__':
  tf.enable_eager_execution()
  tf.test.main()
