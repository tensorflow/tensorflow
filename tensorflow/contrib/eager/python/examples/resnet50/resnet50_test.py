# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests and benchmarks for the ResNet50 model, executed eagerly."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gc
import tempfile
import time

import tensorflow as tf

import tensorflow.contrib.eager as tfe
from tensorflow.contrib.eager.python.examples.resnet50 import resnet50
from tensorflow.contrib.summary import summary_test_util
from tensorflow.python.client import device_lib


def device_and_data_format():
  return ('/gpu:0', 'channels_first') if tfe.num_gpus() else ('/cpu:0',
                                                              'channels_last')


def random_batch(batch_size):
  _, data_format = device_and_data_format()

  shape = (3, 224, 224) if data_format == 'channels_first' else (224, 224, 3)
  shape = (batch_size,) + shape

  num_classes = 1000
  images = tf.random_uniform(shape)
  labels = tf.random_uniform(
      [batch_size], minval=0, maxval=num_classes, dtype=tf.int32)
  one_hot = tf.one_hot(labels, num_classes)

  return images, one_hot


def train_one_step(model, images, labels, optimizer):

  with tfe.GradientTape() as tape:
    logits = model(images, training=True)
    loss = tf.losses.softmax_cross_entropy(
        logits=logits, onehot_labels=labels)
    tf.contrib.summary.scalar(name='loss', tensor=loss)
  grads = tape.gradient(loss, model.variables)
  optimizer.apply_gradients(zip(grads, model.variables))


class ResNet50Test(tf.test.TestCase):

  def _apply(self, defun=False):
    device, data_format = device_and_data_format()
    model = resnet50.ResNet50(data_format)
    if defun:
      model.call = tfe.defun(model.call)
    with tf.device(device):
      images, _ = random_batch(2)
      output = model(images)
    self.assertEqual((2, 1000), output.shape)

  def test_apply(self):
    self._apply(defun=False)

  def test_apply_with_defun(self):
    self._apply(defun=True)

  def test_apply_no_top(self):
    device, data_format = device_and_data_format()
    model = resnet50.ResNet50(data_format, include_top=False)
    with tf.device(device):
      images, _ = random_batch(2)
      output = model(images)
    output_shape = ((2, 2048, 1, 1)
                    if data_format == 'channels_first' else (2, 1, 1, 2048))
    self.assertEqual(output_shape, output.shape)

  def test_apply_with_pooling(self):
    device, data_format = device_and_data_format()
    model = resnet50.ResNet50(data_format, include_top=False, pooling='avg')
    with tf.device(device):
      images, _ = random_batch(2)
      output = model(images)
    self.assertEqual((2, 2048), output.shape)

  def test_train(self):
    device, data_format = device_and_data_format()
    model = resnet50.ResNet50(data_format)
    tf.train.get_or_create_global_step()
    logdir = tempfile.mkdtemp()
    with tf.contrib.summary.create_file_writer(
        logdir, max_queue=0,
        name='t0').as_default(), tf.contrib.summary.always_record_summaries():
      with tf.device(device):
        optimizer = tf.train.GradientDescentOptimizer(0.1)
        images, labels = random_batch(2)
        train_one_step(model, images, labels, optimizer)
        self.assertEqual(320, len(model.variables))
    events = summary_test_util.events_from_logdir(logdir)
    self.assertEqual(len(events), 2)
    self.assertEqual(events[1].summary.value[0].tag, 'loss')

  def test_no_garbage(self):
    device, data_format = device_and_data_format()
    model = resnet50.ResNet50(data_format)
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    with tf.device(device):
      images, labels = random_batch(2)
      gc.disable()
      # Warm up. Note that this first run does create significant amounts of
      # garbage to be collected. The hope is that this is a build-only effect,
      # and a subsequent training loop will create nothing which needs to be
      # collected.
      train_one_step(model, images, labels, optimizer)
      gc.collect()
      previous_gc_debug_flags = gc.get_debug()
      gc.set_debug(gc.DEBUG_SAVEALL)
      for _ in range(2):
        # Run twice to ensure that garbage that is created on the first
        # iteration is no longer accessible.
        train_one_step(model, images, labels, optimizer)
      gc.collect()
      # There should be no garbage requiring collection.
      self.assertEqual(0, len(gc.garbage))
      gc.set_debug(previous_gc_debug_flags)
      gc.enable()


class MockIterator(object):

  def __init__(self, tensors):
    self._tensors = [tf.identity(x) for x in tensors]

  def next(self):
    return self._tensors


class ResNet50Benchmarks(tf.test.Benchmark):

  def _train_batch_sizes(self):
    """Choose batch sizes based on GPU capability."""
    for device in device_lib.list_local_devices():
      if 'GPU:0' in device.name:
        # Avoid OOM errors with larger batch sizes, which seem to cause errors
        # later on even if caught.
        #
        # TODO(allenl): Base this on device memory; memory limit information
        # during the test seems to exclude the amount TensorFlow has allocated,
        # which isn't useful.
        if 'K20' in device.physical_device_desc:
          return (16,)
        if 'P100' in device.physical_device_desc:
          return (16, 32, 64)
    return (16, 32)

  def _report(self, label, start, num_iters, device, batch_size, data_format):
    avg_time = (time.time() - start) / num_iters
    dev = 'cpu' if 'cpu' in device else 'gpu'
    name = '%s_%s_batch_%d_%s' % (label, dev, batch_size, data_format)
    extras = {'examples_per_sec': batch_size / avg_time}
    self.report_benchmark(
        iters=num_iters, wall_time=avg_time, name=name, extras=extras)

  def _force_gpu_sync(self):
    # If this function is called in the context of a GPU device
    # (e.g., inside a 'with tf.device("/gpu:0")' block)
    # then this will force a copy from CPU->GPU->CPU, which forces
    # a sync. This is a roundabout way, yes.
    tf.constant(1.).cpu()

  def _benchmark_eager_apply(self, label, defun=False):
    device, data_format = device_and_data_format()
    model = resnet50.ResNet50(data_format)
    if defun:
      model.call = tfe.defun(model.call)
    batch_size = 64
    num_burn = 5
    num_iters = 30
    with tf.device(device):
      images, _ = random_batch(batch_size)
      for _ in xrange(num_burn):
        model(images).cpu()
      gc.collect()
      start = time.time()
      for _ in xrange(num_iters):
        model(images).cpu()
      self._report(label, start, num_iters, device, batch_size, data_format)

  def benchmark_eager_apply(self):
    self._benchmark_eager_apply('eager_apply', defun=False)

  def benchmark_eager_apply_with_defun(self):
    self._benchmark_eager_apply('eager_apply_with_defun', defun=True)

  def _benchmark_eager_train(self, label, make_iterator, defun=False):
    device, data_format = device_and_data_format()
    for batch_size in self._train_batch_sizes():
      (images, labels) = random_batch(batch_size)
      num_burn = 3
      num_iters = 10
      model = resnet50.ResNet50(data_format)
      if defun:
        model.call = tfe.defun(model.call)
      optimizer = tf.train.GradientDescentOptimizer(0.1)

      with tf.device(device):
        iterator = make_iterator((images, labels))
        for _ in xrange(num_burn):
          (images, labels) = iterator.next()
          train_one_step(model, images, labels, optimizer)
        self._force_gpu_sync()
        gc.collect()

        start = time.time()
        for _ in xrange(num_iters):
          (images, labels) = iterator.next()
          train_one_step(model, images, labels, optimizer)
        self._force_gpu_sync()
        self._report(label, start, num_iters, device, batch_size, data_format)

  def benchmark_eager_train(self):
    self._benchmark_eager_train('eager_train', MockIterator, defun=False)

  def benchmark_eager_train_with_defun(self):
    self._benchmark_eager_train(
        'eager_train_with_defun', MockIterator, defun=True)

  def benchmark_eager_train_datasets(self):

    def make_iterator(tensors):
      with tf.device('/device:CPU:0'):
        ds = tf.data.Dataset.from_tensors(tensors).repeat()
      return tfe.Iterator(ds)

    self._benchmark_eager_train(
        'eager_train_dataset', make_iterator, defun=False)

  def benchmark_eager_train_datasets_with_defun(self):

    def make_iterator(tensors):
      with tf.device('/device:CPU:0'):
        ds = tf.data.Dataset.from_tensors(tensors).repeat()
      return tfe.Iterator(ds)

    self._benchmark_eager_train(
        'eager_train_dataset_with_defun', make_iterator, defun=True)


if __name__ == '__main__':
  tfe.enable_eager_execution()
  tf.test.main()
