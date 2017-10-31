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
"""Tests and benchmarks for ResNet50 under graph execution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
import time

import numpy as np
import tensorflow as tf

from tensorflow.contrib.eager.python.examples.resnet50 import resnet50
from tensorflow.contrib.summary import summary_test_util


def data_format():
  return 'channels_first' if tf.test.is_gpu_available() else 'channels_last'


def image_shape(batch_size):
  if data_format() == 'channels_first':
    return [batch_size, 3, 224, 224]
  return [batch_size, 224, 224, 3]


def random_batch(batch_size):
  images = np.random.rand(*image_shape(batch_size)).astype(np.float32)
  num_classes = 1000
  labels = np.random.randint(
      low=0, high=num_classes, size=[batch_size]).astype(np.int32)
  one_hot = np.zeros((batch_size, num_classes)).astype(np.float32)
  one_hot[np.arange(batch_size), labels] = 1.
  return images, one_hot


class ResNet50GraphTest(tf.test.TestCase):

  def testApply(self):
    batch_size = 64
    with tf.Graph().as_default():
      images = tf.placeholder(tf.float32, image_shape(None))
      model = resnet50.ResNet50(data_format())
      predictions = model(images)

      init = tf.global_variables_initializer()

      with tf.Session() as sess:
        sess.run(init)
        np_images, _ = random_batch(batch_size)
        out = sess.run(predictions, feed_dict={images: np_images})
        self.assertAllEqual([64, 1000], out.shape)

  def testTrainWithSummary(self):
    with tf.Graph().as_default():
      images = tf.placeholder(tf.float32, image_shape(None), name='images')
      labels = tf.placeholder(tf.float32, [None, 1000], name='labels')

      tf.train.get_or_create_global_step()
      logdir = tempfile.mkdtemp()
      with tf.contrib.summary.always_record_summaries():
        with tf.contrib.summary.create_summary_file_writer(
            logdir, max_queue=0,
            name='t0').as_default():
          model = resnet50.ResNet50(data_format())
          logits = model(images, training=True)
          loss = tf.losses.softmax_cross_entropy(
              logits=logits, onehot_labels=labels)
          tf.contrib.summary.scalar(name='loss', tensor=loss)
          optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
          train_op = optimizer.minimize(loss)

      init = tf.global_variables_initializer()
      self.assertEqual(321, len(tf.global_variables()))

      batch_size = 32
      with tf.Session() as sess:
        sess.run(init)
        sess.run(tf.contrib.summary.summary_writer_initializer_op())
        np_images, np_labels = random_batch(batch_size)
        sess.run([train_op, tf.contrib.summary.all_summary_ops()],
                 feed_dict={images: np_images, labels: np_labels})

      events = summary_test_util.events_from_file(logdir)
      self.assertEqual(len(events), 2)
      self.assertEqual(events[1].summary.value[0].tag, 'loss')


class ResNet50Benchmarks(tf.test.Benchmark):

  def _report(self, label, start, num_iters, batch_size):
    avg_time = (time.time() - start) / num_iters
    dev = 'gpu' if tf.test.is_gpu_available() else 'cpu'
    name = 'graph_%s_%s_batch_%d_%s' % (label, dev, batch_size, data_format())
    extras = {'examples_per_sec': batch_size / avg_time}
    self.report_benchmark(
        iters=num_iters, wall_time=avg_time, name=name, extras=extras)

  def benchmark_graph_apply(self):
    with tf.Graph().as_default():
      images = tf.placeholder(tf.float32, image_shape(None))
      model = resnet50.ResNet50(data_format())
      predictions = model(images)

      init = tf.global_variables_initializer()

      batch_size = 64
      with tf.Session() as sess:
        sess.run(init)
        np_images, _ = random_batch(batch_size)
        num_burn, num_iters = (3, 30)
        for _ in range(num_burn):
          sess.run(predictions, feed_dict={images: np_images})
        start = time.time()
        for _ in range(num_iters):
          # Comparison with the eager execution benchmark in resnet50_test.py
          # isn't entirely fair as the time here includes the cost of copying
          # the feeds from CPU memory to GPU.
          sess.run(predictions, feed_dict={images: np_images})
        self._report('apply', start, num_iters, batch_size)

  def benchmark_graph_train(self):
    for batch_size in [16, 32, 64]:
      with tf.Graph().as_default():
        np_images, np_labels = random_batch(batch_size)
        dataset = tf.data.Dataset.from_tensors((np_images, np_labels)).repeat()
        (images, labels) = dataset.make_one_shot_iterator().get_next()

        model = resnet50.ResNet50(data_format())
        logits = model(images, training=True)
        loss = tf.losses.softmax_cross_entropy(
            logits=logits, onehot_labels=labels)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        train_op = optimizer.minimize(loss)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
          sess.run(init)
          (num_burn, num_iters) = (5, 10)
          for _ in range(num_burn):
            sess.run(train_op)
          start = time.time()
          for _ in range(num_iters):
            sess.run(train_op)
          self._report('train', start, num_iters, batch_size)


if __name__ == '__main__':
  tf.test.main()
