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
"""Tests for PTBModel used for graph construction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gc
import time

import numpy as np
import tensorflow as tf

from tensorflow.contrib.eager.python.examples.rnn_ptb import rnn_ptb


class PTBTest(tf.test.TestCase):

  def testTrain(self):
    batch_size = 20
    sequence_length = 35
    with tf.Graph().as_default(), tf.device(tf.test.gpu_device_name()):
      inputs_ph = tf.placeholder(tf.int64, [sequence_length, batch_size],
                                 "inputs")
      labels_ph = tf.placeholder(tf.int64, [sequence_length, batch_size],
                                 "labels")

      inputs = np.ones(inputs_ph.shape.as_list(), dtype=np.int64)
      labels = np.ones(labels_ph.shape.as_list(), dtype=np.int64)

      model = rnn_ptb.test_model(tf.test.is_gpu_available())
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
      loss = rnn_ptb.loss_fn(model, inputs_ph, labels_ph, training=True)
      grads = rnn_ptb.clip_gradients(optimizer.compute_gradients(loss), 0.25)
      train_op = optimizer.apply_gradients(grads)

      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(train_op, feed_dict={inputs_ph: inputs, labels_ph: labels})
        sess.run(
            [train_op, loss], feed_dict={
                inputs_ph: inputs,
                labels_ph: labels
            })


class PTBBenchmark(tf.test.Benchmark):

  BATCH_SIZE = 20
  SEQ_LEN = 35

  def _report(self, label, start, num_iters, device, batch_size):
    wall_time = (time.time() - start) / num_iters
    dev = "cpu" if "cpu" in device.lower() else "gpu"
    name = "%s_%s_batch_%d" % (label, dev, batch_size)
    examples_per_sec = batch_size / wall_time
    self.report_benchmark(
        iters=num_iters,
        wall_time=wall_time,
        name=name,
        extras={
            "examples_per_sec": examples_per_sec
        })

  def _benchmark_apply(self, label, model):
    num_iters = 100
    num_warmup = 10
    dataset = tf.data.Dataset.from_tensors(
        tf.ones(
            [PTBBenchmark.SEQ_LEN, PTBBenchmark.BATCH_SIZE],
            dtype=tf.int64)).repeat(num_iters + num_warmup)
    inputs = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()

    with tf.device(tf.test.gpu_device_name()):
      outputs = model(inputs, training=True)

      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(num_warmup):
          sess.run(outputs)
        gc.collect()

        start = time.time()
        for _ in range(num_iters):
          sess.run(outputs)
        self._report(label, start, num_iters,
                     tf.test.gpu_device_name(), PTBBenchmark.BATCH_SIZE)

  def benchmark_apply_small(self):
    self._benchmark_apply("graph_apply_small", rnn_ptb.small_model(False))

  def benchmark_apply_large(self):
    self._benchmark_apply("graph_apply_large", rnn_ptb.large_model(False))

  def benchmark_cudnn_apply_small(self):
    if not tf.test.is_gpu_available():
      return
    self._benchmark_apply("graph_cudnn_apply_small", rnn_ptb.small_model(True))

  def benchmark_cudnn_apply_large(self):
    if not tf.test.is_gpu_available():
      return
    self._benchmark_apply("graph_cudnn_apply_large", rnn_ptb.large_model(True))

  def _benchmark_train(self, label, model):
    num_iters = 100
    num_warmup = 10
    dataset = tf.data.Dataset.from_tensors(
        tf.ones(
            [PTBBenchmark.SEQ_LEN, PTBBenchmark.BATCH_SIZE],
            dtype=tf.int64)).repeat(num_iters + num_warmup)
    # inputs and labels have the same shape
    dataset = tf.data.Dataset.zip((dataset, dataset))
    (inputs, labels) = tf.compat.v1.data.make_one_shot_iterator(
        dataset).get_next()

    with tf.device(tf.test.gpu_device_name()):
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
      loss = rnn_ptb.loss_fn(model, inputs, labels, training=True)
      grads = rnn_ptb.clip_gradients(optimizer.compute_gradients(loss), 0.25)
      train_op = optimizer.apply_gradients(grads)

      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(num_warmup):
          sess.run(train_op)
        gc.collect()
        start = time.time()
        for _ in range(num_iters):
          sess.run(train_op)
        self._report(label, start, num_iters,
                     tf.test.gpu_device_name(), PTBBenchmark.BATCH_SIZE)

  def benchmark_train_small(self):
    self._benchmark_train("graph_train_small", rnn_ptb.small_model(False))

  def benchmark_train_large(self):
    self._benchmark_train("graph_train_large", rnn_ptb.large_model(False))

  def benchmark_cudnn_train_small(self):
    if not tf.test.is_gpu_available():
      return
    self._benchmark_train("graph_cudnn_train_small", rnn_ptb.small_model(True))

  def benchmark_cudnn_train_large(self):
    if not tf.test.is_gpu_available():
      return
    self._benchmark_train("graph_cudnn_train_large", rnn_ptb.large_model(True))


if __name__ == "__main__":
  tf.test.main()
