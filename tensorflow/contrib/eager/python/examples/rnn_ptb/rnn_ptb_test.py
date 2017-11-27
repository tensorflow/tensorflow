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
"""Tests for PTBModel with eager execution enabled."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gc
import time

import numpy as np
import tensorflow as tf

from tensorflow.contrib.eager.python import tfe
from tensorflow.contrib.eager.python.examples.rnn_ptb import rnn_ptb


def device():
  return "/device:GPU:0" if tfe.num_gpus() else "/device:CPU:0"


class PTBTest(tf.test.TestCase):

  def testTrain(self):
    model = rnn_ptb.test_model(tfe.num_gpus() > 0)
    sequence_length = 35
    data = np.ones([4 * sequence_length, 20], dtype=np.int64)
    with tf.device(device()):
      optimizer = tf.train.GradientDescentOptimizer(1.0)
      # Train two epochs
      rnn_ptb.train(model, optimizer, data, sequence_length, 0.25)
      rnn_ptb.train(model, optimizer, data, sequence_length, 0.25)

  def testApply(self):
    model = rnn_ptb.test_model(tfe.num_gpus() > 0)
    with tf.device(device()):
      model(tf.ones([35, 20], dtype=tf.int64), training=False)


def force_gpu_sync():
  if tfe.num_gpus():
    tf.constant(1).gpu().cpu()


class PTBBenchmark(tf.test.Benchmark):

  BATCH_SIZE = 20
  SEQ_LEN = 35

  def _report(self, label, start, num_iters, dev, batch_size):
    wall_time = (time.time() - start) / num_iters
    dev = "cpu" if "cpu" in dev.lower() else "gpu"
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
    with tf.device(device()):
      sequence_batch = tf.ones(
          [PTBBenchmark.SEQ_LEN, PTBBenchmark.BATCH_SIZE], dtype=tf.int64)

      for _ in range(10):  # Warmup
        model(sequence_batch, training=False).cpu()
      gc.collect()

      start = time.time()
      iters = 100
      for _ in range(iters):
        model(sequence_batch, training=False).cpu()
      self._report(label, start, iters, device(), int(sequence_batch.shape[1]))

  def benchmark_apply_small(self):
    self._benchmark_apply("eager_apply_small", rnn_ptb.small_model(False))

  def benchmark_apply_large(self):
    self._benchmark_apply("eager_apply_large", rnn_ptb.large_model(False))

  def benchmark_cudnn_apply_small(self):
    if not tfe.num_gpus():
      return
    self._benchmark_apply("eager_cudnn_apply_small", rnn_ptb.small_model(True))

  def benchmark_cudnn_apply_large(self):
    if not tfe.num_gpus():
      return
    self._benchmark_apply("eager_cudnn_apply_large", rnn_ptb.large_model(True))

  def _benchmark_train(self, label, model):
    with tf.device(device()):
      optimizer = tf.train.GradientDescentOptimizer(1.)

      def model_loss(inputs, targets):
        return rnn_ptb.loss_fn(model, inputs, targets, training=True)

      grads = tfe.implicit_gradients(model_loss)

      sequence_batch = tf.ones(
          [PTBBenchmark.SEQ_LEN, PTBBenchmark.BATCH_SIZE], dtype=tf.int64)

      def step():
        optimizer.apply_gradients(
            rnn_ptb.clip_gradients(grads(sequence_batch, sequence_batch), 0.25))

      for _ in range(10):  # Warmup
        step()
      force_gpu_sync()
      gc.collect()

      start = time.time()
      iters = 100
      for _ in range(iters):
        step()
      force_gpu_sync()
      self._report(label, start, iters, device(), int(sequence_batch.shape[1]))

  def benchmark_train_small(self):
    self._benchmark_train("eager_train_small", rnn_ptb.small_model(False))

  def benchmark_train_large(self):
    self._benchmark_train("eager_train_large", rnn_ptb.large_model(False))

  def benchmark_cudnn_train_small(self):
    if not tfe.num_gpus():
      return
    self._benchmark_train("eager_cudnn_train_small", rnn_ptb.small_model(True))

  def benchmark_cudnn_train_large(self):
    if not tfe.num_gpus():
      return
    self._benchmark_train("eager_cudnn_train_large", rnn_ptb.large_model(True))


if __name__ == "__main__":
  tfe.enable_eager_execution()
  tf.test.main()
