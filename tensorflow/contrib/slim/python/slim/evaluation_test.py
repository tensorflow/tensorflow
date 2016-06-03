# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for slim.evaluation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import glob
import os

import numpy as np
import tensorflow as tf

import tensorflow.contrib.slim as slim


def GenerateTestData(num_classes, batch_size):
  inputs = np.random.rand(batch_size, num_classes)

  np.random.seed(0)
  labels = np.random.randint(low=0, high=num_classes, size=batch_size)
  labels = labels.reshape((batch_size,))
  return inputs, labels


def TestModel(inputs):
  scale = tf.Variable(1.0, trainable=False)

  # Scaling the outputs wont change the result...
  outputs = tf.mul(inputs, scale)
  return tf.argmax(outputs, 1), scale


def GroundTruthAccuracy(inputs, labels, batch_size):
  predictions = np.argmax(inputs, 1)
  num_correct = np.sum(predictions == labels)
  return float(num_correct) / batch_size


class EvaluationTest(tf.test.TestCase):

  def setUp(self):
    super(EvaluationTest, self).setUp()

    num_classes = 8
    batch_size = 16
    inputs, labels = GenerateTestData(num_classes, batch_size)
    self._expected_accuracy = GroundTruthAccuracy(inputs, labels, batch_size)

    self._global_step = slim.get_or_create_global_step()
    self._inputs = tf.constant(inputs, dtype=tf.float32)
    self._labels = tf.constant(labels, dtype=tf.int64)
    self._predictions, self._scale = TestModel(self._inputs)

  def testUpdateOpsAreEvaluated(self):
    accuracy, update_op = slim.metrics.streaming_accuracy(
        self._predictions, self._labels)
    init_op = tf.group(tf.initialize_all_variables(),
                       tf.initialize_local_variables())

    with self.test_session() as sess:
      slim.evaluation.evaluation(
          sess, init_op=init_op, eval_op=update_op)
      self.assertAlmostEqual(accuracy.eval(), self._expected_accuracy)

  def testSummariesAreFlushedToDisk(self):
    output_dir = os.path.join(self.get_temp_dir(), 'flush_test')
    if tf.gfile.Exists(output_dir):  # For running on jenkins.
      tf.gfile.DeleteRecursively(output_dir)

    accuracy0, update_op0 = tf.contrib.metrics.streaming_accuracy(
        self._predictions, self._labels)
    accuracy1, update_op1 = tf.contrib.metrics.streaming_accuracy(
        self._predictions+1, self._labels)

    names_to_metrics = {
        'Accuracy': accuracy0,
        'Another accuracy': accuracy1,
    }

    for k in names_to_metrics:
      v = names_to_metrics[k]
      tf.scalar_summary(k, v)

    summary_writer = tf.train.SummaryWriter(output_dir)

    init_op = tf.group(tf.initialize_all_variables(),
                       tf.initialize_local_variables())
    eval_op = tf.group(update_op0, update_op1)

    with self.test_session() as sess:
      slim.evaluation.evaluation(
          sess,
          init_op=init_op,
          eval_op=eval_op,
          summary_op=tf.merge_all_summaries(),
          summary_writer=summary_writer,
          global_step=self._global_step)

      # Check that the results were saved. The events file may have additional
      # entries, e.g. the event version stamp, so have to parse things a bit.
      output_filepath = glob.glob(os.path.join(output_dir, '*'))
      self.assertEqual(len(output_filepath), 1)
      events = tf.train.summary_iterator(output_filepath[0])
      summaries = [e.summary for e in events if e.summary.value]
      values = []
      for summary in summaries:
        for value in summary.value:
          values.append(value)
      saved_results = {v.tag: v.simple_value for v in values}
      for name in names_to_metrics:
        self.assertAlmostEqual(names_to_metrics[name].eval(),
                               saved_results[name])

  def testWithFeedDict(self):
    accuracy, update_op = slim.metrics.streaming_accuracy(
        self._predictions, self._labels)
    init_op = tf.group(tf.initialize_all_variables(),
                       tf.initialize_local_variables())

    with self.test_session() as sess:
      slim.evaluation.evaluation(
          sess,
          init_op=init_op,
          eval_op=update_op,
          eval_op_feed_dict={self._scale: np.ones([], dtype=np.float32)})
      self.assertAlmostEqual(accuracy.eval(), self._expected_accuracy)

  def testWithQueueRunning(self):
    strings = ['the', 'cat', 'in', 'the', 'hat']
    _ = tf.train.string_input_producer(strings, capacity=5)

    accuracy, update_op = slim.metrics.streaming_accuracy(
        self._predictions, self._labels)

    init_op = tf.group(tf.initialize_all_variables(),
                       tf.initialize_local_variables())

    with self.test_session() as sess:
      slim.evaluation.evaluation(
          sess, init_op=init_op, eval_op=update_op)
      self.assertAlmostEqual(accuracy.eval(), self._expected_accuracy)


if __name__ == '__main__':
  tf.test.main()
