# Copyright 2016 Google Inc. All Rights Reserved.
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

# pylint: disable=unused-import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class SamplingOpsTest(tf.test.TestCase):

  def testGraphBuildAssertionFailures(self):
    val = tf.zeros([1, 3])
    label = tf.constant([1], shape=[1])  # must have batch dimension
    probs = [.2] * 5
    batch_size = 16

    # Label must have only batch dimension if enqueue_many is True.
    with self.assertRaises(ValueError):
      tf.contrib.framework.sampling_ops.stratified_sample(
          val, tf.zeros([]), probs, batch_size, enqueue_many=True)
    with self.assertRaises(ValueError):
      tf.contrib.framework.sampling_ops.stratified_sample(
          val, tf.zeros([1, 1]), probs, batch_size, enqueue_many=True)

    # Label must not be one-hot.
    with self.assertRaises(ValueError):
      tf.contrib.framework.sampling_ops.stratified_sample(
          val, tf.constant([0, 1, 0, 0, 0]), probs, batch_size)

    # Data must have batch dimension if enqueue_many is True.
    with self.assertRaises(ValueError):
      tf.contrib.framework.sampling_ops.stratified_sample(
          val, tf.constant(1), probs, batch_size, enqueue_many=True)

    # Batch dimensions on data and labels should be equal.
    with self.assertRaises(ValueError):
      tf.contrib.framework.sampling_ops.stratified_sample(
          tf.zeros([2, 1]), label, probs, batch_size, enqueue_many=True)

    # Probabilities must be numpy array or python list.
    with self.assertRaises(ValueError):
      tf.contrib.framework.sampling_ops.stratified_sample(
          val, label, tf.constant([.5, .5]), batch_size)

    # Probabilities should sum to one.
    with self.assertRaises(ValueError):
      tf.contrib.framework.sampling_ops.stratified_sample(
          val, label, np.array([.1] * 5), batch_size)

    # Probabilities must be 1D.
    with self.assertRaises(ValueError):
      tf.contrib.framework.sampling_ops.stratified_sample(
          val, label, np.array([[.25, .25], [.25, .25]]), batch_size)

  def testRuntimeAssertionFailures(self):
    probs = [.2] * 5
    vals = tf.zeros([3, 1])

    illegal_labels = [
        [0, -1, 1],  # classes must be nonnegative
        [5, 1, 1],  # classes must be less than number of classes
        [2, 3],  # data and label batch size must be the same
    ]

    # Set up graph with illegal label vector.
    label_ph = tf.placeholder(tf.int32, shape=[None])
    vals_tf, lbls_tf, _ = tf.contrib.framework.sampling_ops._verify_input(
        vals, label_ph, probs)

    for illegal_label in illegal_labels:
      # Run session that should fail.
      with self.test_session() as sess:
        with self.assertRaises(tf.errors.InvalidArgumentError):
          sess.run([vals_tf, lbls_tf], feed_dict={label_ph: illegal_label})

  def testBatchingBehavior(self):
    batch_size = 20
    input_batch_size = 11
    val_input_batch = tf.zeros([input_batch_size, 2, 3, 4])
    lbl_input_batch = tf.cond(
        tf.greater(.5, tf.random_uniform([])),
        lambda: tf.ones([input_batch_size], dtype=tf.int32) * 1,
        lambda: tf.ones([input_batch_size], dtype=tf.int32) * 3)
    probs = np.array([0, .1, 0, .9, 0])
    data_batch, labels = tf.contrib.framework.sampling_ops.stratified_sample(
        val_input_batch, lbl_input_batch, probs, batch_size, enqueue_many=True)
    with self.test_session() as sess:
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)

      for _ in range(20):
        sess.run([data_batch, labels])

      coord.request_stop()
      coord.join(threads)

  def testCanBeCalledMultipleTimes(self):
    batch_size = 20
    val_input_batch = tf.zeros([2, 3, 4])
    lbl_input_batch = tf.ones([], dtype=tf.int32)
    probs = np.array([0, 1, 0, 0, 0])
    batch1 = tf.contrib.framework.sampling_ops.stratified_sample(
        val_input_batch, lbl_input_batch, probs, batch_size)
    batch2 = tf.contrib.framework.sampling_ops.stratified_sample(
        val_input_batch, lbl_input_batch, probs, batch_size)
    summary_op = tf.merge_summary(tf.get_collection(
        tf.GraphKeys.SUMMARIES))

    with self.test_session() as sess:
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)

      sess.run(batch1 + batch2 + (summary_op,))

      coord.request_stop()
      coord.join(threads)

  def testBatchDimensionNotRequired(self):
    classes = 5
    probs = [1.0/classes] * classes

    # Make sure that these vals/labels pairs don't throw any runtime exceptions.
    legal_input_pairs = [
        (np.zeros([2, 3]), [x % classes for x in range(2)]),  # batch dim 2
        (np.zeros([4, 15]), [x % classes for x in range(4)]),  # batch dim 4
        (np.zeros([10, 1]), [x % classes for x in range(10)]),  # batch dim 10
    ]

    # Set up graph with placeholders.
    vals_ph = tf.placeholder(tf.float32)  # completely undefined shape
    labels_ph = tf.placeholder(tf.int32)  # completely undefined shape
    vals_tf, lbls_tf, _ = tf.contrib.framework.sampling_ops._verify_input(
        vals_ph, labels_ph, probs)

    # Run graph to make sure there are no shape-related runtime errors.
    for vals, labels in legal_input_pairs:
      with self.test_session() as sess:
        sess.run([vals_tf, lbls_tf], feed_dict={vals_ph: vals,
                                                labels_ph: labels})

  def testNormalBehavior(self):
    # Set up graph.
    tf.set_random_seed(1234)
    lbl1 = 0
    lbl2 = 3
    # This cond allows the necessary class queues to be populated.
    label = tf.cond(
        tf.greater(.5, tf.random_uniform([])),
        lambda: tf.constant(lbl1),
        lambda: tf.constant(lbl2))
    val = np.array([1, 4]) * label
    probs = np.array([.8, 0, 0, .2, 0])
    batch_size = 16

    data_batch, labels = tf.contrib.framework.sampling_ops.stratified_sample(
        val, label, probs, batch_size)

    # Run session and keep track of how frequently the labels and values appear.
    data_l = []
    label_l = []
    with self.test_session() as sess:
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)

      for _ in range(20):
        data, lbls = sess.run([data_batch, labels])
        data_l.append(data)
        label_l.append(lbls)

      coord.request_stop()
      coord.join(threads)

    # First check that the data matches the labels.
    for lbl, data in zip(label_l, data_l):
      for i in range(batch_size):
        self.assertListEqual(list(np.array([1, 4]) * lbl[i]), list(data[i, :]))

    # Check that the labels are approximately correct.
    expected_label = probs[0] * lbl1 + probs[3] * lbl2
    lbl_list = range(len(probs))
    lbl_std_dev = np.sqrt(np.sum((np.square(lbl_list - expected_label))))
    lbl_std_dev_of_mean = lbl_std_dev / np.sqrt(len(label_l))  # CLT
    actual_lbl = np.mean(label_l)
    # Tolerance is 3 standard deviations of the mean. According to the central
    # limit theorem, this should cover 99.7% of cases. Note that since the seed
    # is fixed, for a given implementation, this test will pass or fail 100% of
    # the time. This use of assertNear is to cover cases where someone changes
    # an implementation detail, which would cause the random behavior to differ.
    self.assertNear(actual_lbl, expected_label, 3*lbl_std_dev_of_mean)

if __name__ == '__main__':
  tf.test.main()
