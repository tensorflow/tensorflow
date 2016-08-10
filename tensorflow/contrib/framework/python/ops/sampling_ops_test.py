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
from tensorflow.python.platform import tf_logging as logging


class SamplingOpsTest(tf.test.TestCase):

  def testGraphBuildAssertionFailures(self):
    val = [tf.zeros([1, 3]), tf.ones([1, 5])]
    label = tf.constant([1], shape=[1])  # must have batch dimension
    probs = [.2] * 5
    initial_p = [.1, .3, .1, .3, .2]  # only used for stratified_sample
    batch_size = 16

    # Curry the rejection sampler so we can easily run the same tests on both
    # stratified_sample and stratified_sample_unknown_dist.
    def curried_sampler(tensors, labels, probs, batch_size, enqueue_many=True):
      return tf.contrib.framework.sampling_ops.stratified_sample(
          tensors=tensors,
          labels=labels,
          target_probs=probs,
          batch_size=batch_size,
          init_probs=initial_p,
          enqueue_many=enqueue_many)

    samplers = [
        tf.contrib.framework.sampling_ops.stratified_sample_unknown_dist,
        curried_sampler,
    ]

    for sampler in samplers:
      logging.info('Now testing `%s`', sampler.__class__.__name__)
      # Label must have only batch dimension if enqueue_many is True.
      with self.assertRaises(ValueError):
        sampler(val, tf.zeros([]), probs, batch_size, enqueue_many=True)
      with self.assertRaises(ValueError):
        sampler(val, tf.zeros([1, 1]), probs, batch_size, enqueue_many=True)

      # Label must not be one-hot.
      with self.assertRaises(ValueError):
        sampler(val, tf.constant([0, 1, 0, 0, 0]), probs, batch_size)

      # Data must be list, not singleton tensor.
      with self.assertRaises(TypeError):
        sampler(tf.zeros([1, 3]), label, probs, batch_size)

      # Data must have batch dimension if enqueue_many is True.
      with self.assertRaises(ValueError):
        sampler(val, tf.constant(1), probs, batch_size, enqueue_many=True)

      # Batch dimensions on data and labels should be equal.
      with self.assertRaises(ValueError):
        sampler([tf.zeros([2, 1])], label, probs, batch_size, enqueue_many=True)

      # Probabilities must be numpy array, python list, or tensor.
      with self.assertRaises(ValueError):
        sampler(val, label, 1, batch_size)

      # Probabilities shape must be fully defined.
      with self.assertRaises(ValueError):
        sampler(
            val, label, tf.placeholder(
                tf.float32, shape=[None]), batch_size)

    # In the rejection sampling case, make sure that probability lengths are
    # the same.
    with self.assertRaises(ValueError):
      tf.contrib.framework.sampling_ops.stratified_sample(
          val, label, [.1] * 10, batch_size, init_probs=[.2] * 5)

    # In the rejection sampling case, make sure that zero initial probability
    # classes also have zero target probability.
    with self.assertRaises(ValueError):
      tf.contrib.framework.sampling_ops.stratified_sample(
          val, label, [.2, .4, .4], batch_size, init_probs=[0, .5, .5])

    # Probabilities must be 1D.
    with self.assertRaises(ValueError):
      tf.contrib.framework.sampling_ops.stratified_sample_unknown_dist(
          val, label, np.array([[.25, .25], [.25, .25]]), batch_size)

  def testRuntimeAssertionFailures(self):
    valid_probs = [.2] * 5
    valid_labels = [1, 2, 3]
    vals = [tf.zeros([3, 1])]

    illegal_labels = [
        [0, -1, 1],  # classes must be nonnegative
        [5, 1, 1],  # classes must be less than number of classes
        [2, 3],  # data and label batch size must be the same
    ]

    illegal_probs = [
        [.1] * 5,  # probabilities must sum to one
        [-.5, .5, .5, .4, .1],  # probabilities must be non-negative
    ]

    # Set up graph with illegal label vector.
    label_ph = tf.placeholder(tf.int32, shape=[None])
    probs_ph = tf.placeholder(tf.float32, shape=[5])  # shape must be defined
    val_tf, lbl_tf, prob_tf = tf.contrib.framework.sampling_ops._verify_input(
        vals, label_ph, [probs_ph])

    for illegal_label in illegal_labels:
      # Run session that should fail.
      with self.test_session() as sess:
        with self.assertRaises(tf.errors.InvalidArgumentError):
          sess.run([val_tf, lbl_tf],
                   feed_dict={label_ph: illegal_label,
                              probs_ph: valid_probs})

    for illegal_prob in illegal_probs:
      # Run session that should fail.
      with self.test_session() as sess:
        with self.assertRaises(tf.errors.InvalidArgumentError):
          sess.run([prob_tf],
                   feed_dict={label_ph: valid_labels,
                              probs_ph: illegal_prob})

  def batchingBehaviorHelper(self, sampler):
    batch_size = 20
    input_batch_size = 11
    val_input_batch = [tf.zeros([input_batch_size, 2, 3, 4])]
    lbl_input_batch = tf.cond(
        tf.greater(.5, tf.random_uniform([])),
        lambda: tf.ones([input_batch_size], dtype=tf.int32) * 1,
        lambda: tf.ones([input_batch_size], dtype=tf.int32) * 3)
    probs = np.array([0, .2, 0, .8, 0])
    data_batch, labels = sampler(
        val_input_batch, lbl_input_batch, probs, batch_size, enqueue_many=True)
    with self.test_session() as sess:
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)

      sess.run([data_batch, labels])

      coord.request_stop()
      coord.join(threads)

  def testCanBeCalledMultipleTimes(self):
    batch_size = 20
    val_input_batch = [tf.zeros([2, 3, 4])]
    lbl_input_batch = tf.ones([], dtype=tf.int32)
    probs = np.array([0, 1, 0, 0, 0])
    batches = tf.contrib.framework.sampling_ops.stratified_sample(
        val_input_batch, lbl_input_batch, probs, batch_size, init_probs=probs)
    batches += tf.contrib.framework.sampling_ops.stratified_sample(
        val_input_batch, lbl_input_batch, probs, batch_size, init_probs=probs)
    batches += tf.contrib.framework.sampling_ops.stratified_sample_unknown_dist(
        val_input_batch, lbl_input_batch, probs, batch_size)
    batches += tf.contrib.framework.sampling_ops.stratified_sample_unknown_dist(
        val_input_batch, lbl_input_batch, probs, batch_size)
    summary_op = tf.merge_summary(tf.get_collection(tf.GraphKeys.SUMMARIES))

    with self.test_session() as sess:
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)

      sess.run(batches + (summary_op,))

      coord.request_stop()
      coord.join(threads)

  def testBatchingBehavior(self):
    self.batchingBehaviorHelper(
        tf.contrib.framework.sampling_ops.stratified_sample_unknown_dist)

  def testRejectionBatchingBehavior(self):
    initial_p = [0, .3, 0, .7, 0]

    def curried_sampler(val, lbls, probs, batch, enqueue_many=True):
      return tf.contrib.framework.sampling_ops.stratified_sample(
          val,
          lbls,
          probs,
          batch,
          init_probs=initial_p,
          enqueue_many=enqueue_many)

    self.batchingBehaviorHelper(curried_sampler)

  def testProbabilitiesCanBeChanged(self):
    # Set up graph.
    tf.set_random_seed(1234)
    lbl1 = 0
    lbl2 = 3
    # This cond allows the necessary class queues to be populated.
    label = tf.cond(
        tf.greater(.5, tf.random_uniform([])), lambda: tf.constant(lbl1),
        lambda: tf.constant(lbl2))
    val = [np.array([1, 4]) * label]
    probs = tf.placeholder(tf.float32, shape=[5])
    batch_size = 2

    data_batch, labels = tf.contrib.framework.sampling_ops.stratified_sample_unknown_dist(  # pylint: disable=line-too-long
        val, label, probs, batch_size)

    with self.test_session() as sess:
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)

      for _ in range(5):
        [data], lbls = sess.run([data_batch, labels],
                                feed_dict={probs: [1, 0, 0, 0, 0]})
        for data_example in data:
          self.assertListEqual([0, 0], list(data_example))
        self.assertListEqual([0, 0], list(lbls))

      # Now change distribution and expect different output.
      for _ in range(5):
        [data], lbls = sess.run([data_batch, labels],
                                feed_dict={probs: [0, 0, 0, 1, 0]})
        for data_example in data:
          self.assertListEqual([3, 12], list(data_example))
        self.assertListEqual([3, 3], list(lbls))

      coord.request_stop()
      coord.join(threads)

  def testBatchDimensionNotRequired(self):
    classes = 5
    # Probs must be a tensor, since we pass it directly to _verify_input.
    probs = tf.constant([1.0 / classes] * classes)

    # Make sure that these vals/labels pairs don't throw any runtime exceptions.
    legal_input_pairs = [
        (np.zeros([2, 3]), [x % classes for x in range(2)]),  # batch dim 2
        (np.zeros([4, 15]), [x % classes for x in range(4)]),  # batch dim 4
        (np.zeros([10, 1]), [x % classes for x in range(10)]),  # batch dim 10
    ]

    # Set up graph with placeholders.
    vals_ph = tf.placeholder(tf.float32)  # completely undefined shape
    labels_ph = tf.placeholder(tf.int32)  # completely undefined shape
    val_tf, labels_tf, _ = tf.contrib.framework.sampling_ops._verify_input(
        [vals_ph], labels_ph, [probs])

    # Run graph to make sure there are no shape-related runtime errors.
    for vals, labels in legal_input_pairs:
      with self.test_session() as sess:
        sess.run([val_tf, labels_tf],
                 feed_dict={vals_ph: vals,
                            labels_ph: labels})

  def dataListHelper(self, sampler):
    batch_size = 20
    val_input_batch = [tf.zeros([2, 3, 4]), tf.ones([2, 4]), tf.ones(2) * 3]
    lbl_input_batch = tf.ones([], dtype=tf.int32)
    probs = np.array([0, 1, 0, 0, 0])
    val_list, lbls = sampler(val_input_batch, lbl_input_batch, probs,
                             batch_size)

    # Check output shapes.
    self.assertTrue(isinstance(val_list, list))
    self.assertEqual(len(val_list), len(val_input_batch))
    self.assertTrue(isinstance(lbls, tf.Tensor))

    with self.test_session() as sess:
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)

      out = sess.run(val_list + [lbls])

      coord.request_stop()
      coord.join(threads)

    # Check output shapes.
    self.assertEqual(len(out), len(val_input_batch) + 1)

  def testDataListInput(self):
    self.dataListHelper(
        tf.contrib.framework.sampling_ops.stratified_sample_unknown_dist)

  def testRejectionDataListInput(self):
    initial_p = [0, 1, 0, 0, 0]

    def curried_sampler(val, lbls, probs, batch, enqueue_many=False):
      return tf.contrib.framework.sampling_ops.stratified_sample(
          val,
          lbls,
          probs,
          batch,
          init_probs=initial_p,
          enqueue_many=enqueue_many)

    self.dataListHelper(curried_sampler)

  def normalBehaviorHelper(self, sampler):
    # Set up graph.
    tf.set_random_seed(1234)
    lbl1 = 0
    lbl2 = 3
    # This cond allows the necessary class queues to be populated.
    label = tf.cond(
        tf.greater(.5, tf.random_uniform([])), lambda: tf.constant(lbl1),
        lambda: tf.constant(lbl2))
    val = [np.array([1, 4]) * label]
    probs = np.array([.8, 0, 0, .2, 0])
    batch_size = 16

    data_batch, labels = sampler(val, label, probs, batch_size)

    # Run session and keep track of how frequently the labels and values appear.
    data_l = []
    label_l = []
    with self.test_session() as sess:
      # Need to initialize variables that keep running total of classes seen.
      tf.initialize_all_variables().run()

      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)

      for _ in range(20):
        [data], lbls = sess.run([data_batch, labels])
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
    self.assertNear(actual_lbl, expected_label, 3 * lbl_std_dev_of_mean)

  def testNormalBehavior(self):
    self.normalBehaviorHelper(
        tf.contrib.framework.sampling_ops.stratified_sample_unknown_dist)

  def testRejectionNormalBehavior(self):
    initial_p = [.7, 0, 0, .3, 0]

    def curried_sampler(val, lbls, probs, batch, enqueue_many=False):
      return tf.contrib.framework.sampling_ops.stratified_sample(
          val,
          lbls,
          probs,
          batch,
          init_probs=initial_p,
          enqueue_many=enqueue_many)

    self.normalBehaviorHelper(curried_sampler)

  def testRejectionNormalBehaviorWithOnlineInitPEstimate(self):

    def curried_sampler(val, lbls, probs, batch, enqueue_many=False):
      return tf.contrib.framework.sampling_ops.stratified_sample(
          val, lbls, probs, batch, init_probs=None, enqueue_many=enqueue_many)

    self.normalBehaviorHelper(curried_sampler)


if __name__ == '__main__':
  tf.test.main()
