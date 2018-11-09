# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
from tensorflow.contrib.training.python.training import sampling_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import coordinator
from tensorflow.python.training import queue_runner_impl


class StratifiedSampleTest(test.TestCase):

  def testGraphBuildAssertionFailures(self):
    val = [array_ops.zeros([1, 3]), array_ops.ones([1, 5])]
    label = constant_op.constant([1], shape=[1])  # must have batch dimension
    probs = [.2] * 5
    init_probs = [.1, .3, .1, .3, .2]
    batch_size = 16

    # Label must have only batch dimension if enqueue_many is True.
    with self.assertRaises(ValueError):
      sampling_ops.stratified_sample(
          val,
          array_ops.zeros([]),
          probs,
          batch_size,
          init_probs,
          enqueue_many=True)
    with self.assertRaises(ValueError):
      sampling_ops.stratified_sample(
          val,
          array_ops.zeros([1, 1]),
          probs,
          batch_size,
          init_probs,
          enqueue_many=True)

    # Label must not be one-hot.
    with self.assertRaises(ValueError):
      sampling_ops.stratified_sample(val,
                                     constant_op.constant([0, 1, 0, 0, 0]),
                                     probs, batch_size, init_probs)

    # Data must be list, not singleton tensor.
    with self.assertRaises(TypeError):
      sampling_ops.stratified_sample(
          array_ops.zeros([1, 3]), label, probs, batch_size, init_probs)

    # Data must have batch dimension if enqueue_many is True.
    with self.assertRaises(ValueError):
      sampling_ops.stratified_sample(
          val,
          constant_op.constant(1),
          probs,
          batch_size,
          init_probs,
          enqueue_many=True)

    # Batch dimensions on data and labels should be equal.
    with self.assertRaises(ValueError):
      sampling_ops.stratified_sample(
          [array_ops.zeros([2, 1])],
          label,
          probs,
          batch_size,
          init_probs,
          enqueue_many=True)

    # Probabilities must be numpy array, python list, or tensor.
    with self.assertRaises(ValueError):
      sampling_ops.stratified_sample(val, label, 1, batch_size, init_probs)

    # Probabilities shape must be fully defined.
    with self.assertRaises(ValueError):
      sampling_ops.stratified_sample(
          val,
          label,
          array_ops.placeholder(
              dtypes.float32, shape=[None]),
          batch_size,
          init_probs)

    # In the rejection sampling case, make sure that probability lengths are
    # the same.
    with self.assertRaises(ValueError):
      sampling_ops.stratified_sample(
          val, label, [.1] * 10, batch_size, init_probs=[.2] * 5)

    # In the rejection sampling case, make sure that zero initial probability
    # classes also have zero target probability.
    with self.assertRaises(ValueError):
      sampling_ops.stratified_sample(
          val, label, [.2, .4, .4], batch_size, init_probs=[0, .5, .5])

  def testRuntimeAssertionFailures(self):
    valid_probs = [.2] * 5
    valid_labels = [1, 2, 3]
    vals = [array_ops.zeros([3, 1])]

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
    label_ph = array_ops.placeholder(dtypes.int32, shape=[None])
    probs_ph = array_ops.placeholder(
        dtypes.float32, shape=[5])  # shape must be defined
    val_tf, lbl_tf, prob_tf = sampling_ops._verify_input(  # pylint: disable=protected-access
        vals, label_ph, [probs_ph])

    for illegal_label in illegal_labels:
      # Run session that should fail.
      with self.cached_session() as sess:
        with self.assertRaises(errors_impl.InvalidArgumentError):
          sess.run([val_tf, lbl_tf],
                   feed_dict={label_ph: illegal_label,
                              probs_ph: valid_probs})

    for illegal_prob in illegal_probs:
      # Run session that should fail.
      with self.cached_session() as sess:
        with self.assertRaises(errors_impl.InvalidArgumentError):
          sess.run([prob_tf],
                   feed_dict={label_ph: valid_labels,
                              probs_ph: illegal_prob})

  def testCanBeCalledMultipleTimes(self):
    batch_size = 20
    val_input_batch = [array_ops.zeros([2, 3, 4])]
    lbl_input_batch = array_ops.ones([], dtype=dtypes.int32)
    probs = np.array([0, 1, 0, 0, 0])
    batches = sampling_ops.stratified_sample(
        val_input_batch, lbl_input_batch, probs, batch_size, init_probs=probs)
    batches += sampling_ops.stratified_sample(
        val_input_batch, lbl_input_batch, probs, batch_size, init_probs=probs)
    summary_op = logging_ops.merge_summary(
        ops.get_collection(ops.GraphKeys.SUMMARIES))

    with self.cached_session() as sess:
      coord = coordinator.Coordinator()
      threads = queue_runner_impl.start_queue_runners(coord=coord)

      sess.run(batches + (summary_op,))

      coord.request_stop()
      coord.join(threads)

  def testRejectionBatchingBehavior(self):
    batch_size = 20
    input_batch_size = 11
    val_input_batch = [array_ops.zeros([input_batch_size, 2, 3, 4])]
    lbl_input_batch = control_flow_ops.cond(
        math_ops.greater(.5, random_ops.random_uniform([])),
        lambda: array_ops.ones([input_batch_size], dtype=dtypes.int32) * 1,
        lambda: array_ops.ones([input_batch_size], dtype=dtypes.int32) * 3)
    probs = np.array([0, .2, 0, .8, 0])
    data_batch, labels = sampling_ops.stratified_sample(
        val_input_batch,
        lbl_input_batch,
        probs,
        batch_size,
        init_probs=[0, .3, 0, .7, 0],
        enqueue_many=True)
    with self.cached_session() as sess:
      coord = coordinator.Coordinator()
      threads = queue_runner_impl.start_queue_runners(coord=coord)

      sess.run([data_batch, labels])

      coord.request_stop()
      coord.join(threads)

  def testBatchDimensionNotRequired(self):
    classes = 5
    # Probs must be a tensor, since we pass it directly to _verify_input.
    probs = constant_op.constant([1.0 / classes] * classes)

    # Make sure that these vals/labels pairs don't throw any runtime exceptions.
    legal_input_pairs = [
        (np.zeros([2, 3]), [x % classes for x in range(2)]),  # batch dim 2
        (np.zeros([4, 15]), [x % classes for x in range(4)]),  # batch dim 4
        (np.zeros([10, 1]), [x % classes for x in range(10)]),  # batch dim 10
    ]

    # Set up graph with placeholders.
    vals_ph = array_ops.placeholder(
        dtypes.float32)  # completely undefined shape
    labels_ph = array_ops.placeholder(
        dtypes.int32)  # completely undefined shape
    val_tf, labels_tf, _ = sampling_ops._verify_input(  # pylint: disable=protected-access
        [vals_ph], labels_ph, [probs])

    # Run graph to make sure there are no shape-related runtime errors.
    for vals, labels in legal_input_pairs:
      with self.cached_session() as sess:
        sess.run([val_tf, labels_tf],
                 feed_dict={vals_ph: vals,
                            labels_ph: labels})

  def testRejectionDataListInput(self):
    batch_size = 20
    val_input_batch = [
        array_ops.zeros([2, 3, 4]), array_ops.ones([2, 4]), array_ops.ones(2) *
        3
    ]
    lbl_input_batch = array_ops.ones([], dtype=dtypes.int32)
    probs = np.array([0, 1, 0, 0, 0])
    val_list, lbls = sampling_ops.stratified_sample(
        val_input_batch,
        lbl_input_batch,
        probs,
        batch_size,
        init_probs=[0, 1, 0, 0, 0])

    # Check output shapes.
    self.assertTrue(isinstance(val_list, list))
    self.assertEqual(len(val_list), len(val_input_batch))
    self.assertTrue(isinstance(lbls, ops.Tensor))

    with self.cached_session() as sess:
      coord = coordinator.Coordinator()
      threads = queue_runner_impl.start_queue_runners(coord=coord)

      out = sess.run(val_list + [lbls])

      coord.request_stop()
      coord.join(threads)

    # Check output shapes.
    self.assertEqual(len(out), len(val_input_batch) + 1)

  def normalBehaviorHelper(self, sampler):
    # Set up graph.
    random_seed.set_random_seed(1234)
    lbl1 = 0
    lbl2 = 3
    # This cond allows the necessary class queues to be populated.
    label = control_flow_ops.cond(
        math_ops.greater(.5, random_ops.random_uniform([])),
        lambda: constant_op.constant(lbl1), lambda: constant_op.constant(lbl2))
    val = [np.array([1, 4]) * label]
    probs = np.array([.8, 0, 0, .2, 0])
    batch_size = 16

    data_batch, labels = sampler(val, label, probs, batch_size)

    # Run session and keep track of how frequently the labels and values appear.
    data_l = []
    label_l = []
    with self.cached_session() as sess:
      # Need to initialize variables that keep running total of classes seen.
      variables.global_variables_initializer().run()

      coord = coordinator.Coordinator()
      threads = queue_runner_impl.start_queue_runners(coord=coord)

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

  def testRejectionNormalBehavior(self):
    initial_p = [.7, 0, 0, .3, 0]

    def curried_sampler(val, lbls, probs, batch, enqueue_many=False):
      return sampling_ops.stratified_sample(
          val,
          lbls,
          probs,
          batch,
          init_probs=initial_p,
          enqueue_many=enqueue_many)

    self.normalBehaviorHelper(curried_sampler)

  def testRejectionNormalBehaviorWithOnlineInitPEstimate(self):

    def curried_sampler(val, lbls, probs, batch, enqueue_many=False):
      return sampling_ops.stratified_sample(
          val, lbls, probs, batch, init_probs=None, enqueue_many=enqueue_many)

    self.normalBehaviorHelper(curried_sampler)


class RejectionSampleTest(test.TestCase):

  def testGraphConstructionFailures(self):
    accept_prob_fn = lambda _: constant_op.constant(1.0)
    batch_size = 32
    # Data must have batch dimension if `enqueue_many` is `True`.
    with self.assertRaises(ValueError):
      sampling_ops.rejection_sample(
          [array_ops.zeros([])], accept_prob_fn, batch_size, enqueue_many=True)

    # Batch dimensions should be equal if `enqueue_many` is `True`.
    with self.assertRaises(ValueError):
      sampling_ops.rejection_sample(
          [array_ops.zeros([5, 1]), array_ops.zeros([4, 1])],
          accept_prob_fn,
          batch_size,
          enqueue_many=True)

  def testRuntimeFailures(self):
    prob_ph = array_ops.placeholder(dtypes.float32, [])
    accept_prob_fn = lambda _: prob_ph
    batch_size = 32

    # Set up graph.
    random_seed.set_random_seed(1234)
    sampling_ops.rejection_sample(
        [array_ops.zeros([])],
        accept_prob_fn,
        batch_size,
        runtime_checks=True,
        name='rejection_sample')
    prob_tensor = ops.get_default_graph().get_tensor_by_name(
        'rejection_sample/prob_with_checks:0')

    # Run session that should fail.
    with self.cached_session() as sess:
      for illegal_prob in [-0.1, 1.1]:
        with self.assertRaises(errors_impl.InvalidArgumentError):
          sess.run(prob_tensor, feed_dict={prob_ph: illegal_prob})

  def testNormalBehavior(self):
    tensor_list = [
        control_flow_ops.cond(
            math_ops.greater(.5, random_ops.random_uniform([])),
            lambda: constant_op.constant(1.0),
            lambda: constant_op.constant(2.0))
    ]
    accept_prob_fn = lambda x: x[0] - 1.0
    batch_size = 10

    # Set up graph.
    sample = sampling_ops.rejection_sample(tensor_list, accept_prob_fn,
                                           batch_size)

    with self.cached_session() as sess:
      coord = coordinator.Coordinator()
      threads = queue_runner_impl.start_queue_runners(coord=coord)

      for _ in range(5):
        sample_np = sess.run(sample)[0]
        self.assertListEqual([2.0] * batch_size, list(sample_np))

      coord.request_stop()
      coord.join(threads)


if __name__ == '__main__':
  test.main()
