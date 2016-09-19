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

"""Sampling functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.training import input as input_ops
from tensorflow.python.training import queue_runner

__all__ = ['stratified_sample',
           'stratified_sample_unknown_dist',]


def stratified_sample(tensors, labels, target_probs, batch_size,
                      init_probs=None, enqueue_many=False, queue_capacity=16,
                      threads_per_queue=1, name=None):
  """Stochastically creates batches based on per-class probabilities.

  This method discards examples. Internally, it creates one queue to amortize
  the cost of disk reads, and one queue to hold the properly-proportioned
  batch. See `stratified_sample_unknown_dist` for a function that performs
  stratified sampling with one queue per class and doesn't require knowing the
  class data-distribution ahead of time.

  Args:
    tensors: List of tensors for data. All tensors are either one item or a
        batch, according to enqueue_many.
    labels: Tensor for label of data. Label is a single integer or a batch,
        depending on enqueue_many. It is not a one-hot vector.
    target_probs: Target class proportions in batch. An object whose type has a
        registered Tensor conversion function.
    batch_size: Size of batch to be returned.
    init_probs: Class proportions in the data. An object whose type has a
        registered Tensor conversion function, or `None` for estimating the
        initial distribution.
    enqueue_many: Bool. If true, interpret input tensors as having a batch
        dimension.
    queue_capacity: Capacity of the large queue that holds input examples.
    threads_per_queue: Number of threads for the large queue that holds input
        examples and for the final queue with the proper class proportions.
    name: Optional prefix for ops created by this function.
  Raises:
    ValueError: enqueue_many is True and labels doesn't have a batch
        dimension, or if enqueue_many is False and labels isn't a scalar.
    ValueError: enqueue_many is True, and batch dimension on data and labels
        don't match.
    ValueError: if probs don't sum to one.
    ValueError: if a zero initial probability class has a nonzero target
        probability.
    TFAssertion: if labels aren't integers in [0, num classes).
  Returns:
    (data_batch, label_batch), where data_batch is a list of tensors of the same
        length as `tensors`

  Example:
    # Get tensor for a single data and label example.
    data, label = data_provider.Get(['data', 'label'])

    # Get stratified batch according to per-class probabilities.
    target_probs = [...distribution you want...]
    [data_batch], labels = tf.contrib.training.stratified_sample(
        [data], label, target_probs)

    # Run batch through network.
    ...
  """
  with ops.name_scope(name, 'stratified_sample', tensors + [labels]):
    tensor_list = ops.convert_n_to_tensor_or_indexed_slices(tensors)
    labels = ops.convert_to_tensor(labels)
    target_probs = ops.convert_to_tensor(target_probs, dtype=dtypes.float32)
    # Reduce the case of a single example to that of a batch of size 1.
    if not enqueue_many:
      tensor_list = [array_ops.expand_dims(tensor, 0) for tensor in tensor_list]
      labels = array_ops.expand_dims(labels, 0)

    # If `init_probs` is `None`, set up online estimation of data distribution.
    if init_probs is None:
      # We use `target_probs` to get the number of classes, so its shape must be
      # fully defined at graph construction time.
      target_probs.get_shape().assert_is_fully_defined()
      init_probs = _estimate_data_distribution(
          labels, target_probs.get_shape().num_elements())
    else:
      init_probs = ops.convert_to_tensor(init_probs, dtype=dtypes.float32)

    # Validate that input is consistent.
    tensor_list, labels, [init_probs, target_probs] = _verify_input(
        tensor_list, labels, [init_probs, target_probs])

    # Check that all zero initial probabilities also have zero target
    # probabilities.
    assert_op = control_flow_ops.Assert(
        math_ops.reduce_all(math_ops.logical_or(
            math_ops.not_equal(init_probs, 0),
            math_ops.equal(target_probs, 0))),
        ['All classes with zero initial probability must also have zero target '
         'probability: ', init_probs, target_probs])
    init_probs = control_flow_ops.with_dependencies([assert_op], init_probs)

    # Calculate acceptance sampling probabilities.
    accept_probs = _calculate_acceptance_probabilities(init_probs, target_probs)
    proportion_rejected = math_ops.reduce_sum((1 - accept_probs) * init_probs)
    accept_probs = control_flow_ops.cond(
        math_ops.less(proportion_rejected, .5),
        lambda: accept_probs,
        lambda: logging_ops.Print(  # pylint: disable=g-long-lambda
            accept_probs, [accept_probs],
            message='Proportion of examples rejected by sampler is high.',
            first_n=10))

    # Make a single queue to hold input examples. Reshape output so examples
    # don't have singleton batch dimension.
    batched = input_ops.batch(tensor_list + [labels],
                              batch_size=1,
                              num_threads=threads_per_queue,
                              capacity=queue_capacity,
                              enqueue_many=True)
    val_list = [array_ops.squeeze(x, [0]) for x in batched[:-1]]
    label = array_ops.squeeze(batched[-1], [0])

    # Set up second queue containing batches that have the desired class
    # proportions.
    cur_prob = array_ops.gather(accept_probs, label)
    batched = _conditional_batch(
        val_list + [label], cur_prob, batch_size, threads_per_queue)
    return batched[:-1], batched[-1]


def stratified_sample_unknown_dist(tensors, labels, probs, batch_size,
                                   enqueue_many=False, queue_capacity=16,
                                   threads_per_queue=1, name=None):
  """Stochastically creates batches based on per-class probabilities.

  **NOTICE** This sampler can be significantly slower than `stratified_sample`
  due to each thread discarding all examples not in its assigned class.

  This uses a number of threads proportional to the number of classes. See
  `stratified_sample` for an implementation that discards fewer examples and
  uses a fixed number of threads. This function's only advantage over
  `stratified_sample` is that the class data-distribution doesn't need to be
  known ahead of time.

  Args:
    tensors: List of tensors for data. All tensors are either one item or a
        batch, according to enqueue_many.
    labels: Tensor for label of data. Label is a single integer or a batch,
        depending on enqueue_many. It is not a one-hot vector.
    probs: Target class probabilities. An object whose type has a registered
        Tensor conversion function.
    batch_size: Size of batch to be returned.
    enqueue_many: Bool. If true, interpret input tensors as having a batch
        dimension.
    queue_capacity: Capacity of each per-class queue.
    threads_per_queue: Number of threads for each per-class queue.
    name: Optional prefix for ops created by this function.
  Raises:
    ValueError: enqueue_many is True and labels doesn't have a batch
        dimension, or if enqueue_many is False and labels isn't a scalar.
    ValueError: enqueue_many is True, and batch dimension of data and labels
        don't match.
    ValueError: if probs don't sum to one.
    TFAssertion: if labels aren't integers in [0, num classes).
  Returns:
    (data_batch, label_batch), where data_batch is a list of tensors of the same
        length as `tensors`

  Example:
    # Get tensor for a single data and label example.
    data, label = data_provider.Get(['data', 'label'])

    # Get stratified batch according to per-class probabilities.
    init_probs = [1.0/NUM_CLASSES for _ in range(NUM_CLASSES)]
    [data_batch], labels = (
        tf.contrib.training.stratified_sample_unknown_dist(
            [data], label, init_probs, 16))

    # Run batch through network.
    ...
  """
  with ops.name_scope(name, 'stratified_sample_unknown_dist',
                      tensors + [labels]):
    tensor_list = ops.convert_n_to_tensor_or_indexed_slices(tensors)
    labels = ops.convert_to_tensor(labels)
    probs = ops.convert_to_tensor(probs, dtype=dtypes.float32)
    # Reduce the case of a single example to that of a batch of size 1.
    if not enqueue_many:
      tensor_list = [array_ops.expand_dims(tensor, 0) for tensor in tensor_list]
      labels = array_ops.expand_dims(labels, 0)

    # Validate that input is consistent.
    tensor_list, labels, [probs] = _verify_input(tensor_list, labels, [probs])

    # Make per-class queues.
    per_class_queues = _make_per_class_queues(
        tensor_list, labels, probs.get_shape().num_elements(), queue_capacity,
        threads_per_queue)

    # Use the per-class queues to generate stratified batches.
    return _get_batch_from_per_class_queues(
        per_class_queues, probs, batch_size)


def _estimate_data_distribution(labels, num_classes, smoothing_constant=10):
  """Estimate data distribution as labels are seen."""
  # Variable to track running count of classes. Smooth by a nonzero value to
  # avoid division-by-zero. Higher values provide more stability at the cost of
  # slower convergence.
  if smoothing_constant <= 0:
    raise ValueError('smoothing_constant must be nonzero.')
  num_examples_per_class_seen = variables.Variable(
      initial_value=[smoothing_constant] * num_classes, trainable=False,
      name='class_count', dtype=dtypes.int64)

  # Update the class-count based on what labels are seen in batch.
  num_examples_per_class_seen = num_examples_per_class_seen.assign_add(
      math_ops.reduce_sum(array_ops.one_hot(labels, num_classes,
                                            dtype=dtypes.int64), 0))

  # Normalize count into a probability.
  # NOTE: Without the `+= 0` line below, the test
  # `testMultiThreadedEstimateDataDistribution` fails. The reason is that
  # before this line, `num_examples_per_class_seen` is a Tensor that shares a
  # buffer with an underlying `ref` object. When the `ref` is changed by another
  # thread, `num_examples_per_class_seen` changes as well. Since this can happen
  # in the middle of the normalization computation, we get probabilities that
  # are very far from summing to one. Adding `+= 0` copies the contents of the
  # tensor to a new buffer, which will be consistent from the start to the end
  # of the normalization computation.
  num_examples_per_class_seen += 0
  init_prob_estimate = math_ops.truediv(
      num_examples_per_class_seen,
      math_ops.reduce_sum(num_examples_per_class_seen))

  # Must return float32 (not float64) to agree with downstream `_verify_input`
  # checks.
  return math_ops.cast(init_prob_estimate, dtypes.float32)


def _verify_input(tensor_list, labels, probs_list):
  """Verify that batched inputs are well-formed."""
  checked_probs_list = []
  for probs in probs_list:
    # Since number of classes shouldn't change at runtime, probalities shape
    # should be fully defined.
    probs.get_shape().assert_is_fully_defined()

    # Probabilities must be 1D.
    probs.get_shape().assert_has_rank(1)

    # Probabilities must be nonnegative and sum to one.
    tol = 1e-6
    prob_sum = math_ops.reduce_sum(probs)
    checked_probs = control_flow_ops.with_dependencies(
        [check_ops.assert_non_negative(probs),
         check_ops.assert_less(prob_sum, 1.0 + tol),
         check_ops.assert_less(1.0 - tol, prob_sum)],
        probs)
    checked_probs_list.append(checked_probs)

  # All probabilities should be the same length.
  prob_length = checked_probs_list[0].get_shape().num_elements()
  for checked_prob in checked_probs_list:
    if checked_prob.get_shape().num_elements() != prob_length:
      raise ValueError('Probability parameters must have the same length.')

  # Labels tensor should only have batch dimension.
  labels.get_shape().assert_has_rank(1)

  for tensor in tensor_list:
    # Data tensor should have a batch dimension.
    tensor_shape = tensor.get_shape().with_rank_at_least(1)

    # Data and label batch dimensions must be compatible.
    tensor_shape[0].assert_is_compatible_with(labels.get_shape()[0])

  # Data and labels must have the same, strictly positive batch size. Since we
  # can't assume we know the batch size at graph creation, add runtime checks.
  labels_batch_size = array_ops.shape(labels)[0]
  lbl_assert = check_ops.assert_positive(labels_batch_size)

  # Make each tensor depend on its own checks.
  labels = control_flow_ops.with_dependencies([lbl_assert], labels)
  tensor_list = [control_flow_ops.with_dependencies(
      [lbl_assert,
       check_ops.assert_equal(array_ops.shape(x)[0], labels_batch_size)],
      x) for x in tensor_list]

  # Label's classes must be integers 0 <= x < num_classes.
  labels = control_flow_ops.with_dependencies(
      [check_ops.assert_integer(labels),
       check_ops.assert_non_negative(labels),
       check_ops.assert_less(labels, math_ops.cast(prob_length, labels.dtype))],
      labels)

  return tensor_list, labels, checked_probs_list


def _calculate_acceptance_probabilities(init_probs, target_probs):
  """Calculate the per-class acceptance rates.

  Args:
    init_probs: The class probabilities of the data.
    target_probs: The desired class proportion in minibatches.
  Returns:
    A list of the per-class acceptance probabilities.

  This method is based on solving the following analysis:

  Let F be the probability of a rejection (on any example).
  Let p_i be the proportion of examples in the data in class i (init_probs)
  Let a_i is the rate the rejection sampler should *accept* class i
  Let t_i is the target proportion in the minibatches for class i (target_probs)

  ```
  F = sum_i(p_i * (1-a_i))
    = 1 - sum_i(p_i * a_i)     using sum_i(p_i) = 1
  ```

  An example with class `i` will be accepted if `k` rejections occur, then an
  example with class `i` is seen by the rejector, and it is accepted. This can
  be written as follows:

  ```
  t_i = sum_k=0^inf(F^k * p_i * a_i)
      = p_i * a_j / (1 - F)    using geometric series identity, since 0 <= F < 1
      = p_i * a_i / sum_j(p_j * a_j)        using F from above
  ```

  Note that the following constraints hold:
  ```
  0 <= p_i <= 1, sum_i(p_i) = 1
  0 <= a_i <= 1
  0 <= t_i <= 1, sum_i(t_i) = 1
  ```


  A solution for a_i in terms of the other variabes is the following:
    ```a_i = (t_i / p_i) / max_i[t_i / p_i]```
  """
  # Make list of t_i / p_i.
  ratio_l = target_probs / init_probs

  # Replace NaNs with 0s.
  ratio_l = math_ops.select(math_ops.is_nan(ratio_l),
                            array_ops.zeros_like(ratio_l),
                            ratio_l)

  # Calculate list of acceptance probabilities.
  max_ratio = math_ops.reduce_max(ratio_l)
  return ratio_l / max_ratio


def _conditional_batch(tensors, accept_prob, batch_size, queue_threads=10):
  """Conditionally enqueue tensors based on accept_prob.

  Specifically, enqueue the element if accept_prob > rand_unif([0, 1]).

  Args:
      tensors: List of tensors to enqueue.
      accept_prob: Acceptance probability per example.
      batch_size: Size of batch.
      queue_threads: Number of threads enqueuing in the final queue.

  Returns:
      List of batched tensors.

  Raises:
      ValueError: `accept_prob` isn't 0D.
  """
  accept_prob.get_shape().assert_has_rank(0)
  # Determine shapes and types of to-be-enqueued-tensors.
  shapes_list = []
  dtypes_list = []
  for tensor in tensors:
    cur_shape = tensor.get_shape()
    cur_shape.assert_is_fully_defined()
    shapes_list.append(cur_shape)
    dtypes_list.append(tensor.dtype)

  final_q = data_flow_ops.FIFOQueue(capacity=batch_size,
                                    shapes=shapes_list,
                                    dtypes=dtypes_list,
                                    name='batched_queue')
  logging_ops.scalar_summary('queue/%s/size' % final_q.name, final_q.size())

  # Conditionally enqueue.
  # Reshape enqueue op to match no_op's shape.
  eq_tf = math_ops.less(random_ops.random_uniform([]), accept_prob)
  conditional_enqueue = control_flow_ops.cond(
      eq_tf,
      lambda: final_q.enqueue(tensors),
      control_flow_ops.no_op)
  queue_runner.add_queue_runner(queue_runner.QueueRunner(
      final_q, [conditional_enqueue] * queue_threads))

  out_tensor = final_q.dequeue_many(batch_size)
  # Queues return a single tensor if the list of enqued tensors is one. Since we
  # want the type to be the same in all cases, always return a list.
  if isinstance(out_tensor, ops.Tensor):
    out_tensor = [out_tensor]

  return out_tensor


def _make_per_class_queues(tensor_list, labels, num_classes, queue_capacity,
                           threads_per_queue):
  """Creates per-class-queues based on data and labels."""
  # Create one queue per class.
  queues = []
  data_shapes = []
  data_dtypes = []
  for data_tensor in tensor_list:
    per_data_shape = data_tensor.get_shape().with_rank_at_least(1)[1:]
    per_data_shape.assert_is_fully_defined()
    data_shapes.append(per_data_shape)
    data_dtypes.append(data_tensor.dtype)

  for i in range(num_classes):
    q = data_flow_ops.FIFOQueue(capacity=queue_capacity,
                                shapes=data_shapes, dtypes=data_dtypes,
                                name='stratified_sample_class%d_queue' % i)
    logging_ops.scalar_summary(
        'queue/%s/stratified_sample_class%d' % (q.name, i), q.size())
    queues.append(q)

  # Partition tensors according to labels. `partitions` is a list of lists, of
  # size num_classes X len(tensor_list). The number of tensors in partition `i`
  # should be the same for all tensors.
  all_partitions = [data_flow_ops.dynamic_partition(data, labels, num_classes)
                    for data in tensor_list]
  partitions = [[cur_partition[i] for cur_partition in all_partitions] for i in
                range(num_classes)]

  # Enqueue each tensor on the per-class-queue.
  for i in range(num_classes):
    enqueue_op = queues[i].enqueue_many(partitions[i]),
    queue_runner.add_queue_runner(queue_runner.QueueRunner(
        queues[i], [enqueue_op] * threads_per_queue))

  return queues


def _get_batch_from_per_class_queues(per_class_queues, probs, batch_size):
  """Generates batches according to per-class-probabilities."""
  num_classes = probs.get_shape().num_elements()
  # Number of examples per class is governed by a multinomial distribution.
  # Note: multinomial takes unnormalized log probabilities for its first
  # argument, of dimension [batch_size, num_classes].
  examples = random_ops.multinomial(
      array_ops.expand_dims(math_ops.log(probs), 0), batch_size)

  # Prepare the data and label batches.
  val_list = []
  label_list = []
  for i in range(num_classes):
    num_examples = math_ops.reduce_sum(
        math_ops.cast(math_ops.equal(examples, i), dtypes.int32))
    tensors = per_class_queues[i].dequeue_many(num_examples)

    # If you enqueue a list with a single tensor, only a single tensor is
    # returned. If you enqueue a list with multiple tensors, then a list is
    # returned. We want to handle both cases, so reduce the case of the single
    # tensor to the case of multiple tensors.
    if not isinstance(tensors, list):
      tensors = [tensors]

    val_list.append(tensors)
    label_list.append(array_ops.ones([num_examples], dtype=dtypes.int32) * i)

  # Create a list of tensor of values. val_list is of dimension
  # [num_classes x len(tensors)]. We want list_batch_vals to be of dimension
  # [len(tensors)].
  num_data = len(val_list[0])
  list_batch_vals = [array_ops.concat(
      0, [val_list[i][j] for i in range(num_classes)]) for j in range(num_data)]

  # Create a tensor of labels.
  batch_labels = array_ops.concat(0, label_list)
  batch_labels.set_shape([batch_size])

  # Debug instrumentation.
  sample_tags = ['stratified_sample/%s/samples_class%i' % (batch_labels.name, i)
                 for i in range(num_classes)]
  logging_ops.scalar_summary(sample_tags, math_ops.reduce_sum(
      array_ops.one_hot(batch_labels, num_classes), 0))

  return list_batch_vals, batch_labels
