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
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import input as input_ops

__all__ = [
    'rejection_sample',
    'stratified_sample',
]


def rejection_sample(tensors,
                     accept_prob_fn,
                     batch_size,
                     queue_threads=1,
                     enqueue_many=False,
                     prebatch_capacity=16,
                     prebatch_threads=1,
                     runtime_checks=False,
                     name=None):
  """Stochastically creates batches by rejection sampling.

  Each list of non-batched tensors is evaluated by `accept_prob_fn`, to produce
  a scalar tensor between 0 and 1. This tensor corresponds to the probability of
  being accepted. When `batch_size` tensor groups have been accepted, the batch
  queue will return a mini-batch.

  Args:
    tensors: List of tensors for data. All tensors are either one item or a
        batch, according to enqueue_many.
    accept_prob_fn: A python lambda that takes a non-batch tensor from each
        item in `tensors`, and produces a scalar tensor.
    batch_size: Size of batch to be returned.
    queue_threads: The number of threads for the queue that will hold the final
      batch.
    enqueue_many: Bool. If true, interpret input tensors as having a batch
        dimension.
    prebatch_capacity: Capacity for the large queue that is used to convert
      batched tensors to single examples.
    prebatch_threads: Number of threads for the large queue that is used to
      convert batched tensors to single examples.
    runtime_checks: Bool. If true, insert runtime checks on the output of
        `accept_prob_fn`. Using `True` might have a performance impact.
    name: Optional prefix for ops created by this function.
  Raises:
    ValueError: enqueue_many is True and labels doesn't have a batch
        dimension, or if enqueue_many is False and labels isn't a scalar.
    ValueError: enqueue_many is True, and batch dimension on data and labels
        don't match.
    ValueError: if a zero initial probability class has a nonzero target
        probability.
  Returns:
    A list of tensors of the same length as `tensors`, with batch dimension
    `batch_size`.

  Example:
    # Get tensor for a single data and label example.
    data, label = data_provider.Get(['data', 'label'])

    # Get stratified batch according to data tensor.
    accept_prob_fn = lambda x: (tf.tanh(x[0]) + 1) / 2
    data_batch = tf.contrib.training.rejection_sample(
        [data, label], accept_prob_fn, 16)

    # Run batch through network.
    ...
  """
  with variable_scope.variable_scope(name, 'rejection_sample', tensors):
    tensor_list = ops.convert_n_to_tensor_or_indexed_slices(tensors)
    # Reduce the case of a batched example to that of a batch of a single
    # example by taking a batch of size one.
    if enqueue_many:
      # Validate that batch dimension of the input is consistent.
      tensor_list = _verify_data_inputs(tensor_list)

      # Make a single queue to hold input examples. Reshape output so examples
      # don't have singleton batch dimension.
      batched = input_ops.batch(
          tensor_list,
          batch_size=1,
          num_threads=prebatch_threads,
          capacity=prebatch_capacity,
          enqueue_many=True)
      tensor_list = [array_ops.squeeze(x, [0]) for x in batched]

    # Set up a queue containing batches that have the distribution.
    cur_prob = accept_prob_fn(tensor_list)
    if runtime_checks:
      cur_prob = array_ops.identity(
          control_flow_ops.with_dependencies([
              check_ops.assert_less_equal(0.0, cur_prob),
              check_ops.assert_less_equal(cur_prob, 1.0)
          ], cur_prob),
          name='prob_with_checks')
    minibatch = input_ops.maybe_batch(
        tensor_list,
        keep_input=random_ops.random_uniform([]) < cur_prob,
        batch_size=batch_size,
        num_threads=queue_threads)

    # Queues return a single tensor if the list of enqued tensors is one. Since
    # we want the type to always be the same, always return a list.
    if isinstance(minibatch, ops.Tensor):
      minibatch = [minibatch]

    return minibatch


def stratified_sample(tensors,
                      labels,
                      target_probs,
                      batch_size,
                      init_probs=None,
                      enqueue_many=False,
                      queue_capacity=16,
                      threads_per_queue=1,
                      name=None):
  """Stochastically creates batches based on per-class probabilities.

  This method discards examples. Internally, it creates one queue to amortize
  the cost of disk reads, and one queue to hold the properly-proportioned
  batch.

  Args:
    tensors: List of tensors for data. All tensors are either one item or a
        batch, according to enqueue_many.
    labels: Tensor for label of data. Label is a single integer or a batch,
        depending on `enqueue_many`. It is not a one-hot vector.
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
    ValueError: If `tensors` isn't iterable.
    ValueError: `enqueue_many` is True and labels doesn't have a batch
        dimension, or if `enqueue_many` is False and labels isn't a scalar.
    ValueError: `enqueue_many` is True, and batch dimension on data and labels
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
  with ops.name_scope(name, 'stratified_sample', list(tensors) + [labels]):
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
        math_ops.reduce_all(
            math_ops.logical_or(
                math_ops.not_equal(init_probs, 0),
                math_ops.equal(target_probs, 0))),
        ['All classes with zero initial probability must also have zero target '
         'probability: ', init_probs, target_probs
        ])
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
    batched = input_ops.batch(
        tensor_list + [labels],
        batch_size=1,
        num_threads=threads_per_queue,
        capacity=queue_capacity,
        enqueue_many=True)
    val_list = [array_ops.squeeze(x, [0]) for x in batched[:-1]]
    label = array_ops.squeeze(batched[-1], [0])

    # Set up second queue containing batches that have the desired class
    # proportions.
    cur_prob = array_ops.gather(accept_probs, label)
    batched = input_ops.maybe_batch(
        val_list + [label],
        keep_input=random_ops.random_uniform([]) < cur_prob,
        batch_size=batch_size,
        num_threads=threads_per_queue)
    return batched[:-1], batched[-1]


def _estimate_data_distribution(labels, num_classes, smoothing_constant=10):
  """Estimate data distribution as labels are seen."""
  # Variable to track running count of classes. Smooth by a nonzero value to
  # avoid division-by-zero. Higher values provide more stability at the cost of
  # slower convergence.
  if smoothing_constant <= 0:
    raise ValueError('smoothing_constant must be nonzero.')
  num_examples_per_class_seen = variable_scope.variable(
      initial_value=[smoothing_constant] * num_classes,
      trainable=False,
      name='class_count',
      dtype=dtypes.int64)

  # Update the class-count based on what labels are seen in batch.
  num_examples_per_class_seen = num_examples_per_class_seen.assign_add(
      math_ops.reduce_sum(
          array_ops.one_hot(
              labels, num_classes, dtype=dtypes.int64), 0))

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


def _verify_data_inputs(tensor_list):
  """Verify that batched data inputs are well-formed."""
  for tensor in tensor_list:
    # Data tensor should have a batch dimension.
    tensor_shape = tensor.get_shape().with_rank_at_least(1)

    # Data batch dimensions must be compatible.
    tensor_shape[0].assert_is_compatible_with(tensor_list[0].get_shape()[0])

  return tensor_list


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
    checked_probs = control_flow_ops.with_dependencies([
        check_ops.assert_non_negative(probs),
        check_ops.assert_less(prob_sum, 1.0 + tol),
        check_ops.assert_less(1.0 - tol, prob_sum)
    ], probs)
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
  tensor_list = [
      control_flow_ops.with_dependencies([
          lbl_assert,
          check_ops.assert_equal(array_ops.shape(x)[0], labels_batch_size)
      ], x) for x in tensor_list
  ]

  # Label's classes must be integers 0 <= x < num_classes.
  labels = control_flow_ops.with_dependencies([
      check_ops.assert_integer(labels), check_ops.assert_non_negative(labels),
      check_ops.assert_less(labels, math_ops.cast(prob_length, labels.dtype))
  ], labels)

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
  ratio_l = array_ops.where(
      math_ops.is_nan(ratio_l), array_ops.zeros_like(ratio_l), ratio_l)

  # Calculate list of acceptance probabilities.
  max_ratio = math_ops.reduce_max(ratio_l)
  return ratio_l / max_ratio
