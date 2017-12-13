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
"""Helper functions to add support for magnitude-based model pruning.

  # Adds variables and ops to the graph to enable
  # elementwise masking of weights
  apply_mask(weights)

  # Returns a list containing the sparsity of each of the weight tensors
  get_weight_sparsity()

  # Returns a list of all the masked weight tensorflow variables
  get_masked_weights()

  # Returns a list of all the mask tensorflow variables
  get_masks()

  # Returns a list of all the thresholds
  get_thresholds()

  # Returns a list of all the weight tensors that have been masked
  get_weights()

  The Pruning class uses a proto (defined in pruning.proto) to set up the
  parameters for a pruning specification. Here's a typical usage:

  # Initialize a pruning spec from a proto
  pruning_spec = '/tmp/pruning.pb'
  p = Pruning(pruning_spec)

  # Add mask update ops to the graph
  mask_update_op = p.conditional_mask_update_op()

  # Add the summaries
  p.add_pruning_summaries()

  # Run the op
  session.run(mask_update_op)

  # An object of the pruning also accepts externally defined sparsity:
  sparsity = tf.Variable(0.5, name = "ConstantSparsity")
  pruning_spec = '/tmp/pruning.pb'
  p = Pruning(pruning_spec, sparsity=sparsity)

"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.model_pruning.python.layers import core_layers as core
from tensorflow.contrib.training.python.training import hparam
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary
from tensorflow.python.training import training_util

_MASK_COLLECTION = core.MASK_COLLECTION
_THRESHOLD_COLLECTION = core.THRESHOLD_COLLECTION
_MASKED_WEIGHT_COLLECTION = core.MASKED_WEIGHT_COLLECTION
_WEIGHT_COLLECTION = core.WEIGHT_COLLECTION
_MASKED_WEIGHT_NAME = core.MASKED_WEIGHT_NAME


def _weight_mask_variable(var, scope):
  """Create a mask for the weights.

  This function adds a variable 'mask' to the graph.

  Args:
    var: the weight variable that needs to be masked
    scope: The variable scope of the variable var

  Returns:
    the mask variable of the same size and shape as var, initialized to all 1s.
  """
  with variable_scope.variable_scope(scope):
    mask = variable_scope.get_variable(
        'mask',
        var.get_shape(),
        initializer=init_ops.ones_initializer(),
        trainable=False,
        dtype=var.dtype)
  return mask


def _weight_threshold_variable(var, scope):
  """Create a scalar threshold for the weights.

  This function adds a variable
  'threshold' to the graph.

  Args:
    var: The weight variable that needs to be masked
    scope: The variable scope of the variable var

  Returns:
    a scalar threshold variable initialized to 0.
  """
  with variable_scope.variable_scope(scope):
    threshold = variable_scope.get_variable(
        'threshold', [],
        initializer=init_ops.zeros_initializer(),
        trainable=False,
        dtype=var.dtype)
    return threshold


def _histogram(values, value_range, nbins=100, dtype=np.int32, name=None):
  """Return histogram of values.

  Given the tensor `values`, this operation returns a rank 1 histogram counting
  the number of entries in `values` that fell into every bin.  The bins are
  equal width and determined by the arguments `value_range` and `nbins`.

  Args:
    values:  Numeric `Tensor`.
    value_range:  Shape [2] `Tensor` of same `dtype` as `values`.
      values <= value_range[0] will be mapped to hist[0],
      values >= value_range[1] will be mapped to hist[-1].
    nbins:  Scalar `int32 Tensor`.  Number of histogram bins.
    dtype:  dtype for returned histogram.
    name:  A name for this operation (defaults to 'histogram').

  Returns:
    A 1-D `Tensor` holding histogram of values.

  """
  with ops.name_scope(name, 'histogram', [values, value_range, nbins]) as scope:
    values = ops.convert_to_tensor(values, name='values')
    values = gen_array_ops.reshape(values, [-1])
    value_range = ops.convert_to_tensor(value_range, name='value_range')
    nbins = ops.convert_to_tensor(nbins, dtype=np.int32, name='nbins')
    nbins_float = math_ops.cast(nbins, values.dtype)

    # Map tensor values that fall within value_range to [0, 1].
    scaled_values = math_ops.truediv(
        values - value_range[0],
        value_range[1] - value_range[0],
        name='scaled_values')

    # map tensor values within the open interval value_range to {0,.., nbins-1},
    # values outside the open interval will be zero or less, or nbins or more.
    indices = math_ops.floor(nbins_float * scaled_values, name='indices')

    # Clip edge cases (e.g. value = value_range[1]) or "outliers."
    indices = math_ops.cast(
        clip_ops.clip_by_value(indices, 0, nbins_float - 1), np.int32)

    return math_ops.unsorted_segment_sum(
        array_ops.ones_like(indices, dtype=dtype), indices, nbins, name=scope)


def _determine_partitioned_axis(partitioned_variable):
  partitioned_axis = 0
  concatenated_variable_shape = partitioned_variable.get_shape()
  for partition in partitioned_variable:
    partition_shape = partition.get_shape()
    maybe_partitioned_axis = np.less(partition_shape,
                                     concatenated_variable_shape)
    # Sanity check: make sure number of partitioned axis == 1
    if np.count_nonzero(maybe_partitioned_axis) != 1:
      raise ValueError('Number of partitioned axes %s not equal to 1' %
                       np.count_nonzero(maybe_partitioned_axis))
    partitioned_axis = np.where(maybe_partitioned_axis)[0][0]
  return partitioned_axis


def _variable_assign(var, new_value):
  return state_ops.assign(var, new_value, name=var.op.name + '_assign')


def _partitioned_variable_assign(partitioned_var, new_value):
  """Assign op for partitioned variables.

  Args:
    partitioned_var: A partitioned tensotflow variable
    new_value: Value to be assigned to the variable var

  Returns:
    A tensorflow op that groups the assign ops for each of the variable slices
  """
  # Determine which axis was used to partition the variable. Currently
  # tensorflow allows partitioning variable only along 1 axis.
  axis = 0 if len(partitioned_var) == 1 else _determine_partitioned_axis(
      partitioned_var)

  partition_sizes = np.array(
      [partition.get_shape()[axis] for partition in partitioned_var])
  new_partitioned_values = array_ops.split(
      new_value,
      ops.convert_to_tensor(partition_sizes, dtype=np.int32),
      axis=axis)
  op_list = []
  for partition in partitioned_var:
    op_list.append(
        _variable_assign(partition, new_partitioned_values[len(op_list)]))
  return control_flow_ops.group(
      *op_list, name=partitioned_var.name + '_group_assign')


def apply_mask(x, scope=''):
  """Apply mask to a given weight tensor.

  Args:
    x: Input weight tensor
    scope: The current variable scope. Defaults to ""
  Returns:
    Tensor representing masked_weights
  """

  mask = _weight_mask_variable(x, scope)
  threshold = _weight_threshold_variable(x, scope)
  # Add masked_weights in the weights namescope so as to make it easier
  # for the quantization library to add quant ops.
  masked_weights = math_ops.multiply(mask, x, _MASKED_WEIGHT_NAME)

  # Make sure the mask for a given variable are not added multiple times to the
  # collection. This is particularly important when applying mask to RNN's
  # weight variables
  if mask not in ops.get_collection_ref(_MASK_COLLECTION):
    ops.add_to_collection(_THRESHOLD_COLLECTION, threshold)
    ops.add_to_collection(_MASK_COLLECTION, mask)
    ops.add_to_collection(_MASKED_WEIGHT_COLLECTION, masked_weights)
    ops.add_to_collection(_WEIGHT_COLLECTION, x)
  return masked_weights


def get_masked_weights():
  return ops.get_collection(_MASKED_WEIGHT_COLLECTION)


def get_masks():
  return ops.get_collection(_MASK_COLLECTION)


def get_thresholds():
  return ops.get_collection(_THRESHOLD_COLLECTION)


def get_weights():
  return ops.get_collection(_WEIGHT_COLLECTION)


def get_weight_sparsity():
  """Get sparsity of the weights.

  Args:
    None

  Returns:
    A list containing the sparsity of each of the weight tensors
  """
  masks = get_masks()
  return [nn_impl.zero_fraction(mask) for mask in masks]


def get_pruning_hparams():
  """Get a tf.HParams object with the default values for the hyperparameters.

    name: string
      name of the pruning specification. Used for adding summaries and ops under
      a common tensorflow name_scope
    begin_pruning_step: integer
      the global step at which to begin pruning
    end_pruning_step: integer
      the global step at which to terminate pruning. Defaults to -1 implying
      that pruning continues till the training stops
    do_not_prune: list of strings
      list of layers that are not pruned
    threshold_decay: float
      the decay factor to use for exponential decay of the thresholds
    pruning_frequency: integer
      How often should the masks be updated? (in # of global_steps)
    nbins: integer
      number of bins to use for histogram computation
    initial_sparsity: float
      initial sparsity value
    target_sparsity: float
      target sparsity value
    sparsity_function_begin_step: integer
      the global step at this which the gradual sparsity function begins to
      take effect
    sparsity_function_end_step: integer
      the global step used as the end point for the gradual sparsity function
    sparsity_function_exponent: float
      exponent = 1 is linearly varying sparsity between initial and final.
      exponent > 1 varies more slowly towards the end than the beginning

    We use the following sparsity function:

    num_steps = (sparsity_function_end_step -
                 sparsity_function_begin_step)/pruning_frequency
    sparsity(step) = (initial_sparsity - target_sparsity)*
                     [1-step/(num_steps -1)]**exponent + target_sparsity

  Args:
    None

  Returns:
    tf.HParams object initialized to default values

  """
  return hparam.HParams(
      name='model_pruning',
      begin_pruning_step=0,
      end_pruning_step=-1,
      do_not_prune=[''],
      threshold_decay=0.9,
      pruning_frequency=10,
      nbins=255,
      initial_sparsity=0,
      target_sparsity=0.5,
      sparsity_function_begin_step=0,
      sparsity_function_end_step=100,
      sparsity_function_exponent=3)


class Pruning(object):

  def __init__(self, spec=None, global_step=None, sparsity=None):
    """Set up the specification for model pruning.

    If a spec is provided, the sparsity is set up based on the sparsity_function
    in the spec. The effect of sparsity_function is overridden if the sparsity
    variable is passed to the constructor. This enables setting up arbitrary
    sparsity profiles externally and passing it to this pruning functions.

    Args:
      spec: Pruning spec as defined in pruning.proto
      global_step: A tensorflow variable that is used while setting up the
        sparsity function
      sparsity: A tensorflow scalar variable storing the sparsity
    """
    # Pruning specification
    self._spec = spec if spec else get_pruning_hparams()

    # A tensorflow variable that tracks the sparsity function.
    # If not provided as input, the graph must already contain the global_step
    # variable before calling this constructor.
    self._global_step = self._setup_global_step(global_step)

    # Stores the tensorflow sparsity variable.
    # Built using self._setup_sparsity() or provided externally
    self._sparsity = sparsity if sparsity else self._setup_sparsity()

    # List of tensorflow assignments ops for new masks and thresholds
    self._assign_ops = []

    # Tensorflow variable keeping track of the last global step when the masks
    # were updated
    self._last_update_step = self._setup_last_update_step()

  def _setup_global_step(self, global_step):
    graph_global_step = global_step
    if graph_global_step is None:
      graph_global_step = training_util.get_global_step()

    return math_ops.cast(graph_global_step, np.int32)

  def _setup_sparsity(self):
    begin_step = self._spec.sparsity_function_begin_step
    end_step = self._spec.sparsity_function_end_step
    initial_sparsity = self._spec.initial_sparsity
    target_sparsity = self._spec.target_sparsity
    exponent = self._spec.sparsity_function_exponent

    if begin_step >= end_step:
      raise ValueError(
          'Pruning must begin before it can end. begin_step=%d, end_step=%d' %
          (begin_step, end_step))

    with ops.name_scope(self._spec.name):
      p = math_ops.minimum(1.0,
                           math_ops.maximum(
                               0.0,
                               math_ops.div(
                                   math_ops.cast(self._global_step - begin_step,
                                                 np.float32),
                                   end_step - begin_step)))
      sparsity = math_ops.add(
          math_ops.multiply(initial_sparsity - target_sparsity,
                            math_ops.pow(1 - p, exponent)),
          target_sparsity,
          name='sparsity')

    return sparsity

  def _setup_last_update_step(self):
    with variable_scope.variable_scope(self._spec.name) as scope:
      try:
        last_update_step = variable_scope.get_variable(
            'last_mask_update_step', [],
            initializer=init_ops.zeros_initializer(),
            trainable=False,
            dtype=np.int32)
      except ValueError:
        scope.reuse_variables()
        last_update_step = variable_scope.get_variable(
            'last_mask_update_step', dtype=np.int32)
    return last_update_step

  def _exists_in_do_not_prune_list(self, tensor_name):
    do_not_prune_list = self._spec.do_not_prune
    if not do_not_prune_list[0]:
      return False
    for layer_name in do_not_prune_list:
      if tensor_name.find(layer_name) != -1:
        return True

    return False

  def _update_mask(self, weights, threshold):
    """Updates the mask for a given weight tensor.

    This functions first computes the cdf of the weight tensor, and estimates
    the threshold value such that 'desired_sparsity' fraction of weights
    have magnitude less than the threshold.

    Args:
      weights: The weight tensor that needs to be masked.
      threshold: The current threshold value. The function will compute a new
        threshold and return the exponential moving average using the current
        value of threshold

    Returns:
      new_threshold: The new value of the threshold based on weights, and
        desired_sparsity
      new_mask: A n-D numpy array containing 0 or 1 to indicate which of the
        values in weights falls below the threshold

    Raises:
      ValueError: if sparsity is not defined
    """
    if self._sparsity is None:
      raise ValueError('Sparsity variable undefined')

    with ops.name_scope(weights.op.name + '_pruning_ops'):
      abs_weights = math_ops.abs(weights)
      max_value = math_ops.reduce_max(abs_weights)
      histogram = _histogram(
          abs_weights, [0.0, max_value],
          nbins=self._spec.nbins,
          dtype=np.float32)

      cdf = math_ops.cumsum(histogram)
      norm_cdf = math_ops.div(cdf, math_ops.reduce_sum(histogram))
      current_threshold = math_ops.multiply(
          math_ops.div(
              math_ops.reduce_sum(
                  math_ops.cast(
                      math_ops.less(norm_cdf, self._sparsity), np.float32)),
              float(self._spec.nbins)), max_value)

      smoothed_threshold = math_ops.add_n([
          math_ops.multiply(current_threshold, 1 - self._spec.threshold_decay),
          math_ops.multiply(threshold, self._spec.threshold_decay)
      ])
      new_mask = math_ops.cast(
          math_ops.greater(abs_weights, smoothed_threshold), np.float32)
    return smoothed_threshold, new_mask

  def _get_mask_assign_ops(self):
    # Make sure the assignment ops have not already been added to the list
    if self._assign_ops:
      raise ValueError(
          'Assign op list not empty. _get_mask_assign_ops() called twice?')

    masks = get_masks()
    weights = get_weights()
    thresholds = get_thresholds()

    if len(masks) != len(thresholds):
      raise ValueError(
          'Number of masks %s and number of thresholds %s mismatch' %
          (len(masks), len(thresholds)))

    for index, mask in enumerate(masks):
      threshold = thresholds[index]
      weight = weights[index]
      is_partitioned = isinstance(weight, variables.PartitionedVariable)
      if is_partitioned:
        weight = weight.as_tensor()

      if self._spec.do_not_prune:
        if self._exists_in_do_not_prune_list(mask.name):
          continue

      new_threshold, new_mask = self._update_mask(weight, threshold)
      self._assign_ops.append(_variable_assign(threshold, new_threshold))

      self._assign_ops.append(
          _partitioned_variable_assign(mask, new_mask)
          if is_partitioned else _variable_assign(mask, new_mask))

  def mask_update_op(self):
    with ops.name_scope(self._spec.name):
      if not self._assign_ops:
        self._get_mask_assign_ops()
      with ops.control_dependencies([
          state_ops.assign(
              self._last_update_step,
              self._global_step,
              name='last_mask_update_step_assign')
      ]):
        with ops.control_dependencies(self._assign_ops):
          logging.info('Updating masks.')
          return control_flow_ops.no_op('mask_update')

  def conditional_mask_update_op(self):

    def maybe_update_masks():
      with ops.name_scope(self._spec.name):
        is_step_within_pruning_range = math_ops.logical_and(
            math_ops.greater_equal(self._global_step,
                                   self._spec.begin_pruning_step),
            # If end_pruning_step is negative, keep pruning forever!
            math_ops.logical_or(
                math_ops.less_equal(self._global_step,
                                    self._spec.end_pruning_step),
                math_ops.less(self._spec.end_pruning_step, 0)))
        is_pruning_step = math_ops.less_equal(
            math_ops.add(self._last_update_step, self._spec.pruning_frequency),
            self._global_step)
        return math_ops.logical_and(is_step_within_pruning_range,
                                    is_pruning_step)

    def mask_update_op():
      return self.mask_update_op()

    def no_update_op():
      return control_flow_ops.no_op()

    return control_flow_ops.cond(maybe_update_masks(), mask_update_op,
                                 no_update_op)

  def add_pruning_summaries(self):
    """Adds summaries for this pruning spec.

    Args: none

    Returns: none
    """
    with ops.name_scope(self._spec.name + '_summaries'):
      summary.scalar('sparsity', self._sparsity)
      summary.scalar('last_mask_update_step', self._last_update_step)
      masks = get_masks()
      thresholds = get_thresholds()
      for index, mask in enumerate(masks):
        if not self._exists_in_do_not_prune_list(mask.name):
          summary.scalar(mask.name + '/sparsity', nn_impl.zero_fraction(mask))
          summary.scalar(thresholds[index].op.name + '/threshold',
                         thresholds[index])

  def print_hparams(self):
    logging.info(self._spec.to_json())
