# Copyright 2015 Google Inc. All Rights Reserved.
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

# pylint: disable=g-short-docstring-punctuation
"""## Higher level ops related to regularization and building layers.

This package provides several ops that take care of creating variables that are
used internally in a consistent way and provide the building blocks for many
common machine learning algorithms.

@@fully_connected

## Regularizers

Regularization can help prevent overfitting.
These have the signature `fn(weights)`. The loss is typically added to
`tf.GraphKeys.REGULARIZATION_LOSS`

@@l1_regularizer
@@l2_regularizer

## Initializations

This also includes a common initialization for connecting multiple layers.

@@xavier_initializer
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numbers

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import logging


__all__ = ['xavier_initializer', 'fully_connected', 'l1_regularizer',
           'l2_regularizer']


def xavier_initializer(n_inputs, n_outputs, uniform=True):
  """Set the parameter initialization using the method described in paper.

  Xavier Glorot and Yoshua Bengio (2010):
           Understanding the difficulty of training deep feedforward neural
           networks. International conference on artificial intelligence and
           statistics.

  This method is designed to keep the scale of the gradients roughly the same
  in all layers. In uniform distribution this ends up being the range:
  `x = sqrt(6. / (in + out)); [-x, x]` and for normal distribution a standard
  deviation of `sqrt(3. / (in + out))` is used.

  Args:
    n_inputs: The number of input nodes into each output.
    n_outputs: The number of output nodes for each input.
    uniform: If true use a uniform distribution, otherwise use a truncated
      normal.

  Returns:
    An initializer.
  """
  if uniform:
    # 6 was used in the paper.
    init_range = math.sqrt(6.0 / (n_inputs + n_outputs))
    return standard_ops.random_uniform_initializer(-init_range, init_range)
  else:
    # 3 gives us approximately the same limits as above since this repicks
    # values greater than 2 standard deviations from the mean.
    stddev = math.sqrt(3.0 / (n_inputs + n_outputs))
    return standard_ops.truncated_normal_initializer(stddev=stddev)


def _assert_summary_tag_unique(tag):
  for summary in ops.get_collection(ops.GraphKeys.SUMMARIES):
    old_tag = tensor_util.ConstantValue(summary.op.inputs[0])
    if tag == str(old_tag):
      raise ValueError('Conflict with summary tag: %s exists on summary %s %s' %
                       (tag, summary, old_tag))


def _add_scalar_summary(tensor, tag=None):
  """Add a summary operation for the tensor.

  Args:
    tensor: The tensor to summarize.
    tag: The tag to use, if None then use tensor's op's name.

  Returns:
    The created histogram summary.

  Raises:
    ValueError: If the tag is already in use or the rank is not 0.
  """
  tensor.get_shape().assert_has_rank(0)
  tag = tag or tensor.op.name
  _assert_summary_tag_unique(tag)
  return standard_ops.scalar_summary(tag, tensor, name='%s_summary' % tag)


def _add_histogram_summary(tensor, tag=None):
  """Add a summary operation for the histogram of a tensor.

  Args:
    tensor: The tensor to summarize.
    tag: The tag to use, if None then use tensor's op's name.

  Returns:
    The created histogram summary.

  Raises:
    ValueError: If the tag is already in use.
  """
  # TODO(opensource): A global or scoped mechanism to disable summaries.
  tag = tag or tensor.op.name
  _assert_summary_tag_unique(tag)
  return standard_ops.histogram_summary(tag, tensor, name='%s_summary' % tag)


def _apply_activation_with_summaries(x, activation_fn):
  """Returns activation_fn(x).

  This applies the given activation and adds useful summaries specific to the
  activation.

  Args:
    x: The tensor to apply activation to.
    activation_fn: An activation function.
  Returns:
    A tensor with activation applied to x.
  """
  if activation_fn is None:
    return x
  y = activation_fn(x)
  if activation_fn in (nn.relu, nn.softplus, nn.relu6):
    # Using x for comparison to avoid floating point equality and/or epsilons.
    _add_scalar_summary(
        standard_ops.reduce_mean(standard_ops.to_float(standard_ops.less(
            x, 0.0))), '%s/zeros' % y.op.name)
  if activation_fn is nn.relu6:
    _add_scalar_summary(
        standard_ops.reduce_mean(standard_ops.to_float(standard_ops.greater(
            x, 6.0))), '%s/sixes' % y.op.name)
  if activation_fn is nn.l2_normalize:
    _add_scalar_summary(
        standard_ops.reduce_mean(standard_ops.sqrt(standard_ops.sum(
            standard_ops.square(x), 1))), '%s/length' % y.op.name)
  _add_histogram_summary(y, '%s/activations' % y.op.name)
  return y


def _apply_regularization(w, regularizer):
  loss = regularizer(w)
  if loss:
    ops.add_to_collection(ops.GraphKeys.REGULARIZATION_LOSSES, loss)


def l1_regularizer(scale):
  """Returns a function that can be used to apply L1 regularization to weights.

  L1 regularization encourages sparsity.

  Args:
    scale: A scalar multiplier `Tensor`. 0.0 disables the regularizer.

  Returns:
    A function with signature `l1(weights, name=None)` that apply L1
    regularization.

  Raises:
    ValueError: If scale is outside of the range [0.0, 1.0] or if scale is not a
    float.
  """
  if isinstance(scale, numbers.Integral):
    raise ValueError('scale cannot be an integer: %s' % scale)
  if isinstance(scale, numbers.Real):
    if scale < 0.:
      raise ValueError('Setting a scale less than 0 on a regularizer: %g' %
                       scale)
    if scale >= 1.:
      raise ValueError('Setting a scale greater than 1 on a regularizer: %g' %
                       scale)
    if scale == 0.:
      logging.info('Scale of 0 disables regularizer.')
      return lambda _, name=None: None
  def l1(weights, name=None):
    """Applies L1 regularization to weights."""
    with ops.op_scope([weights], name, 'l1_regularizer') as scope:
      my_scale = ops.convert_to_tensor(scale,
                                       dtype=weights.dtype.base_dtype,
                                       name='scale')
      return standard_ops.mul(
          my_scale,
          standard_ops.reduce_sum(standard_ops.abs(weights)),
          name=scope)
  return l1


def l2_regularizer(scale):
  """Returns a function that can be used to apply L2 regularization to weights.

  Small values of L2 can help prevent overfitting the training data.

  Args:
    scale: A scalar multiplier `Tensor`. 0.0 disables the regularizer.

  Returns:
    A function with signature `l2(weights, name=None)` that applies L2
    regularization.

  Raises:
    ValueError: If scale is outside of the range [0.0, 1.0] or if scale is not a
    float.
  """
  if isinstance(scale, numbers.Integral):
    raise ValueError('scale cannot be an integer: %s' % (scale,))
  if isinstance(scale, numbers.Real):
    if scale < 0.:
      raise ValueError('Setting a scale less than 0 on a regularizer: %g.' %
                       scale)
    if scale >= 1.:
      raise ValueError('Setting a scale greater than 1 on a regularizer: %g.' %
                       scale)
    if scale == 0.:
      logging.info('Scale of 0 disables regularizer.')
      return lambda _, name=None: None
  def l2(weights, name=None):
    """Applies l2 regularization to weights."""
    with ops.op_scope([weights], name, 'l2_regularizer') as scope:
      my_scale = ops.convert_to_tensor(scale,
                                       dtype=weights.dtype.base_dtype,
                                       name='scale')
      return standard_ops.mul(my_scale, nn.l2_loss(weights), name=scope)
  return l2


def fully_connected(x,
                    num_output_nodes,
                    activation_fn=None,
                    weight_init=None,
                    bias_init=standard_ops.constant_initializer(0.),
                    num_input_nodes=None,
                    name=None,
                    weight_collections=None,
                    bias_collections=None,
                    weight_regularizer=None,
                    create_summaries=True):
  """Adds the parameters for a fully connected layer and returns the output.

  A fully connected layer is generally defined as a matrix multiply:
  \\\\(y = f(w * x + b)\\\\) where **f** is given by `activation_fn`

  This op creates `w` and optionally `b` (disable with `bias_init=None`) and
  adds various summaries that can be useful for visualizing learning or
  diagnosing training problems. The variable creation is compatible with
  `tf.variable_scope` and so can be reused with `tf.variable_scope` or
  `tf.make_template`.

  In almost all cases, the number of input nodes can be inferred from the shape
  of `x`, but if it is unspecified or additional size checks are desired, then
  `num_input_nodes` can be specified.

  Most of the details of variable creation can be controlled by specifying the
  initializers (`weight_init` and `bias_init`) and which collections to place
  the created variables in (`weight_collections` and `bias_collections`).

  A per layer regularization can be specified by setting `weight_regularizer`.
  This is only applied to weights and not the bias.

  Args:
    x: The input tensor.
    num_output_nodes: The size of the output.
    activation_fn: A function that requires a single Tensor that is applied as a
      non-linearity. If None is used, then this is a linear layer.
    weight_init: An optional initialization. If not specified, uses Xavier
      initialization (see `tf.learn.xavier_initializer`).
    bias_init: An initializer for the bias, defaults to 0.
    num_input_nodes: The number of input nodes.
    name: The name for this operation is used to name operations and to find
      variables. If specified it must be unique for this scope, otherwise a
      unique name starting with "fully_connected" will be created.  See
      `tf.variable_op_scope` for details.
    weight_collections: List of graph collections for just weights.
    bias_collections: List of graph collections for just bias.
    weight_regularizer: A regularizer like the result of
      `tf.learn.l1_regularizer` or `tf.learn.l2_regularizer`.
    create_summaries: Set to false to disable summaries.

  Returns:
    The result of applying a fully connected layer.

  Raises:
    ValueError: if `x` is not rank 2; or `x`'s second dimension is not known
    and `num_input_nodes` is not specified.
  """
  with variable_scope.variable_op_scope([x], name, 'fully_connected') as vs:
    # Check rank and if num_input_nodes is specified, make sure it matches.
    x.get_shape().assert_is_compatible_with([None, num_input_nodes])

    if not num_input_nodes:
      if x.get_shape().dims is None or x.get_shape().dims[1].value is None:
        raise ValueError(
            'If x has an unknown first dimension then num_input_nodes '
            'must be specified; shape: %s num_input_nodes: %s'
            % (x.get_shape(), num_input_nodes))
      else:
        num_input_nodes = x.get_shape().dims[1].value

    weight_init = weight_init or xavier_initializer(
        num_input_nodes, num_output_nodes)

    dtype = x.dtype
    w = variable_scope.get_variable('weights',
                                    shape=[num_input_nodes, num_output_nodes],
                                    dtype=dtype,
                                    initializer=weight_init,
                                    collections=weight_collections)

    if not vs.reuse and create_summaries:
      _add_histogram_summary(w)

    y = standard_ops.matmul(x, w)
    # Regularization is only applied to the weights and not bias.
    if weight_regularizer:
      _apply_regularization(w, weight_regularizer)
    if bias_init:
      b = variable_scope.get_variable('bias',
                                      shape=[num_output_nodes],
                                      dtype=dtype,
                                      initializer=bias_init,
                                      collections=bias_collections)
      if not vs.reuse and create_summaries:
        _add_histogram_summary(b)

      y = nn.bias_add(y, b)

    if create_summaries:
      return _apply_activation_with_summaries(y, activation_fn)
    else:
      return activation_fn(y)
