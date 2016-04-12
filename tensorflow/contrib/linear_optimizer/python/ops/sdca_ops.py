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
"""Proximal stochastic dual coordinate ascent optimizer for linear models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import uuid

from six.moves import range  # pylint: disable=redefined-builtin

from tensorflow.contrib.linear_optimizer.ops import gen_sdca_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework.load_library import load_op_library
from tensorflow.python.framework.ops import convert_to_tensor
from tensorflow.python.framework.ops import name_scope
from tensorflow.python.framework.ops import op_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables as var_ops
from tensorflow.python.ops.nn import sigmoid_cross_entropy_with_logits
from tensorflow.python.platform import resource_loader

__all__ = ['SdcaModel']

_sdca_ops = load_op_library(resource_loader.get_path_to_datafile(
    '_sdca_ops.so'))
assert _sdca_ops, 'Could not load _sdca_ops.so'


# TODO(sibyl-Aix6ihai): add op_scope to appropriate methods.
class SdcaModel(object):
  """Stochastic dual coordinate ascent solver for linear models.

    This class currently only supports a single machine (multi-threaded)
    implementation. We expect the weights and duals to fit in a single machine.

    Loss functions supported:
     * Binary logistic loss
     * Squared loss
     * Hinge loss

    This class defines an optimizer API to train a linear model.

    ### Usage

    ```python
    # Create a solver with the desired parameters.
    lr = tf.contrib.linear_optimizer.SdcaModel(
        container, examples, variables, options)
    opt_op = lr.minimize()

    predictions = lr.predictions(examples)
    # Primal loss + L1 loss + L2 loss.
    regularized_loss = lr.regularized_loss(examples)
    # Primal loss only
    unregularized_loss = lr.unregularized_loss(examples)

    container: Name of the container (eg a hex-encoded UUID) where internal
      state of the optimizer can be stored. The container can be safely shared
      across many models.
    examples: {
      sparse_features: list of SparseTensors of value type float32.
      dense_features: list of dense tensors of type float32.
      example_labels: a tensor of type float32 and shape [Num examples]
      example_weights: a tensor of type float32 and shape [Num examples]
      example_ids: a tensor of type string and shape [Num examples]
    }
    variables: {
      sparse_features_weights: list of tensors of shape [vocab size]
      dense_features_weights: list of tensors of shape [1]
    }
    options: {
      symmetric_l1_regularization: 0.0
      symmetric_l2_regularization: 1.0
      loss_type: "logistic_loss"
    }
    ```

    In the training program you will just have to run the returned Op from
    minimize(). You should also eventually cleanup the temporary state used by
    the model, by resetting its (possibly shared) container.

    ```python
    # Execute opt_op and train for num_steps.
    for _ in xrange(num_steps):
      opt_op.run()

    # You can also check for convergence by calling
    # lr.approximate_duality_gap()
    ```
  """

  def __init__(self, container, examples, variables, options):
    """Create a new sdca optimizer."""

    if not container or not examples or not variables or not options:
      raise ValueError('All arguments must be specified.')

    supported_losses = ('logistic_loss', 'squared_loss', 'hinge_loss')
    if options['loss_type'] not in supported_losses:
      raise ValueError('Unsupported loss_type: ', options['loss_type'])

    self._assertSpecified(
        ['example_labels', 'example_weights', 'example_ids', 'sparse_features',
         'dense_features'], examples)
    self._assertList(['sparse_features', 'dense_features'], examples)

    self._assertSpecified(
        ['sparse_features_weights', 'dense_features_weights'], variables)
    self._assertList(
        ['sparse_features_weights', 'dense_features_weights'], variables)

    self._assertSpecified(
        ['loss_type', 'symmetric_l2_regularization',
         'symmetric_l1_regularization'], options)

    for name in ['symmetric_l1_regularization', 'symmetric_l2_regularization']:
      value = options[name]
      if value < 0.0:
        raise ValueError('%s should be non-negative. Found (%f)' %
                         (name, value))

    self._container = container
    self._examples = examples
    self._variables = variables
    self._options = options
    self._solver_uuid = uuid.uuid4().hex
    self._create_slots()

  def _symmetric_l2_regularization(self):
    # Algorithmic requirement (for now) is to have minimal l2 of 1.0
    return max(self._options['symmetric_l2_regularization'], 1.0)

  # TODO(sibyl-Aix6ihai): Use optimizer interface to make use of slot creation logic.
  def _create_slots(self):
    # Make internal variables which have the updates before applying L1
    # regularization.
    self._slots = {
        'unshrinked_sparse_features_weights': [],
        'unshrinked_dense_features_weights': [],
    }
    for name in ['sparse_features_weights', 'dense_features_weights']:
      for var in self._variables[name]:
        self._slots['unshrinked_' + name].append(var_ops.Variable(
            array_ops.zeros_like(var.initialized_value(), dtypes.float32)))

  def _assertSpecified(self, items, check_in):
    for x in items:
      if check_in[x] is None:
        raise ValueError(check_in[x] + ' must be specified.')

  def _assertList(self, items, check_in):
    for x in items:
      if not isinstance(check_in[x], list):
        raise ValueError(x + ' must be a list.')

  def _l1_loss(self):
    """Computes the l1 loss of the model."""
    with name_scope('l1_loss'):
      sum = 0.0
      for name in ['sparse_features_weights', 'dense_features_weights']:
        for weights in self._convert_n_to_tensor(self._variables[name]):
          sum += math_ops.reduce_sum(math_ops.abs(weights))
      # SDCA L1 regularization cost is: l1 * sum(|weights|)
      return self._options['symmetric_l1_regularization'] * sum

  def _l2_loss(self, l2):
    """Computes the l2 loss of the model."""
    with name_scope('l2_loss'):
      sum = 0.0
      for name in ['sparse_features_weights', 'dense_features_weights']:
        for weights in self._convert_n_to_tensor(self._variables[name]):
          sum += math_ops.reduce_sum(math_ops.square(weights))
      # SDCA L2 regularization cost is: l2 * sum(weights^2) / 2
      return l2 * sum / 2.0

  def _convert_n_to_tensor(self, input_list, as_ref=False):
    """Converts input list to a set of tensors."""
    return [convert_to_tensor(x, as_ref=as_ref) for x in input_list]

  def _linear_predictions(self, examples):
    """Returns predictions of the form w*x."""
    with name_scope('sdca/prediction'):
      sparse_variables = self._convert_n_to_tensor(self._variables[
          'sparse_features_weights'])
      result = 0.0
      for st_i, sv in zip(examples['sparse_features'], sparse_variables):
        ei, fi = array_ops.split(1, 2, st_i.indices)
        ei = array_ops.reshape(ei, [-1])
        fi = array_ops.reshape(fi, [-1])
        fv = array_ops.reshape(st_i.values, [-1])
        # TODO(sibyl-Aix6ihai): This does not work if examples have empty features.
        result += math_ops.segment_sum(
            math_ops.mul(array_ops.gather(sv, fi), fv), ei)
      dense_features = self._convert_n_to_tensor(examples['dense_features'])
      dense_variables = self._convert_n_to_tensor(self._variables[
          'dense_features_weights'])

      for i in range(len(dense_variables)):
        result += dense_features[i] * dense_variables[i]

    # Reshaping to allow shape inference at graph construction time.
    return array_ops.reshape(result, [-1])

  def predictions(self, examples):
    """Add operations to compute predictions by the model.

    If logistic_loss is being used, predicted probabilities are returned.
    Otherwise, (raw) linear predictions (w*x) are returned.

    Args:
      examples: Examples to compute predictions on.

    Returns:
      An Operation that computes the predictions for examples.

    Raises:
      ValueError: if examples are not well defined.
    """
    self._assertSpecified(
        ['example_weights', 'sparse_features', 'dense_features'], examples)
    self._assertList(['sparse_features', 'dense_features'], examples)

    result = self._linear_predictions(examples)
    if self._options['loss_type'] == 'logistic_loss':
      # Convert logits to probability for logistic loss predictions.
      with name_scope('sdca/logistic_prediction'):
        result = math_ops.sigmoid(result)
    return result

  def minimize(self, global_step=None, name=None):
    """Add operations to train a linear model by minimizing the loss function.

    Args:
      global_step: Optional `Variable` to increment by one after the
        variables have been updated.
      name: Optional name for the returned operation.

    Returns:
      An Operation that updates the variables passed in the constructor.
    """
    # Technically, the op depends on a lot more than the variables,
    # but we'll keep the list short.
    with op_scope([], name, 'sdca/minimize'):
      sparse_features_indices = []
      sparse_features_values = []
      for sf in self._examples['sparse_features']:
        sparse_features_indices.append(convert_to_tensor(sf.indices))
        sparse_features_values.append(convert_to_tensor(sf.values))

      step_op = _sdca_ops.sdca_solver(
          sparse_features_indices,
          sparse_features_values,
          self._convert_n_to_tensor(self._examples['dense_features']),
          convert_to_tensor(self._examples['example_weights']),
          convert_to_tensor(self._examples['example_labels']),
          convert_to_tensor(self._examples['example_ids']),
          self._convert_n_to_tensor(
              self._slots['unshrinked_sparse_features_weights'],
              as_ref=True),
          self._convert_n_to_tensor(
              self._slots['unshrinked_dense_features_weights'],
              as_ref=True),
          l1=self._options['symmetric_l1_regularization'],
          l2=self._symmetric_l2_regularization(),
          # TODO(sibyl-Aix6ihai): Provide empirical evidence for this. It is better
          # to run more than one iteration on single mini-batch as we want to
          # spend more time in compute. SDCA works better with larger
          # mini-batches and there is also recent work that shows its better to
          # reuse old samples than train on new samples.
          # See: http://arxiv.org/abs/1602.02136.
          num_inner_iterations=2,
          loss_type=self._options['loss_type'],
          container=self._container,
          solver_uuid=self._solver_uuid)
      with ops.control_dependencies([step_op]):
        assign_ops = []
        for name in ['sparse_features_weights', 'dense_features_weights']:
          for var, slot_var in zip(self._variables[name],
                                   self._slots['unshrinked_' + name]):
            assign_ops.append(var.assign(slot_var))
        assign_group = control_flow_ops.group(*assign_ops)
        with ops.control_dependencies([assign_group]):
          shrink_l1 = _sdca_ops.sdca_shrink_l1(
              self._convert_n_to_tensor(
                  self._variables['sparse_features_weights'],
                  as_ref=True),
              self._convert_n_to_tensor(
                  self._variables['dense_features_weights'],
                  as_ref=True),
              l1=self._options['symmetric_l1_regularization'],
              l2=self._symmetric_l2_regularization())
      if not global_step:
        return shrink_l1
      with ops.control_dependencies([shrink_l1]):
        with ops.colocate_with(global_step):
          return state_ops.assign_add(global_step, 1, name=name).op

  def approximate_duality_gap(self):
    """Add operations to compute the approximate duality gap.

    Returns:
      An Operation that computes the approximate duality gap over all
      examples.
    """
    (primal_loss, dual_loss, example_weights) = _sdca_ops.sdca_training_stats(
        container=self._container,
        solver_uuid=self._solver_uuid)
    # Note that example_weights is guaranteed to be positive by
    # sdca_training_stats so dividing by it is safe.
    return (primal_loss + dual_loss + math_ops.to_double(self._l1_loss()) +
            (2.0 * math_ops.to_double(self._l2_loss(
                self._symmetric_l2_regularization())))) / example_weights

  def unregularized_loss(self, examples):
    """Add operations to compute the loss (without the regularization loss).

    Args:
      examples: Examples to compute unregularized loss on.

    Returns:
      An Operation that computes mean (unregularized) loss for given set of
      examples.

    Raises:
      ValueError: if examples are not well defined.
    """
    self._assertSpecified(
        ['example_labels', 'example_weights', 'sparse_features',
         'dense_features'], examples)
    self._assertList(['sparse_features', 'dense_features'], examples)
    with name_scope('sdca/unregularized_loss'):
      predictions = self._linear_predictions(examples)
      labels = convert_to_tensor(examples['example_labels'])
      weights = convert_to_tensor(examples['example_weights'])

      if self._options['loss_type'] == 'logistic_loss':
        return math_ops.reduce_sum(math_ops.mul(
            sigmoid_cross_entropy_with_logits(
                predictions, labels), weights)) / math_ops.reduce_sum(weights)

      if self._options['loss_type'] == 'hinge_loss':
        # hinge_loss = max{0, 1 - y_i w*x} where y_i \in {-1, 1}. So, we need to
        # first convert 0/1 labels into -1/1 labels.
        all_ones = array_ops.ones_like(predictions)
        adjusted_labels = math_ops.sub(2 * labels, all_ones)
        all_zeros = array_ops.zeros_like(predictions)
        # Tensor that contains (unweighted) error (hinge loss) per
        # example.
        error = math_ops.maximum(all_zeros, math_ops.sub(
            all_ones, math_ops.mul(adjusted_labels, predictions)))
        weighted_error = math_ops.mul(error, weights)
        return math_ops.reduce_sum(weighted_error) / math_ops.reduce_sum(
            weights)

      # squared loss
      err = math_ops.sub(labels, predictions)

      weighted_squared_err = math_ops.mul(math_ops.square(err), weights)
      # SDCA squared loss function is sum(err^2) / (2*sum(weights))
      return (math_ops.reduce_sum(weighted_squared_err) /
              (2.0 * math_ops.reduce_sum(weights)))

  def regularized_loss(self, examples):
    """Add operations to compute the loss with regularization loss included.

    Args:
      examples: Examples to compute loss on.

    Returns:
      An Operation that computes mean (regularized) loss for given set of
      examples.
    Raises:
      ValueError: if examples are not well defined.
    """
    self._assertSpecified(
        ['example_labels', 'example_weights', 'sparse_features',
         'dense_features'], examples)
    self._assertList(['sparse_features', 'dense_features'], examples)
    with name_scope('sdca/regularized_loss'):
      weights = convert_to_tensor(examples['example_weights'])
      return (((
          self._l1_loss() +
          # Note that here we are using the raw regularization
          # (as specified by the user) and *not*
          # self._symmetric_l2_regularization().
          self._l2_loss(self._options['symmetric_l2_regularization'])) /
               math_ops.reduce_sum(weights)) +
              self.unregularized_loss(examples))
