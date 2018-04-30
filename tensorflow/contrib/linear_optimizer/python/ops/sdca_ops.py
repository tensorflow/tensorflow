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
"""Proximal stochastic dual coordinate ascent optimizer for linear models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from six.moves import range

from tensorflow.contrib.linear_optimizer.python.ops.sharded_mutable_dense_hashtable import ShardedMutableDenseHashTable
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework.ops import internal_convert_to_tensor
from tensorflow.python.framework.ops import name_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_sdca_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables as var_ops
from tensorflow.python.ops.nn import sigmoid_cross_entropy_with_logits
from tensorflow.python.summary import summary

__all__ = ['SdcaModel']


# TODO(sibyl-Aix6ihai): add name_scope to appropriate methods.
class SdcaModel(object):
  """Stochastic dual coordinate ascent solver for linear models.

    This class currently only supports a single machine (multi-threaded)
    implementation. We expect the weights and duals to fit in a single machine.

    Loss functions supported:

     * Binary logistic loss
     * Squared loss
     * Hinge loss
     * Smooth hinge loss

    This class defines an optimizer API to train a linear model.

    ### Usage

    ```python
    # Create a solver with the desired parameters.
    lr = tf.contrib.linear_optimizer.SdcaModel(examples, variables, options)
    min_op = lr.minimize()
    opt_op = lr.update_weights(min_op)

    predictions = lr.predictions(examples)
    # Primal loss + L1 loss + L2 loss.
    regularized_loss = lr.regularized_loss(examples)
    # Primal loss only
    unregularized_loss = lr.unregularized_loss(examples)

    examples: {
      sparse_features: list of SparseFeatureColumn.
      dense_features: list of dense tensors of type float32.
      example_labels: a tensor of type float32 and shape [Num examples]
      example_weights: a tensor of type float32 and shape [Num examples]
      example_ids: a tensor of type string and shape [Num examples]
    }
    variables: {
      sparse_features_weights: list of tensors of shape [vocab size]
      dense_features_weights: list of tensors of shape [dense_feature_dimension]
    }
    options: {
      symmetric_l1_regularization: 0.0
      symmetric_l2_regularization: 1.0
      loss_type: "logistic_loss"
      num_loss_partitions: 1 (Optional, with default value of 1. Number of
      partitions of the global loss function, 1 means single machine solver,
      and >1 when we have more than one optimizer working concurrently.)
      num_table_shards: 1 (Optional, with default value of 1. Number of shards
      of the internal state table, typically set to match the number of
      parameter servers for large data sets.
    }
    ```

    In the training program you will just have to run the returned Op from
    minimize().

    ```python
    # Execute opt_op and train for num_steps.
    for _ in range(num_steps):
      opt_op.run()

    # You can also check for convergence by calling
    lr.approximate_duality_gap()
    ```
  """

  def __init__(self, examples, variables, options):
    """Create a new sdca optimizer."""

    if not examples or not variables or not options:
      raise ValueError('examples, variables and options must all be specified.')

    supported_losses = ('logistic_loss', 'squared_loss', 'hinge_loss',
                        'smooth_hinge_loss')
    if options['loss_type'] not in supported_losses:
      raise ValueError('Unsupported loss_type: ', options['loss_type'])

    self._assertSpecified([
        'example_labels', 'example_weights', 'example_ids', 'sparse_features',
        'dense_features'
    ], examples)
    self._assertList(['sparse_features', 'dense_features'], examples)

    self._assertSpecified(['sparse_features_weights', 'dense_features_weights'],
                          variables)
    self._assertList(['sparse_features_weights', 'dense_features_weights'],
                     variables)

    self._assertSpecified([
        'loss_type', 'symmetric_l2_regularization',
        'symmetric_l1_regularization'
    ], options)

    for name in ['symmetric_l1_regularization', 'symmetric_l2_regularization']:
      value = options[name]
      if value < 0.0:
        raise ValueError('%s should be non-negative. Found (%f)' %
                         (name, value))

    self._examples = examples
    self._variables = variables
    self._options = options
    self._create_slots()
    self._hashtable = ShardedMutableDenseHashTable(
        key_dtype=dtypes.int64,
        value_dtype=dtypes.float32,
        num_shards=self._num_table_shards(),
        default_value=[0.0, 0.0, 0.0, 0.0],
        # SdcaFprint never returns 0 or 1 for the low64 bits, so this a safe
        # empty_key (that will never collide with actual payloads).
        empty_key=[0, 0])

    summary.scalar('approximate_duality_gap', self.approximate_duality_gap())
    summary.scalar('examples_seen', self._hashtable.size())

  def _symmetric_l1_regularization(self):
    return self._options['symmetric_l1_regularization']

  def _symmetric_l2_regularization(self):
    # Algorithmic requirement (for now) is to have minimal l2 of 1.0.
    return max(self._options['symmetric_l2_regularization'], 1.0)

  def _num_loss_partitions(self):
    # Number of partitions of the global objective.
    # TODO(andreasst): set num_loss_partitions automatically based on the number
    # of workers
    return self._options.get('num_loss_partitions', 1)

  def _adaptive(self):
    # Perform adaptive sampling.
    return self._options.get('adaptive', True)

  def _num_table_shards(self):
    # Number of hash table shards.
    # Return 1 if not specified or if the value is 'None'
    # TODO(andreasst): set num_table_shards automatically based on the number
    # of parameter servers
    num_shards = self._options.get('num_table_shards')
    return 1 if num_shards is None else num_shards

  # TODO(sibyl-Aix6ihai): Use optimizer interface to make use of slot creation logic.
  def _create_slots(self):
    # Make internal variables which have the updates before applying L1
    # regularization.
    self._slots = collections.defaultdict(list)
    for name in ['sparse_features_weights', 'dense_features_weights']:
      for var in self._variables[name]:
        with ops.device(var.device):
          # TODO(andreasst): remove SDCAOptimizer suffix once bug 30843109 is
          # fixed
          self._slots['unshrinked_' + name].append(
              var_ops.Variable(
                  array_ops.zeros_like(var.initialized_value(), dtypes.float32),
                  name=var.op.name + '_unshrinked/SDCAOptimizer'))

  def _assertSpecified(self, items, check_in):
    for x in items:
      if check_in[x] is None:
        raise ValueError(check_in[x] + ' must be specified.')

  def _assertList(self, items, check_in):
    for x in items:
      if not isinstance(check_in[x], list):
        raise ValueError(x + ' must be a list.')

  def _l1_loss(self):
    """Computes the (un-normalized) l1 loss of the model."""
    with name_scope('sdca/l1_loss'):
      sums = []
      for name in ['sparse_features_weights', 'dense_features_weights']:
        for weights in self._convert_n_to_tensor(self._variables[name]):
          with ops.device(weights.device):
            sums.append(
                math_ops.reduce_sum(
                    math_ops.abs(math_ops.cast(weights, dtypes.float64))))
      # SDCA L1 regularization cost is: l1 * sum(|weights|)
      return self._options['symmetric_l1_regularization'] * math_ops.add_n(sums)

  def _l2_loss(self, l2):
    """Computes the (un-normalized) l2 loss of the model."""
    with name_scope('sdca/l2_loss'):
      sums = []
      for name in ['sparse_features_weights', 'dense_features_weights']:
        for weights in self._convert_n_to_tensor(self._variables[name]):
          with ops.device(weights.device):
            sums.append(
                math_ops.reduce_sum(
                    math_ops.square(math_ops.cast(weights, dtypes.float64))))
      # SDCA L2 regularization cost is: l2 * sum(weights^2) / 2
      return l2 * math_ops.add_n(sums) / 2.0

  def _convert_n_to_tensor(self, input_list, as_ref=False):
    """Converts input list to a set of tensors."""
    return [internal_convert_to_tensor(x, as_ref=as_ref) for x in input_list]

  def _linear_predictions(self, examples):
    """Returns predictions of the form w*x."""
    with name_scope('sdca/prediction'):
      sparse_variables = self._convert_n_to_tensor(self._variables[
          'sparse_features_weights'])
      result_sparse = 0.0
      for sfc, sv in zip(examples['sparse_features'], sparse_variables):
        # TODO(sibyl-Aix6ihai): following does not take care of missing features.
        result_sparse += math_ops.segment_sum(
            math_ops.multiply(
                array_ops.gather(sv, sfc.feature_indices), sfc.feature_values),
            sfc.example_indices)
      dense_features = self._convert_n_to_tensor(examples['dense_features'])
      dense_variables = self._convert_n_to_tensor(self._variables[
          'dense_features_weights'])

      result_dense = 0.0
      for i in range(len(dense_variables)):
        result_dense += math_ops.matmul(dense_features[i],
                                        array_ops.expand_dims(
                                            dense_variables[i], -1))

    # Reshaping to allow shape inference at graph construction time.
    return array_ops.reshape(result_dense, [-1]) + result_sparse

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
    with name_scope(name, 'sdca/minimize'):
      sparse_example_indices = []
      sparse_feature_indices = []
      sparse_features_values = []
      for sf in self._examples['sparse_features']:
        sparse_example_indices.append(sf.example_indices)
        sparse_feature_indices.append(sf.feature_indices)
        # If feature values are missing, sdca assumes a value of 1.0f.
        if sf.feature_values is not None:
          sparse_features_values.append(sf.feature_values)

      # pylint: disable=protected-access
      example_ids_hashed = gen_sdca_ops.sdca_fprint(
          internal_convert_to_tensor(self._examples['example_ids']))
      # pylint: enable=protected-access
      example_state_data = self._hashtable.lookup(example_ids_hashed)
      # Solver returns example_state_update, new delta sparse_feature_weights
      # and delta dense_feature_weights.

      weights_tensor = self._convert_n_to_tensor(self._slots[
          'unshrinked_sparse_features_weights'])
      sparse_weights = []
      sparse_indices = []
      for w, i in zip(weights_tensor, sparse_feature_indices):
        # Find the feature ids to lookup in the variables.
        with ops.device(w.device):
          sparse_indices.append(
              math_ops.cast(
                  array_ops.unique(math_ops.cast(i, dtypes.int32))[0],
                  dtypes.int64))
          sparse_weights.append(array_ops.gather(w, sparse_indices[-1]))

      # pylint: disable=protected-access
      esu, sfw, dfw = gen_sdca_ops.sdca_optimizer(
          sparse_example_indices,
          sparse_feature_indices,
          sparse_features_values,
          self._convert_n_to_tensor(self._examples['dense_features']),
          internal_convert_to_tensor(self._examples['example_weights']),
          internal_convert_to_tensor(self._examples['example_labels']),
          sparse_indices,
          sparse_weights,
          self._convert_n_to_tensor(self._slots[
              'unshrinked_dense_features_weights']),
          example_state_data,
          loss_type=self._options['loss_type'],
          l1=self._options['symmetric_l1_regularization'],
          l2=self._symmetric_l2_regularization(),
          num_loss_partitions=self._num_loss_partitions(),
          num_inner_iterations=1,
          adaptative=self._adaptive())
      # pylint: enable=protected-access

      with ops.control_dependencies([esu]):
        update_ops = [self._hashtable.insert(example_ids_hashed, esu)]
        # Update the weights before the proximal step.
        for w, i, u in zip(self._slots['unshrinked_sparse_features_weights'],
                           sparse_indices, sfw):
          update_ops.append(state_ops.scatter_add(w, i, u))
        for w, u in zip(self._slots['unshrinked_dense_features_weights'], dfw):
          update_ops.append(w.assign_add(u))

      if not global_step:
        return control_flow_ops.group(*update_ops)
      with ops.control_dependencies(update_ops):
        return state_ops.assign_add(global_step, 1, name=name).op

  def update_weights(self, train_op):
    """Updates the model weights.

    This function must be called on at least one worker after `minimize`.
    In distributed training this call can be omitted on non-chief workers to
    speed up training.

    Args:
      train_op: The operation returned by the `minimize` call.

    Returns:
      An Operation that updates the model weights.
    """
    with ops.control_dependencies([train_op]):
      update_ops = []
      # Copy over unshrinked weights to user provided variables.
      for name in ['sparse_features_weights', 'dense_features_weights']:
        for var, slot_var in zip(self._variables[name],
                                 self._slots['unshrinked_' + name]):
          update_ops.append(var.assign(slot_var))

    # Apply proximal step.
    with ops.control_dependencies(update_ops):
      update_ops = []
      for name in ['sparse_features_weights', 'dense_features_weights']:
        for var in self._variables[name]:
          with ops.device(var.device):
            # pylint: disable=protected-access
            update_ops.append(
                gen_sdca_ops.sdca_shrink_l1(
                    self._convert_n_to_tensor(
                        [var], as_ref=True),
                    l1=self._symmetric_l1_regularization(),
                    l2=self._symmetric_l2_regularization()))
      return control_flow_ops.group(*update_ops)

  def approximate_duality_gap(self):
    """Add operations to compute the approximate duality gap.

    Returns:
      An Operation that computes the approximate duality gap over all
      examples.
    """
    with name_scope('sdca/approximate_duality_gap'):
      _, values_list = self._hashtable.export_sharded()
      shard_sums = []
      for values in values_list:
        with ops.device(values.device):
          # For large tables to_double() below allocates a large temporary
          # tensor that is freed once the sum operation completes. To reduce
          # peak memory usage in cases where we have multiple large tables on a
          # single device, we serialize these operations.
          # Note that we need double precision to get accurate results.
          with ops.control_dependencies(shard_sums):
            shard_sums.append(
                math_ops.reduce_sum(math_ops.to_double(values), 0))
      summed_values = math_ops.add_n(shard_sums)

      primal_loss = summed_values[1]
      dual_loss = summed_values[2]
      example_weights = summed_values[3]
      # Note: we return NaN if there are no weights or all weights are 0, e.g.
      # if no examples have been processed
      return (primal_loss + dual_loss + self._l1_loss() +
              (2.0 * self._l2_loss(self._symmetric_l2_regularization()))
             ) / example_weights

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
    self._assertSpecified([
        'example_labels', 'example_weights', 'sparse_features', 'dense_features'
    ], examples)
    self._assertList(['sparse_features', 'dense_features'], examples)
    with name_scope('sdca/unregularized_loss'):
      predictions = math_ops.cast(
          self._linear_predictions(examples), dtypes.float64)
      labels = math_ops.cast(
          internal_convert_to_tensor(examples['example_labels']),
          dtypes.float64)
      weights = math_ops.cast(
          internal_convert_to_tensor(examples['example_weights']),
          dtypes.float64)

      if self._options['loss_type'] == 'logistic_loss':
        return math_ops.reduce_sum(math_ops.multiply(
            sigmoid_cross_entropy_with_logits(labels=labels,
                                              logits=predictions),
            weights)) / math_ops.reduce_sum(weights)

      if self._options['loss_type'] in ['hinge_loss', 'smooth_hinge_loss']:
        # hinge_loss = max{0, 1 - y_i w*x} where y_i \in {-1, 1}. So, we need to
        # first convert 0/1 labels into -1/1 labels.
        all_ones = array_ops.ones_like(predictions)
        adjusted_labels = math_ops.subtract(2 * labels, all_ones)
        # Tensor that contains (unweighted) error (hinge loss) per
        # example.
        error = nn_ops.relu(
            math_ops.subtract(all_ones,
                              math_ops.multiply(adjusted_labels, predictions)))
        weighted_error = math_ops.multiply(error, weights)
        return math_ops.reduce_sum(weighted_error) / math_ops.reduce_sum(
            weights)

      # squared loss
      err = math_ops.subtract(labels, predictions)

      weighted_squared_err = math_ops.multiply(math_ops.square(err), weights)
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
    self._assertSpecified([
        'example_labels', 'example_weights', 'sparse_features', 'dense_features'
    ], examples)
    self._assertList(['sparse_features', 'dense_features'], examples)
    with name_scope('sdca/regularized_loss'):
      weights = internal_convert_to_tensor(examples['example_weights'])
      return ((
          self._l1_loss() +
          # Note that here we are using the raw regularization
          # (as specified by the user) and *not*
          # self._symmetric_l2_regularization().
          self._l2_loss(self._options['symmetric_l2_regularization'])) /
              math_ops.reduce_sum(math_ops.cast(weights, dtypes.float64)) +
              self.unregularized_loss(examples))
