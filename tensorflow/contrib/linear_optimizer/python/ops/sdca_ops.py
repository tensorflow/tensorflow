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
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework.ops import internal_convert_to_tensor
from tensorflow.python.framework.ops import name_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gen_sdca_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables as var_ops
from tensorflow.python.ops.nn import log_poisson_loss
from tensorflow.python.ops.nn import sigmoid_cross_entropy_with_logits
from tensorflow.python.summary import summary

__all__ = ['SdcaModel']


# TODO(sibyl-Aix6ihai): add name_scope to appropriate methods.
class SdcaModel(object):
  """Stochastic dual coordinate ascent solver for linear models.

    Loss functions supported:

     * Binary logistic loss
     * Squared loss
     * Hinge loss
     * Smooth hinge loss
     * Poisson log loss

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
                        'smooth_hinge_loss', 'poisson_loss')
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
    """Make unshrinked internal variables (slots)."""
    # Unshrinked variables have the updates before applying L1 regularization.
    # Each unshrinked slot variable is either a `Variable` or list of
    # `Variable`, depending on the value of its corresponding primary variable.
    # We avoid using `PartitionedVariable` for the unshrinked slots since we do
    # not need any of the extra information.
    self._slots = collections.defaultdict(list)
    for name in ['sparse_features_weights', 'dense_features_weights']:
      for var in self._variables[name]:
        # Our primary variable may be either a PartitionedVariable, or a list
        # of Variables (each representing a partition).
        if (isinstance(var, var_ops.PartitionedVariable) or
            isinstance(var, list)):
          var_list = []
          # pylint: disable=protected-access
          for v in var:
            with ops.colocate_with(v):
              # TODO(andreasst): remove SDCAOptimizer suffix once bug 30843109
              # is fixed.
              slot_var = var_ops.Variable(
                  initial_value=array_ops.zeros_like(v.initialized_value(),
                                                     dtypes.float32),
                  name=v.op.name + '_unshrinked/SDCAOptimizer')
              var_list.append(slot_var)
          self._slots['unshrinked_' + name].append(var_list)
          # pylint: enable=protected-access
        else:
          with ops.device(var.device):
            # TODO(andreasst): remove SDCAOptimizer suffix once bug 30843109 is
            # fixed.
            self._slots['unshrinked_' + name].append(
                var_ops.Variable(
                    array_ops.zeros_like(var.initialized_value(),
                                         dtypes.float32),
                    name=var.op.name + '_unshrinked/SDCAOptimizer'))

  def _assertSpecified(self, items, check_in):
    for x in items:
      if check_in[x] is None:
        raise ValueError(check_in[x] + ' must be specified.')

  def _assertList(self, items, check_in):
    for x in items:
      if not isinstance(check_in[x], list):
        raise ValueError(x + ' must be a list.')

  def _var_to_list(self, var):
    """Wraps var in a list if it is not a list or PartitionedVariable."""
    if not (isinstance(var, list) or
            isinstance(var, var_ops.PartitionedVariable)):
      var = [var]
    return var

  def _l1_loss(self):
    """Computes the (un-normalized) l1 loss of the model."""
    with name_scope('sdca/l1_loss'):
      sums = []
      for name in ['sparse_features_weights', 'dense_features_weights']:
        for var in self._variables[name]:
          for v in self._var_to_list(var):
            weights = internal_convert_to_tensor(v)
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
        for var in self._variables[name]:
          for v in self._var_to_list(var):
            weights = internal_convert_to_tensor(v)
            with ops.device(weights.device):
              sums.append(math_ops.reduce_sum(math_ops.square(math_ops.cast(
                  weights, dtypes.float64))))
      # SDCA L2 regularization cost is: l2 * sum(weights^2) / 2
      return l2 * math_ops.add_n(sums) / 2.0

  def _convert_n_to_tensor(self, input_list, as_ref=False):
    """Converts input list to a set of tensors."""
    # input_list can be a list of Variables (that are implicitly partitioned),
    # in which case the underlying logic in internal_convert_to_tensor will not
    # concatenate the partitions together.  This method takes care of the
    # concatenating (we only allow partitioning on the first axis).
    output_list = []
    for x in input_list:
      tensor_to_convert = x
      if isinstance(x, list) or isinstance(x, var_ops.PartitionedVariable):
        # We only allow for partitioning on the first axis.
        tensor_to_convert = array_ops.concat(x, axis=0)
      output_list.append(internal_convert_to_tensor(
          tensor_to_convert, as_ref=as_ref))
    return output_list

  def _get_first_dimension_size_statically(self, w, num_partitions):
    """Compute the static size of the first dimension for a sharded variable."""
    dim_0_size = w[0].get_shape()[0]
    for p in range(1, num_partitions):
      dim_0_size += w[p].get_shape()[0]
    return dim_0_size

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
    If poisson_loss is being used, predictions are exponentiated.
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
    elif self._options['loss_type'] == 'poisson_loss':
      # Exponeniate the prediction for poisson loss predictions.
      with name_scope('sdca/poisson_prediction'):
        result = math_ops.exp(result)
    return result

  def _get_partitioned_update_ops(self,
                                  v_num,
                                  num_partitions_by_var,
                                  p_assignments_by_var,
                                  gather_ids_by_var,
                                  weights,
                                  full_update,
                                  p_assignments,
                                  num_partitions):
    """Get updates for partitioned variables."""
    num_partitions = num_partitions_by_var[v_num]
    p_assignments = p_assignments_by_var[v_num]
    gather_ids = gather_ids_by_var[v_num]
    updates = data_flow_ops.dynamic_partition(
        full_update, p_assignments, num_partitions)
    update_ops = []
    for p in range(num_partitions):
      with ops.colocate_with(weights[p]):
        result = state_ops.scatter_add(weights[p], gather_ids[p], updates[p])
      update_ops.append(result)
    return update_ops

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

      sparse_weights = []
      sparse_indices = []
      # If we have partitioned variables, keep a few lists of Tensors around
      # that we need for the assign_add after the op call to
      # gen_sdca_ops.sdca_optimizer().
      num_partitions_by_var = []
      p_assignments_by_var = []
      gather_ids_by_var = []
      for w, i in zip(self._slots['unshrinked_sparse_features_weights'],
                      sparse_feature_indices):
        # Append the sparse_indices (in full-variable space).
        sparse_idx = math_ops.cast(
            array_ops.unique(math_ops.cast(i, dtypes.int32))[0],
            dtypes.int64)
        sparse_indices.append(sparse_idx)
        if isinstance(w, list) or isinstance(w, var_ops.PartitionedVariable):
          num_partitions = len(w)
          flat_ids = array_ops.reshape(sparse_idx, [-1])
          # We use div partitioning, which is easiest to support downstream.
          # Compute num_total_ids as the sum of dim-0 of w, then assign
          # to partitions based on a constant number of ids per partition.
          # Optimize if we already know the full shape statically.
          dim_0_size = self._get_first_dimension_size_statically(
              w, num_partitions)

          if dim_0_size.value:
            num_total_ids = constant_op.constant(dim_0_size.value,
                                                 flat_ids.dtype)
          else:
            dim_0_sizes = []
            for p in range(num_partitions):
              if w[p].get_shape()[0].value is not None:
                dim_0_sizes.append(w[p].get_shape()[0].value)
              else:
                with ops.colocate_with(w[p]):
                  dim_0_sizes.append(array_ops.shape(w[p])[0])
            num_total_ids = math_ops.reduce_sum(
                math_ops.cast(array_ops.stack(dim_0_sizes), flat_ids.dtype))
          ids_per_partition = num_total_ids // num_partitions
          extras = num_total_ids % num_partitions

          p_assignments = math_ops.maximum(
              flat_ids // (ids_per_partition + 1),
              (flat_ids - extras) // ids_per_partition)

          # Emulate a conditional using a boolean indicator tensor
          new_ids = array_ops.where(p_assignments < extras,
                                    flat_ids % (ids_per_partition + 1),
                                    (flat_ids - extras) % ids_per_partition)

          # Cast partition assignments to int32 for use in dynamic_partition.
          # There really should not be more than 2^32 partitions.
          p_assignments = math_ops.cast(p_assignments, dtypes.int32)
          # Partition list of ids based on assignments into num_partitions
          # separate lists.
          gather_ids = data_flow_ops.dynamic_partition(new_ids,
                                                       p_assignments,
                                                       num_partitions)
          # Append these to the lists for use in the later update.
          num_partitions_by_var.append(num_partitions)
          p_assignments_by_var.append(p_assignments)
          gather_ids_by_var.append(gather_ids)

          # Gather the weights from each partition.
          partition_gathered_weights = []
          for p in range(num_partitions):
            with ops.colocate_with(w[p]):
              partition_gathered_weights.append(
                  array_ops.gather(w[p], gather_ids[p]))

          # Stitch the weights back together in the same order they were before
          # we dynamic_partitioned them.
          condition_indices = data_flow_ops.dynamic_partition(
              math_ops.range(array_ops.shape(new_ids)[0]),
              p_assignments, num_partitions)
          batch_gathered_weights = data_flow_ops.dynamic_stitch(
              condition_indices, partition_gathered_weights)
        else:
          w_as_tensor = internal_convert_to_tensor(w)
          with ops.device(w_as_tensor.device):
            batch_gathered_weights = array_ops.gather(
                w_as_tensor, sparse_idx)
        sparse_weights.append(batch_gathered_weights)

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
        for v_num, (w, i, u) in enumerate(
            zip(self._slots['unshrinked_sparse_features_weights'],
                sparse_indices, sfw)):
          if (isinstance(w, var_ops.PartitionedVariable) or
              isinstance(w, list)):
            update_ops += self._get_partitioned_update_ops(
                v_num, num_partitions_by_var, p_assignments_by_var,
                gather_ids_by_var, w, u, p_assignments, num_partitions)
          else:
            update_ops.append(state_ops.scatter_add(w, i, u))
        for w, u in zip(self._slots['unshrinked_dense_features_weights'], dfw):
          if (isinstance(w, var_ops.PartitionedVariable) or
              isinstance(w, list)):
            split_updates = array_ops.split(
                u, num_or_size_splits=[v.shape.as_list()[0] for v in w])
            for v, split_update in zip(w, split_updates):
              update_ops.append(state_ops.assign_add(v, split_update))
          else:
            update_ops.append(state_ops.assign_add(w, u))
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
          for v, sv in zip(self._var_to_list(var), self._var_to_list(slot_var)):
            update_ops.append(v.assign(sv))

    # Apply proximal step.
    with ops.control_dependencies(update_ops):
      update_ops = []
      for name in ['sparse_features_weights', 'dense_features_weights']:
        for var in self._variables[name]:
          for v in self._var_to_list(var):
            with ops.device(v.device):
              # pylint: disable=protected-access
              update_ops.append(
                  gen_sdca_ops.sdca_shrink_l1(
                      self._convert_n_to_tensor([v], as_ref=True),
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

      if self._options['loss_type'] == 'poisson_loss':
        return math_ops.reduce_sum(math_ops.multiply(
            log_poisson_loss(targets=labels, log_input=predictions),
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
