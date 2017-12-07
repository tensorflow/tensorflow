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
"""Utilities to replicate model_fn's over local GPUs.

This file contains util that allow to replicate `Estimator.model_fn` over
GPUs.  Replicated version of a `model_fn` is returned that can subsequently
be used with `Estimator`.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import six

from tensorflow.core.framework import node_def_pb2
from tensorflow.python.client import device_lib
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.estimator import util
from tensorflow.python.estimator.export import export_output as export_output_lib
from tensorflow.python.framework import device as framework_device
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging
from tensorflow.python.training import device_setter as device_setter_lib
from tensorflow.python.training import training_util


def replicate_model_fn(model_fn, optimizer_fn, devices=None):
  """Replicate `Estimator.model_fn` over GPUs within a single host.

  The given `model_fn` specifies a single forward pass of a model.  To replicate
  such a model over GPUs, each GPU gets its own instance of the forward pass
  (a.k.a. a tower).  The input features and labels get sharded into the chunks
  that correspond to the number of GPUs.  Each tower computes its own loss based
  on its input.  For each such loss, gradients are computed.  After that, the
  available losses are summed to form aggregated loss.  The available
  gradients are summed too.  Then, they update weights using the specified
  optimizer.

  If `devices` are `None`, then all available GPUs are going to be used for
  replication.  If no GPUs are available, then the model is going to be
  placed on the CPU.

  Two modes of local replication over available GPUs are supported:
    1)  If exactly 1 GPU is detected, then variables and operations are placed
        onto GPU.
    2)  If more than 1 GPU is detected, then variables are going to be placed on
        the CPU.  Replicas of operations are placed on each individual GPU.

  Here is an example of how one might use their `model_fn` to run over GPUs:
    ```python
       def optimizer_fn():
         return tf.train.GradientDescentOptimizer(learning_rate=0.001)
       ...
       def model_fn(...):  # See `model_fn` in `Estimator`.
         loss = ...
         if mode == tf.estimator.ModeKeys.TRAIN:
           #  See the section below on `EstimatorSpec.train_op`.
           return EstimatorSpec(mode=mode, loss=loss, train_op=tf.noop())

         #  No change for `ModeKeys.EVAL` or `ModeKeys.PREDICT`.
         return EstimatorSpec(...)
       ...
       classifier = tf.estimator.Estimator(
         model_fn=replicate_model_fn.replicate_model_fn(model_fn, optimizer_fn))
    ```

  On `EstimatorSpec.train_op`:
  `model_fn` returns `EstimatorSpec.train_op` for
  `tf.estimator.GraphKeys.TRAIN`. It is typically derived using an optimizer.
  `replicate_model_fn` ignores the returned `EstimatorSpec.train_op`, so there
  is no need to use an optimizer inside the user's `model_fn`.  The
  `EstimatorSpec.loss` subgraph is going to be executed, while
  `EstimatorSpec.train_op` isn't going to be executed. One could pass
  `train_op=tf.noop()` to `EstimatorSpec`.

  On sharding input features and labels:
  Input features and labels are split for consumption by each tower. They are
  split across the dimension 0.  Features and labels need to be batch major.

  On reduction algorithms:
  Certain algorithms were chosen for aggregating results of computations on
  multiple towers:
    - Losses from all towers are reduced using sum.
    - Gradients are reduced using sum for each trainable variable.
    - `eval_metrics_ops` are reduced per metric using `reduce_mean`.
    - `EstimatorSpec.predictions` and `EstimatorSpec.export_outputs` are
      reduced using concatenation.
    - For all other fields of `EstimatorSpec` the values of the first tower
      are taken.

  On distribution of variables:
  Variables are not duplicated between towers.  Instead, they are placed on a
  single device as defined above and shared across towers.

  Other current limitations:
    - `predictions` are not supported for `ModeKeys.EVAL`.  That is required for
      `tf.contrib.estimator.add_metrics`.

  Args:
    model_fn: `model_fn` as defined in `Estimator`.  See the section above about
      the train_op argument of `EstimatorSpec`.
    optimizer_fn: a function that returns an optimizer instance.  The function
      may accept one `params` argument.  This is the `params` argument as
      defined by `Estimator`.  See  the `Estimator` documentation for details.
    devices: Optional list of devices to replicate the model across.  This
      argument can be used to replice only on the subset of available GPUs.
      If `None`, then all available GPUs are going to be used for replication.
      If no GPUs are available, then the model is going to be placed on the CPU.

  Returns:
    A replicated version of the supplied `model_fn`. Returned function that
      conforms to the requirements of `Estimator`'s `model_fn` and can be used
      instead of the supplied `model_fn`.
  """
  return _replicate_model_fn_with_mode(
      model_fn,
      optimizer_fn,
      devices,
      # TODO(isaprykin): Query system configuration to choose modes other than
      # `SHARED_LOCAL_PARAMETER_SERVER`, even though it is often appropriate.
      mode=_VariableDistributionMode.SHARED_LOCAL_PARAMETER_SERVER)


class _VariableDistributionMode(object):
  """Modes for variable distribution used for forcing a particular one.

  Forcing a mode is meant for performance experimentation purposes rather than
  for general use cases.
  """

  SHARED_LOCAL_PARAMETER_SERVER = 1
  """Variables are placed on a single device and shared across all devices.

  Two ways to achieve this distribution over available GPUs are supported:
    1)  If exactly 1 GPU is detected, then variables and operations are placed
        onto GPU.
    2)  If more than 1 GPU is detected, then variables are going to be placed on
        the CPU.  Replicas of operations are placed on each individual GPU.
  """

  SHARED_ROUND_ROBIN = 2
  """Variables are placed on all devices in a round-robin fashion.

  Every subsequent variable is placed on the next device.  There is only one
  copy of each variable that is shared across all devices.
  """


def _replicate_model_fn_with_mode(
    model_fn,
    optimizer_fn,
    devices=None,
    mode=_VariableDistributionMode.SHARED_LOCAL_PARAMETER_SERVER):
  """A version of `replicate_model_fn` that allows to specify a `mode`."""
  if not devices:
    devices = _get_local_devices('GPU') or _get_local_devices('CPU')

  is_a_single_gpu_case = len(devices) == 1 and 'GPU' in devices[0]
  consolidation_device = '/{}:0'.format('GPU'
                                        if is_a_single_gpu_case else 'CPU')

  ps_devices = [consolidation_device]
  if mode == _VariableDistributionMode.SHARED_ROUND_ROBIN:
    ps_devices = devices

  tf_logging.info('Replicating the `model_fn` across {}.  Variables are going '
                  'to be placed on {}.  Consolidation device is going to be {}.'
                  .format(devices, ps_devices, consolidation_device))

  def replicated_model_fn(features, labels, mode, params=None, config=None):
    """Replicated version of `model_fn` to be used instead."""
    feature_shards, label_shards = _split_batch(
        features, labels, len(devices), device=consolidation_device)
    tower_specs = _get_loss_towers(
        model_fn=model_fn,
        mode=mode,
        features=feature_shards,
        labels=label_shards,
        params=params,
        config=config,
        devices=devices,
        local_ps_devices=ps_devices)

    if mode == model_fn_lib.ModeKeys.TRAIN:
      train_op = _minimize_towers(tower_specs,
                                  _call_optimizer_fn(optimizer_fn, params))
      return _train_spec(
          tower_specs, train_op, aggregation_device=consolidation_device)
    elif mode == model_fn_lib.ModeKeys.EVAL:
      return _eval_spec(tower_specs, aggregation_device=consolidation_device)
    elif mode == model_fn_lib.ModeKeys.PREDICT:
      return _predict_spec(tower_specs, aggregation_device=consolidation_device)

  return replicated_model_fn


def _get_local_devices(device_type):
  local_device_protos = device_lib.list_local_devices()
  return [
      device.name
      for device in local_device_protos
      if device.device_type == device_type
  ]


def _split_batch(features, labels, number_of_shards, device):
  """Split input features and labes into batches."""

  def split_dictionary(dictionary):
    """Split a dictionary into shards."""
    shards = [{} for _ in range(number_of_shards)]
    for name, tensor in six.iteritems(dictionary):
      if isinstance(tensor, sparse_tensor.SparseTensor):
        for i, shard in enumerate(
            sparse_ops.sparse_split(
                sp_input=tensor, num_split=number_of_shards, axis=0)):
          shards[i][name] = shard
      else:
        for i, shard in enumerate(array_ops.split(tensor, number_of_shards)):
          shards[i][name] = shard
    return shards

  with ops_lib.name_scope('split_inputs'):
    with ops_lib.device(device):
      if isinstance(features, dict):
        feature_shards = split_dictionary(features)
      else:
        feature_shards = array_ops.split(features, number_of_shards)

      if labels is None:
        label_shards = None
      elif isinstance(labels, dict):
        label_shards = split_dictionary(labels)
      else:
        label_shards = array_ops.split(labels, number_of_shards)
  return feature_shards, label_shards


_DEFAULT_NAME_SCOPE_PATTERN = 'tower_{}'


def _get_loss_towers(model_fn,
                     mode,
                     features,
                     labels,
                     params,
                     config,
                     devices,
                     local_ps_devices,
                     name_scope_pattern=_DEFAULT_NAME_SCOPE_PATTERN):
  """Replicate the loss computation across devices."""
  tower_specs = []

  model_fn_args = util.fn_args(model_fn)
  optional_params = {}
  if 'params' in model_fn_args:
    optional_params['params'] = copy.deepcopy(params)
  if 'config' in model_fn_args:
    optional_params['config'] = copy.deepcopy(config)

  # pylint: disable=protected-access
  round_robin_strategy = device_setter_lib._RoundRobinStrategy(
      num_tasks=len(local_ps_devices))
  # pylint: enable=protected-access

  for i, device in enumerate(devices):
    is_the_first_tower = (i == 0)

    device_setter = _local_device_setter(
        worker_device=device,
        ps_devices=local_ps_devices,
        ps_strategy=round_robin_strategy)

    # We would like to preserve the names of the variables and ops that the user
    # might be relying on. Names without a prefix are going to resolve to
    # variables and ops of the first tower.
    name_scope = name_scope_pattern
    if is_the_first_tower:
      name_scope = ''

    with variable_scope.variable_scope('', reuse=not is_the_first_tower):
      with ops_lib.name_scope(name_scope.format(i)):
        with ops_lib.device(device_setter):
          labels_shard = None
          if labels:
            labels_shard = labels[i]

          tower_specs.append(
              model_fn(
                  mode=mode,
                  features=features[i],
                  labels=labels_shard,
                  **optional_params))
  return tower_specs


def _local_device_setter(worker_device, ps_devices, ps_strategy):
  """A device setter that puts distributes Var/Ops to PS/workers."""
  ps_ops = ['Variable', 'VariableV2', 'VarHandleOp']

  def local_device_chooser(op):
    current_device = framework_device.DeviceSpec.from_string(op.device or '')

    node_def = op if isinstance(op, node_def_pb2.NodeDef) else op.node_def
    if node_def.op in ps_ops:
      ps_device_spec = framework_device.DeviceSpec.from_string(
          '{}'.format(ps_devices[ps_strategy(op)]))

      ps_device_spec.merge_from(current_device)
      return ps_device_spec.to_string()
    else:
      worker_device_spec = framework_device.DeviceSpec.from_string(
          worker_device or '')
      worker_device_spec.merge_from(current_device)
      return worker_device_spec.to_string()

  return local_device_chooser


def _minimize_towers(tower_specs, optimizer):
  """Aggregate and apply gradients for computed losses."""
  grad_lists = {}
  for tower_spec in tower_specs:
    with ops_lib.device(tower_spec.loss.device):
      for grad, var in optimizer.compute_gradients(tower_spec.loss):
        if grad is not None:
          grad_lists.setdefault(var, []).append(grad)

  aggregated_grads = []
  with ops_lib.name_scope('gradient_aggregating'):
    for var, grads in six.iteritems(grad_lists):
      grad = _compute_sum_on_device(grads, var.device)
      aggregated_grads.append((grad, var))

  train_op = optimizer.apply_gradients(
      aggregated_grads, global_step=training_util.get_global_step())

  return train_op


def _call_optimizer_fn(optimizer_fn, params):
  arguments = {}
  optimizer_fn_arguments = util.fn_args(optimizer_fn)
  if 'params' in optimizer_fn_arguments:
    arguments['params'] = params
  return optimizer_fn(**arguments)


def _compute_sum_on_device(values, device, name=None):
  with ops_lib.device(device):
    if isinstance(values[0], ops_lib.IndexedSlices):
      if name:
        raise ValueError('The name {} is not expected to be given to '
                         'IndexedSlices {}'.format(name, values))

      values_concat = array_ops.concat([v.values for v in values], axis=0)
      indices_concat = array_ops.concat([v.indices for v in values], axis=0)
      return ops_lib.IndexedSlices(values_concat, indices_concat,
                                   values[0].dense_shape)
    else:
      return math_ops.add_n(values, name=name)


def _train_spec(tower_specs,
                train_op,
                aggregation_device,
                aggregated_loss_name='loss'):
  """Populate replicated EstimatorSpec for `GraphKeys.TRAIN`."""
  estimator_spec = tower_specs[0]._asdict()
  estimator_spec['mode'] = model_fn_lib.ModeKeys.TRAIN
  estimator_spec['train_op'] = train_op
  estimator_spec['loss'] = _compute_sum_on_device(
      [spec.loss for spec in tower_specs], aggregation_device,
      aggregated_loss_name)
  return model_fn_lib.EstimatorSpec(**estimator_spec)


def _eval_spec(tower_specs, aggregation_device, aggregated_loss_name='loss'):
  """Populate replicated EstimatorSpec for `GraphKeys.EVAL`."""
  estimator_spec = tower_specs[0]._asdict()
  estimator_spec['mode'] = model_fn_lib.ModeKeys.EVAL
  estimator_spec['loss'] = _compute_sum_on_device(
      [spec.loss for spec in tower_specs], aggregation_device,
      aggregated_loss_name)

  update_ops = []
  for tower_spec in tower_specs:
    for name, (_, update_op) in six.iteritems(tower_spec.eval_metric_ops):
      update_ops.append(update_op)

  with ops_lib.control_dependencies(update_ops):
    reduced_update_op = _reduce_metric_variables(len(tower_specs))

  eval_metric_ops = {}
  for name, (metric_tensor, _) in six.iteritems(tower_specs[0].eval_metric_ops):
    eval_metric_ops[name] = (metric_tensor, reduced_update_op)
  estimator_spec['eval_metric_ops'] = eval_metric_ops
  return model_fn_lib.EstimatorSpec(**estimator_spec)


def _reduce_metric_variables(number_of_towers):
  """Aggregate local variables used in metrics into the first tower."""
  if number_of_towers == 1:
    return control_flow_ops.no_op()

  metric_variables = ops_lib.get_collection(ops_lib.GraphKeys.METRIC_VARIABLES)
  variables_per_tower = len(metric_variables) // number_of_towers

  if len(metric_variables) % number_of_towers != 0:
    raise ValueError(
        'Different `EstimatorSpec.eval_metric_ops` across `model_fn()` calls.'
        ' Expected {} local variables, but got {} instead.'.format(
            variables_per_tower * number_of_towers, len(metric_variables)))

  # `metric_variables` has the size of `variables_per_tower` x
  #  number_of_towers.  Each tower is produced by calling the same model_fn.
  #  First `variables_per_tower` correspond to the first tower.  Each such
  #  variable has an replica at the `(variables_per_tower * i)` position, where
  #  `i` is `[1.. number_of_towers]`.  We are going to add values from replicas
  #  to each variable of the first tower.  We then zero out replica values, so
  #  that `_reduce_metric_variables` operation is idempotent.  If a metric
  #  is then computed based on local variables from the first tower, then the
  #  resulting metric is an estimate for all `number_of_towers` towers.
  ops = []
  for i in range(0, variables_per_tower):
    next_replica_id = i + variables_per_tower
    replicas = [
        metric_variables[replica_id]
        for replica_id in range(next_replica_id, len(metric_variables),
                                variables_per_tower)
    ]  #  `replicas` doesn't contain the first-tower variable.

    reduce_op = state_ops.assign_add(metric_variables[i],
                                     math_ops.add_n(replicas))

    with ops_lib.control_dependencies([reduce_op]):
      for replica in replicas:
        zeros_for_replica = array_ops.zeros(
            array_ops.shape(replica), dtype=replica.dtype)
        zero_out_replica_op = state_ops.assign(replica, zeros_for_replica)
        ops.append(zero_out_replica_op)

  return control_flow_ops.group(*ops)


def _predict_spec(tower_specs, aggregation_device):
  """Populate replicated EstimatorSpec for `GraphKeys.PREDICT`."""
  estimator_spec = tower_specs[0]._asdict()
  estimator_spec['mode'] = model_fn_lib.ModeKeys.PREDICT

  with ops_lib.device(aggregation_device):
    estimator_spec['predictions'] = _concat_tensor_dicts(
        *[tower_spec.predictions for tower_spec in tower_specs])

    export_outputs_dict = _dict_concat(
        *[tower_spec.export_outputs for tower_spec in tower_specs])

    export_outputs = {}
    for name, export_output_list in six.iteritems(export_outputs_dict):
      if isinstance(export_output_list[0], export_output_lib.PredictOutput):
        export_outputs[name] = export_output_lib.PredictOutput(
            outputs=_concat_tensor_dicts(*[
                export_output.outputs for export_output in export_output_list
            ]))
      elif isinstance(export_output_list[0],
                      export_output_lib.RegressionOutput):
        export_outputs[name] = export_output_lib.RegressionOutput(
            value=array_ops.concat(
                [export_output.value for export_output in export_output_list],
                axis=0))
      elif isinstance(export_output_list[0],
                      export_output_lib.ClassificationOutput):
        scores = None
        if export_output_list[0].scores is not None:
          scores = array_ops.concat(
              [export_output.scores for export_output in export_output_list],
              axis=0)

        classes = None
        if export_output_list[0].classes is not None:
          classes = array_ops.stack(
              [export_output.classes for export_output in export_output_list],
              axis=0)

        export_outputs[name] = export_output_lib.ClassificationOutput(
            scores=scores, classes=classes)

  estimator_spec['export_outputs'] = export_outputs
  return model_fn_lib.EstimatorSpec(**estimator_spec)


def _concat_tensor_dicts(*tensor_dicts):
  return {
      name: array_ops.concat(tensors, axis=0, name=name)
      for name, tensors in six.iteritems(_dict_concat(*tensor_dicts))
  }


def _dict_concat(*dicts):
  list_dict = {}
  for d in dicts:
    if d is None:
      continue

    for k, v in six.iteritems(d):
      list_dict.setdefault(k, []).append(v)
  return list_dict
