# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Logging tensorflow::tfprof::OpLog.

OpLog is used to add extra model information for offline analysis by tfprof.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import six
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.tools.tfprof import tfprof_log_pb2

TRAINABLE_VARIABLES = '_trainable_variables'
REGISTERED_FLOP_STATS = 'flops'


def _fill_missing_graph_shape(graph, run_meta):
  """Fill Tensor shapes in 'graph' with run time shape from 'run_meta'."""
  for dev_stat in run_meta.step_stats.dev_stats:
    for node_stat in dev_stat.node_stats:
      if not node_stat.output:
        continue
      try:
        op = graph.get_operation_by_name(node_stat.node_name)
      except KeyError as e:
        # Graph doesn't contains the node_stat, usually RecvTensor.
        continue
      if len(node_stat.output) != len(op.outputs):
        # For example, conditional op has only 1 output at run time.
        continue
      for (i, node_stat_out) in enumerate(node_stat.output):
        if op.outputs[i].get_shape().is_fully_defined():
          continue
        node_stat_dims = node_stat_out.tensor_description.shape.dim
        node_stat_shape = tensor_shape.TensorShape(
            [d.size for d in node_stat_dims])
        try:
          op.outputs[i].set_shape(op.outputs[i].get_shape().merge_with(
              node_stat_shape))
        except ValueError as e:
          sys.stderr.write('Node %s incompatible shapes: %s.\n' %
                           (node_stat.node_name, e))
  return graph


def _get_logged_ops(graph, run_meta=None):
  """Extract trainable model parameters and FLOPs for ops from a Graph.

  Args:
    graph: tf.Graph.
    run_meta: RunMetadata proto used to complete shape information.
  Returns:
    logged_ops: dict mapping from op_name to OpLogEntry.
  """
  if run_meta:
    graph = _fill_missing_graph_shape(graph, run_meta)

  op_missing_shape = 0
  logged_ops = {}
  graph_def = graph.as_graph_def()
  for node in graph_def.node:
    try:
      stats = ops.get_stats_for_node_def(graph, node, REGISTERED_FLOP_STATS)
    except ValueError:
      # Catch Exception When shape is incomplete. Skip it.
      op_missing_shape += 1
      stats = None

    if not stats or not stats.value:
      continue
    if node.name not in logged_ops:
      entry = tfprof_log_pb2.OpLogEntry()
      entry.name = node.name
      entry.float_ops = int(stats.value)
      logged_ops[entry.name] = entry

  for v in graph.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES):
    if v.op.name not in logged_ops:
      entry = tfprof_log_pb2.OpLogEntry()
      entry.name = v.op.name
      entry.types.append(TRAINABLE_VARIABLES)
      logged_ops[entry.name] = entry
    else:
      logged_ops[v.op.name].types.append(TRAINABLE_VARIABLES)
  if op_missing_shape > 0 and not run_meta:
    sys.stderr.write('%d ops no flops stats due to incomplete shapes. '
                     'Consider passing run_meta to use run_time shapes.\n' %
                     op_missing_shape)
  return logged_ops


def _merge_default_with_oplog(graph, op_log=None, run_meta=None):
  """Merge the tfprof default extra info with caller's op_log.

  Args:
    graph: tf.Graph.
    op_log: OpLog proto.
    run_meta: RunMetadata proto used to complete shape information.
  Returns:
    tmp_op_log: Merged OpLog proto.
  """
  tmp_op_log = tfprof_log_pb2.OpLog()
  logged_ops = _get_logged_ops(graph, run_meta)
  if not op_log:
    tmp_op_log.log_entries.extend(logged_ops.values())
  else:
    all_ops = dict()
    for entry in op_log.log_entries:
      all_ops[entry.name] = entry
    for op_name, entry in six.iteritems(logged_ops):
      if op_name in all_ops:
        all_ops[op_name].types.extend(entry.types)
        if entry.float_ops > 0 and all_ops[op_name].float_ops == 0:
          all_ops[op_name].float_ops = entry.float_ops
      else:
        all_ops[op_name] = entry
    tmp_op_log.log_entries.extend(all_ops.values())
  return tmp_op_log


def write_op_log(graph, log_dir, op_log=None, run_meta=None):
  """Log provided 'op_log', and add additional model information below.

    The API also assigns ops in tf.trainable_variables() an op type called
    '_trainable_variables'.
    The API also logs 'flops' statistics for ops with op.RegisterStatistics()
    defined. flops calculation depends on Tensor shapes defined in 'graph',
    which might not be complete, 'run_meta', if provided, completes the shape
    information with best effort.

  Args:
    graph: tf.Graph.
    log_dir: directory to write the log file.
    op_log: (Optional) OpLog proto to be written. If not provided, an new
        one is created.
    run_meta: (Optional) RunMetadata proto that helps flops computation using
        run time shape information.
  """
  op_log = _merge_default_with_oplog(graph, op_log, run_meta)

  with gfile.Open(os.path.join(log_dir, 'tfprof_log'), 'w') as log:
    log.write(op_log.SerializeToString())
