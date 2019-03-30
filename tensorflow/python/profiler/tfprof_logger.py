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
"""Logging tensorflow::tfprof::OpLogProto.

OpLogProto is used to add extra model information for offline analysis.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import six
from tensorflow.core.profiler import tfprof_log_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.profiler.internal import flops_registry  # pylint: disable=unused-import
from tensorflow.python.util.tf_export import tf_export

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


def _str_id(s, str_to_id):
  """Maps string to id."""
  num = str_to_id.get(s, None)
  if num is None:
    num = len(str_to_id)
    str_to_id[s] = num
  return num


def _get_logged_ops(graph, run_meta=None, add_trace=True,
                    add_trainable_var=True):
  """Extract trainable model parameters and FLOPs for ops from a Graph.

  Args:
    graph: tf.Graph.
    run_meta: RunMetadata proto used to complete shape information.
    add_trace: Whether to add op trace information.
    add_trainable_var: Whether to assign tf.trainable_variables() op type
      '_trainable_variables'.
  Returns:
    logged_ops: dict mapping from op_name to OpLogEntry.
    string_to_id: dict mapping from string to id.
  """
  if run_meta:
    graph = _fill_missing_graph_shape(graph, run_meta)

  op_missing_shape = 0
  logged_ops = {}
  string_to_id = {}
  string_to_id['none'] = len(string_to_id)
  # TODO(xpan): Work with Profiler more efficiently.
  for op in graph.get_operations():
    try:
      stats = ops.get_stats_for_node_def(
          graph, op.node_def, REGISTERED_FLOP_STATS)
    except ValueError:
      # Catch Exception When shape is incomplete. Skip it.
      op_missing_shape += 1
      stats = None

    entry = tfprof_log_pb2.OpLogEntry()
    entry.name = op.name
    add_entry = False
    if stats and stats.value:
      entry.float_ops = int(stats.value)
      add_entry = True

    if add_trace:
      for tb in op.traceback_with_start_lines:
        trace = entry.code_def.traces.add()
        trace.file_id = _str_id(tb[0], string_to_id) if tb[0] else 0
        trace.lineno = tb[1] if tb[1] else -1
        trace.function_id = _str_id(tb[2], string_to_id) if tb[2] else 0
        trace.line_id = _str_id(tb[3], string_to_id) if tb[3] else 0
        trace.func_start_line = tb[4] if tb[4] else -1
      add_entry = True

    if add_entry:
      logged_ops[entry.name] = entry

  if add_trainable_var:
    for v in graph.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES):
      if v.op.name not in logged_ops:
        entry = tfprof_log_pb2.OpLogEntry()
        entry.name = v.op.name
        entry.types.append(TRAINABLE_VARIABLES)
        logged_ops[entry.name] = entry
      else:
        logged_ops[v.op.name].types.append(TRAINABLE_VARIABLES)

  if op_missing_shape > 0 and not run_meta:
    sys.stderr.write('%d ops no flops stats due to incomplete shapes.\n' %
                     op_missing_shape)
  return logged_ops, string_to_id


def merge_default_with_oplog(graph, op_log=None, run_meta=None,
                             add_trace=True, add_trainable_var=True):
  """Merge the tfprof default extra info with caller's op_log.

  Args:
    graph: tf.Graph. If None and eager execution is not enabled, use
        default graph.
    op_log: OpLogProto proto.
    run_meta: RunMetadata proto used to complete shape information.
    add_trace: Whether to add op trace information.
    add_trainable_var: Whether to assign tf.trainable_variables() op type
      '_trainable_variables'.
  Returns:
    tmp_op_log: Merged OpLogProto proto.
  """
  if not graph and not context.executing_eagerly():
    graph = ops.get_default_graph()

  tmp_op_log = tfprof_log_pb2.OpLogProto()
  if not graph:
    return tmp_op_log

  logged_ops, string_to_id = _get_logged_ops(
      graph, run_meta, add_trace=add_trace, add_trainable_var=add_trainable_var)

  if not op_log:
    tmp_op_log.log_entries.extend(logged_ops.values())
  else:
    all_ops = {}
    for entry in op_log.log_entries:
      all_ops[entry.name] = entry
    for op_name, entry in six.iteritems(logged_ops):
      if op_name in all_ops:
        all_ops[op_name].types.extend(entry.types)
        if entry.float_ops > 0 and all_ops[op_name].float_ops == 0:
          all_ops[op_name].float_ops = entry.float_ops
        if entry.code_def.traces and not all_ops[op_name].code_def.traces:
          all_ops[op_name].code_def.MergeFrom(entry.code_def)
      else:
        all_ops[op_name] = entry
    tmp_op_log.log_entries.extend(all_ops.values())

  for s, i in six.iteritems(string_to_id):
    tmp_op_log.id_to_string[i] = s
  return tmp_op_log


@tf_export(v1=['profiler.write_op_log'])
def write_op_log(graph, log_dir, op_log=None, run_meta=None, add_trace=True):
  """Log provided 'op_log', and add additional model information below.

    The API also assigns ops in tf.trainable_variables() an op type called
    '_trainable_variables'.
    The API also logs 'flops' statistics for ops with op.RegisterStatistics()
    defined. flops calculation depends on Tensor shapes defined in 'graph',
    which might not be complete. 'run_meta', if provided, completes the shape
    information with best effort.

  Args:
    graph: tf.Graph. If None and eager execution is not enabled, use
        default graph.
    log_dir: directory to write the log file.
    op_log: (Optional) OpLogProto proto to be written. If not provided, an new
        one is created.
    run_meta: (Optional) RunMetadata proto that helps flops computation using
        run time shape information.
    add_trace: Whether to add python code trace information.
        Used to support "code" view.
  """
  if not graph and not context.executing_eagerly():
    graph = ops.get_default_graph()
  op_log = merge_default_with_oplog(graph, op_log, run_meta, add_trace)

  with gfile.Open(os.path.join(log_dir, 'tfprof_log'), 'w') as log:
    log.write(op_log.SerializeToString())
