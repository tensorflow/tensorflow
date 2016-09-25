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

import tensorflow as tf
from tensorflow.contrib.tfprof.python.tools.tfprof import tfprof_log_pb2
from tensorflow.python.framework import ops

TRAINABLE_VARIABLES = '_trainable_variables'
REGISTERED_FLOP_STATS = 'flops'


def _get_logged_ops(graph):
  """Extract trainable model parameters and FLOPs for ops from a Graph.

  Args:
    graph: tf.Graph.
  Returns:
    logged_ops: dict mapping from op_name to OpLogEntry.
  """
  logged_ops = {}

  graph_def = graph.as_graph_def()
  for node in graph_def.node:
    try:
      stats = ops.get_stats_for_node_def(graph, node, REGISTERED_FLOP_STATS)
    except ValueError:
      # Catch Exception When shape is incomplete. Skip it.
      stats = None

    if not stats or not stats.value:
      continue
    if node.name not in logged_ops:
      entry = tfprof_log_pb2.OpLogEntry()
      entry.name = node.name
      entry.float_ops = stats.value
      logged_ops[entry.name] = entry

  for v in graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
    if v.op.name not in logged_ops:
      entry = tfprof_log_pb2.OpLogEntry()
      entry.name = v.op.name
      entry.types.append(TRAINABLE_VARIABLES)
      logged_ops[entry.name] = entry
    else:
      logged_ops[v.op.name].types.append(TRAINABLE_VARIABLES)
  return logged_ops


def _merge_default_with_oplog(graph, op_log=None):
  """Merge the tfprof default extra info with caller's op_log.

  Args:
    graph: tf.Graph.
    op_log: OpLog proto.
  Returns:
    tmp_op_log: Merged OpLog proto.
  """
  tmp_op_log = tfprof_log_pb2.OpLog()
  logged_ops = _get_logged_ops(graph)
  if not op_log:
    tmp_op_log.log_entries.extend(logged_ops.values())
  else:
    all_ops = dict()
    for entry in op_log.log_entries:
      all_ops[entry.name] = entry
    for op_name, entry in logged_ops.iteritems():
      if op_name in all_ops:
        all_ops[op_name].types.extend(entry.types)
        if entry.float_ops > 0 and all_ops[op_name].float_ops == 0:
          all_ops[op_name].float_ops = entry.float_ops
      else:
        all_ops[op_name] = entry
    tmp_op_log.log_entries.extend(all_ops.values())
  return tmp_op_log


def write_op_log(graph, log_dir, op_log=None):
  """Log provided 'op_log', and add additional model information below.

    The API also assigns ops in tf.trainable_variables() an op type called
    '_trainable_variables'.
    The API also logs 'flops' statistics for ops with op.RegisterStatistics()
    defined.

  Args:
    graph: tf.Graph.
    log_dir: directory to write the log file.
    op_log: OpLog proto.
  """
  op_log = _merge_default_with_oplog(graph, op_log)

  with tf.gfile.Open(os.path.join(log_dir, 'tfprof_log'), 'w') as log:
    log.write(op_log.SerializeToString())
