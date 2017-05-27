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
"""Model Analyzer.

Analyze model, including shape, params, time, memory, structure, etc.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.tfprof.python.tools.tfprof import pywrap_tensorflow_print_model_analysis_lib as print_mdl
from tensorflow.contrib.tfprof.python.tools.tfprof import tfprof_logger
from tensorflow.python.framework import errors
from tensorflow.tools.tfprof import tfprof_options_pb2
from tensorflow.tools.tfprof import tfprof_output_pb2

# pylint: disable=bad-whitespace
# pylint: disable=bad-continuation
# 2 example tfprof_options for print_model_analysis API.
#
# Show the parameter statistics of trainable variables.
TRAINABLE_VARS_PARAMS_STAT_OPTIONS = {
    'max_depth': 10000,
    'min_bytes': 0,
    'min_micros': 0,
    'min_params': 0,
    'min_float_ops': 0,
    'order_by': 'name',
    'account_type_regexes': [tfprof_logger.TRAINABLE_VARIABLES],
    'start_name_regexes': ['.*'],
    'trim_name_regexes': [],
    'show_name_regexes': ['.*'],
    'hide_name_regexes': [],
    'account_displayed_op_only': True,
    'select': ['params'],
    'output': 'stdout',
    'dump_to_file': ''
}

# Show the number float operations.
FLOAT_OPS_OPTIONS = {
    'max_depth': 10000,
    'min_bytes': 0,
    'min_micros': 0,
    'min_params': 0,
    'min_float_ops': 1,
    'order_by': 'float_ops',
    'account_type_regexes': ['.*'],
    'start_name_regexes': ['.*'],
    'trim_name_regexes': [],
    'show_name_regexes': ['.*'],
    'hide_name_regexes': [],
    'account_displayed_op_only': True,
    'select': ['float_ops'],
    'output': 'stdout',
    'dump_to_file': ''
}

# Show number of parameters on parameter server 0.
# It is recommended to provide`run_meta` argument
# to have complete device placement info.
PRINT_PARAMS_ON_DEVICE = {
    'max_depth': 1,
    'min_bytes': 0,
    'min_micros': 0,
    'min_params': 0,
    'min_float_ops': 0,
    'order_by': 'name',
    'account_type_regexes': ['.*ps.*task:0.*'],
    'start_name_regexes': ['.*'],
    'trim_name_regexes': [],
    'show_name_regexes': ['.*'],
    'hide_name_regexes': [],
    'account_displayed_op_only': False,
    'select': ['device', 'params'],
    'output': 'stdout',
    'dump_to_file': ''
}

# Show the timing stats and memory demands.
PRINT_ALL_TIMING_MEMORY = {
    'max_depth': 10000,
    'min_bytes': 1,  # Only >=1
    'min_micros': 1,  # Only >=1
    'min_params': 0,
    'min_float_ops': 0,
    'order_by': 'name',
    'account_type_regexes': ['.*'],
    'start_name_regexes': ['.*'],
    'trim_name_regexes': [],
    'show_name_regexes': ['.*'],
    'hide_name_regexes': [],
    'account_displayed_op_only': True,
    'select': ['micros', 'bytes'],
    'output': 'stdout',
    'dump_to_file': ''
}

# pylint: enable=bad-whitespace
# pylint: enable=bad-continuation


def print_model_analysis(graph,
                         run_meta=None,
                         op_log=None,
                         tfprof_cmd='scope',
                         tfprof_options=TRAINABLE_VARS_PARAMS_STAT_OPTIONS):
  """Print model statistics.

    See go/tfprof or README for examples and tutorials.
    Run tfprof tool for help:
    'bazel run third_party/tensorflow/tools/tfprof help'

  Args:
    graph: tf.Graph.
    run_meta: tensorflow::RunMetadata proto. When provided, also shows valid
              timing and memory information when 'select' option contains
              'micros' and 'bytes'.
    op_log: tensorflow::tfprof::OpLog proto. users can use this proto to
            group together ops and use a op_type to select the group.
    tfprof_cmd: string. Either 'op', 'scope', 'graph', 'code'.
                'op' view organize outputs using operation type. (e.g. MatMul)
                'scope' view organize outputs using graph node name scope.
                'graph' view organize outputs using graph node inputs/outputs.
                'code' view organize outputs using Python call stack.
    tfprof_options: See 'tfprof help' for details.
  Returns:
    If tfprof_cmd is 'scope' or 'graph', returns TFGraphNodeProto proto.
    If tfprof_cmd is 'op' or 'code', returns TFMultiGraphNodeProto proto.
    Side effect: stdout/file/timeline.json depending on tfprof_options['output']
  """
  # pylint: disable=protected-access
  op_log = tfprof_logger._merge_default_with_oplog(
      graph, op_log, run_meta, add_trace=tfprof_cmd == 'code')
  # pylint: enable=protected-access
  opts = tfprof_options_pb2.OptionsProto()
  opts.max_depth = tfprof_options['max_depth']
  opts.min_bytes = tfprof_options['min_bytes']
  opts.min_micros = tfprof_options['min_micros']
  opts.min_params = tfprof_options['min_params']
  opts.min_float_ops = tfprof_options['min_float_ops']
  if 'min_occurrence' in tfprof_options:
    opts.min_occurrence = tfprof_options['min_occurrence']
  else:
    opts.min_occurrence = 0

  opts.order_by = tfprof_options['order_by']
  for p in tfprof_options['account_type_regexes']:
    opts.account_type_regexes.append(p)
  for p in tfprof_options['start_name_regexes']:
    opts.start_name_regexes.append(p)
  for p in tfprof_options['trim_name_regexes']:
    opts.trim_name_regexes.append(p)
  for p in tfprof_options['show_name_regexes']:
    opts.show_name_regexes.append(p)
  for p in tfprof_options['hide_name_regexes']:
    opts.hide_name_regexes.append(p)
  opts.account_displayed_op_only = tfprof_options['account_displayed_op_only']
  for p in tfprof_options['select']:
    opts.select.append(p)
  opts.output = tfprof_options['output']
  opts.dump_to_file = tfprof_options['dump_to_file']

  run_meta_str = run_meta.SerializeToString() if run_meta else b''

  if tfprof_cmd == 'code' or tfprof_cmd == 'op':
    tfprof_node = tfprof_output_pb2.TFMultiGraphNodeProto()
    tfprof_node.ParseFromString(
        print_mdl.PrintModelAnalysis(
            graph.as_graph_def().SerializeToString(),
            run_meta_str,
            op_log.SerializeToString(),
            tfprof_cmd.encode('utf-8'),
            opts.SerializeToString()))
  elif tfprof_cmd == 'graph' or tfprof_cmd == 'scope':
    tfprof_node = tfprof_output_pb2.TFGraphNodeProto()
    tfprof_node.ParseFromString(
        print_mdl.PrintModelAnalysis(
            graph.as_graph_def().SerializeToString(),
            run_meta_str,
            op_log.SerializeToString(),
            tfprof_cmd.encode('utf-8'),
            opts.SerializeToString()))
  else:
    raise errors.InvalidArgumentError(
        None, None, 'unknown tfprof_cmd: %s\n' % tfprof_cmd)

  return tfprof_node
