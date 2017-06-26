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

from tensorflow.contrib.tfprof.python.tools.tfprof import tfprof_logger
from tensorflow.contrib.tfprof.python.tools.tfprof.internal import pywrap_tensorflow_print_model_analysis_lib as print_mdl
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


def _build_options(tfprof_options):
  """Build tfprof.OptionsProto.

  Args:
    tfprof_options: A dictionary of options.
  Returns:
    tfprof.OptionsProto.
  """
  opts = tfprof_options_pb2.OptionsProto()
  opts.max_depth = tfprof_options.get('max_depth', 10)
  opts.min_bytes = tfprof_options.get('min_bytes', 0)
  opts.min_micros = tfprof_options.get('min_micros', 0)
  opts.min_params = tfprof_options.get('min_params', 0)
  opts.min_float_ops = tfprof_options.get('min_float_ops', 0)
  opts.min_occurrence = tfprof_options.get('min_occurrence', 0)

  opts.step = tfprof_options.get('step', -1)

  opts.order_by = tfprof_options.get('order_by', 'name')

  for p in tfprof_options.get('account_type_regexes', []):
    opts.account_type_regexes.append(p)
  for p in tfprof_options.get('start_name_regexes', []):
    opts.start_name_regexes.append(p)
  for p in tfprof_options.get('trim_name_regexes', []):
    opts.trim_name_regexes.append(p)
  for p in tfprof_options.get('show_name_regexes', []):
    opts.show_name_regexes.append(p)
  for p in tfprof_options.get('hide_name_regexes', []):
    opts.hide_name_regexes.append(p)
  opts.account_displayed_op_only = tfprof_options.get(
      'account_displayed_op_only', False)

  for p in tfprof_options.get('select', []):
    opts.select.append(p)

  opts.output = tfprof_options.get('output', 'stdout')
  opts.dump_to_file = tfprof_options.get('dump_to_file', '')

  return opts


class Profiler(object):
  """TensorFlow multi-step profiler.

  See go/tfprof or README for details.

  Typical use case:
    # Currently we are only allowed to create 1 profiler per process.
    profiler = Profile(sess.graph)

    for i in xrange(total_steps):
      if i % 10000 == 0:
        run_meta = tf.RunMetadata()
        _ = sess.run(...,
                     options=tf.RunOptions(
                         trace_level=tf.RunOptions.FULL_TRACE),
                     run_metadata=run_meta)
        profiler.add_step(i, run_meta)

        # Profile the parameters of your model.
        profiler.profile_name_scope(options=TRAINABLE_VARS_PARAMS_STAT_OPTIONS)

        # Or profile the timing of your model operations.
        opts = PRINT_ALL_TIMING_MEMORY.copy()
        opts['order_by'] = 'micros'
        opts['select'] = ['micros', 'occurrence']
        opts['max_depth'] = 20
        profiler.profile_operations(options=opts)

        # Or you can generate a timeline:
        opts = PRINT_ALL_TIMING_MEMORY.copy()
        opts['output'] = 'timeline:outfile=' + filename
        opts['step'] = i
        profiler.profile_graph(options=opts)
      else:
        _ = sess.run(...)
    # Auto detect problems and generate advice.
    profiler.advise()
  """

  def __init__(self, graph, op_log=None):
    """Constructor.

    Args:
      graph: tf.Graph.
      op_log: optional. tensorflow::tfprof::OpLog proto. Used to define
          extra op types.
    """
    self._graph = graph
    # pylint: disable=protected-access
    op_log = tfprof_logger._merge_default_with_oplog(
        self._graph, op_log=op_log)
    # pylint: enable=protected-access

    print_mdl.NewProfiler(
        self._graph.as_graph_def(add_shapes=True).SerializeToString(),
        op_log.SerializeToString())

  def __del__(self):
    print_mdl.DeleteProfiler()

  def add_step(self, step, run_meta):
    """Add statistics of a step.

    Args:
      step: A step uint64 used to identify the RunMetadata. Must be different
         across different AddStep() calls.
      run_meta: RunMetadata proto that contains statistics of a session run.
    """
    # pylint: disable=protected-access
    op_log = tfprof_logger._merge_default_with_oplog(
        self._graph, run_meta=run_meta, add_trace=False,
        add_trainable_var=False)
    # pylint: enable=protected-access
    print_mdl.AddStep(
        step, run_meta.SerializeToString(), op_log.SerializeToString())

  def profile_python_codes(self, options):
    """Profile the statistics of the Python codes.

      Hint: set options['show_name_regexes'] = ['.*my_code.py.*']

    Args:
      options: A dict of profiler options.
    Returns:
      a TFMultiGraphNodeProto that records the results.
    """
    opts = _build_options(options)
    tfprof_node = tfprof_output_pb2.TFMultiGraphNodeProto()
    tfprof_node.ParseFromString(
        print_mdl.Profile('code'.encode('utf-8'), opts.SerializeToString()))
    return tfprof_node

  def profile_operations(self, options):
    """Profile the statistics of the Operation types (e.g. MatMul, Conv2D).

    Args:
      options: A dict of profiler options.
    Returns:
      a TFMultiGraphNodeProto that records the results.
    """
    opts = _build_options(options)
    tfprof_node = tfprof_output_pb2.TFMultiGraphNodeProto()
    tfprof_node.ParseFromString(
        print_mdl.Profile('op'.encode('utf-8'), opts.SerializeToString()))
    return tfprof_node

  def profile_name_scope(self, options):
    """Profile the statistics of graph nodes, organized by name scope.

    Args:
      options: A dict of profiler options.
    Returns:
      a TFGraphNodeProto that records the results.
    """
    opts = _build_options(options)
    tfprof_node = tfprof_output_pb2.TFGraphNodeProto()
    tfprof_node.ParseFromString(
        print_mdl.Profile('scope'.encode('utf-8'), opts.SerializeToString()))
    return tfprof_node

  def profile_graph(self, options):
    """Profile the statistics of graph nodes, organized by dataflow graph.

    Args:
      options: A dict of profiler options.
    Returns:
      a TFGraphNodeProto that records the results.
    """
    opts = _build_options(options)
    tfprof_node = tfprof_output_pb2.TFGraphNodeProto()
    tfprof_node.ParseFromString(
        print_mdl.Profile('graph'.encode('utf-8'), opts.SerializeToString()))
    return tfprof_node

  def advise(self):
    """Automatically detect problems and generate reports."""
    print_mdl.Advise()


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

  opts = _build_options(tfprof_options)

  run_meta_str = run_meta.SerializeToString() if run_meta else b''

  if tfprof_cmd == 'code' or tfprof_cmd == 'op':
    tfprof_node = tfprof_output_pb2.TFMultiGraphNodeProto()
    tfprof_node.ParseFromString(
        print_mdl.PrintModelAnalysis(
            graph.as_graph_def(add_shapes=True).SerializeToString(),
            run_meta_str,
            op_log.SerializeToString(),
            tfprof_cmd.encode('utf-8'),
            opts.SerializeToString()))
  elif tfprof_cmd == 'graph' or tfprof_cmd == 'scope':
    tfprof_node = tfprof_output_pb2.TFGraphNodeProto()
    tfprof_node.ParseFromString(
        print_mdl.PrintModelAnalysis(
            graph.as_graph_def(add_shapes=True).SerializeToString(),
            run_meta_str,
            op_log.SerializeToString(),
            tfprof_cmd.encode('utf-8'),
            opts.SerializeToString()))
  else:
    raise errors.InvalidArgumentError(
        None, None, 'unknown tfprof_cmd: %s\n' % tfprof_cmd)

  return tfprof_node
