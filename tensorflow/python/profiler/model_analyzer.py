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

import six

from google.protobuf import message
from tensorflow.core.profiler import tfprof_options_pb2
from tensorflow.core.profiler import tfprof_output_pb2
from tensorflow.python import pywrap_tensorflow as print_mdl
from tensorflow.python.framework import errors
from tensorflow.python.profiler import option_builder
from tensorflow.python.profiler import tfprof_logger

_DEFAULT_PROFILE_OPTIONS = 0
_DEFAULT_ADVISE_OPTIONS = 0

# The following options are for 'advise' cmd.
# Show all advice.
ALL_ADVICE = {
    'ExpensiveOperationChecker': {},
    'AcceleratorUtilizationChecker': {},
    'JobChecker': {},  # Only available internally.
    'OperationChecker': {},
}


def _build_options(options):
  """Build tfprof.OptionsProto.

  Args:
    options: A dictionary of options.
  Returns:
    tfprof.OptionsProto.
  """
  opts = tfprof_options_pb2.OptionsProto()
  opts.max_depth = options.get('max_depth', 10)
  opts.min_bytes = options.get('min_bytes', 0)
  opts.min_peak_bytes = options.get('min_peak_bytes', 0)
  opts.min_residual_bytes = options.get('min_residual_bytes', 0)
  opts.min_output_bytes = options.get('min_output_bytes', 0)
  opts.min_micros = options.get('min_micros', 0)
  opts.min_accelerator_micros = options.get('min_accelerator_micros', 0)
  opts.min_cpu_micros = options.get('min_cpu_micros', 0)
  opts.min_params = options.get('min_params', 0)
  opts.min_float_ops = options.get('min_float_ops', 0)
  opts.min_occurrence = options.get('min_occurrence', 0)

  opts.step = options.get('step', -1)

  opts.order_by = options.get('order_by', 'name')

  for p in options.get('account_type_regexes', []):
    opts.account_type_regexes.append(p)
  for p in options.get('start_name_regexes', []):
    opts.start_name_regexes.append(p)
  for p in options.get('trim_name_regexes', []):
    opts.trim_name_regexes.append(p)
  for p in options.get('show_name_regexes', []):
    opts.show_name_regexes.append(p)
  for p in options.get('hide_name_regexes', []):
    opts.hide_name_regexes.append(p)
  opts.account_displayed_op_only = options.get('account_displayed_op_only',
                                               False)

  for p in options.get('select', []):
    opts.select.append(p)

  opts.output = options.get('output', 'stdout')
  opts.dump_to_file = options.get('dump_to_file', '')

  return opts


def _build_advisor_options(options):
  """Build tfprof.AdvisorOptionsProto.

  Args:
    options: A dictionary of options. See ALL_ADVICE example.
  Returns:
    tfprof.AdvisorOptionsProto.
  """
  opts = tfprof_options_pb2.AdvisorOptionsProto()
  if options is None:
    return opts
  for checker, checker_opts in six.iteritems(options):
    checker_ops_pb = tfprof_options_pb2.AdvisorOptionsProto.CheckerOption()
    for k, v in six.iteritems(checker_opts):
      checker_ops_pb[k] = v
    opts.checkers[checker].MergeFrom(checker_ops_pb)
  return opts


class Profiler(object):
  """TensorFlow multi-step profiler.

  https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/profiler/README.md

  ```python
  Typical use case:
    # Currently we are only allowed to create 1 profiler per process.
    profiler = Profiler(sess.graph)

    for i in xrange(total_steps):
      if i % 10000 == 0:
        run_meta = tf.RunMetadata()
        _ = sess.run(...,
                     options=tf.RunOptions(
                         trace_level=tf.RunOptions.FULL_TRACE),
                     run_metadata=run_meta)
        profiler.add_step(i, run_meta)

        # Profile the parameters of your model.
        profiler.profile_name_scope(options=(option_builder.ProfileOptionBuilder
            .trainable_variables_parameter()))

        # Or profile the timing of your model operations.
        opts = option_builder.ProfileOptionBuilder.time_and_memory()
        profiler.profile_operations(options=opts)

        # Or you can generate a timeline:
        opts = (option_builder.ProfileOptionBuilder(
                option_builder.ProfileOptionBuilder.time_and_memory())
                .with_step(i)
                .with_timeline_output(filename).build())
        profiler.profile_graph(options=opts)
      else:
        _ = sess.run(...)
    # Auto detect problems and generate advice.
    profiler.advise()
  ```
  """

  def __init__(self, graph, op_log=None):
    """Constructor.

    Args:
      graph: tf.Graph.
      op_log: optional. tensorflow::tfprof::OpLogProto proto. Used to define
          extra op types.
    """
    self._coverage = 0.0
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
      step: int, A step used to identify the RunMetadata. Must be different
         across different AddStep() calls.
      run_meta: RunMetadata proto that contains statistics of a session run.
    """
    # pylint: disable=protected-access
    op_log = tfprof_logger._merge_default_with_oplog(
        self._graph, run_meta=run_meta)
    # pylint: enable=protected-access
    # TODO(xpan): P1: Better to find the current graph.
    self._coverage = print_mdl.AddStep(
        step,
        self._graph.as_graph_def(add_shapes=True).SerializeToString(),
        run_meta.SerializeToString(), op_log.SerializeToString())

  def profile_python(self, options):
    """Profile the statistics of the Python codes.

      By default, it shows the call stack from root. To avoid
      redundant output, you may use options to filter as below
        options['show_name_regexes'] = ['.*my_code.py.*']

    Args:
      options: A dict of options. See core/profiler/g3doc/options.md.
    Returns:
      a MultiGraphNodeProto that records the results.
    """
    opts = _build_options(options)
    tfprof_node = tfprof_output_pb2.MultiGraphNodeProto()
    try:
      tfprof_node.ParseFromString(
          print_mdl.Profile('code'.encode('utf-8'), opts.SerializeToString()))
    except message.DecodeError as _:
      pass
    return tfprof_node

  def profile_operations(self, options):
    """Profile the statistics of the Operation types (e.g. MatMul, Conv2D).

    Args:
      options: A dict of options. See core/profiler/g3doc/options.md.
    Returns:
      a MultiGraphNodeProto that records the results.
    """
    opts = _build_options(options)
    tfprof_node = tfprof_output_pb2.MultiGraphNodeProto()
    try:
      tfprof_node.ParseFromString(
          print_mdl.Profile('op'.encode('utf-8'), opts.SerializeToString()))
    except message.DecodeError as _:
      pass
    return tfprof_node

  def profile_name_scope(self, options):
    """Profile the statistics of graph nodes, organized by name scope.

    Args:
      options: A dict of options. See core/profiler/g3doc/options.md.
    Returns:
      a GraphNodeProto that records the results.
    """
    opts = _build_options(options)
    tfprof_node = tfprof_output_pb2.GraphNodeProto()
    try:
      tfprof_node.ParseFromString(
          print_mdl.Profile('scope'.encode('utf-8'), opts.SerializeToString()))
    except message.DecodeError as _:
      pass
    return tfprof_node

  def profile_graph(self, options):
    """Profile the statistics of graph nodes, organized by dataflow graph.

    Args:
      options: A dict of options. See core/profiler/g3doc/options.md.
    Returns:
      a GraphNodeProto that records the results.
    """
    opts = _build_options(options)
    tfprof_node = tfprof_output_pb2.GraphNodeProto()
    try:
      tfprof_node.ParseFromString(
          print_mdl.Profile('graph'.encode('utf-8'), opts.SerializeToString()))
    except message.DecodeError as _:
      pass
    return tfprof_node

  def advise(self, options):
    """Automatically detect problems and generate reports.

    Args:
      options: A dict of options. See ALL_ADVICE example above.
    Returns:
      A Advise proto that conains the reports from all checkers.
    """
    advise_pb = tfprof_output_pb2.AdviceProto()
    opts = _build_advisor_options(options)
    advise_pb.ParseFromString(
        print_mdl.Profile('advise'.encode('utf-8'), opts.SerializeToString()))
    return advise_pb

  def _write_profile(self, filename):
    """Writes the profile to a file."""
    print_mdl.WriteProfile(filename)


def profile(graph,
            run_meta=None,
            op_log=None,
            cmd='scope',
            options=_DEFAULT_PROFILE_OPTIONS):
  """Profile model.

    Tutorials and examples can be found in:
    https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/profiler/README.md

  Args:
    graph: required tf.Graph.
    run_meta: optional tensorflow.RunMetadata proto. It is necessary to
        to support run time information profiling, such as time and memory.
    op_log: tensorflow.tfprof.OpLogProto proto. User can assign "types" to
        graph nodes with op_log. "types" allow user to flexibly group and
        account profiles using options['accounted_type_regexes'].
    cmd: string. Either 'op', 'scope', 'graph' or 'code'.
        'op' view organizes profile using operation type. (e.g. MatMul)
        'scope' view organizes profile using graph node name scope.
        'graph' view organizes profile using graph node inputs/outputs.
        'code' view organizes profile using Python call stack.
    options: A dict of options. See core/profiler/g3doc/options.md.
  Returns:
    If cmd is 'scope' or 'graph', returns GraphNodeProto proto.
    If cmd is 'op' or 'code', returns MultiGraphNodeProto proto.
    Side effect: stdout/file/timeline.json depending on options['output']
  """
  if options == _DEFAULT_PROFILE_OPTIONS:
    options = (option_builder.ProfileOptionBuilder
               .trainable_variables_parameter())

  # pylint: disable=protected-access
  op_log = tfprof_logger._merge_default_with_oplog(
      graph, op_log, run_meta, add_trace=cmd == 'code')
  # pylint: enable=protected-access

  opts = _build_options(options)

  run_meta_str = run_meta.SerializeToString() if run_meta else b''

  if cmd == 'code' or cmd == 'op':
    tfprof_node = tfprof_output_pb2.MultiGraphNodeProto()
    ret = print_mdl.PrintModelAnalysis(
        graph.as_graph_def(add_shapes=True).SerializeToString(),
        run_meta_str,
        op_log.SerializeToString(),
        cmd.encode('utf-8'),
        opts.SerializeToString())
    try:
      tfprof_node.ParseFromString(ret)
    except message.DecodeError as _:
      pass
      # sys.stderr.write('Cannot parse returned proto: %s.\n' % e)

  elif cmd == 'graph' or cmd == 'scope':
    tfprof_node = tfprof_output_pb2.GraphNodeProto()
    ret = print_mdl.PrintModelAnalysis(
        graph.as_graph_def(add_shapes=True).SerializeToString(),
        run_meta_str,
        op_log.SerializeToString(),
        cmd.encode('utf-8'),
        opts.SerializeToString())
    try:
      tfprof_node.ParseFromString(ret)
    except message.DecodeError as _:
      pass
      # sys.stderr.write('Cannot parse returned proto: %s.\n' % e)
  else:
    raise errors.InvalidArgumentError(
        None, None, 'unknown cmd: %s\n' % cmd)

  return tfprof_node


def advise(graph, run_meta=None, options=_DEFAULT_ADVISE_OPTIONS):
  """Auto profile and advise.

    Builds profiles and automatically check anomalies of various
    aspects. For more details:
    https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/profiler/README.md

  Args:
    graph: required tf.Graph.
    run_meta: optional tensorflow.RunMetadata proto. It is necessary to
        to support run time information profiling, such as time and memory.
    options: see ALL_ADVICE example above. Default checks everything.
  Returns:
    Returns AdviceProto proto
  """
  if options == _DEFAULT_ADVISE_OPTIONS:
    options = ALL_ADVICE.copy()

  # pylint: disable=protected-access
  op_log = tfprof_logger._merge_default_with_oplog(
      graph, None, run_meta, add_trace=True)
  # pylint: enable=protected-access

  run_meta_str = run_meta.SerializeToString() if run_meta else b''

  opts = _build_advisor_options(options)
  ret = tfprof_output_pb2.AdviceProto()
  ret.ParseFromString(
      print_mdl.PrintModelAnalysis(
          graph.as_graph_def(add_shapes=True).SerializeToString(),
          run_meta_str,
          op_log.SerializeToString(),
          'advise'.encode('utf-8'),
          opts.SerializeToString()))
  return ret
