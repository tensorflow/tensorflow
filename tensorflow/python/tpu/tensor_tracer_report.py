# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ========================================================================
"""Tensor Tracer report generation utilities."""

import collections
import hashlib
import os

from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tensor_tracer_pb2

_TRACER_LOG_PREFIX = ' [>>>TT>>>]'
_MARKER_SECTION_BEGIN = '!!!!!!! section-begin:'
_MARKER_SECTION_END = '!!!!!!! section-end:'

_SECTION_NAME_CONFIG = 'configuration'
_SECTION_NAME_REASON = 'reason'
_SECTION_NAME_OP_LIST = 'op-list'
_SECTION_NAME_TENSOR_LIST = 'tensor-list'
_SECTION_NAME_CACHE_INDEX_MAP = 'cache-index-map'
_SECTION_NAME_GRAPH = 'graph'
_SECTION_NAME_TENSOR_TRACER_CHECKPOINT = 'tensor_tracer_checkpoint'

_FIELD_NAME_VERSION = 'version:'
_FIELD_NAME_DEVICE = 'device:'
_FIELD_NAME_TRACE_MODE = 'trace-mode:'
_FIELD_NAME_SUBMODE = 'submode:'
_FIELD_NAME_NUM_REPLICAS = 'num-replicas:'
_FIELD_NAME_NUM_REPLICAS_PER_HOST = 'num-replicas-per-host:'
_FIELD_NAME_NUM_HOSTS = 'num-hosts:'
_FIELD_NAME_NUM_OPS = 'number-of-ops:'
_FIELD_NAME_NUM_TENSORS = 'number-of-tensors:'
_FIELD_NAME_NUM_CACHE_INDICES = 'number-of-indices:'
_FIELD_NAME_TOPOLOGICAL_SORT_SUCCEED = 'topological-sort-succeed:'

_CURRENT_VERSION = 'use-outside-compilation'

_TT_REPORT_PROTO = 'tensor_tracer_report.report_pb'


def topological_sort(g):
  """Performs topological sort on the given graph.

  Args:
     g: the graph.

  Returns:
     A pair where the first element indicates if the topological
     sort succeeded (True if there is no cycle found; False if a
     cycle is found) and the second element is either the sorted
     list of nodes or the cycle of nodes found.
  """
  def _is_loop_edge(op):
    """Returns true if the op is the end of a while-loop creating a cycle."""
    return op.type in ['NextIteration']

  def _in_op_degree(op):
    """Returns the number of incoming edges to the given op.

    The edge calculation skips the edges that come from 'NextIteration' ops.
    NextIteration creates a cycle in the graph. We break cycles by treating
    this op as 'sink' and ignoring all outgoing edges from it.
    Args:
      op: Tf.Operation
    Returns:
      the number of incoming edges.
    """
    count = 0
    for op in op.control_inputs + [in_tensor.op for in_tensor in op.inputs]:
      if not _is_loop_edge(op):
        count += 1
    return count

  sorted_ops = []
  op_in_degree = {op: _in_op_degree(op) for op in g.get_operations()}

  frontier = [op for (op, degree) in op_in_degree.items() if degree == 0]
  frontier.sort(key=lambda op: op.name)
  while frontier:
    op = frontier.pop()
    # Remove the op from graph, and remove its outgoing edges.
    sorted_ops.append(op)
    if _is_loop_edge(op):
      continue
    # pylint: disable=protected-access
    consumers = list(op._control_outputs)
    # pylint: enable=protected-access
    for out_tensor in op.outputs:
      consumers += [consumer_op for consumer_op in out_tensor.consumers()]
    consumers.sort(key=lambda op: op.name)
    for consumer in consumers:
      # For each deleted edge shift the bucket of the vertex.
      op_in_degree[consumer] -= 1
      if op_in_degree[consumer] == 0:
        frontier.append(consumer)
      if op_in_degree[consumer] < 0:
        raise ValueError('consumer:%s degree mismatch'%consumer.name)

  left_ops = set(op for (op, degree) in op_in_degree.items() if degree > 0)
  if left_ops:
    return (True, left_ops)
  else:
    assert len(g.get_operations()) == len(sorted_ops)
    return (False, sorted_ops)


class TensorTracerConfig(object):
  """Tensor Tracer config object."""

  def __init__(self):
    self.version = _CURRENT_VERSION
    self.device_type = None
    self.num_replicas = None
    self.num_replicas_per_host = None
    self.num_hosts = None


class TensorTraceOrder(object):
  """Class that is responsible from storing the trace-id of the tensors."""

  def __init__(self, graph_order, traced_tensors):
    self.graph_order = graph_order
    self.traced_tensors = traced_tensors
    self._create_tensor_maps()

  def _create_tensor_maps(self):
    """Creates tensor to cache id maps."""
    self.tensorname_to_cache_idx = {}
    self.cache_idx_to_tensor_idx = []
    for out_tensor in self.traced_tensors:
      tensor_name = out_tensor.name
      if tensor_name in self.tensorname_to_cache_idx:
        raise ValueError('Tensor name {} should not be already in '
                         'tensorname_to_cache_idx'.format(tensor_name))
      if tensor_name not in self.graph_order.tensor_to_idx:
        raise ValueError(
            'Tensor name {} is not in the tensor_to_idx, tensor_to_idx={} '
            .format(tensor_name, self.graph_order.tensor_to_idx))
      tensor_idx = self.graph_order.tensor_to_idx[tensor_name]
      cache_idx = len(self.tensorname_to_cache_idx)
      self.tensorname_to_cache_idx[tensor_name] = cache_idx
      self.cache_idx_to_tensor_idx.append(tensor_idx)
      if len(self.tensorname_to_cache_idx) != len(
          self.cache_idx_to_tensor_idx):
        raise RuntimeError(
            'len(self.tensorname_to_cache_idx) must equal'
            'len(self.cache_idx_to_tensor_idx), got '
            'len(self.tensorname_to_cache_idx)={}, '
            'len(self.cache_idx_to_tensor_idx)={}'
            .format(
                len(self.tensorname_to_cache_idx),
                len(self.cache_idx_to_tensor_idx)))


def sort_tensors_and_ops(graph):
  """Returns a wrapper that has consistent tensor and op orders."""
  graph_wrapper = collections.namedtuple('GraphWrapper',
                                         ['graph', 'operations', 'op_to_idx',
                                          'tensors', 'tensor_to_idx',
                                          'contains_cycle',
                                          'topological_order_or_cycle'])
  contains_cycle, topological_order_or_cycle = topological_sort(graph)
  if not contains_cycle:
    operations = topological_order_or_cycle
  else:
    operations = graph.get_operations()
  op_to_idx = {op.name: index for index, op
               in enumerate(operations)}
  tensors = []
  for op in operations:
    tensors.extend(op.outputs)
  tensor_to_idx = {tensor.name: index for index, tensor in
                   enumerate(tensors)}
  return graph_wrapper(graph=graph, operations=operations, op_to_idx=op_to_idx,
                       tensors=tensors, tensor_to_idx=tensor_to_idx,
                       contains_cycle=contains_cycle,
                       topological_order_or_cycle=topological_order_or_cycle)


class OpenReportFile(object):
  """Context manager for writing report file."""

  def __init__(self, tt_parameters):
    if not tt_parameters.report_file_path:
      self._report_file = None
      return
    try:
      self._report_file = gfile.Open(tt_parameters.report_file_path, 'w')
    except IOError as e:
      raise e

  def __enter__(self):
    return self._report_file

  def __exit__(self, unused_type, unused_value, unused_traceback):
    if self._report_file:
      self._report_file.close()


def proto_fingerprint(message_proto):
  serialized_message = message_proto.SerializeToString()
  hasher = hashlib.sha256(serialized_message)
  return hasher.hexdigest()


class TTReportHandle(object):
  """Utility class responsible from creating a tensor tracer report."""

  def __init__(self):
    self.instrument_records = {}
    self._report_file = None

  def instrument(self, name, explanation):
    self.instrument_records[name] = explanation

  def instrument_op(self, op, explanation):
    self.instrument(op.name, explanation)

  def instrument_tensor(self, tensor, explanation):
    self.instrument(tensor.name, explanation)

  def create_report_proto(self, tt_config, tt_parameters, tensor_trace_order,
                          tensor_trace_points, collected_signature_types):
    """Creates and returns a proto that stores tensor tracer configuration.

    Args:
      tt_config: TensorTracerConfig object holding information about the run
        environment (device, # cores, # hosts), and tensor tracer version
        information.
      tt_parameters: TTParameters objects storing the user provided parameters
        for tensor tracer.
      tensor_trace_order: TensorTraceOrder object storing a topological order of
        the graph.
      tensor_trace_points: Progromatically added trace_points/checkpoints.
      collected_signature_types: The signature types collected, e,g, norm,
        max, min, mean...
    Returns:
      TensorTracerReport proto.
    """
    report = tensor_tracer_pb2.TensorTracerReport()
    report.config.version = tt_config.version
    report.config.device = tt_config.device_type
    report.config.num_cores = tt_config.num_replicas
    report.config.num_hosts = tt_config.num_hosts
    report.config.num_cores_per_host = tt_config.num_replicas_per_host
    report.config.submode = tt_parameters.submode
    report.config.trace_mode = tt_parameters.trace_mode

    for signature_name, _ in sorted(collected_signature_types.items(),
                                    key=lambda x: x[1]):
      report.config.signatures.append(signature_name)

    for tensor in tensor_trace_order.graph_order.tensors:
      tensor_def = tensor_tracer_pb2.TensorTracerReport.TracedTensorDef()
      tensor_def.name = tensor.name
      if tensor.name in tensor_trace_order.tensorname_to_cache_idx:
        tensor_def.is_traced = True
        tensor_def.cache_index = (
            tensor_trace_order.tensorname_to_cache_idx[tensor.name])
      else:
        # To prevent small changes affecting the fingerprint calculation, avoid
        # writing the untraced tensors to metadata. Fingerprints will be
        # different only when the list of the traced tensors are different.
        if tt_parameters.use_fingerprint_subdir:
          continue
        tensor_def.is_traced = False

      if tensor.name in tensor_trace_points:
        tensor_def.trace_point_name = tensor_trace_points[tensor.name]
      if tensor.name in self.instrument_records:
        tensor_def.explanation = self.instrument_records[tensor.name]
      elif tensor.op.name in self.instrument_records:
        tensor_def.explanation = self.instrument_records[tensor.op.name]
      report.tensordef[tensor.name].CopyFrom(tensor_def)
    report.fingerprint = proto_fingerprint(report)
    logging.info('TensorTracerProto fingerprint is %s.',
                 report.fingerprint)
    tf_graph = tensor_trace_order.graph_order.graph
    report.graphdef.CopyFrom(tf_graph.as_graph_def())
    return report

  def report_proto_path(self, trace_dir, summary_tag_name):
    """Returns the path where report proto should be written.

    Args:
      trace_dir: String denoting the trace directory.
      summary_tag_name: Name of the unique tag that relates to
                        the report.
    Returns:
      A string denoting the path to the report proto.
    """
    filename = _TT_REPORT_PROTO + '.' + summary_tag_name.replace('/', '_')
    return os.path.join(trace_dir, filename)

  def write_report_proto(self, report_path, report_proto, tt_parameters):
    """Writes the given report proto under trace_dir."""
    gfile.MakeDirs(tt_parameters.trace_dir)
    with gfile.GFile(report_path, 'wb') as f:
      f.write(report_proto.SerializeToString())

  def create_report(self, tt_config, tt_parameters,
                    tensor_trace_order, tensor_trace_points):
    """Creates a report file and writes the trace information."""
    with OpenReportFile(tt_parameters) as self._report_file:
      self._write_config_section(tt_config, tt_parameters)
      self._write_op_list_section(tensor_trace_order.graph_order)
      self._write_tensor_list_section(tensor_trace_order.graph_order)
      self._write_trace_points(tensor_trace_points)
      self._write_cache_index_map_section(tensor_trace_order)
      self._write_reason_section()
      self._write_graph_section(tensor_trace_order.graph_order)

  def _write_trace_points(self, tensor_trace_points):
    """Writes the list of checkpoints."""
    self._write_report('%s %s\n'%(_MARKER_SECTION_BEGIN,
                                  _SECTION_NAME_TENSOR_TRACER_CHECKPOINT))
    for (tensor, checkpoint_name) in tensor_trace_points:
      self._write_report('%s %s\n'%(tensor.name, checkpoint_name))
    self._write_report('%s %s\n'%(_MARKER_SECTION_END,
                                  _SECTION_NAME_TENSOR_TRACER_CHECKPOINT))

  def _write_report(self, content):
    """Writes the given content to the report."""

    line = '%s %s'%(_TRACER_LOG_PREFIX, content)
    if self._report_file:
      self._report_file.write(line)
    else:
      logging.info(line)

  def _write_config_section(self, tt_config, tt_parameters):
    """Writes the config section of the report."""

    self._write_report('%s %s\n'%(_MARKER_SECTION_BEGIN, _SECTION_NAME_CONFIG))
    self._write_report('%s %s\n'%(_FIELD_NAME_VERSION, tt_config.version))
    self._write_report('%s %s\n'%(_FIELD_NAME_DEVICE, tt_config.device_type))
    self._write_report('%s %s\n'%(_FIELD_NAME_TRACE_MODE,
                                  tt_parameters.trace_mode))
    self._write_report('%s %s\n'%(_FIELD_NAME_SUBMODE,
                                  tt_parameters.submode))
    self._write_report('%s %s\n'%(_FIELD_NAME_NUM_REPLICAS,
                                  tt_config.num_replicas))
    self._write_report('%s %s\n'%(_FIELD_NAME_NUM_REPLICAS_PER_HOST,
                                  tt_config.num_replicas_per_host))
    self._write_report('%s %s\n'%(_FIELD_NAME_NUM_HOSTS, tt_config.num_hosts))
    self._write_report('%s %s\n'%(_MARKER_SECTION_END, _SECTION_NAME_CONFIG))

  def _write_reason_section(self):
    """Writes the reason section of the report."""

    self._write_report('%s %s\n'%(_MARKER_SECTION_BEGIN, _SECTION_NAME_REASON))
    for key in sorted(self.instrument_records):
      self._write_report('"%s" %s\n'%(key, self.instrument_records[key]))
    self._write_report('%s %s\n'%(_MARKER_SECTION_END, _SECTION_NAME_REASON))

  def _write_op_list_section(self, graph_order):
    """Writes the Op-list section of the report."""

    self._write_report('%s %s\n'%(_MARKER_SECTION_BEGIN, _SECTION_NAME_OP_LIST))
    self._write_report('%s %d\n'%(_FIELD_NAME_NUM_OPS,
                                  len(graph_order.operations)))
    for i in range(0, len(graph_order.operations)):
      op = graph_order.operations[i]
      line = '%d "%s" %s'%(i, op.name, op.type)
      for out_tensor in op.outputs:
        if out_tensor.name not in graph_order.tensor_to_idx:
          raise ValueError(
              'out_tensor is not in tensor_to_idx. out_tensor={}, '
              'tensor_to_idx={}'
              .format(out_tensor.name, graph_order.tensor_to_idx))
        line += ' %d'%graph_order.tensor_to_idx[out_tensor.name]
      line += '\n'
      self._write_report(line)
    self._write_report('%s %s\n'%(_MARKER_SECTION_END, _SECTION_NAME_OP_LIST))

  def _write_tensor_list_section(self, graph_order):
    """Writes the tensor-list section of the report."""

    self._write_report('%s %s\n'%(_MARKER_SECTION_BEGIN,
                                  _SECTION_NAME_TENSOR_LIST))
    self._write_report('%s %d\n'%(_FIELD_NAME_NUM_TENSORS,
                                  len(graph_order.tensors)))
    for i in range(0, len(graph_order.tensors)):
      tensor = graph_order.tensors[i]
      line = '%d "%s"'%(i, tensor.name)
      consumers = tensor.consumers()
      consumers.sort(key=lambda op: op.name)
      for consumer_op in consumers:
        if consumer_op.name not in graph_order.op_to_idx:
          raise ValueError(
              'consumer_op is not in op_to_idx.  '
              'got consumer_op={}, op_to_idx={}'
              .format(consumer_op.name, graph_order.op_to_idx))
        line += ' %d'%graph_order.op_to_idx[consumer_op.name]
      line += '\n'
      self._write_report(line)
    self._write_report('%s %s\n'%(_MARKER_SECTION_END,
                                  _SECTION_NAME_TENSOR_LIST))

  def _write_cache_index_map_section(self, tensor_trace_order):
    """Writes the mapping from cache index to tensor index to the report."""
    self._write_report('%s %s\n'%(_MARKER_SECTION_BEGIN,
                                  _SECTION_NAME_CACHE_INDEX_MAP))
    self._write_report('%s %d\n'%(
        _FIELD_NAME_NUM_CACHE_INDICES,
        len(tensor_trace_order.cache_idx_to_tensor_idx)))
    for cache_idx in range(0, len(tensor_trace_order.cache_idx_to_tensor_idx)):
      tensor_idx = tensor_trace_order.cache_idx_to_tensor_idx[cache_idx]
      line = '%d %d\n'%(cache_idx, tensor_idx)
      self._write_report(line)
    self._write_report('%s %s\n'%(_MARKER_SECTION_END,
                                  _SECTION_NAME_CACHE_INDEX_MAP))

  def _write_graph_section(self, graph_order):
    """Writes the graph section of the report."""

    self._write_report('%s %s\n'%(_MARKER_SECTION_BEGIN, _SECTION_NAME_GRAPH))
    self._write_report('%s %s\n'%(_FIELD_NAME_TOPOLOGICAL_SORT_SUCCEED,
                                  not graph_order.contains_cycle))
    l = list(graph_order.topological_order_or_cycle)
    for i in range(0, len(l)):
      self._write_report('%d "%s"\n'%(i, l[i].name))
    self._write_report('%s %s\n'%(_MARKER_SECTION_END, _SECTION_NAME_GRAPH))
