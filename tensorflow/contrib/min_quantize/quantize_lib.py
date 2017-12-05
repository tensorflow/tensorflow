from __future__ import absolute_import, division, print_function

import math
import re

import numpy as np
from tensorflow.contrib.min_quantize.quantized_pb2 import QuantizedGraph, QuantizedItem, RAW, SIMPLE, TABLE
from tensorflow.core.framework.node_def_pb2 import NodeDef
from tensorflow.core.protobuf.config_pb2 import ConfigProto, OptimizerOptions

from tensorflow.python.framework.dtypes import float32, int32
from tensorflow.python.framework.graph_util_impl import extract_sub_graph
from tensorflow.python.framework.importer import import_graph_def
from tensorflow.python.framework.ops import Graph
from tensorflow.python.framework.tensor_shape import as_shape
from tensorflow.python.framework.tensor_util import MakeNdarray

__all__ = [
  'quantize_graph_def',
  'import_graph',
]

while_node_name_reg = re.compile(r'(^|.*/)while(_\d+)?($|/.*)')


def quantize_graph_def(graph_def, skip=None, output_nodes=None, rel_tol=None, only=None):
  """
  :type graph_def: GraphDef
  :type skip: set|list
  :type output_nodes: list
  :type rel_tol: float
  :type only: str
  :return: QuantizedGraph
  """
  if output_nodes is not None and len(output_nodes) > 0:
    graph_def = extract_sub_graph(graph_def, output_nodes)

  nodes = []
  items = []
  for node in graph_def.node:
    # check skip
    if should_skip(node, skip):
      nodes.append(node)
      continue

    # try convert to constant
    try:
      value = MakeNdarray(node.attr['value'].tensor)  # type: np.ndarray
    except TypeError:
      nodes.append(node)
      continue

    # check repeated field
    same_value = all_same_value(value, rel_tol)
    if same_value is not None:
      nodes.append(const_node(node.attr['dtype'].type,
                              np.array([same_value], dtype=value.dtype),
                              value.shape))
      continue

    # check data size
    elif value.size < 4096:
      nodes.append(node)
      continue

    # finally
    processed_node = NodeDef()
    processed_node.name = node.name
    processed_node.op = 'Placeholder'
    processed_node.attr['dtype'].type = node.attr['dtype'].type
    processed_node.attr['shape'].shape.CopyFrom(as_shape(value.shape).as_proto())
    nodes.append(processed_node)

    item = QuantizedItem()
    item.name = node.name
    item.dtype = node.attr['dtype'].type
    item.shape.extend(value.shape)
    print('quantize {}'.format(node.name))
    _fill(item, value, only=only)
    items.append(item)
  graph = QuantizedGraph()
  graph.graph.versions.CopyFrom(graph_def.versions)
  graph.graph.library.CopyFrom(graph_def.library)
  graph.graph.node.extend(nodes)
  graph.items.extend(items)
  return graph


def import_graph(path, name=None):
  graph = load_graph(path)
  import_graph_def(graph.graph, name=name)
  return feeds_of_graph(graph)


def _fill_raw(item, value):
  print('\traw')
  base = np.min(value) if value.dtype == np.float32 else int(np.mean(value))
  data = value - base
  if value.dtype == np.int32:
    _extend(item.int_raw, data)
  elif value.dtype == np.float32:
    _extend(item.float_raw, data)
  else:
    raise Exception('unknown dtype {}'.format(value.dtype.name))
  item.vtype = RAW
  item.base = base


def _fill_simple(item, base, step, index):
  print('\tsimple')
  item.vtype = SIMPLE
  item.base = base
  item.step = step
  _extend(item.index, index)


def _fill_table(item, table, index):
  print('\ttable')
  item.vtype = TABLE
  if table.dtype == np.int32:
    _extend(item.int_table, table)
  elif table.dtype == np.float32:
    _extend(item.float_table, table)
  else:
    raise Exception('unknown dtype {}'.format(table.dtype))
  _extend(item.index, index)


def _fill(item, value, only=None):
  if only == 'raw':
    _fill_raw(item, value)
    return

  fallback_base, fallback_step, fallback_index = average_slice(value, 256)
  if only == 'simple':
    _fill_simple(item, fallback_base, fallback_step, fallback_index)
    return

  fallback_rmse = rmse_of(value,
                          restore_average_slice(fallback_base, fallback_step, fallback_index, dtype=value.dtype))
  rmse_threshold = min(np.std(value) * 0.05, fallback_rmse)
  print('\t256 average slice rmse {}'.format(fallback_rmse))
  print('\trmse threshold {}'.format(rmse_threshold))
  bit_size = 2
  while bit_size < 9:
    try:
      k_mean_table, k_means_index = k_means_slice(value, 2 ** bit_size, n_jobs=-1)
    except ImportError:
      print('\tsklearn not found, please install via pip');
      break
    k_means_rmse = rmse_of(value, restore_k_means_slice(k_mean_table, k_means_index))
    print('\tk-means table {}, rmse {}'.format(2 ** bit_size, k_means_rmse))
    if k_means_rmse <= rmse_threshold:
      _fill_table(item, k_mean_table, k_means_index)
      return
    diff = int(math.floor(math.log2(k_means_rmse / rmse_threshold)))
    if diff > 8 - bit_size:
      diff = 8 - bit_size
    if diff < 1:
      diff = 1
    bit_size += diff

  if fallback_rmse <= rmse_threshold:
    _fill_simple(item, fallback_base, fallback_step, fallback_index)
    return

  _fill_raw(item, value)


def _extend(field, data):
  if len(data.shape) == 0:
    if data.dtype == np.int32:
      field.append(int(data))
    elif data.dtype == np.float32:
      field.append(float(data))
    else:
      raise Exception('unknown dtype {}'.format(data.dtype))
  else:
    field.extend(reshaped_view(data))


def const_node(dtype, value, shape):
  node = NodeDef()
  node.op = 'Const'
  node.attr['dtype'].type = dtype.as_datatype_enum
  node.attr['value'].tensor.dtype = dtype.as_datatype_enum
  node.attr['value'].tensor.tensor_shape.CopyFrom(as_shape(shape).as_proto())

  if value.dtype == np.float32:
    _extend(node.attr['value'].tensor.float_val, value)
  elif value.dtype == np.int32:
    _extend(node.attr['value'].tensor.int_val, value)
  else:
    raise Exception('const_node, unknown dtype {}'.format(value.dtype))
  return node


def all_same_value(v, rel_tol=None):
  flat = reshaped_view(v)
  mean = np.mean(flat)
  for v in flat:
    if not isclose(v, mean, rel_tol):
      return None
  if isclose(mean, 0, rel_tol * 0.5):
    return 0
  return mean


def should_skip(node, skip):
  if skip is not None:
    if node.name in skip:
      return True
  if node.op != 'Const':
    return True
  if while_node_name_reg.match(node.name):  # variable in 'while' loop should not be quantized, cause frame error
    return True
  return False


def isclose(a, b, rel_tol=None):
  if rel_tol is None:
    rel_tol = 1e-9
  return abs(a - b) <= rel_tol


def reshaped_view(ary, shape=None):
  """
  :param ary: source data
  :param shape: target shape, default (-1,) as flat view
  :return: reshaped view

  :type ary: np.ndarray
  :rtype: np.ndarray
  """
  v = ary.view()
  v.shape = (-1,) if shape is None else shape
  return v


def rmse_of(v1, v2):
  """
  :type v1: np.ndarray
  :type v2: np.ndarray
  :return: root mean square error of (v1, v2)
  """
  e = reshaped_view(v1) - reshaped_view(v2)  # error
  e = np.square(e)  # square
  e = np.mean(e)  # mean
  e = np.sqrt(e)  # root
  return e


def average_slice(v, size, zero_rel_tol=None):
  """
  :type size: int
  :type v: np.ndarray
  :type zero_rel_tol: float
  :rtype: tuple[float, float, np.ndarray]
  """
  v_min = np.min(v)  # min value
  v_max = np.max(v)  # max value
  index = np.ndarray(shape=v.shape, dtype=np.uint32)  # quantized index

  cluster_size = size - 1
  step = (v_max - v_min) / (cluster_size - 1)
  for i in range(v.size):
    if isclose(v.flat[i], 0, rel_tol=zero_rel_tol):
      index.flat[i] = 0
    else:
      index.flat[i] = round((v.flat[i] - v_min) / step) + 1
  return v_min - step, step, index


def k_means_slice(v, size, zero_is_special=True, zero_rel_tol=None, n_jobs=1):
  from sklearn.cluster import KMeans

  """
  :type v: np.ndarray
  :type size: int
  :type zero_is_special: bool
  :type zero_rel_tol: float
  :type n_jobs: int
  :return: tuple[np.ndarray, np.ndarray]
  """
  v_min = np.min(v)
  v_max = np.max(v)

  if zero_is_special and v_min < 0 < v_max:
    has_zero_center = True
  else:
    has_zero_center = False

  k_means = KMeans(n_clusters=size - (1 if has_zero_center else 0), n_jobs=n_jobs)

  if has_zero_center:
    fit_data = []
    for i in range(v.size):
      if not isclose(v.flat[i], 0, rel_tol=zero_rel_tol):
        fit_data.append([i, v.flat[i]])
    fit_data = np.array(fit_data, dtype=np.float32)
    k_means.fit(reshaped_view(fit_data[:, 1], (-1, 1)))

    index = np.ndarray(shape=v.shape, dtype=np.uint32)
    index.fill(0)
    for i in range(len(fit_data)):
      real_index = int(fit_data[i][0])
      if not isclose(v.flat[real_index], 0, rel_tol=zero_rel_tol):
        index.flat[real_index] = k_means.labels_[i]
  else:
    k_means.fit(reshaped_view(v, (-1, 1)))
    index = reshaped_view(k_means.labels_)
  table = reshaped_view(k_means.cluster_centers_)
  return table.astype(v.dtype), index.astype(np.uint32)


def restore_average_slice(base, step, index, dtype=np.float32, zero_is_special=True):
  v = np.ndarray(shape=index.shape, dtype=dtype)
  for i in range(v.size):
    v.flat[i] = 0 if zero_is_special and isclose(0, index.flat[i]) else (index.flat[i] * step + base)
  return v


def restore_k_means_slice(table, index):
  return np.take(table, index)


def quantized_raw_item_to_ndarray(item):
  if item.dtype == int32.as_datatype_enum:
    return reshaped_view(np.array(item.int_raw), item.shape) + int(item.base)
  elif item.dtype == float32.as_datatype_enum:
    return reshaped_view(np.array(item.float_raw), item.shape) + float(item.base)
  else:
    raise Exception('unknown dtype {}'.format(item.dtype))


def quantized_simple_item_to_ndarray(item):
  if item.dtype == int32.as_datatype_enum:
    return reshaped_view(restore_average_slice(item.base, item.step, np.array(item.index), dtype=np.int32),
                         item.shape)
  elif item.dtype == float32.as_datatype_enum:
    return reshaped_view(restore_average_slice(item.base, item.step, np.array(item.index), dtype=np.float32),
                         item.shape)
  else:
    raise Exception('unknown dtype {}'.format(item.dtype))


def quantized_k_means_item_to_ndarray(item):
  if item.dtype == int32.as_datatype_enum:
    return reshaped_view(restore_k_means_slice(item.int_table, item.index), item.shape)
  elif item.dtype == float32.as_datatype_enum:
    return reshaped_view(restore_k_means_slice(item.float_table, item.index), item.shape)
  else:
    raise Exception('unknown dtype {}'.format(item.dtype))


def quantized_item_to_ndarray(item):
  if item.vtype == RAW:
    return quantized_raw_item_to_ndarray(item)
  elif item.vtype == SIMPLE:
    return quantized_simple_item_to_ndarray(item)
  elif item.vtype == TABLE:
    return quantized_k_means_item_to_ndarray(item)
  else:
    raise Exception('unknown vtype {}'.format(item.vtype))


def load_graph(path):
  g = QuantizedGraph()
  with open(path, 'rb') as fp:
    g.ParseFromString(fp.read())
  return g


def feeds_of_graph(graph):
  feeds = dict()
  for item in graph.items:
    feeds['{}:0'.format(item.name)] = quantized_item_to_ndarray(item)
  return feeds
