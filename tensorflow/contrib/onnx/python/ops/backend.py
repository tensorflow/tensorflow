"""Backend for running ONNX on Tensorflow

To run this, you will need to have Tensorflow installed as well.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
import re
import warnings
import sys

import numpy as np
from onnx import checker
from onnx.onnx_pb2 import GraphProto, TensorProto, AttributeProto
import onnx.numpy_helper
import onnx.defs

from onnx.backend.base import (
    Backend,
    BackendRep,
    Device,
    DeviceType,
    namedtupledict,
)

from onnx import helper
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.client import device_lib
from tensorflow.contrib.layers.python.layers import layers

class TensorflowNet(object):
  """
  Placeholder class for a protobuf definition.
  """
  def __init__(self):
    self.op = []
    self.external_input = []
    self.external_output = []
    self.output = []

    self.output_dict = {}

# TODO: allow more flexible placement
def get_device_option(device):
  m = {DeviceType.CPU: '/cpu',
       DeviceType.CUDA: '/gpu'}
  return m[device.type]

# TODO: Move this into ONNX main library
def convertAttributeProto(onnx_arg):
  """
  Convert an ONNX AttributeProto into an appropriate Python object
  for the type.
  NB: Tensor attribute gets returned as the straight proto.
  """
  if onnx_arg.HasField('f'):
    return onnx_arg.f
  elif onnx_arg.HasField('i'):
    return onnx_arg.i
  elif onnx_arg.HasField('s'):
    return str(onnx_arg.s, 'utf-8') \
      if sys.version_info[0] >= 3 else onnx_arg.s
  elif onnx_arg.HasField('t'):
    return onnx_arg.t  # this is a proto!
  elif onnx_arg.floats:
    return list(onnx_arg.floats)
  elif onnx_arg.ints:
    return list(onnx_arg.ints)
  elif onnx_arg.strings:
    str_list = list(onnx_arg.strings)
    if sys.version_info[0] >= 3:
      str_list = map(lambda x: str(x, 'utf-8'), str_list)
    return str_list
  else:
    raise ValueError("Unsupported ONNX attribute: {}".format(onnx_arg))

class OnnxAttributes(dict):
  """
  This is a more convenient way to work with ONNX/Caffe2 attributes
  that is not the protobuf representation.
  """
  @staticmethod
  def from_onnx(args):
    d = OnnxAttributes()
    for arg in args:
      d[arg.name] = convertAttributeProto(arg)
    return d

  def caffe2(self, kmap=lambda x: x):
    for k, v in self.items():
      yield caffe2.python.utils.MakeArgument(kmap(k), v)

# TODO: Move this into ONNX main library
class OnnxNode(object):
  """
  Reimplementation of NodeProto from ONNX, but in a form
  more convenient to work with from Python.
  We may temporarily edit these nodes to get them into Caffe2 form,
  before actually translating into the Caffe2 protobuf, since this
  is easier than decomposing everything, and putting it back together
  when we're ready.
  """
  def __init__(self, node):
    self.name = str(node.name)
    self.op_type = str(node.op_type)
    self.attrs = OnnxAttributes.from_onnx(node.attribute)
    self.consumed_inputs = self.attrs.pop("consumed_inputs", None)
    self.inputs = list(node.input)
    self.outputs = list(node.output)

class TensorflowBackend(Backend):
  """ Tensorflow Backend for ONNX
  """

  onnx_tf_attribute_map = {
      "scale": "stddev",
      "high": "maxval",
      "low": "minval",
      "axes": "axis",
      "keepdims": "keep_dims",
      "axis": "dim",
      "to": "dtype",
  }

  onnx_tf_per_op_attr_map = {}

  onnx_tf_op_map = {
      "abs": math_ops.abs,
      "cast": math_ops.cast,
      "ceil": math_ops.ceil,
      "relu": gen_nn_ops.relu,
      # "dot": tf.contrib.keras.backend.dot,
      "exp": math_ops.exp,
      "floor": math_ops.floor,
      "gather": array_ops.gather,
      "log": math_ops.log,
      "neg": math_ops.negative,
      "pow": math_ops.pow,
      "random_normal": random_ops.random_normal,
      "random_uniform": random_ops.random_uniform,
      "reciprocal": math_ops.reciprocal,
      "reduce_log_sum_exp": math_ops.reduce_logsumexp,
      "reduce_max": math_ops.reduce_max,
      "reduce_mean": math_ops.reduce_mean,
      "reduce_min": math_ops.reduce_min,
      "reduce_prod": math_ops.reduce_prod,
      "reduce_sum": math_ops.reduce_sum,
      "sigmoid": math_ops.sigmoid,
      "sqrt": math_ops.sqrt,
      "squeeze": array_ops.squeeze,
      "tanh": math_ops.tanh,
      "transpose": array_ops.transpose,
  }

  tensor_type_to_tf_type = {
      TensorProto.FLOAT: dtypes.float32,
      TensorProto.UINT8: dtypes.uint8,
      TensorProto.INT8: dtypes.int8,
      TensorProto.UINT16: dtypes.uint16,
      TensorProto.INT16: dtypes.int16,
      TensorProto.INT32: dtypes.int32,
      TensorProto.INT64: dtypes.int64,
      TensorProto.BOOL: dtypes.bool,
      TensorProto.FLOAT16: dtypes.float16,
      TensorProto.DOUBLE: dtypes.float64,
      TensorProto.COMPLEX64: dtypes.complex64,
      TensorProto.COMPLEX128: dtypes.complex128,
      # TODO: uncomment this in the future
      # TensorProto.UINT32: dtypes.uint32,
      # TensorProto.UINT64: dtypes.uint64,
  }

  tensor_type_enum = [
      "undefined",
      dtypes.float32,
      dtypes.uint8,
      dtypes.int8,
      dtypes.uint16,
      dtypes.int16,
      dtypes.int32,
      dtypes.int64,
      dtypes.bool,
      dtypes.float16,
      dtypes.float64,
      dtypes.complex64,
      dtypes.complex128,
      # TODO: uncomment this in the future
      # dtypes.uint32,
      # dtypes.uint64,
  ]

  type_string_to_tf_type = {
      "float": dtypes.float32,
      "uint8": dtypes.uint8,
      "int8": dtypes.int8,
      "uint16": dtypes.uint16,
      "int16": dtypes.int16,
      "int32": dtypes.int32,
      "int64": dtypes.int64,
      "bool": dtypes.bool,
      "float16": dtypes.float16,
      "double": dtypes.float64,
      "complex64": dtypes.complex64,
      "complex128": dtypes.complex128,
      # TODO: uncomment this in the future
      # "uint32": dtypes.uint32,
      # "uint64": dtypes.uint64,
  }

  attr_translator = {
      "dtype": lambda cls, x: cls.tensor_type_to_tf_type[x],
      "keepdims": lambda cls, x: bool(x),
      "to": lambda cls, x: cls.type_string_to_tf_type[x],
  }

  # TODO: Remove this.
  @classmethod
  def guess_tf_pad(cls, pads):
    tf_pad = "VALID" if pads == None \
              or pads[-1] == 0 \
              or (pads[0] != pads[2]) else "SAME"
    warnings.warn("Unsupported pads attribute by Tensorflow in "
                  "pool operator. Your padding is {}, we guess "
                  "you want {} padding.".format(str(pads), tf_pad),
                  UserWarning)
    return tf_pad

  @classmethod
  def get_padding_as_op(cls, x, pads):
    num_dim = int(len(pads)/2)

    tf_pads = np.transpose(np.array(pads).reshape([2, num_dim]))
    tf_pads = [0, 0, 0, 0] + tf_pads.flatten().tolist()

    padding = constant_op.constant(np.array(tf_pads)
                                   .reshape([num_dim + 2, 2])
                                   # tf requires int32 paddings
                                   .astype(np.int32))
    return array_ops.pad(x, padding)

  # TODO: better broadcast
  @classmethod
  def _explicit_broadcast(cls, tensor, broadcast=1):
    if broadcast == 0:
      return tensor
    warnings.warn("Currently, support for broadcasting is limited "
                  "and may result in unexpected results",
                  UserWarning)
    tensor = array_ops.expand_dims(tensor, 0)
    tensor = array_ops.expand_dims(tensor, 2)
    tensor = array_ops.expand_dims(tensor, 3)
    return tensor

  @classmethod
  def _bin_op(cls, node, input_dict, op_func):
    x = input_dict[node.inputs[0]]
    y = input_dict[node.inputs[1]]
    broadcast = node.attrs.get("broadcast", 1)
    if broadcast == 0:
      warnings.warn("Definition of {} with broadcast disabled is not "
                    "yet supported.".format(node.type), UserWarning)

    if "axis" in node.attrs.keys():
      num_ones_to_append = len(x.get_shape()) - \
                           len(y.get_shape()) - \
                           node.attrs["axis"]
      if num_ones_to_append > 0:
        ones = array_ops.ones([num_ones_to_append], dtypes.int32)
        broadcasted_shape = array_ops.concat([gen_array_ops.shape(y), ones],
                                             axis=0)
        y = gen_array_ops.reshape(y, broadcasted_shape)

    return op_func(x, y)

  @classmethod
  def onnx_graph_to_tensorflow_net(cls, graph_def):
    # initializer: TensorProtos representing the values to initialize
    # a given tensor.
    # initialized: A list of names of the initialized tensors.
    if graph_def.initializer:
      input_dict_items = cls.onnx_initializer_to_input_dict_items( \
        graph_def.initializer)
      initialized = {init.name for init in graph_def.initializer}
    else:
      input_dict_items = []
      initialized = set()
    predict_net = TensorflowNet()
    predict_net.name = graph_def.name

    predict_net.external_input.extend(
        value_info.name for value_info in graph_def.input)
    predict_net.external_output.extend(
        value_info.name for value_info in graph_def.output)

    # creating placeholders for currently unkown inputs
    for value_info in graph_def.input:
      if value_info.name in initialized:
        continue

      shape = list(d.dim_value for d in \
        value_info.type.tensor_type.shape.dim)
      x = array_ops.placeholder(
          cls.tensor_type_enum[value_info.type.tensor_type.elem_type],
          name=value_info.name, shape=shape)
      input_dict_items.append([value_info.name, x])

    # input dict: this dictionary is a map from variable names
    # to the latest produced tensors of the given name.
    # This dictionary will get updated as build the graph because
    # some ops may produce a result tensor with the same name as
    # the input tensor. The input dict tracks the latest produced
    # tensors.
    input_dict = dict(input_dict_items)
    # Since input dict may be updated, we need to keep a copy
    # of the original input dict where we track the earliest
    # defined tensors so we can have access to the placeholders
    # to feed in input tensors when we run the graph.
    original_input_dict = dict(input_dict_items)
    output_dict = dict()

    for node in graph_def.node:
      node = OnnxNode(node)
      output_ops = cls._onnx_node_to_tensorflow_op(node, input_dict)
      curr_node_output_map = list(zip(node.outputs, output_ops))
      input_dict = dict(list(input_dict.items()) +
                        curr_node_output_map)

      output_dict = dict(list(output_dict.items()) +
                         curr_node_output_map)
      predict_net.op.extend(output_ops)

    predict_net.output_dict = output_dict
    return original_input_dict, predict_net

  @classmethod
  def prepare(cls, model, device='CPU', **kwargs):
    super(TensorflowBackend, cls).prepare(model, device, **kwargs)

    original_input_dict, predict_net = \
      cls.onnx_graph_to_tensorflow_net(model.graph)

    initialized = {init.name for init in model.graph.initializer}
    uninitialized = [x for x in predict_net.external_input
                     if not x in initialized]

    original_input_dict = dict([(key, original_input_dict[key]) \
      for key in uninitialized])

    return original_input_dict, predict_net.output_dict

  @classmethod
  def onnx_initializer_to_input_dict_items(cls, initializer, init_net_name='init'):
    def tensor2list(onnx_tensor):
      # Use the onnx.numpy_helper because the data may be raw
      return onnx.numpy_helper.to_array(onnx_tensor).flatten().tolist()
    input_dict = [(tp.name,
                   constant_op.constant(tensor2list(tp),
                                        shape=tp.dims,
                                        dtype=cls. \
                                        tensor_type_to_tf_type[tp.data_type]))
                  for tp in initializer]
    return input_dict

  @classmethod
  def op_name_to_lower(cls, name):
    return re.sub('(?<!^)(?=[A-Z])', '_', name).lower()

  @classmethod
  def _onnx_node_to_tensorflow_op(cls, node, input_dict):
    op_name_lowered = cls.op_name_to_lower(node.op_type)
    if op_name_lowered in cls.onnx_tf_op_map.keys():
      return cls.handle_trivial(node, input_dict)

    handler_name = "handle_" + op_name_lowered
    # Check if specialized handler exists.
    if handler_name in dir(cls):
      method_to_call = getattr(cls, handler_name)
      return method_to_call(node, input_dict)

  @classmethod
  def handle_trivial(cls, node, input_dict):
    # Perform automatic attribute value translation.
    attrs = dict([(x, cls.attr_translator[x](cls, node.attrs[x]) \
      if x in cls.attr_translator else node.attrs[x]) \
      for x in node.attrs.keys()])

    # Create an identity map from onnx attribute names to tf
    # attribute names.
    attr_map = dict([(x, x) for x in node.attrs.keys()])

    # Modify the map accoridng to onnx_tf_attribute_map.
    attr_map = dict([(x, cls.onnx_tf_attribute_map[x] \
      if x in cls.onnx_tf_attribute_map.keys() else x) \
      for x in attr_map.keys()])

    # TODO: Per op attribute name mapping has the final say.

    # Substitute attribute names in attrs.
    attrs = dict([(attr_map[x], y) for (x, y) in attrs.items()])
    inputs = [input_dict[name] for name in node.inputs]
    return [cls.onnx_tf_op_map[cls.op_name_to_lower(node.op_type)] \
      (*inputs, **attrs)]

  @classmethod
  def handle_add(cls, node, input_dict):
    return [cls._bin_op(node, input_dict, math_ops.add)]

  @classmethod
  def handle_arg_max(cls, node, input_dict):
    data = input_dict[node.inputs[0]]
    axis = node.attrs["axis"]
    keepdims = node.attrs.get("keepdims", 1)
    if keepdims == 1:
      warnings.warn("Definition of ArgMax with keepdims enabled is "
                    "incompatible between onnx and tensorflow.",
                    UserWarning)
    return [math_ops.argmax(data, axis=axis)]

  @classmethod
  def handle_arg_min(cls, node, input_dict):
    data = input_dict[node.inputs[0]]
    axis = node.attrs["axis"]
    keepdims = node.attrs.get("keepdims", 1)
    if keepdims == 1:
      warnings.warn("Definition of ArgMin with keepdims enabled is "
                    "incompatible between onnx and tensorflow.",
                    UserWarning)
    return [math_ops.argmin(data, axis=axis)]

  # pylint: disable=line-too-long
  @classmethod
  def _compatibility_pool(cls, node, input_dict, pool_func, guess_or_manual_pad):
    from math import ceil

    x = input_dict[node.inputs[0]]
    x_rank = len(x.get_shape())

    kernel_shape = node.attrs["kernel_shape"]
    strides = node.attrs["strides"]

    pads = node.attrs.get("pads", [0, 0, 0, 0])

    def py_pool(x, kernel_shape, strides, pad):
      out_h = int((x.shape[2] + pads[0] + pads[2] - kernel_shape[0]) // strides[0]) + 1
      out_w = int((x.shape[3] + pads[1] + pads[3] - kernel_shape[1]) // strides[1]) + 1

      out = np.zeros([x.shape[0], x.shape[1], out_h, out_w], dtype=np.float32)
      for n in range(0, x.shape[0]):
        for c in range(0, x.shape[1]):
          for h in range(0 - pad[0], x.shape[2] + pad[2], strides[0]):
            for w in range(0 - pad[1], x.shape[3] + pad[3], strides[1]):
              # skip window if window is outside padded region
              if (h + kernel_shape[0] > x.shape[2] + pad[2]) or \
                 (w + kernel_shape[1] > x.shape[3] + pad[3]):
                continue
              count = 0
              val = 0
              for kh in range(0, kernel_shape[0]):
                for kw in range(0, kernel_shape[1]):
                  current_h = h+kh
                  current_w = w+kw
                  if (current_h >= 0) and (current_w >= 0) and \
                     (current_h < x.shape[2]) and (current_w < x.shape[3]):
                    count += 1
                    val += x[n][c][current_h][current_w]
              out[n][c][int((h + pad[0])//strides[0])][int((w + pad[1])//strides[1])] = val/count
      return out

    pooled = script_ops.py_func(py_pool, [x, kernel_shape, strides, pads], dtypes.float32)
    x_shape = list(x.get_shape())

    out_h = int((x_shape[2] + pads[0] + pads[2] - kernel_shape[0]) // strides[0]) + 1
    out_w = int((x_shape[3] + pads[1] + pads[3] - kernel_shape[1]) // strides[1]) + 1
    pooled.set_shape([x_shape[0], x_shape[1], out_h, out_w])

    return [pooled]

  # pylint: enable=line-too-long
  @classmethod
  def _pool(cls, node, input_dict, pool_func, guess_or_manual_pad):
    from math import ceil

    x = input_dict[node.inputs[0]]
    x_rank = len(x.get_shape())

    support_cuda = cls.supports_device("CUDA")
    data_format = cls.get_data_format(x_rank, support_cuda)

    kernel_shape = node.attrs["kernel_shape"]
    strides = node.attrs["strides"]

    # By default, do not pad
    pad = "VALID"
    if "pads" in node.attrs.keys():
      if guess_or_manual_pad == 0:
        pad = cls.guess_tf_pad(node.attrs["pads"])
      else:
        x = cls.get_padding_as_op(x, node.attrs["pads"])
        pad = "VALID"

    if support_cuda:
      pooled = pool_func(x, [1, 1] + kernel_shape, [1, 1] + strides, pad,
                         data_format=data_format)
    else:
      x = array_ops.transpose(x, perm=[0, 2, 3, 1])
      pooled = pool_func(x, [1] + kernel_shape + [1], [1] + strides + [1], pad,
                         data_format=data_format)
      pooled = array_ops.transpose(pooled, perm=[0, 3, 1, 2])

    return [pooled]

  @classmethod
  def handle_average_pool(cls, node, input_dict):
    spatial_dim = list(input_dict[node.inputs[0]].get_shape()[2:])
    kernel_shape = node.attrs.get("kernel_shape", [])
    global_pool = True
    for i in range(len(spatial_dim)):
      global_pool = global_pool and (spatial_dim[i] < kernel_shape[i])

    if global_pool:
      return cls.handle_global_average_pool(node, input_dict)

    # 0 = guess padding
    return cls._compatibility_pool(node, input_dict, nn_ops.avg_pool, 0)

  @classmethod
  def handle_batch_normalization(cls, node, input_dict):
    x = input_dict[node.inputs[0]]
    scale = cls._explicit_broadcast(input_dict[node.inputs[1]])
    bias = cls._explicit_broadcast(input_dict[node.inputs[2]])
    mean = cls._explicit_broadcast(input_dict[node.inputs[3]])
    variance = cls._explicit_broadcast(input_dict[node.inputs[4]])

    variance_epsilon = node.attrs.get("epsilon", 0.00001)
    if node.attrs.get("is_test", 0):
      return [nn_impl.batch_normalization(x, mean, variance, bias, scale,
                                          variance_epsilon)]
    if "momentum" in node.attrs.keys():
      warnings.warn("Unsupported momentum attribute by Tensorflow in "
                    "batch_normalization. This attribute will be ignored.",
                    UserWarning)
    if "spatial" in node.attrs.keys():
      warnings.warn("Unsupported spatial attribute by Tensorflow in "
                    "batch_normalization. This attribute will be ignored.",
                    UserWarning)
    # TODO: need to conform to the documentation here
    return [nn_impl.batch_normalization(x, mean, variance, bias, scale,
                                        variance_epsilon)]

  @classmethod
  def handle_concat(cls, node, input_dict):
    values = [input_dict[a] for a in node.inputs]
    # apparently this is what's needed for squeezenet to work
    axis = node.attrs.get("axis", 1)
    return [array_ops.concat(values, axis=axis)]

  @classmethod
  def handle_constant(cls, node, input_dict):
    value = node.attrs["value"]
    elements = onnx.numpy_helper.to_array(value).flatten().tolist()
    dtype = cls.tensor_type_to_tf_type[value.data_type]
    return [array_ops.constant(elements, dtype=dtype, shape=value.dims)]

  @classmethod
  def get_data_format(cls, x_rank, support_cuda):
    if support_cuda:
      data_format = "NCDHW"
      if x_rank == 3:
        data_format = "NCW"
      elif x_rank == 4:
        data_format = "NCHW"
    else:
      data_format = "NDHWC"
      if x_rank == 3:
        data_format = "NWC"
      elif x_rank == 4:
        data_format = "NHWC"
    return data_format

  @classmethod
  def _conv(cls, node, input_dict, transpose=False):
    x = input_dict[node.inputs[0]]
    x_rank = len(x.get_shape())

    support_cuda = cls.supports_device("CUDA")
    data_format = cls.get_data_format(x_rank, support_cuda)

    in_weights = input_dict[node.inputs[1]]
    weights_rank = len(in_weights.get_shape())
    if transpose:
      # Translate weights from (C x M x KH x KW) to (KH x KW X C X M)
      perm = list(range(2, weights_rank)) + [0, 1]
    else:
      # Translate weights from (M x C x KH x KW) to (KH x KW X C X M)
      perm = list(range(2, weights_rank)) + [1, 0]

    weights = array_ops.transpose(in_weights, perm)
    dilations = node.attrs.get("dilations", None)
    strides = node.attrs.get("strides", None)

    if "kernel_shape" in node.attrs.keys():
      warnings.warn("Unsupported kernel_shape attribute by Tensorflow in "
                    "Conv operator. The attribute will be ignored.",
                    UserWarning)

    if "pads" in node.attrs.keys():
      x = cls.get_padding_as_op(x, node.attrs["pads"])

    if "group" in node.attrs:

      weight_groups = array_ops.split(weights, \
        num_or_size_splits=node.attrs["group"], axis=3)

      if support_cuda:
        xs = array_ops.split(x, num_or_size_splits=node.attrs["group"], axis=1)
      else:
        x = array_ops.transpose(x, perm=[0, 2, 3, 1])
        xs = array_ops.split(x, num_or_size_splits=node.attrs["group"], axis=3)

      convolved = [nn_ops.convolution(x, weight, "VALID", strides=strides,
                                      dilation_rate=dilations,
                                      data_format=data_format)
                   for (x, weight) in zip(xs, weight_groups)]

      if len(node.inputs) == 2:
        if support_cuda:
          output = array_ops.concat(convolved, axis=1)
        else:
          output = array_ops.concat(convolved, axis=3)
          output = array_ops.transpose(output, perm=[0, 3, 1, 2])
      else:
        bias = input_dict[node.inputs[2]]

        if support_cuda:
          output = array_ops.concat(convolved, axis=1)
          output = nn_ops.bias_add(output, bias, data_format=data_format)
        else:
          output = array_ops.concat(convolved, axis=3)
          output = nn_ops.bias_add(output, bias, data_format=data_format)
          output = array_ops.transpose(output, perm=[0, 3, 1, 2])

      return [output]

    if not support_cuda:
      x = array_ops.transpose(x, perm=[0, 2, 3, 1])

    convolved = nn_ops.convolution(x, weights, "VALID", strides=strides,
                                   dilation_rate=dilations,
                                   data_format=data_format)

    if not support_cuda:
      convolved = array_ops.transpose(convolved, perm=[0, 3, 1, 2])

    if len(node.inputs) == 2:
      return [convolved]
    else:
      bias = input_dict[node.inputs[2]]
      if not support_cuda:
        convolved = array_ops.transpose(convolved, perm=[0, 2, 3, 1])
      output = nn_ops.bias_add(convolved, bias, data_format=data_format)
      if not support_cuda:
        output = array_ops.transpose(output, perm=[0, 3, 1, 2])
      return [output]

  @classmethod
  def handle_conv(cls, node, input_dict):
    return cls._conv(node, input_dict)

  @classmethod
  def handle_conv_transpose(cls, node, input_dict):
    return cls._conv(node, input_dict, transpose=True)

  @classmethod
  def handle_div(cls, node, input_dict):
    return [cls._bin_op(node, input_dict, math_ops.divide)]

  @classmethod
  def handle_dropout(cls, node, input_dict):
    x = input_dict[node.inputs[0]]
    # Not supported by TF
    is_test = node.attrs["is_test"] if "is_test" in node.attrs.keys() else 0
    if is_test:
      return [x]
    ratio = node.attrs["ratio"] if "ratio" in node.attrs.keys() else 0.5
    return [nn_ops.dropout(x, 1 - ratio)]

  @classmethod
  def handle_elu(cls, node, input_dict):
    x = input_dict[node.inputs[0]]
    if "alpha" in node.attrs.keys():
      warnings.warn("Unsupported alpha attribute by Tensorflow in Elu."
                    "This attribute will be ignored.", UserWarning)
    return [gen_nn_ops.elu(x)]

  @classmethod
  def handle_flatten(cls, node, input_dict):
    tensor = input_dict[node.inputs[0]]
    axis = node.attrs["axis"] if "axis" in node.attrs.keys() else 1
    shape = array_ops.shape(tensor)
    split0, split1 = array_ops.split(shape,
                                     [axis, array_ops.size(shape) - axis])
    split0 = math_ops.reduce_prod(split0)
    split1 = math_ops.reduce_prod(split1)
    output_shape = array_ops.stack([split0, split1])
    return [array_ops.reshape(tensor, output_shape)]

  @classmethod
  def handle_gemm(cls, node, input_dict):
    x = input_dict[node.inputs[0]]
    x = layers.flatten(x)
    y = input_dict[node.inputs[1]]
    z = input_dict[node.inputs[2]]
    if "transA" in node.attrs.keys() and node.attrs["transA"] == 1:
      x = array_ops.transpose(x)
    if "transB" in node.attrs.keys() and node.attrs["transB"] == 1:
      y = array_ops.transpose(y)
    alpha = node.attrs["alpha"] if "alpha" in node.attrs.keys() else 1.0
    beta = node.attrs["beta"] if "beta" in node.attrs.keys() else 1.0
    return [alpha * math_ops.matmul(x, y) + beta * z]

  @classmethod
  def handle_global_average_pool(cls, node, input_dict):
    x = input_dict[node.inputs[0]]
    dims = math_ops.range(array_ops.rank(x))
    _, dim_window = array_ops.split(dims, [2, array_ops.size(dims) - 2])
    return [math_ops.reduce_mean(x, axis=dim_window, keep_dims=True)]

  @classmethod
  def handle_global_max_pool(cls, node, input_dict):
    x = input_dict[node.inputs[0]]
    dims = math_ops.range(array_ops.rank(x))
    _, dim_window = array_ops.split(dims, [2, array_ops.size(dims) - 2])
    return [math_ops.reduce_max(x, axis=dim_window, keep_dims=True)]

  @classmethod
  def handle_l_r_n(cls, node, input_dict):
    x = input_dict[node.inputs[0]]
    alpha = node.attrs["alpha"]
    beta = node.attrs["beta"]
    bias = node.attrs["bias"]
    size = node.attrs["size"]
    tf_alpha = alpha / size
    depth_radius = np.floor([(size - 1) / 2.0])[0]
    # TODO: LRN in tf accepts radius
    # but in ONNX/Caffe accepts diameter.
    # This could be a problem.
    x_t = array_ops.transpose(x, perm=[0, 2, 3, 1])
    normed = gen_nn_ops.lrn(x_t, depth_radius=depth_radius,
                            bias=bias, alpha=tf_alpha, beta=beta)
    normed = array_ops.transpose(normed, perm=[0, 3, 1, 2])
    return [normed]

  @classmethod
  def handle_leaky_relu(cls, node, input_dict):
    x = input_dict[node.inputs[0]]
    if not "alpha" in node.attrs.keys():
      warnings.warn("Provide an alpha value.", UserWarning)
      alpha = 1.0
    else:
      alpha = node.attrs["alpha"]
    tf_op = gen_nn_ops.relu(x) - alpha * gen_nn_ops.relu(-x)
    return [tf_op]

  @classmethod
  def handle_max(cls, node, input_dict):
    values = [input_dict[a] for a in node.inputs]
    return [math_ops.reduce_max(array_ops.stack(values), axis=0)]

  @classmethod
  def handle_max_pool(cls, node, input_dict):
    # 1 = pad manually
    return cls._pool(node, input_dict, nn_ops.max_pool, 1)

  @classmethod
  def handle_min(cls, node, input_dict):
    values = [input_dict[a] for a in node.inputs]
    return [math_ops.reduce_min(array_ops.stack(values), axis=0)]

  @classmethod
  def handle_mul(cls, node, input_dict):
    return [cls._bin_op(node, input_dict, math_ops.multiply)]

  # #TODO: better support optimized rnn
  # @classmethod
  # def handle_optimized_r_n_n(cls, node, input_dict):
  #   if "direction" in node.attrs.keys():
  #     direction = node.attrs["direction"]
  #   else:
  #     direction = 1
  #   if "num_layers" in node.attrs.keys():
  #     num_layers = node.attrs["num_layers"]
  #   else:
  #     num_layers = 1
  #   if "skip_input_transform" in node.attrs.keys():
  #     warnings.warn("We currently do not support skipping input transformation.")

  #   hidden_size = node.attrs["hidden_size"]
  #   if node.attrs["cell_type"] == "relu":
  #     relu_layer = tf.contrib.rnn.BasicCell(hidden_size, activation=gen_nn_ops.relu)
  #     cell = tf.contrib.rnn.MultiRNNCell([relu_layer] * num_layers)
  #   elif node.attrs["cell_type"] == "tanh":
  #     tanh_layer = tf.contrib.rnn.BasicCell(hidden_size)
  #     cell = tf.contrib.rnn.MultiRNNCell([tanh_layer] * num_layers)
  #   elif node.attrs["cell_type"] == "gru":
  #     gru_layer = tf.contrib.rnn.GRUCell(hidden_size)
  #     cell = tf.contrib.rnn.MultiRNNCell([gru_layer] * num_layers)
  #   elif node.attrs["cell_type"] == "lstm":
  #     lstm_layer = tf.contrib.rnn.LSTMCell(hidden_size)
  #     cell = tf.contrib.rnn.MultiRNNCell([lstm_layer] * num_layers)
  #   else:
  #     raise RuntimeError("unexpected cell type")

  #   warnings.warn("Initial weight, hidden/cell states will be ignored for now.")
  #   # TODO: handle data types
  #   if direction == 1:
  #     output, state = tf.nn.dynamic_rnn(cell,
  #                                       input_dict[node.inputs[1]],
  #                                       time_major=True,
  #                                       dtype=tf.float32)
  #   else:
  #     output, state = tf.nn.bidirectional_dynamic_rnn(cell,
  #                                                     input_dict[node.inputs[1]],
  #                                                     time_major=True,
  #                                                     dtype=tf.float32)

  #   if node.attrs["cell_type"] == "lstm":
  #     state = state[0]
  #     c, h = state
  #     states = [h, c]
  #   else:
  #     states = [state]
  #   outputs = [output]
  #   outputs.extend(states)
  #   return outputs

  @classmethod
  def handle_p_relu(cls, node, input_dict):
    """
    Reference implementation at
    https://github.com/tflearn/tflearn/blob/4ba8c8d78bf1bbdfc595bf547bad30580cb4c20b/tflearn/activations.py#L191
    """
    x = input_dict[node.inputs[0]]
    slope = input_dict[node.inputs[1]]
    pos = gen_nn_ops.relu(x)
    neg = slope * (x - abs(x)) * 0.5
    return [pos + neg]

  @classmethod
  def handle_pad(cls, node, input_dict):
    num_dim = int(len(node.attrs["paddings"])/2)
    mode = node.attrs["mode"]

    def _compatibility_edge_pad(x, pads):
      x = np.pad(x, pads, mode="edge")
      return x

    value = node.attrs.get("value", 0)
    padding = constant_op.constant(np.array(node.attrs["paddings"])
                                   .reshape([num_dim, 2])
                                   .astype(np.int32)) # tf requires int32 paddings

    x = input_dict[node.inputs[0]]
    if mode.lower() == "edge":
      return [script_ops.py_func(_compatibility_edge_pad, [x, padding], x.dtype)]

    return [array_ops.pad(input_dict[node.inputs[0]], padding, mode, None, value)]

  @classmethod
  def handle_random_normal_like(cls, node, input_dict):
    shape = array_ops.shape(input_dict[node.inputs[0]])
    mean = node.attrs["mean"]
    stddev = node.attrs["scale"]
    dtype = cls.tensor_type_to_tf_type[node.attrs["dtype"]]
    seed = node.attrs["seed"] if "seed" in node.attrs.keys() else None
    return [random_ops.random_normal(shape, mean, stddev, dtype, seed)]

  @classmethod
  def handle_random_uniform_like(cls, node, input_dict):
    shape = array_ops.shape(input_dict[node.inputs[0]])
    minval = node.attrs["low"]
    maxval = node.attrs["high"]
    dtype = cls.tensor_type_to_tf_type[node.attrs["dtype"]]
    seed = node.attrs["seed"] if "seed" in node.attrs.keys() else None
    return [random_ops.random_uniform(shape, minval, maxval, dtype, seed)]

  @classmethod
  def handle_reshape(cls, node, input_dict):
    tensor = input_dict[node.inputs[0]]
    shape = constant_op.constant(node.attrs["shape"])
    return [gen_array_ops.reshape(tensor, shape)]

  @classmethod
  def handle_selu(cls, node, input_dict):
    warnings.warn("Definition of Selu is different "
                  "between onnx and tensorflow.", UserWarning)
    return [gen_nn_ops.selu(input_dict[node.inputs[0]])]

  @classmethod
  def handle_slice(cls, node, input_dict):
    x = input_dict[node.inputs[0]]

    full_sizes = x.get_shape().as_list()
    full_begin = [0] * len(full_sizes)

    starts = node.attrs.get("starts")
    ends = node.attrs.get("ends")
    slice_len = len(starts)
    axes = node.attrs.get("axes", list(range(slice_len)))

    for i in range(slice_len):
      ends[i] = full_sizes[axes[i]] + ends[i] \
                if ends[i] < 0 else ends[i]
      full_sizes[axes[i]] = ends[i] - starts[i]
      full_begin[axes[i]] = starts[i]

    return [array_ops.slice(input_dict[node.inputs[0]],
                            constant_op.constant(full_begin),
                            constant_op.constant(full_sizes))]

  @classmethod
  def handle_softmax(cls, node, input_dict):
    if "axis" in node.attrs:
      axis = node.attrs["axis"]
    else:
      axis = 1
    return [nn_ops.softmax(input_dict[node.inputs[0]], dim=axis)]

  @classmethod
  def handle_split(cls, node, input_dict):
    split = constant_op.constant(node.attrs["split"]) \
      if "split" in node.attrs else \
      input_dict[node.inputs[1]]
    axis = node.attrs["axis"]
    # return value is naturally a list
    return array_ops.split(input_dict[node.inputs[0]], split, axis)

  @classmethod
  def handle_sub(cls, node, input_dict):
    return [cls._bin_op(node, input_dict, math_ops.subtract)]

  @classmethod
  def handle_sum(cls, node, input_dict):
    values = [input_dict[a] for a in node.inputs]
    return [math_ops.reduce_sum(array_ops.stack(values), axis=0)]

  @classmethod
  def handle_mat_mul(cls, node, input_dict):
    return [math_ops.matmul(input_dict[node.inputs[0]],
                            input_dict[node.inputs[1]])]

  @classmethod
  def supports_device(cls, device):
    if device == "CUDA":
      local_device_protos = device_lib.list_local_devices()
      return len([x.name for x in \
        local_device_protos if x.device_type == 'GPU']) > 0
    elif device == "CPU":
      return True
    return False

prepare = TensorflowBackend.prepare

supports_device = TensorflowBackend.supports_device
