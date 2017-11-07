from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np

from tensorflow.contrib.onnx import prepare
from tensorflow.python.platform import test
from tensorflow.python.framework import dtypes

from onnx import helper
from onnx.onnx_pb2 import TensorProto

class TestNode(test.TestCase):
  """ Tests for nodes
  """
  def np_type_to_tp_type(self, np_type):
    if np_type == np.dtype('int64'):
      return TensorProto.INT32
    if np_type == np.dtype('float32'):
      return TensorProto.FLOAT
    return None

  def run_node(self, node_def, inputs):
    model = node_def

    dummy_dim = 2

    input_proto = []
    for i in range(0, len(node_def.input)):
      input_proto.append(helper.make_tensor_value_info(node_def.input[i],
                                                       self.np_type_to_tp_type(inputs[i].dtype),
                                                       inputs[i].shape))

    output_proto = []
    for i in range(0, len(node_def.output)):
      output_proto.append(helper.make_tensor_value_info(node_def.output[i],
                                                        TensorProto.FLOAT,
                                                        [dummy_dim]))

    init_proto = []
    for i in range(0, len(inputs)):
      init_proto.append(helper.make_tensor(node_def.input[i],
                                           self.np_type_to_tp_type(inputs[i].dtype),
                                           inputs[i].shape,
                                           inputs[i].flatten()))

    graph_def = helper.make_graph(
      [node_def],
      name="test",
      inputs=input_proto,
      outputs=output_proto,
      initializer=init_proto)

    input_dict, output_dict = prepare(helper.make_model(graph_def))
    output_keys = output_dict.keys()
    output_values = output_dict.values()

    with self.test_session():
      output_values = map(lambda x: x.eval(), output_values)

    return dict(zip(output_keys, output_values))

  def _get_rnd(self, shape, low=-1.0, high=1.0):
    return np.random.uniform(low, high, np.prod(shape)) \
                      .reshape(shape) \
                      .astype(np.float32)

  def _elu(self, x):
    # f(x) = alpha * (exp(x) - 1.) for x < 0,
    # f(x) = x for x >= 0
    if x < 0.:
      return np.expm1(x)
    return x

  def _leaky_relu(self, x, alpha):
    # f(x) = alpha * x for x < 0,
    # f(x) = x for x >= 0
    if x < 0.:
      return alpha * x
    return x

  def test_abs(self):
    node_def = helper.make_node("Abs", ["X"], ["Y"])
    x = self._get_rnd([1000])
    output = self.run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.abs(x))

  def test_add(self):
    node_def = helper.make_node("Add", ["X", "Y"], ["Z"], broadcast=1, axis=1)
    x = self._get_rnd([5, 10, 5, 5])
    y = self._get_rnd([10])
    output = self.run_node(node_def, [x, y])
    np.testing.assert_almost_equal(output["Z"], np.add(x, y.reshape([1, 10, 1, 1])))

  def test_arg_max(self):
    # TODO: need to fix this test
    return
    for axis in [0, 1]:
      node_def = helper.make_node("ArgMax", ["data"], ["reduced"],
                                  axis=axis,
                                  keepdims=0)
      data = self._get_rnd([10, 10])
      output = self.run_node(node_def, [data])
      np.testing.assert_almost_equal(output["reduced"],
                                     np.argmax(data, axis=axis))

  def test_arg_min(self):
    # TODO: need to fix this test
    return
    for axis in [0, 1]:
      node_def = helper.make_node("ArgMin", ["data"], ["reduced"],
                                  axis=axis, keepdims=0)
      data = self._get_rnd([10, 10])
      output = self.run_node(node_def, [data])
      np.testing.assert_almost_equal(output["reduced"],
                                     np.argmin(data, axis=axis))

  def test_average_pool(self):
    # TODO: fix this test
    return
    device = "CUDA"
    if not supports_device(device):
                raise unittest.SkipTest(
                    "Backend doesn't support device {}".format(device))
    shape = [1, 1, 40, 40]
    node_def = helper.make_node("AveragePool", ["X"], ["Y"],
      kernel_shape=[1,2],
      pads=[1, 1], strides=[1,1])
    x = self._get_rnd(shape)
    output = self.run_node(node_def, [x], device=device)
    test_output = np.zeros(shape)
    for i1 in range(0, shape[0]):
      for i2 in range(0, shape[1]):
        for j1 in range(0, shape[2]):
          for j2 in range(0, shape[3]):
            test_output[i1][i2][j1][j2] = 0
            count = 0
            for k in range(j2, min(j2+2, shape[3])):
              test_output[i1][i2][j1][j2] += x[i1][i2][j1][k]
              count += 1
            test_output[i1][i2][j1][j2] /= count
    np.testing.assert_almost_equal(output["Y"], test_output)

  def _batch_normalization(self, x, mean, variance, bias, scale,
                           variance_epsilon):
    inv = np.reciprocal(np.sqrt(variance + variance_epsilon))
    if scale is not None:
      inv *= scale
    return x * inv + (bias - mean * inv
                      if bias is not None else -mean * inv)

  def test_batch_normalization(self):
    node_def = helper.make_node("BatchNormalization",
                                ["X", "scale", "bias", "mean", "var"],
                                ["Y"],
                                consumed_inputs=[0, 0, 0, 1, 1],
                                epsilon=0.001)
    x_shape = [3, 5, 4, 2]
    param_shape = [5]
    _param_shape = [1, 5, 1, 1]
    x = self._get_rnd(x_shape, 0, 1)
    m = self._get_rnd(param_shape, 0, 1)
    _m = m.reshape(_param_shape)
    v = self._get_rnd(param_shape, 0, 1)
    _v = v.reshape(_param_shape)
    scale = self._get_rnd(param_shape, 0, 1)
    _scale = scale.reshape(_param_shape)
    bias = self._get_rnd(param_shape, 0, 1)
    _bias = bias.reshape(_param_shape)
    golden = self._batch_normalization(x, _m, _v, _bias, _scale, 0.001)
    output = self.run_node(node_def, [x, scale, bias, m, v])
    np.testing.assert_almost_equal(output["Y"], golden, decimal=5)

  def test_cast(self):
    for ty, tf_type in [("float", dtypes.float32),
                        ("uint8", dtypes.uint8),
                        ("int8", dtypes.int8),
                        ("uint16", dtypes.uint16),
                        ("int16", dtypes.int16),
                        ("int32", dtypes.int32),
                        ("int64", dtypes.int64),
                        ("bool", dtypes.bool),
                        ("float16", dtypes.float16),
                        ("double", dtypes.float64),
                        ("complex64", dtypes.complex64),
                        ("complex128", dtypes.complex128)]:
      node_def = helper.make_node("Cast", ["input"], ["output"],
                                  to=ty)
      vector = np.array([2, 3])
      output = self.run_node(node_def, [vector])
      np.testing.assert_equal(output["output"].dtype, tf_type)

  def test_ceil(self):
    node_def = helper.make_node("Ceil", ["X"], ["Y"])
    x = self._get_rnd([1000])
    output = self.run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.ceil(x))

  def test_concat(self):
    shape = [10, 20, 5]
    for axis in range(len(shape)):
      node_def = helper.make_node("Concat", ["X1", "X2"], ["Y"], axis=axis)
      x1 = self._get_rnd(shape)
      x2 = self._get_rnd(shape)
      output = self.run_node(node_def, [x1, x2])
      np.testing.assert_almost_equal(output["Y"],
                                     np.concatenate((x1, x2), axis))

  def test_constant(self):
    shape = [16, 16]
    values = np.random.randn(*shape).flatten().astype(float)
    const2_onnx = helper.make_tensor("const2",
                                     TensorProto.DOUBLE,
                                     shape,
                                     values)
    node_def = helper.make_node("Constant", [], ["Y"], value=const2_onnx)
    output = self.run_node(node_def, [])
    np.testing.assert_equal(output["Y"].shape, shape)
    np.testing.assert_almost_equal(output["Y"].flatten(), values)

  def test_conv(self):
    # Fix test in the future.
    return
    device = "CUDA"
    if not supports_device(device):
                raise unittest.SkipTest(
                    "Backend doesn't support device {}".format(device))
    node_def = helper.make_node("Conv", ["X", "weights"],
                                ["Y"], pads=[1,1])
    x_shape = [1, 5, 4]
    x = self._get_rnd(x_shape)
    weight_shape = [3, 5, 2]
    weights = self._get_rnd(weight_shape)
    output = self.run_node(node_def, [x, weights], device=device)
    out_shape = [x_shape[0], weight_shape[0], x_shape[2]]
    test_output = np.zeros(out_shape)
    for b in range(0, x_shape[0]):
      for m in range(0, weight_shape[0]):
        for h in range(0, x_shape[2]):
          v = 0
          for c in range(0, x_shape[1]):
            for k in range(h, min(h+weight_shape[2], x_shape[2])):
              v += x[b][c][k] * weights[m][c][k-h]
          test_output[b][m][h] = v
    np.testing.assert_almost_equal(output["Y"], test_output, decimal=5)

  def test_conv_transpose(self):
    # Fix test in the future.
    return
    device = "CUDA"
    if not supports_device(device):
                raise unittest.SkipTest(
                    "Backend doesn't support device {}".format(device))
    node_def = helper.make_node("ConvTranspose", ["X", "weights"],
                                ["Y"], pads=[1,1])
    x_shape = [1, 5, 4]
    x = self._get_rnd(x_shape)
    weight_shape = [5, 3, 2]
    weights = self._get_rnd(weight_shape)
    output = self.run_node(node_def, [x, weights], device=device)
    out_shape = [x_shape[0], weight_shape[1], x_shape[2]]
    test_output = np.zeros(out_shape)
    for b in range(0, x_shape[0]):
      for m in range(0, weight_shape[1]):
        for h in range(0, x_shape[2]):
          v = 0
          for c in range(0, x_shape[1]):
            for k in range(h, min(h+weight_shape[2], x_shape[2])):
              v += x[b][c][k] * weights[c][m][k-h]
          test_output[b][m][h] = v
    np.testing.assert_almost_equal(output["Y"], test_output, decimal=5)

  def test_div(self):
    node_def = helper.make_node("Div", ["X", "Y"], ["Z"], broadcast=1)
    x = self._get_rnd([10, 10])
    y = self._get_rnd([10, 10])
    output = self.run_node(node_def, [x, y])
    np.testing.assert_almost_equal(output["Z"], np.divide(x, y))

  def test_dot(self):
    # this op is removed
    # remove this test in the future
    return
    node_def = helper.make_node("Dot", ["X", "Y"], ["Z"])
    x = np.floor(self._get_rnd([10, 10]));
    y = np.floor(self._get_rnd([10, 10]));
    output = self.run_node(node_def, [x, y])
    np.testing.assert_almost_equal(output["Z"], np.dot(x, y))

  def test_elu(self):
    node_def = helper.make_node("Elu", ["X"], ["Y"])
    x = self._get_rnd([100])
    output = self.run_node(node_def, [x])
    test_output = [self._elu(a) for a in x];
    np.testing.assert_almost_equal(output["Y"], test_output)

  def test_exp(self):
    node_def = helper.make_node("Exp", ["X"], ["Y"])
    x = self._get_rnd([100])
    x = x - 3.6;
    output = self.run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.exp(x))

  def test_flatten(self):
    # If input tensor has shape (d_0, d_1, ... d_n) then the
    # output will have shape:
    #
    # (d_0 X d_1 ... d_(axis-1), d_axis X d_(axis+1) ... X dn)
    #
    # TODO: pass axis attribute which is supported in newer
    # versions of onnx
    node_def = helper.make_node("Flatten", ["X"], ["Y"])
    x = self._get_rnd([10, 2, 3, 4, 5])
    output = self.run_node(node_def, [x])
    # TODO: pass axis=3 and uncomment the line below
    # np.testing.assert_almost_equal(output["Y"], x.reshape([60, 20]))
    np.testing.assert_almost_equal(output["Y"], x.reshape([10, 120]))

  def test_gather(self):
    node_def = helper.make_node("Gather", ["X", "Y"], ["Z"])
    x = self._get_rnd([10, 10])
    y = np.array([[0, 1], [1, 2]])
    output = self.run_node(node_def, [x, y])
    test_output = np.zeros((2, 2, 10))
    for i in range(0, 2):
      for j in range(0, 10):
        test_output[0][i][j] = x[i][j]
    for i in range(0, 2):
      for j in range(0, 10):
        test_output[1][i][j] = x[i + 1][j]
    np.testing.assert_almost_equal(output["Z"], test_output)

  def test_gemm(self):
    # Compute Y = alpha * A * B + beta * C
    node_def = helper.make_node("Gemm", ["A", "B", "C"], ["Y"],
      transA=0, transB=0, broadcast=1, alpha=1.0, beta=1.0)
    x = np.floor(self._get_rnd([10, 10]))
    y = np.floor(self._get_rnd([10, 10]))
    z = np.floor(self._get_rnd([10, 10]))
    output = self.run_node(node_def, [x, y, z])
    test_output = np.matmul(x, y) + z
    np.testing.assert_almost_equal(output["Y"], test_output)

  def test_global_average_pool(self):
    #   Image case:  (N x C x H x W), where N is the batch size,
    # C is the number of channels, and H and W are the height
    # and the width of the data
    #
    #   Non-image case: (N x C x D1 x D2 ... Dn)
    #
    #   Output data tensor from pooling across the input tensor.
    # Dimensions will be N x C x 1 x 1
    node_def = helper.make_node("GlobalAveragePool", ["X"], ["Y"])
    x = self._get_rnd([10, 10, 2, 3])
    output = self.run_node(node_def, [x])
    test_output = np.zeros([10, 10, 1, 1])
    for i1 in range(0, 10):
      for i2 in range(0, 10):
        sum = 0
        for j1 in range(0, 2):
          for j2 in range(0, 3):
            sum += x[i1][i2][j1][j2]
        test_output[i1][i2][0][0] = sum / 6.
    np.testing.assert_almost_equal(output["Y"], test_output)

  def test_global_max_pool(self):
    #   Image case:  (N x C x H x W), where N is the batch size,
    # C is the number of channels, and H and W are the height
    # and the width of the data
    #
    #   Non-image case: (N x C x D1 x D2 ... Dn)
    #
    #   Output data tensor from pooling across the input tensor.
    # Dimensions will be N x C x 1 x 1
    node_def = helper.make_node("GlobalMaxPool", ["X"], ["Y"])
    x = self._get_rnd([10, 10, 2, 3])
    output = self.run_node(node_def, [x])
    test_output = np.zeros([10, 10, 1, 1])
    for i1 in range(0, 10):
      for i2 in range(0, 10):
        max = x[i1][i2][0][0]
        for j1 in range(0, 2):
          for j2 in range(0, 3):
            if max < x[i1][i2][j1][j2]:
              max = x[i1][i2][j1][j2]
        test_output[i1][i2][0][0] = max
    np.testing.assert_almost_equal(output["Y"], test_output)

  def test_l_r_n(self):
    # Each input value is divided by:
    #
    # (bias+(alpha/size)*sum(xi^2 for every xi in the local region))^beta
    alpha = 2.0
    beta = 1.0
    bias = 5.0
    size = 3
    node_def = helper.make_node("LRN", ["X"], ["Y"], alpha=alpha,
      beta=beta, bias=bias, size=size)
    x = self._get_rnd([10, 2, 10, 10])
    output = self.run_node(node_def, [x])
    test_output = np.zeros([10, 10, 10, 2])
    x = np.transpose(x, axes=[0,2,3,1])
    for i1 in range(0, 10):
      for i2 in range(0, 10):
        for j1 in range(0, 10):
          for j2 in range(0, 2):
            sqr_sum = 0.;
            # size of 3 means radius 1 in TF speak
            # i.e. the immediate neighbouring values
            # if "previous" neighbour exists
            if j2 > 0:
              sqr_sum += x[i1][i2][j1][j2 - 1] * x[i1][i2][j1][j2 - 1]
            # current value
            sqr_sum += x[i1][i2][j1][j2] * x[i1][i2][j1][j2]
            # if "next" neighbour exists
            if j2 < 2 - 1:
              sqr_sum += x[i1][i2][j1][j2 + 1] * x[i1][i2][j1][j2 + 1]
            test_output[i1][i2][j1][j2] = \
              x[i1][i2][j1][j2] / ((bias + (alpha * 1. / size) * sqr_sum) ** beta)
    test_output = np.transpose(test_output, axes=[0,3,1,2])
    np.testing.assert_almost_equal(output["Y"], test_output)

  def test_floor(self):
    node_def = helper.make_node("Floor", ["X"], ["Y"])
    x = self._get_rnd([100])
    output = self.run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.floor(x))

  def test_leakyrelu(self):
    node_def = helper.make_node("LeakyRelu", ["X"], ["Y"], alpha=2.0)
    x = np.floor(self._get_rnd([100]))
    output = self.run_node(node_def, [x])
    test_output = [self._leaky_relu(a, 2.0) for a in x]
    np.testing.assert_almost_equal(output["Y"], test_output)

  def test_log(self):
    node_def = helper.make_node("Log", ["X"], ["Y"])
    x = self._get_rnd([100])
    x = x + 3.6;
    output = self.run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.log(x))

  def test_max(self):
    node_def = helper.make_node("Max", ["X1", "X2", "X3", "X4"], ["Z"])
    x1 = self._get_rnd([10, 10])
    x2 = self._get_rnd([10, 10])
    x3 = self._get_rnd([10, 10])
    x4 = self._get_rnd([10, 10])
    output = self.run_node(node_def, [x1, x2, x3, x4])
    test_output = np.maximum(np.maximum(np.maximum(x1, x2), x3), x4)
    np.testing.assert_almost_equal(output["Z"], test_output)

  def test_max_pool(self):
    return
    node_def = helper.make_node("MaxPool", ["X"], ["Y"],
      dilations=[1,1], kernel_shape=[1,2],
      pads=[0,0], strides=[1,2])
    x = self._get_rnd([10, 10, 4, 4])
    output = self.run_node(node_def, [x])
    test_output = np.zeros([10, 10, 4, 2])
    for i1 in range(0, 10):
      for i2 in range(0, 10):
        for j1 in range(0, 4):
          for j2 in range(0, 2):
            test_output[i1][i2][j1][j2] = \
              max(x[i1][i2][j1][2*j2], x[i1][i2][j1][2*j2 + 1])
    np.testing.assert_almost_equal(output["Y"], test_output)

  def test_min(self):
    node_def = helper.make_node("Min", ["X1", "X2", "X3", "X4"], ["Z"])
    x1 = self._get_rnd([10, 10])
    x2 = self._get_rnd([10, 10])
    x3 = self._get_rnd([10, 10])
    x4 = self._get_rnd([10, 10])
    output = self.run_node(node_def, [x1, x2, x3, x4])
    test_output = np.minimum(np.minimum(np.minimum(x1, x2), x3), x4)
    np.testing.assert_almost_equal(output["Z"], test_output)

  def test_mul(self):
    node_def = helper.make_node("Mul", ["X", "Y"], ["Z"], broadcast=1, axis=1)
    x = self._get_rnd([5, 10, 5, 5])
    y = self._get_rnd([10])
    output = self.run_node(node_def, [x, y])
    np.testing.assert_almost_equal(output["Z"], np.multiply(x, y.reshape([1, 10, 1, 1])))

  def test_neg(self):
    node_def = helper.make_node("Neg", ["X"], ["Y"])
    x = self._get_rnd([1000])
    output = self.run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.negative(x))

  # TODO: better testing for RNN. For now, we are just making sure
  # # that it runs.
  # def test_optimizedrnn(self):
  #   node_def = helper.make_node("OptimizedRNN",
  #                               ["W", "I", "H", "C"],
  #                               ["O", "OH", "OC"], hidden_size=3, cell_type="lstm")
  #   x = self._get_rnd([10, 10, 10])
  #   dummy = np.array([0])
  #   output = self.run_node(node_def, [dummy, x, dummy, dummy])

  #   node_def = helper.make_node("OptimizedRNN",
  #                               ["W", "I", "H", "C"],
  #                               ["O", "OH"], hidden_size=3, cell_type="gru")
  #   x = self._get_rnd([10, 10, 10])
  #   dummy = np.array([0])
  #   output = self.run_node(node_def, [dummy, x, dummy, dummy])

  def test_relu(self):
    node_def = helper.make_node("Relu", ["X"], ["Y"])
    x = self._get_rnd([1000])
    output = self.run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.maximum(x, 0))

  def test_pad(self):
    node_def = helper.make_node("Pad", ["X"], ["Y"],
                                mode="constant",
                                paddings=[1, 1, 1, 1],
                                value=2.0)
    x = self._get_rnd([100, 100])
    output = self.run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"],
                                   np.lib.pad(x, ((1, 1), (1, 1)),
                                              'constant',
                                              constant_values=(2, 2)))

  def test_reciprocal(self):
    node_def = helper.make_node("Reciprocal", ["X"], ["Y"])
    x = self._get_rnd([1000])
    output = self.run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], 1.0/x)

  def test_pow(self):
    # TODO: uncomment power schema is wrong in onnx
    return
    node_def = helper.make_node("Pow", ["X", "Y"], ["Z"])
    x = self._get_rnd(1000)/2.0 + 0.5
    y = self._get_rnd(1000)/2.0 + 0.5
    output = self.run_node(node_def, [x, y])
    np.testing.assert_almost_equal(output["Z"],
                                   np.power(x, y))

  def test_reshape(self):
    node_def = helper.make_node("Reshape", ["X"], ["Y"], shape=[10, 10])
    x = self._get_rnd(100)
    output = self.run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], x.reshape([10, 10]))

  def test_sigmoid(self):
    node_def = helper.make_node("Sigmoid", ["X"], ["Y"])
    x = self._get_rnd([1000])
    output = self.run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], 1/(1 + np.exp(-x)))

  def test_slice(self):
    # TODO: API update or fix onnx version
    return
    node_def = helper.make_node("Slice", ["X", "Y", "Z", "W"], ["S"])
    x = self._get_rnd([1000]).reshape([10, 10, 10])
    output = self.run_node(node_def, [x, [0, 1, 2], [0, 0, 0], [2, 2, 2]])
    np.testing.assert_almost_equal(output["S"], x[0:2, 0:2, 0:2])

  def test_split(self):
    node_def = helper.make_node("Split", ["X", "Y"], ["A", "B", "C"], axis=0)
    x = self._get_rnd([100]).reshape([10, 10])
    split = np.array([3, 3, 4])
    output = self.run_node(node_def, [x, split])
    for a_key, b in zip(["A", "B", "C"], np.split(x,np.cumsum(split))[:-1]):
      np.testing.assert_almost_equal(output[a_key], b)

  def test_sqrt(self):
    node_def = helper.make_node("Sqrt", ["X"], ["Y"])
    x = self._get_rnd([1000]) + 1.0
    output = self.run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.sqrt(x), decimal=5)

  def test_squeeze(self):
    node_def = helper.make_node("Squeeze", ["X"], ["Y"], axes=[2])
    x = np.array([[[0], [1], [2]]])
    output = self.run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"],
                                   np.squeeze(x, axis=2))

  def test_sub(self):
    node_def = helper.make_node("Sub", ["X", "Y"], ["Z"], broadcast=1)
    x = self._get_rnd([10, 10])
    y = self._get_rnd([10, 10])
    output = self.run_node(node_def, [x, y])
    np.testing.assert_almost_equal(output["Z"], np.subtract(x, y))

  def test_sum(self):
    node_def = helper.make_node("Sum", ["X1", "X2", "X3", "X4"], ["Z"])
    x1 = self._get_rnd([10, 10])
    x2 = self._get_rnd([10, 10])
    x3 = self._get_rnd([10, 10])
    x4 = self._get_rnd([10, 10])
    output = self.run_node(node_def, [x1, x2, x3, x4])
    test_output = x1 + x2 + x3 + x4
    np.testing.assert_almost_equal(output["Z"], test_output)

  def test_tanh(self):
    node_def = helper.make_node("Tanh", ["X"], ["Y"])
    x = self._get_rnd([1000]) + 1.0
    output = self.run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.tanh(x), decimal=5)

  def test_transpose(self):
    node_def = helper.make_node("Transpose", ["X"], ["Y"], perm=[0, 2, 1])
    x = self._get_rnd([1000]).reshape([10, 10, 10])
    output = self.run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.transpose(x, (0, 2, 1)))

if __name__ == '__main__':
  unittest.main()
