# Copyright 2019 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import test_utils as tu

from tensorflow.python.platform import googletest
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops


class OneHotTopK(test_util.TensorFlowTestCase):
  def testOneHot(self):
    def executeModel(inputs, expected):

      # Decide what the output type should be.
      data_type = inputs["on"].dtype

      # The actual model function which perfoms the one-hot operation based on the inputs given to executeModel.
      def model(a):
        return array_ops.one_hot(
            a,
            inputs["n_classes"],
            dtype=data_type,
            on_value=inputs["on"],
            off_value=inputs["off"],
            axis=inputs["axis"])

      # We run once on the CPU to get the expected result, then on the IPU to compare the two.
      cpuRun = expected == None

      with ops.device('cpu'):
        pa = array_ops.placeholder(np.int32, inputs["shape"], name="a")
        report = gen_ipu_ops.ipu_event_trace()

      # Check if we should be running on IPU or cpu.
      device = "/device:IPU:0"
      if cpuRun:
        device = "cpu:0"

      with ops.device(device):
        out = model(pa)

      tu.configure_ipu_system()

      with tu.ipu_session() as sess:
        sess.run(report)

        in_data = np.array(inputs["in_values"])

        fd = {pa: in_data}
        result = sess.run(out, fd)

        if cpuRun:
          return result
        else:
          self.assertAllClose(result, expected)

    # Generate a multi dimensional matrix.
    largish_matrix_size = [4, 3, 4, 2, 2]
    largish_matrix_data = np.random.randint(1, np.prod(largish_matrix_size),
                                            largish_matrix_size)

    # Generate a vector as well, as using just the matrix will increase test times unnecessarily
    vector_size = [4, 3, 4, 2, 2]
    vector_data = np.random.randint(1, np.prod(largish_matrix_size),
                                    largish_matrix_size)

    inputs = [
        # Test different dimensions.
        {
            "n_classes": 10,
            "shape": [4],
            "in_values": [1, 2, 3, 4],
            "on": np.float32(2.0),
            "off": np.float32(0.0),
            "axis": -1
        },
        {
            "n_classes": 1200,
            "shape": [4, 2],
            "in_values": [[1, 1], [2, 5], [4, 3], [4, 6]],
            "on": np.float32(1.0),
            "off": np.float32(0.0),
            "axis": -1
        },
        {
            "n_classes": 1200,
            "shape": largish_matrix_size,
            "in_values": largish_matrix_data,
            "on": np.float32(1.0),
            "off": np.float32(0.0),
            "axis": -1
        },
        # Test different depths
        {
            "n_classes": 1,
            "shape": [4],
            "in_values": [1, 2, 3, 4],
            "on": np.float32(1.0),
            "off": np.float32(0.0),
            "axis": -1
        },
        {
            "n_classes": 12000,
            "shape": [4, 2],
            "in_values": [[1, 1], [2, 5], [4, 3], [4, 6]],
            "on": np.float32(1.0),
            "off": np.float32(0.0),
            "axis": -1
        },

        # Test different axes.
        {
            "n_classes": 100,
            "shape": vector_size,
            "in_values": vector_data,
            "on": np.float32(1.0),
            "off": np.float32(0.0),
            "axis": 0
        },
        {
            "n_classes": 1200,
            "shape": largish_matrix_size,
            "in_values": largish_matrix_data,
            "on": np.float32(1.0),
            "off": np.float32(0.0),
            "axis": 3
        },
        {
            "n_classes": 100,
            "shape": largish_matrix_size,
            "in_values": largish_matrix_data,
            "on": np.float32(1.0),
            "off": np.float32(0.0),
            "axis": 2
        },
        # Test different on/off.
        {
            "n_classes": 100,
            "shape": vector_size,
            "in_values": vector_data,
            "on": np.float32(0.25),
            "off": np.float32(0.1),
            "axis": 0
        },
        {
            "n_classes": 100,
            "shape": vector_size,
            "in_values": vector_data,
            "on": np.float32(20.0),
            "off": np.float32(-1.0),
            "axis": 0
        },
        # Float16 is the only data type we will run on assembly so we have specific cases for that.
        {
            "n_classes": 100,
            "shape": vector_size,
            "in_values": vector_data,
            "on": np.float16(1.0),
            "off": np.float16(0.0),
            "axis": 0
        },
        {
            "n_classes": 100,
            "shape": vector_size,
            "in_values": vector_data,
            "on": np.float16(2.0),
            "off": np.float16(3.0),
            "axis": 1
        },

        # Check int32 works as well
        {
            "n_classes": 100,
            "shape": vector_size,
            "in_values": vector_data,
            "on": np.int32(4.0),
            "off": np.int32(2.0),
            "axis": 0
        },
        {
            "n_classes": 100,
            "shape": vector_size,
            "in_values": vector_data,
            "on": np.int32(4.0),
            "off": np.int32(2.0),
            "axis": 1
        },
    ]

    for test_case in inputs:
      # Run on CPU first
      result = executeModel(test_case, None)

      # Run on IPU and test against CPU out.
      executeModel(test_case, result)

  def testTopK(self):

    n_categories = 1200
    topn = 24

    def model(a):
      values, indices = nn.top_k(a, topn)
      return indices

    with ops.device('cpu'):
      pa = array_ops.placeholder(np.float32, [n_categories], name="a")
      report = gen_ipu_ops.ipu_event_trace()

    with ops.device("/device:IPU:0"):
      out = model(pa)

    tu.configure_ipu_system()

    with tu.ipu_session() as sess:
      sess.run(report)

      input = np.random.random(n_categories)
      expected = (-input).argsort()[:topn]

      fd = {pa: input}
      result = sess.run(out, fd)
      self.assertAllClose(result, expected)

      result = sess.run(report)
      self.assertTrue(len(result) == 3)

  def testArgMax(self):

    batchsize = 4
    n_categories = 1200

    def model(a):
      return math_ops.argmax(a, axis=1, output_type=dtypes.int32)

    with ops.device('cpu'):
      pa = array_ops.placeholder(np.float32, [batchsize, n_categories])
      report = gen_ipu_ops.ipu_event_trace()

    with ops.device("/device:IPU:0"):
      out = model(pa)

    tu.configure_ipu_system()

    with tu.ipu_session() as sess:
      sess.run(report)

      input = np.random.rand(batchsize, n_categories)

      fd = {pa: input}
      result = sess.run(out, fd)
      self.assertAllClose(result, np.argmax(input, axis=1))

      result = sess.run(report)
      self.assertTrue(len(result) == 3)

  def testInTopK(self):

    batchsize = 4
    n_categories = 1200
    topn = 8

    def model(a, b):
      return nn.in_top_k(a, b, topn)

    with ops.device('cpu'):
      pa = array_ops.placeholder(np.float32, [batchsize, n_categories])
      pb = array_ops.placeholder(np.int32, [batchsize])
      report = gen_ipu_ops.ipu_event_trace()

    with ops.device("/device:IPU:0"):
      out = model(pa, pb)

    tu.configure_ipu_system()

    with tu.ipu_session() as sess:
      sess.run(report)

      input = np.random.rand(batchsize, n_categories)
      input = input / np.sqrt(np.sum(input**2))

      ref = (-input).argsort(axis=1)[:, :1]
      ref = ref.reshape([batchsize])

      expected = [True] * batchsize

      fd = {pa: input, pb: ref}
      result = sess.run(out, fd)
      self.assertAllClose(result, [True, True, True, True])

      result = sess.run(report)
      self.assertTrue(len(result) == 3)


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = (
      '--tf_xla_min_cluster_size=1 ' + os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()
