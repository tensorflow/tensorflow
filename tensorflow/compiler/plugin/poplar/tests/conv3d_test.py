# Copyright 2018 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import test_utils as tu

from tensorflow.python.platform import googletest
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops


class IpuXlaConvTest(test_util.TensorFlowTestCase):

  def test3DConv1x1x1_Stride2x1x1_In1x1x5(self):
    with ops.device("/device:IPU:0"):
      pa = array_ops.placeholder(np.float32, [1,1,1,5,1], name="a")
      pb = array_ops.placeholder(np.float32, [1,1,1,1,1], name="b")
      output = nn_ops.convolution(pa, pb, strides=[1,1,2], padding="VALID")

    with session_lib.Session() as sess:
      fd = {
        pa: [[[[[1], [2], [3], [4], [5]]]]],
        pb: [[[[[10]]]]]
      }
      result = sess.run(output, fd)
      self.assertAllClose(result, [[[[[10], [30], [50]]]]])

  def test3DConv3x3x3_Pad1x1x1(self):
    with ops.device("/device:IPU:0"):
      pa = array_ops.placeholder(np.float32, [1,14,14,14,64], name="a")
      pb = array_ops.placeholder(np.float32, [3,3,3,64,128], name="b")
      output = nn_ops.convolution(pa, pb, padding="SAME")

    with session_lib.Session() as sess:
      fd = {
        pa: np.zeros([1,14,14,14,64]),
        pb: np.zeros([3,3,3,64,128])
      }
      result = sess.run(output, fd)
      self.assertAllClose(result, np.zeros([1,14,14,14,128]))

  def test3DConv3x3x3_WithBias(self):
    with ops.device("/device:IPU:0"):
      pa = array_ops.placeholder(np.float32, [1,14,14,14,16], name="a")
      pb = array_ops.placeholder(np.float32, [3,3,3,16,32], name="b")
      bi = array_ops.placeholder(np.float32, [32], name="b")
      output = nn_ops.convolution(pa, pb, padding="SAME")
      output = nn_ops.bias_add(output, bi)

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    with tu.ipu_session() as sess:
      sess.run(report)

      fd = {
        pa: np.zeros([1,14,14,14,16]),
        pb: np.zeros([3,3,3,16,32]),
        bi: np.zeros([32]),
      }
      result = sess.run(output, fd)
      self.assertAllClose(result, np.zeros([1,14,14,14,32]))

      result = sess.run(report)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      ok = ['host-exchange-local-copy-',
            'Copy_',
            'convolution/convolution.*/Conv_3x3x3',
            'BiasAdd/fusion/addToChannel']
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

  def test3DConv8x8x8_WithBias(self):
    with ops.device("/device:IPU:0"):
      inp = array_ops.placeholder(np.float32, [1,84,84,84,2], name="inp")
      wei = array_ops.placeholder(np.float32, [8,8,8,2,4], name="wei")
      bia = array_ops.placeholder(np.float32, [4], name="bia")
      output = nn_ops.conv3d(inp, wei, strides=[1,4,4,4,1], padding="VALID")
      output = nn_ops.bias_add(output, bia)

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    with tu.ipu_session() as sess:
      sess.run(report)

      fd = {
        inp: np.zeros([1,84,84,84,2]),
        wei: np.zeros([8,8,8,2,4]),
        bia: np.zeros([4]),
      }
      result = sess.run(output, fd)
      self.assertAllClose(result, np.zeros([1, 20, 20, 20, 4]))

      result = sess.run(report)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      ok = ['host-exchange-local-copy-',
            'Copy_',
            'Conv3D/convolution.*/Conv_8x8x8_stride4x4x4',
            'BiasAdd/fusion/addToChannel']
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

  def test3DConv1x1x1_WithBias(self):
    with ops.device("/device:IPU:0"):
      inp = array_ops.placeholder(np.float32, [1,1,1,1,4], name="inp")
      wei = array_ops.placeholder(np.float32, [1,1,1,4,8], name="wei")
      bia = array_ops.placeholder(np.float32, [8], name="bia")
      output = nn_ops.conv3d(inp, wei, strides=[1,1,1,1,1], padding="VALID")
      output = output + bia

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    with tu.ipu_session() as sess:
      sess.run(report)

      fd = {
        inp: np.zeros([1,1,1,1,4]),
        wei: np.zeros([1,1,1,4,8]),
        bia: np.zeros([8]),
      }
      result = sess.run(output, fd)
      self.assertAllClose(result, np.zeros([1, 1, 1, 1, 8]))

      result = sess.run(report)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      ok = ['Copy_*actsRearranged',
            'host-exchange-local-copy-',
            'Conv3D/convolution.*/Conv_1x1',
            'add/fusion/addToChannel']
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

  def test3DConvBackpropInput(self):
    with ops.device("/device:IPU:0"):
      ins = constant_op.constant([2,8,8,8,3], np.int32)
      fil = array_ops.placeholder(np.float32, [2,2,2,3,5], name="inp")
      bck = array_ops.placeholder(np.float32, [2,8,8,8,5], name="wei")

      output = nn_ops.conv3d_backprop_input_v2(ins, fil, bck, strides=[1,1,1,1,1],
                                            padding="SAME")

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    with tu.ipu_session() as sess:
      sess.run(report)

      fd = {
        fil: np.zeros([2,2,2,3,5]),
        bck: np.zeros([2,8,8,8,5]),
      }
      result = sess.run(output, fd)
      self.assertAllClose(result, np.zeros([2,8,8,8,3]))

      result = sess.run(report)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      ok = ['Copy_',
            'Conv3DBackpropInputV2/fusion*/WeightTranspose',
            'Conv3DBackpropInputV2/fusion*/Conv_2x2x2']

      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

  def test3DConvBackpropFilter(self):
    with ops.device("/device:IPU:0"):
      inp = array_ops.placeholder(np.float32, [2,8,8,8,3])
      fil = constant_op.constant([2,2,2,3,5], np.int32)
      bck = array_ops.placeholder(np.float32, [2,8,8,8,5], name="wei")

      output = nn_ops.conv3d_backprop_filter_v2(inp, fil, bck, strides=[1,1,1,1,1],
                                            padding="SAME")

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    with tu.ipu_session() as sess:
      sess.run(report)

      fd = {
        inp: np.zeros([2,8,8,8,3]),
        bck: np.zeros([2,8,8,8,5]),
      }
      result = sess.run(output, fd)
      self.assertAllClose(result, np.zeros([2,2,2,3,5]))

      result = sess.run(report)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      ok = ['host-exchange-local-copy-',
            'Copy_',
            'Conv3DBackpropFilterV2/convolution.*/Conv_8x8x8']
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

if __name__ == "__main__":
    googletest.main()
