# Copyright 2017 Graphcore Ltd
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
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops


class IpuXlaConvTest(test_util.TensorFlowTestCase):

  def testConv1x1_Stride2x1_In1x5(self):
    with ops.device("/device:IPU:0"):
      pa = array_ops.placeholder(np.float32, [1,1,5,1], name="a")
      pb = array_ops.placeholder(np.float32, [1,1,1,1], name="b")
      output = nn_ops.convolution(pa, pb, strides=[1,2], padding="VALID")

    with session_lib.Session() as sess:
      fd = {
        pa: [[[[1], [2], [3], [4], [5]]]],
        pb: [[[[10]]]]
      }
      result = sess.run(output, fd)
      self.assertAllClose(result, [[[[10], [30], [50]]]])

  def testConv3x3_Pad1x1(self):
    with ops.device("/device:IPU:0"):
      pa = array_ops.placeholder(np.float32, [1,14,14,64], name="a")
      pb = array_ops.placeholder(np.float32, [3,3,64,128], name="b")
      output = nn_ops.convolution(pa, pb, padding="SAME")

    with session_lib.Session() as sess:
      fd = {
        pa: np.zeros([1,14,14,64]),
        pb: np.zeros([3,3,64,128])
      }
      result = sess.run(output, fd)
      self.assertAllClose(result, np.zeros([1,14,14,128]))

  def testConv3x3_WithBias(self):
    with ops.device("/device:IPU:0"):
      pa = array_ops.placeholder(np.float32, [1,14,14,64], name="a")
      pb = array_ops.placeholder(np.float32, [3,3,64,128], name="b")
      bi = array_ops.placeholder(np.float32, [128], name="b")
      output = nn_ops.convolution(pa, pb, padding="SAME")
      output = nn_ops.bias_add(output, bi)

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    with tu.ipu_session() as sess:
      sess.run(report)

      fd = {
        pa: np.zeros([1,14,14,64]),
        pb: np.zeros([3,3,64,128]),
        bi: np.zeros([128]),
      }
      result = sess.run(output, fd)
      self.assertAllClose(result, np.zeros([1,14,14,128]))

      result = sess.run(report)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      ok = ['convolution/convolution.*clone/Conv_3x3',
            'Copy_{<const>,XLA_Args/arg0.*_input}',
            'BiasAdd/call/addToChannel']
      self.assertTrue(tu.check_all_compute_sets_in_list(cs_list, ok))

  def testConv8x8_WithBias(self):
    with ops.device("/device:IPU:0"):
      inp = array_ops.placeholder(np.float32, [1,84,84,4], name="inp")
      wei = array_ops.placeholder(np.float32, [8,8,4,16], name="wei")
      bia = array_ops.placeholder(np.float32, [16], name="bia")
      output = nn_ops.conv2d(inp, wei, strides=[1,4,4,1], padding="VALID")
      output = nn_ops.bias_add(output, bia)

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    with tu.ipu_session() as sess:
      sess.run(report)

      fd = {
        inp: np.zeros([1,84,84,4]),
        wei: np.zeros([8,8,4,16]),
        bia: np.zeros([16]),
      }
      result = sess.run(output, fd)
      self.assertAllClose(result, np.zeros([1, 20, 20, 16]))

      result = sess.run(report)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      ok = ['Conv2D/convolution.*clone/Conv_8x8_stride4x4',
            'BiasAdd/call/addToChannel']
      self.assertTrue(tu.check_all_compute_sets_in_list(cs_list, ok))

  def testConv1x1_WithBias(self):
    with ops.device("/device:IPU:0"):
      inp = array_ops.placeholder(np.float32, [1,1,1,4], name="inp")
      wei = array_ops.placeholder(np.float32, [1,1,4,16], name="wei")
      bia = array_ops.placeholder(np.float32, [16], name="bia")
      output = nn_ops.conv2d(inp, wei, strides=[1,1,1,1], padding="VALID")
      output = output + bia

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    with tu.ipu_session() as sess:
      sess.run(report)

      fd = {
        inp: np.zeros([1,1,1,4]),
        wei: np.zeros([1,1,4,16]),
        bia: np.zeros([16]),
      }
      result = sess.run(output, fd)
      self.assertAllClose(result, np.zeros([1, 1, 1, 16]))

      result = sess.run(report)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      ok = ['Conv2D/convolution.*clone/Conv_1x1',
            'add/add.*/Op/Add']
# TODO = should be addToChannel T3170           'BiasAdd/call/addToChannel']
      self.assertTrue(tu.check_all_compute_sets_in_list(cs_list, ok))

  def testConvBackpropInput(self):
    with ops.device("/device:IPU:0"):
      ins = constant_op.constant([2,8,8,3], np.int32)
      fil = array_ops.placeholder(np.float32, [2,2,3,5], name="inp")
      bck = array_ops.placeholder(np.float32, [2,8,8,5], name="wei")

      output = nn_ops.conv2d_backprop_input(ins, fil, bck, strides=[1,1,1,1],
                                            padding="SAME")

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    with tu.ipu_session() as sess:
      sess.run(report)

      fd = {
        fil: np.zeros([2,2,3,5]),
        bck: np.zeros([2,8,8,5]),
      }
      result = sess.run(output, fd)
      self.assertAllClose(result, np.zeros([2,8,8,3]))

      result = sess.run(report)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      ok = ['Copy_XLA_Args/arg0.*_to_bwdWeights',
            'Copy_bwdWeights_to_weightsRearranged',
            'Conv2DBackpropInput/convolution.*clone/Conv_2x2']
      self.assertTrue(tu.check_all_compute_sets_in_list(cs_list, ok))


  def testConvBackpropFilter(self):
    with ops.device("/device:IPU:0"):
      inp = array_ops.placeholder(np.float32, [2,8,8,3])
      fil = constant_op.constant([2,2,3,5], np.int32)
      bck = array_ops.placeholder(np.float32, [2,8,8,5], name="wei")

      output = nn_ops.conv2d_backprop_filter(inp, fil, bck, strides=[1,1,1,1],
                                            padding="SAME")

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    with tu.ipu_session() as sess:
      sess.run(report)

      fd = {
        inp: np.zeros([2,8,8,3]),
        bck: np.zeros([2,8,8,5]),
      }
      result = sess.run(output, fd)
      self.assertAllClose(result, np.zeros([2,2,3,5]))

      result = sess.run(report)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      ok = ['Copy_XLA_Args/arg1.*_weights_to',
            'Conv2DBackpropFilter/convolution.*clone/Conv_8x8']
      self.assertTrue(tu.check_all_compute_sets_in_list(cs_list, ok))

  def testDepthwiseConv3x2(self):
    with ops.device("/device:IPU:0"):
      pa = array_ops.placeholder(np.float32, [1,2,2,3], name="a")
      pb = array_ops.placeholder(np.float32, [1,1,3,2], name="b")
      pc = array_ops.placeholder(np.float32, [6], name="c")
      c = nn.depthwise_conv2d(pa, pb, strides=[1,1,1,1], padding="SAME")
      output = c + pc

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    with tu.ipu_session() as sess:
      sess.run(report)

      fd = {
        pa: [[[[1,2,3],
               [4,5,6]],
              [[7,8,9],
               [10,11,12]]]],
        pb: [[[[6,5],
               [4,3],
               [2,1]]]],
        pc: [1,1,1,1,1,1]
      }
      result = sess.run(output, fd)
      self.assertAllClose(result, [[[[7, 6, 9, 7, 7, 4],
                                     [25, 21, 21, 16, 13, 7]],
                                    [[43, 36, 33, 25, 19, 10],
                                     [61, 51, 45, 34, 25, 13]]]])

      result = sess.run(report)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      ok = ['depthwise/call.*clone/Conv_1x1',
            'add/call.*/addToChannel']
      self.assertTrue(tu.check_all_compute_sets_in_list(cs_list, ok))

  def testDepthwiseConv3x1(self):
    with ops.device("/device:IPU:0"):
      pa = array_ops.placeholder(np.float32, [1,2,2,3], name="a")
      pb = array_ops.placeholder(np.float32, [1,1,3,1], name="b")
      pc = array_ops.placeholder(np.float32, [3], name="c")
      c = nn.depthwise_conv2d(pa, pb, strides=[1,1,1,1], padding="SAME")
      output = c + pc

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    with tu.ipu_session() as sess:
      sess.run(report)

      fd = {
        pa: [[[[1,2,3],
               [4,5,6]],
              [[7,8,9],
               [10,11,12]]]],
        pb: [[[[6],
               [4],
               [2]]]],
        pc: [1,1,1]
      }
      result = sess.run(output, fd)
      self.assertAllClose(result, [[[[7, 9, 7],
                                     [25, 21, 13]],
                                    [[43, 33, 19],
                                     [61, 45, 25]]]])

      result = sess.run(report)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      ok = ['depthwise/call.*clone/Conv_1x1',
            'Copy_partials_to_depthwise/call',
            'add/call.*/addToChannel']
      self.assertTrue(tu.check_all_compute_sets_in_list(cs_list, ok))


  def testDepthwiseConvBackpropInput(self):
    with ops.device("/device:IPU:0"):
      pa = constant_op.constant([1,8,8,3], dtype=np.int32) # input sizes
      filt = array_ops.placeholder(np.float32, [3,3,3,2], name="filt")
      outb = array_ops.placeholder(np.float32, [1,8,8,6], name="outb")
      c = nn.depthwise_conv2d_native_backprop_input(pa, filt, outb,
                                                       strides=[1,1,1,1],
                                                       padding="SAME")

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    with tu.ipu_session() as sess:
      sess.run(report)

      fd = {
        filt: np.zeros([3,3,3,2]),
        outb: np.zeros([1,8,8,6])
      }
      result = sess.run(c, fd)
      self.assertAllClose(result, np.zeros([1,8,8,3]))

      result = sess.run(report)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      ok = ['Copy_XLA_Args/arg0.*_to_bwdWeights',
            'Copy_bwdWeights_to_weightsRearranged/OnTileCopy',
            'DepthwiseConv2dNativeBackpropInput/convolution.*clone/Conv_3x3']

      self.assertTrue(tu.check_all_compute_sets_in_list(cs_list, ok))


  def testDepthwiseConvBackpropInput1x1(self):
    with ops.device("/device:IPU:0"):
      pa = constant_op.constant([1,8,8,3], dtype=np.int32) # input sizes
      pb = array_ops.placeholder(np.float32, [1,1,3,2], name="b")
      pc = array_ops.placeholder(np.float32, [1,8,8,6], name="c")
      c = nn.depthwise_conv2d_native_backprop_input(pa, pb, pc,
                                                       strides=[1,1,1,1],
                                                       padding="SAME")

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    with tu.ipu_session() as sess:
      sess.run(report)

      fd = {
        pb: np.zeros([1,1,3,2]),
        pc: np.zeros([1,8,8,6])
      }
      result = sess.run(c, fd)
      self.assertAllClose(result, np.zeros([1,8,8,3]))

      result = sess.run(report)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      ok = ['Copy_XLA_Args/arg0.*_to_bwdWeights/',
            'DepthwiseConv2dNativeBackpropInput/convolution.*.clone/Conv_1x1']

      self.assertTrue(tu.check_all_compute_sets_in_list(cs_list, ok))


  def testDepthwiseConvBackpropFilter(self):
    with ops.device("/device:IPU:0"):
      pa = array_ops.placeholder(np.float32, [1,6,6,3], name="a")
      pb = constant_op.constant([3,3,3,2], dtype=np.int32) # filter sizes
      pc = array_ops.placeholder(np.float32, [1,6,6,6], name="c")
      c = nn.depthwise_conv2d_native_backprop_filter(pa, pb, pc,
                                                        strides=[1,1,1,1],
                                                        padding="SAME")

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    with tu.ipu_session() as sess:
      fd = {
        pa: np.zeros([1,6,6,3]),
        pc: np.zeros([1,6,6,6])
      }
      result = sess.run(c, fd)
      self.assertAllClose(result, np.zeros([3,3,3,2]))

      result = sess.run(report)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      ok = ['Copy_{<const>,XLA_Args/arg0.*,XLA_Args/arg1.*}_to',
            'DepthwiseConv2dNativeBackpropFilter/convolution.*.clone/Conv_6x6']
      self.assertTrue(tu.check_all_compute_sets_in_list(cs_list, ok))

  def testDepthwiseConvBackpropFilter1x1(self):
    with ops.device("/device:IPU:0"):
      pa = array_ops.placeholder(np.float32, [1,6,6,3], name="a")
      pb = constant_op.constant([1,1,3,2], dtype=np.int32) # filter sizes
      pc = array_ops.placeholder(np.float32, [1,6,6,6], name="c")
      c = nn.depthwise_conv2d_native_backprop_filter(pa, pb, pc,
                                                        strides=[1,1,1,1],
                                                        padding="SAME")

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    with tu.ipu_session() as sess:
      sess.run(report)

      fd = {
        pa: np.zeros([1,6,6,3]),
        pc: np.zeros([1,6,6,6])
      }
      result = sess.run(c, fd)
      self.assertAllClose(result, np.zeros([1,1,3,2]))

      result = sess.run(report)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      ok = ['Copy_XLA_Args/arg0.*_to',
            'DepthwiseConv2dNativeBackpropFilter/convolution.*.clone/Conv_6x6']
      self.assertTrue(tu.check_all_compute_sets_in_list(cs_list, ok))

  def testDepthwiseConvBackpropFilter1x1WithRelu(self):
    with ops.device("/device:IPU:0"):
      pa = array_ops.placeholder(np.float32, [1,6,6,3], name="a")
      pb = constant_op.constant([1,1,3,2], dtype=np.int32) # filter sizes
      pc = array_ops.placeholder(np.float32, [1,6,6,6], name="c")
      c = nn.depthwise_conv2d_native_backprop_filter(pa, pb, pc,
                                                        strides=[1,1,1,1],
                                                        padding="SAME")
      c = nn.relu(c)

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    with tu.ipu_session() as sess:
      sess.run(report)

      fd = {
        pa: np.zeros([1,6,6,3]),
        pc: np.zeros([1,6,6,6])
      }
      result = sess.run(c, fd)
      self.assertAllClose(result, np.zeros([1,1,3,2]))

      result = sess.run(report)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      ok = ['Copy_XLA_Args/arg0.*_to',
            'DepthwiseConv2dNativeBackpropFilter/convolution.*.clone/Conv_6x6',
            'Relu/call.*/Nonlinearity']
      self.assertTrue(tu.check_all_compute_sets_in_list(cs_list, ok))


if __name__ == "__main__":
    googletest.main()
