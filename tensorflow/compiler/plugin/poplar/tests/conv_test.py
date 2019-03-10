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

  data_formats = ['NHWC', 'NCHW']

  def _ip_shp(self, nhwc, fmt):
    if fmt == 'NHWC':
      return nhwc
    else:
      return [nhwc[0], nhwc[3], nhwc[1], nhwc[2]]

  def testConv1x1_Stride2x1_In1x5(self):
    for fmt in self.data_formats:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, self._ip_shp([1,1,5,1], fmt),
                                   name="a")
        pb = array_ops.placeholder(np.float32, [1,1,1,1], name="b")
        output = nn_ops.convolution(pa, pb, strides=[1,2], padding="VALID",
                                    data_format=fmt, name='cnv1')

      with session_lib.Session() as sess:
        fd = {
          pa: np.zeros(self._ip_shp([1,1,5,1], fmt)),
          pb: np.zeros([1,1,1,1])
        }
        result = sess.run(output, fd)
        self.assertAllClose(result, np.zeros(self._ip_shp([1,1,3,1], fmt)))

  def testConv3x3_Pad1x1(self):
    for fmt in self.data_formats:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, self._ip_shp([1,14,14,64], fmt),
                                   name="a")
        pb = array_ops.placeholder(np.float32, [3,3,64,128], name="b")
        output = nn_ops.convolution(pa, pb, padding="SAME", data_format=fmt,
                                    name='cnv2')

      with session_lib.Session() as sess:
        fd = {
          pa: np.zeros(self._ip_shp([1,14,14,64], fmt)),
          pb: np.zeros([3,3,64,128])
        }
        result = sess.run(output, fd)
        self.assertAllClose(result, np.zeros(self._ip_shp([1,14,14,128], fmt)))

  def testConv3x3_WithBias(self):
    for fmt in self.data_formats:
      with ops.device("/device:IPU:0"):
        pa = array_ops.placeholder(np.float32, self._ip_shp([1,14,14,64], fmt),
                                   name="a")
        pb = array_ops.placeholder(np.float32, [3,3,64,128], name="b")
        bi = array_ops.placeholder(np.float32, [128], name="b")
        output = nn_ops.convolution(pa, pb, padding="SAME", data_format=fmt,
                                    name='cnv3')
        output = nn_ops.bias_add(output, bi, data_format=fmt, name='ba3')

      with ops.device('cpu'):
        report = gen_ipu_ops.ipu_event_trace()

      with tu.ipu_session() as sess:
        sess.run(report)

        fd = {
          pa: np.zeros(self._ip_shp([1,14,14,64], fmt)),
          pb: np.zeros([3,3,64,128]),
          bi: np.zeros([128]),
        }
        result = sess.run(output, fd)
        self.assertAllClose(result, np.zeros(self._ip_shp([1,14,14,128], fmt)))

        result = sess.run(report)

        s = tu.extract_all_strings_from_event_trace(result)
        cs_list = tu.get_compute_sets_from_report(s)

        ok = ['Copy_*actsRearranged',
              'host-exchange-local-copy-',
              'cnv3*/convolution.*/Conv_3x3',
              'ba3*/fusion/addToChannel']

        self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

  def testConv8x8_WithBias(self):
    for fmt in self.data_formats:
      with ops.device("/device:IPU:0"):
        inp = array_ops.placeholder(np.float32, self._ip_shp([1,84,84,4], fmt),
                                    name="inp")
        wei = array_ops.placeholder(np.float32, [8,8,4,16], name="wei")
        bia = array_ops.placeholder(np.float32, [16], name="bia")
        output = nn_ops.conv2d(inp, wei, strides=self._ip_shp([1,4,4,1], fmt),
                               padding="VALID", data_format=fmt, name='cnv4')
        output = nn_ops.bias_add(output, bia, data_format=fmt, name='ba4')

      with ops.device('cpu'):
        report = gen_ipu_ops.ipu_event_trace()

      with tu.ipu_session() as sess:
        sess.run(report)

        fd = {
          inp: np.zeros(self._ip_shp([1,84,84,4], fmt)),
          wei: np.zeros([8,8,4,16]),
          bia: np.zeros([16]),
        }
        result = sess.run(output, fd)
        self.assertAllClose(result,
                            np.zeros(self._ip_shp([1, 20, 20, 16], fmt)))

        result = sess.run(report)

        s = tu.extract_all_strings_from_event_trace(result)
        cs_list = tu.get_compute_sets_from_report(s)

        ok = ['host-exchange-local-copy-',
              'Copy_XLA_Args/arg2.*_weights_to_cnv4*/convolution.*/Conv_8x8_stride4x4/weightsRearranged',
              'cnv4*/convolution.*/Conv_8x8_stride4x4',
              'ba4*/fusion/addToChannel']
        self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

  def testConv1x1_WithBias(self):
    for fmt in self.data_formats:
      with ops.device("/device:IPU:0"):
        inp = array_ops.placeholder(np.float32, self._ip_shp([1,1,1,4], fmt),
                                    name="inp")
        wei = array_ops.placeholder(np.float32, [1,1,4,16], name="wei")
        bia = array_ops.placeholder(np.float32, [16], name="bia")
        output = nn_ops.conv2d(inp, wei, strides=[1,1,1,1], padding="VALID",
                               data_format=fmt, name='cnv5')
        output = nn_ops.bias_add(output, bia, data_format=fmt, name='ba5')

      with ops.device('cpu'):
        report = gen_ipu_ops.ipu_event_trace()

      with tu.ipu_session() as sess:
        sess.run(report)

        fd = {
          inp: np.zeros(self._ip_shp([1,1,1,4], fmt)),
          wei: np.zeros([1,1,4,16]),
          bia: np.zeros([16]),
        }
        result = sess.run(output, fd)
        self.assertAllClose(result, np.zeros(self._ip_shp([1, 1, 1, 16], fmt)))

        result = sess.run(report)

        s = tu.extract_all_strings_from_event_trace(result)
        cs_list = tu.get_compute_sets_from_report(s)

        ok = ['Copy_*actsRearranged',
              'host-exchange-local-copy-',
              'cnv5*/convolution.*/Conv_1x1',
              'ba5*/fusion/addToChannel']
        self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

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

      ok = ['Copy_',
            'Conv2DBackpropInput/fusion*/Conv_2x2',
            'Conv2DBackpropInput/fusion*/WeightTranspose']
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))


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

      ok = ['host-exchange-local-copy-',
            'Copy_',
            'Conv2DBackpropFilter/convolution.*/Conv_8x8']
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

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

      ok = ['host-exchange-local-copy-',
            'depthwise/convolution.*/Conv_1x1',
            'add/fusion*/addToChannel']
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

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

      ok = ['host-exchange-local-copy-',
            'Copy_',
            'depthwise/convolution.*/Conv_1x1',
            'Copy_depthwise/convolution.*/Conv_1x1/partials_to_depthwise/convolution.*/Conv_1x1/partials[[]cloned[]]',
            'add/fusion*/addToChannel']
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))


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

      ok = ['DepthwiseConv2dNativeBackpropInput/fusion*/WeightTranspose',
            'DepthwiseConv2dNativeBackpropInput/fusion*/Conv_3x3',
            'Copy_']

      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))


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

      ok = ['DepthwiseConv2dNativeBackpropInput/fusion*/WeightTranspose',
            'DepthwiseConv2dNativeBackpropInput/fusion*/Conv_1x1',
            'Copy_']

      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))


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

      ok = ['DepthwiseConv2dNativeBackpropFilter/fusion*/Conv_6x6']
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

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

      ok = ['DepthwiseConv2dNativeBackpropFilter/fusion*/Conv_6x6']
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

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

      ok = ['DepthwiseConv2dNativeBackpropFilter/fusion*/Conv_6x6',
            'Relu/fusion.*/Nonlinearity']
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

  def testDataLayout(self):
    with ops.device("/device:IPU:0"):
      pa1 = array_ops.placeholder(np.float32, [1,14,14,64], name="a")
      pb1 = array_ops.placeholder(np.float32, [3,3,64,128], name="b")
      bi1 = array_ops.placeholder(np.float32, [128], name="b")
      op1 = nn_ops.convolution(pa1, pb1, padding="SAME", data_format='NHWC')
      op1 = nn_ops.bias_add(op1, bi1, data_format='NHWC')

      pa2 = array_ops.placeholder(np.float32, [1,64,14,14], name="a")
      pb2 = array_ops.placeholder(np.float32, [3,3,64,128], name="b")
      bi2 = array_ops.placeholder(np.float32, [128], name="b")
      op2 = nn_ops.convolution(pa2, pb2, padding="SAME", data_format='NCHW')
      op2 = nn_ops.bias_add(op2, bi2, data_format='NCHW')

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    with tu.ipu_session() as sess:
      sess.run(report)

      fd = {
        pa1: np.zeros([1,14,14,64]),
        pb1: np.zeros([3,3,64,128]),
        bi1: np.zeros([128]),
        pa2: np.zeros([1,64,14,14]),
        pb2: np.zeros([3,3,64,128]),
        bi2: np.zeros([128]),
      }
      result = sess.run(op1, fd)
      self.assertAllClose(result, np.zeros([1,14,14,128]))

      result = sess.run(report)
      s = tu.extract_all_strings_from_event_trace(result)
      mem_nhwc = tu.get_total_memory_from_report(s)

      result = sess.run(op2, fd)
      self.assertAllClose(result, np.zeros([1,128,14,14]))

      result = sess.run(report)
      s = tu.extract_all_strings_from_event_trace(result)
      mem_nchw = tu.get_total_memory_from_report(s)

      self.assertTrue((mem_nhwc - mem_nchw) / mem_nhwc > -0.1)

if __name__ == "__main__":
    googletest.main()
