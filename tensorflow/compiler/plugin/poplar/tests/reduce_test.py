# Copyright 2017 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import test_utils as tu

from tensorflow.python.platform import googletest
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops

class IpuXlaConvTest(test_util.TensorFlowTestCase):

    def testReductionMeanDim12(self):
        with ops.device("/device:IPU:0"):
            with session_lib.Session() as sess:
                pa = array_ops.placeholder(np.float32, [2,7,7,32], name="a")
                output = math_ops.reduce_mean(pa, reduction_indices=[1,2])

                fd = {
                    pa: np.ones([2,7,7,32])
                }
                result = sess.run(output, fd)
                self.assertAllClose(result,
                                    np.ones([2,32]))

    def testReductionMeanDim03(self):
        with ops.device("/device:IPU:0"):
            with session_lib.Session() as sess:
                pa = array_ops.placeholder(np.float32, [2,7,7,32], name="a")
                output = math_ops.reduce_mean(pa, reduction_indices=[0,3])

                fd = {
                    pa: np.ones([2,7,7,32])
                }
                result = sess.run(output, fd)
                self.assertAllClose(result,
                                    np.ones([7,7]))

    def testReductionMeanDim13(self):
        with ops.device("/device:IPU:0"):
            with session_lib.Session() as sess:
                pa = array_ops.placeholder(np.float32, [2,7,7,32], name="a")
                output = math_ops.reduce_mean(pa, reduction_indices=[1,3])

                fd = {
                    pa: np.ones([2,7,7,32])
                }
                result = sess.run(output, fd)
                self.assertAllClose(result,
                                    np.ones([2,7]))

    def testReductionMeanDim23(self):
        with ops.device("/device:IPU:0"):
            with session_lib.Session() as sess:
                pa = array_ops.placeholder(np.float32, [2,7,7,32], name="a")
                output = math_ops.reduce_mean(pa, reduction_indices=[2,3])

                fd = {
                    pa: np.ones([2,7,7,32])
                }
                result = sess.run(output, fd)
                self.assertAllClose(result,
                                    np.ones([2,7]))

    def testAvgPoolSamePaddingWithStridesF32(self):
        with ops.device("/device:IPU:0"):
            with session_lib.Session() as sess:
                pa = array_ops.placeholder(np.float32, [1,1,10,10], name="a")
                output = nn.avg_pool(pa, ksize=[1,1,5,5], strides=[1,1,2,2],
                                        data_format='NCHW',
                                        padding='SAME', name="avg")

                fd = {
                    pa: np.ones([1,1,10,10])
                }
                result = sess.run(output, fd)
                self.assertAllClose(result,
                                    np.ones([1,1,5,5]))

    def testAvgPoolSamePaddingWithStridesF16(self):
        with ops.device("/device:IPU:0"):
            with session_lib.Session() as sess:
                pa = array_ops.placeholder(np.float16, [1,1,10,10], name="a")
                output = nn.avg_pool(pa, ksize=[1,1,5,5], strides=[1,1,2,2],
                                        data_format='NCHW',
                                        padding='SAME')

                fd = {
                    pa: np.ones([1,1,10,10])
                }
                result = sess.run(output, fd)
                self.assertAllClose(result,
                                    np.ones([1,1,5,5]))

    def testAvgPoolValidPaddingWithStridesF32(self):
        with ops.device("/device:IPU:0"):
            with session_lib.Session() as sess:
                pa = array_ops.placeholder(np.float32, [1,1,10,10], name="a")
                output = nn.avg_pool(pa, ksize=[1,1,5,5], strides=[1,1,2,2],
                                        data_format='NCHW',
                                        padding='VALID')

                fd = {
                    pa: np.ones([1,1,10,10])
                }
                result = sess.run(output, fd)
                self.assertAllClose(result,
                                    np.ones([1,1,3,3]))

    def testAvgPoolValidPaddingWithStridesF16(self):
        with ops.device("/device:IPU:0"):
            with session_lib.Session() as sess:
                pa = array_ops.placeholder(np.float16, [1,1,10,10], name="a")
                output = nn.avg_pool(pa, ksize=[1,1,5,5], strides=[1,1,2,2],
                                        data_format='NCHW',
                                        padding='VALID')

                fd = {
                    pa: np.ones([1,1,10,10])
                }
                result = sess.run(output, fd)
                self.assertAllClose(result,
                                    np.ones([1,1,3,3]))

    def testMaxPoolSamePaddingWithStridesF32(self):
        with ops.device("/device:IPU:0"):
            with session_lib.Session() as sess:
                pa = array_ops.placeholder(np.float32, [1,1,10,10], name="a")
                output = nn.max_pool(pa, ksize=[1,1,5,5], strides=[1,1,2,2],
                                        data_format='NCHW',
                                        padding='SAME', name="max")

                fd = {
                    pa: np.ones([1,1,10,10])
                }
                result = sess.run(output, fd)
                self.assertAllClose(result,
                                    np.ones([1,1,5,5]))

    def testMaxPoolValidPaddingWithStridesF32(self):
        with ops.device("/device:IPU:0"):
            with session_lib.Session() as sess:
                pa = array_ops.placeholder(np.float32, [1,1,10,10], name="a")
                output = nn.max_pool(pa, ksize=[1,1,5,5], strides=[1,1,2,2],
                                        data_format='NCHW',
                                        padding='VALID', name="max")

                fd = {
                    pa: np.ones([1,1,10,10])
                }
                result = sess.run(output, fd)
                self.assertAllClose(result,
                                    np.ones([1,1,3,3]))

    def testAvgPoolSamePaddingWithStridesF32Dim12(self):
        with ops.device("/device:IPU:0"):
            with session_lib.Session() as sess:
                pa = array_ops.placeholder(np.float32, [1,10,10,1], name="a")
                output = nn.avg_pool(pa, ksize=[1,5,5,1], strides=[1,2,2,1],
                                        data_format='NHWC',
                                        padding='SAME', name="avg")

                fd = {
                    pa: np.ones([1,10,10,1])
                }
                result = sess.run(output, fd)
                self.assertAllClose(result,
                                    np.ones([1,5,5,1]))

    def testAvgPoolValidPaddingWithStridesF32Dim12(self):
        with ops.device("/device:IPU:0"):
            with session_lib.Session() as sess:
                pa = array_ops.placeholder(np.float32, [1,10,10,1], name="a")
                output = nn.avg_pool(pa, ksize=[1,5,5,1], strides=[1,2,2,1],
                                        data_format='NHWC',
                                        padding='VALID', name="avg")

                fd = {
                    pa: np.ones([1,10,10,1])
                }
                result = sess.run(output, fd)
                self.assertAllClose(result,
                                    np.ones([1,3,3,1]))

    def testReductionSumVectorF16NoConverts(self):
        with ops.device("/device:IPU:0"):
            pa = array_ops.placeholder(np.float16, [4096], name="a")
            output = math_ops.reduce_sum(pa, reduction_indices=[0])

        with ops.device('cpu'):
            report = gen_ipu_ops.ipu_event_trace()

        with tu.ipu_session() as sess:
            sess.run(report)
            fd = {
                pa: np.ones([4096])
            }
            result = sess.run(output, fd)
            self.assertAllClose(result, 4096)

            result = sess.run(report)

            s = tu.extract_all_strings_from_event_trace(result)
            cs_list = tu.get_compute_sets_from_report(s)

            # Check that there are no casts to float at the beginning
            # Note that intermidiates are still floats, so there is a final cast
            ok = ['Execution',
                  '/ExchangePre',
                  'Sum/reduce.*.clone_f16/final_stage/Cast/Cast']
            self.assertTrue(tu.check_all_compute_sets_in_list(cs_list, ok))

if __name__ == "__main__":
    import time
    time.sleep(20)
    googletest.main()
