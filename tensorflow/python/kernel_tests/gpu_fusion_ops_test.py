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
# ==============================================================================
"""Tests for _ROCmFusion* ops (internal)"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import collections

from tensorflow.python.framework import ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test
from tensorflow.python.framework import test_util
from tensorflow.python.client import session

import numpy as np

class FusionOpsTestCase(test.TestCase) :

    def runTest(self, test_func, dtype) :

        # run the test with fusion turned OFF
        os.environ["TF_ROCM_FUSION_ENABLE"] = "0"
        with session.Session() as sess:
            results_without_fusion = sess.run(test_func(dtype))

        # explicitly clear the graph
        ops.reset_default_graph()
        
        # run the test with fusion turned ON
        os.environ["TF_ROCM_FUSION_ENABLE"] = "1"
        with session.Session() as sess:
            results_with_fusion = sess.run(test_func(dtype))

        # compare the results with and without fusion
        tol = 1e-3 if dtype == dtypes.float16 else 1e-5
        for with_fusion, without_fusion in zip(results_with_fusion, results_without_fusion) :
            self.assertAllClose(with_fusion, without_fusion, atol=tol, rtol=tol)
        

class CBAForwardTestSuite(FusionOpsTestCase):

    def _test01(self, dtype):
        with test_util.device(True):
            
            x = constant_op.constant([5,5,5,5,5,7,7,7,7,7,9,9,9,9,9,11,11,11,11,11,13,13,13,13,13], dtype=dtype, shape=[1,5,5,1])
            k = constant_op.constant([1,0,1,1,0,1,1,0,1], dtype=dtype, shape=[3,3,1,1])
            offset = constant_op.constant([2], dtype=dtype)
            
            conv = nn_ops.conv2d(x, k, [1,1,1,1], "VALID")
            bias = nn_ops.bias_add(conv, offset)
            relu = nn_ops.relu(bias)

            y1 = array_ops.identity(relu)
            
            return (y1,)
        
    
    def test01(self):
        for dtype in [dtypes.float32, dtypes.float16]:
            self.runTest(self._test01, dtype)

            
class BnAForwardTestSuite(FusionOpsTestCase):

    def _test01(self, dtype):
        with test_util.device(True):
            
            x = constant_op.constant([5,5,7,7,9,9,11,11,13,13,15,15], dtype=dtype, shape=[1,1,6,2])
            scale = constant_op.constant([4,5], dtype=float)
            offset = constant_op.constant([2,3], dtype=float)
            
            batch_norm, batch_mean, batch_var = nn_impl.fused_batch_norm(x, scale, offset, is_training=True)
            relu = nn_ops.relu(batch_norm)

            y1 = array_ops.identity(relu)
            y2 = array_ops.identity(batch_mean)
            y3 = array_ops.identity(batch_var)
            
            return (y1, y2, y3)
        
    
    def test01(self):
        for dtype in [dtypes.float32, dtypes.float16]:
            self.runTest(self._test01, dtype)


            

class BnABackwardTestSuite(FusionOpsTestCase):

    def _test01(self, dtype):
        with test_util.device(True):

            x_val = [5,5,7,7,9,9,11,11,13,13,15,15]
            shape = [1,1,6,2]
            
            x = constant_op.constant(x_val, dtype=dtype, shape=shape)
            scale = constant_op.constant([4,5], dtype=float)
            offset = constant_op.constant([2,3], dtype=float)
            
            batch_norm, batch_mean, batch_var = nn_impl.fused_batch_norm(x, scale, offset, is_training=True)
            relu = nn_ops.relu(batch_norm)
            grad = gradients_impl.gradients(relu, x)
            
            y1 = array_ops.identity(grad)
            
            return (y1,)
        
    
    def test01(self):
        for dtype in [dtypes.float32, dtypes.float16]:
            self.runTest(self._test01, dtype)

            

class BnAInferenceTestSuite(FusionOpsTestCase):

    def _test01(self, dtype):
        with test_util.device(True):
            x = constant_op.constant([5,5,7,7,9,9,11,11,13,13,15,15], dtype=dtype, shape=[1,1,6,2])
            scale = constant_op.constant([4,5], dtype=float)
            offset = constant_op.constant([2,3], dtype=float)
            batch_mean = constant_op.constant([10,10], dtype=float)
            batch_var = constant_op.constant([14,14], dtype=float)
            
            batch_norm, _, _ = nn_impl.fused_batch_norm(x, scale, offset, mean=batch_mean, variance=batch_var, is_training=False)
            relu = nn_ops.relu(batch_norm)
            
            y1 = array_ops.identity(relu)

            return (y1,)
        
    
    def test01(self):
        for dtype in [dtypes.float32, dtypes.float16]:
            self.runTest(self._test01, dtype)



class AddReluTestSuite(FusionOpsTestCase):

    def _test01(self, dtype):
        with test_util.device(True):
            x1 = constant_op.constant([-3,-2,-1,0,1,2], dtype=dtype)
            x2 = constant_op.constant([1,1,1,1,1,1], dtype=dtype)

            add = math_ops.add(x1, x2)
            relu = nn_ops.relu(add)

            y1 = array_ops.identity(relu)

            return (y1,)
        
    def test01(self):
        for dtype in [dtypes.float32, dtypes.float16]:
            self.runTest(self._test01, dtype)

            
    def _test02(self, dtype):
        with test_util.device(True):
            x1 = constant_op.constant([[-3,-2,-1],[0,1,2]], dtype=dtype)
            x2 = constant_op.constant([1,1,1], dtype=dtype)

            add = math_ops.add(x1, x2)
            relu = nn_ops.relu(add)

            y1 = array_ops.identity(relu)

            return (y1,)
        
    def test02(self):
        for dtype in [dtypes.float32]:
            self.runTest(self._test02, dtype)


            
            
class AddNReluGradTestSuite(FusionOpsTestCase):

    def _test01(self, dtype):
        with test_util.device(True):
            x1 = constant_op.constant([-3,-2,-1,0,1,2], dtype=dtype)
            x2 = constant_op.constant([1,1,1,1,1,1], dtype=dtype)
            x3 = constant_op.constant([-1,-1,-1,-1,-1,-1], dtype=dtype)

            relu1 = nn_ops.relu(x1)
            add1 = math_ops.add(relu1, x2)
            add2 = math_ops.add(relu1, x3)
            add3 = math_ops.add(add1, add2)
            
            grad = gradients_impl.gradients(add3, x1)
            
            y1 = array_ops.identity(grad)

            return (y1,)
        
    def test01(self):
        for dtype in [dtypes.float32, dtypes.float16]:
            self.runTest(self._test01, dtype)

            

if __name__ == "__main__":
    test.main()

            

