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
"""This test checks a situation where the same tensor is considered as an output

multiple times because it has been duplicated by 2+ indentity ops. Previously,
the tensor would be renamed multiple times, overwriting the output binding name
which resulted in a runtime error when the binding would not be found.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class IdentityTest(trt_test.TfTrtIntegrationTestBase):

  def _ConstOp(self, shape):
    return constant_op.constant(np.random.randn(*shape), dtype=dtypes.float32)

  def GetParams(self):
    """Testing engine with the same tensor repeated as output via identity."""
    input_name = 'input'
    input_dims = [100, 32]
    g = ops.Graph()
    with g.as_default():
      x = array_ops.placeholder(
          dtype=dtypes.float32, shape=input_dims, name=input_name)

      b = self._ConstOp((32, 4))
      x1 = math_ops.matmul(x, b)
      b = self._ConstOp((1, 4))
      x1 = x1 + b

      out1 = array_ops.identity(x1, name='output1')
      out2 = array_ops.identity(x1, name='output2')
      iden1 = array_ops.identity(x1)
      out3 = array_ops.identity(iden1, name='output3')

    return trt_test.TfTrtIntegrationTestParams(
        gdef=g.as_graph_def(),
        input_names=[input_name],
        input_dims=[[input_dims]],
        output_names=['output1', 'output2', 'output3'],
        expected_output_dims=[[[100, 4]] * 3])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return ['TRTEngineOp_0']


if __name__ == '__main__':
  test.main()
