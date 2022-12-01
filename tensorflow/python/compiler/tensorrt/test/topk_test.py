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
"""Model script to test TF-TensorRT integration."""

from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class TopKTest(trt_test.TfTrtIntegrationTestBase):
  """Testing Top-K in TF-TRT conversion."""

  def GraphFn(self, x):
    k = 5
    k_tensor = constant_op.constant(k, dtype=dtypes.int32, name="Const")
    values, indices = nn_ops.top_k(x, k_tensor, name="TopK")
    values = array_ops.identity(values, name="output_0")
    indices = array_ops.identity(indices, name="output_1")
    return values, indices

  def GetParams(self):
    k = 5
    return self.BuildParams(self.GraphFn, dtypes.float32, [[100, 100]],
                            [[100, k], [100, k]])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return {"TRTEngineOp_000": ["Const", "TopK"]}


class TopKOutputTypeTest(trt_test.TfTrtIntegrationTestBase):
  """Testing that output type of engine using Top-K is set correctly."""

  def GraphFn(self, x):
    k = 5
    k_tensor = constant_op.constant(k, dtype=dtypes.int32, name="Const")
    values, indices = nn_ops.top_k(x, k_tensor, name="TopK")
    # Reshape will act as a layer between the TopK output and the engine
    # output, requiring the output tensor of reshape to be set explicitly to
    # int32.
    indices = array_ops.reshape(indices, [100, 1, 5], name="Reshape")
    values = array_ops.identity(values, name="output_0")
    indices = array_ops.identity(indices, name="output_1")
    return values, indices

  def GetParams(self):
    k = 5
    return self.BuildParams(self.GraphFn, dtypes.float32, [[100, 100]],
                            [[100, k], [100, 1, k]])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return {"TRTEngineOp_000": ["Const", "TopK", "Reshape", "Reshape/shape"]}


if __name__ == "__main__":
  test.main()
