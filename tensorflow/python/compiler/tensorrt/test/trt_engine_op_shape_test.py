# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""This script test input and output shapes and dtype of the TRTEngineOp."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python import saved_model
from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.platform import test
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants


class TRTEngineOpInputOutputShapeTest(trt_test.TfTrtIntegrationTestBase):
  """Testing the output shape of a TRTEngine."""

  def GraphFn(self, inp):
    b = array_ops.squeeze(inp, axis=[2])
    c = nn.relu(b)
    d1 = c + c
    d2 = math_ops.reduce_sum(d1)

    d1 = array_ops.identity(d1, name="output_0")
    d2 = array_ops.identity(d2, name="output_1")
    return d1, d2

  def GetParams(self):
    return self.BuildParams(self.GraphFn, dtypes.float32, [[1, 2, 1, 4]],
                            [[1, 2, 4], []])

  def _GetInferGraph(self, *args, **kwargs):
    trt_saved_model_dir = super(TRTEngineOpInputOutputShapeTest,
                                self)._GetInferGraph(*args, **kwargs)

    def get_func_from_saved_model(saved_model_dir):
      saved_model_loaded = saved_model.load.load(
          saved_model_dir, tags=[tag_constants.SERVING])
      graph_func = saved_model_loaded.signatures[
          signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
      return graph_func, saved_model_loaded

    func, _ = get_func_from_saved_model(trt_saved_model_dir)

    input_shape = func.inputs[0].shape
    if isinstance(input_shape, tensor_shape.TensorShape):
      input_shape = input_shape.as_list()

    output_shapes = [
        out_shape.shape.as_list() if isinstance(
            out_shape.shape, tensor_shape.TensorShape) else out_shape.shape
        for out_shape in func.outputs
    ]

    self.assertEqual(func.inputs[0].dtype, dtypes.float32)
    self.assertEqual(func.outputs[0].dtype, dtypes.float32)
    self.assertEqual(func.outputs[1].dtype, dtypes.float32)

    self.assertEqual(input_shape, [None, 2, 1, 4])
    self.assertEqual(output_shapes[0], [None, 2, 4])
    self.assertEqual(output_shapes[1], [])

    return trt_saved_model_dir

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return ["TRTEngineOp_0"]


if __name__ == "__main__":
  test.main()
