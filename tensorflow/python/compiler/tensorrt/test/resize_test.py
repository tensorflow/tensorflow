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
"""This module tests resize_bilinear ops."""

import unittest
import os

import numpy as np

from tensorflow.python.compiler.tensorrt.test import (
  tf_trt_integration_test_base as trt_test,
)
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops.image_ops import resize_bilinear
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


def build_dynamic_graph(inp, align_corners, half_pixel_centers):
  inp_shape = array_ops.shape(inp)[1:3]
  resize_multiplier = constant_op.constant(
    np.asarray([2, ]).reshape(
        -1,
    ),
    dtype=dtypes.int32,
  )
  inp_shape_2x = math_ops.mul(inp_shape, resize_multiplier)
  x1 = resize_bilinear(
    inp, inp_shape_2x, align_corners=align_corners, 
    half_pixel_centers=half_pixel_centers, name="resize")
  return array_ops.identity(x1, name="output_0")


def build_static_graph(inp, align_corners, half_pixel_centers,
   target_shape=(64, 64)):
  b = constant_op.constant(np.asarray(
    target_shape).reshape(-1), dtype=np.int32)
  x1 = resize_bilinear(inp, b, align_corners=align_corners,
    half_pixel_centers=half_pixel_centers)
  return array_ops.identity(x1, name="output_0")


class ResizeTestBase(trt_test.TfTrtIntegrationTestBase):
  """Base class for resize tests."""

  @classmethod
  def setUpClass(cls):
    if cls is ResizeTestBase:
      raise unittest.SkipTest(
          "ResizeTestBase defines base class for other test."
      )
    super(ResizeTestBase, cls).setUpClass()

  def BuildParamsStaticShape(self, graph_fn, input_shapes, output_shapes):
    input_mask = [[True] * 4]
    output_mask = [[True] * 4]
    return self.BuildParamsWithMask(
        graph_fn,
        dtype=np.float32,
        input_shapes=input_shapes,
        output_shapes=output_shapes,
        input_mask=input_mask,
        output_mask=output_mask,
        extra_inputs=[],
        extra_outputs=[],
    )


class ResizeTestSizeDynamic(ResizeTestBase):
  """Test bilinear resize with dynamic target size."""

  def BuildParams(self, graph_fn, input_shapes, output_shapes):
    input_mask = [[False, False, False, True]]
    output_mask = [[False, False, False, True]]
    return self.BuildParamsWithMask(
      graph_fn,
      dtype=np.float32,
      input_shapes=input_shapes,
      output_shapes=output_shapes,
      input_mask=input_mask,
      output_mask=output_mask,
      extra_inputs=[],
      extra_outputs=[],
    )

  def GraphFn(self, inp):
    return build_dynamic_graph(inp, align_corners=False, 
      half_pixel_centers=False)

  def GetParams(self):
    return self.BuildParams(self.GraphFn, [[1, 32, 32, 1]], [[1, 64, 64, 1]])

  def ExpectedEnginesToBuild(self, run_params):
    if not run_params.dynamic_shape:
      return {}
    return {"TRTEngineOp_0": []}


class ResizeTestSizeConst(ResizeTestBase):
  """Test bilinear resize with constant for target size."""

  def BuildParams(self, graph_fn, input_shapes, output_shapes):
    return self.BuildParamsStaticShape(graph_fn, input_shapes, output_shapes)

  def GraphFn(self, inp):
    return build_static_graph(inp, align_corners=False,
      half_pixel_centers=False)

  def GetParams(self):
    return self.BuildParams(self.GraphFn, [[1, 32, 32, 1]], [[1, 64, 64, 1]])

  def ExpectedEnginesToBuild(self, run_params):
    return {"TRTEngineOp_0": []}


class ResizeTestSizeDynamicAligned(ResizeTestBase):
  """Test bilinear resize with dynamic target size, with aligned_corners=True"""

  def BuildParams(self, graph_fn, input_shapes, output_shapes):
    input_mask = [[False, False, False, True]]
    output_mask = [[False, False, False, True]]
    return self.BuildParamsWithMask(
      graph_fn,
      dtype=np.float32,
      input_shapes=input_shapes,
      output_shapes=output_shapes,
      input_mask=input_mask,
      output_mask=output_mask,
      extra_inputs=[],
      extra_outputs=[],
    )

  def GraphFn(self, inp):
    return build_dynamic_graph(inp, align_corners=True, 
      half_pixel_centers=False)

  def GetParams(self):
    return self.BuildParams(self.GraphFn, [[1, 32, 32, 1]], [[1, 64, 64, 1]])

  def ExpectedEnginesToBuild(self, run_params):
    if not run_params.dynamic_shape:
      return {}
    return {"TRTEngineOp_0": []}


class ResizeTestSizeDynamicHalfPixel(ResizeTestBase):
  """Test bilinear resize with dynamic target size, with aligned_corners=False, 
    half_pixel_centers=True"""

  def BuildParams(self, graph_fn, input_shapes, output_shapes):
    input_mask = [[False, False, False, True]]
    output_mask = [[False, False, False, True]]
    return self.BuildParamsWithMask(
      graph_fn,
      dtype=np.float32,
      input_shapes=input_shapes,
      output_shapes=output_shapes,
      input_mask=input_mask,
      output_mask=output_mask,
      extra_inputs=[],
      extra_outputs=[],
    )

  def GraphFn(self, inp):
    return build_dynamic_graph(inp, align_corners=False, 
      half_pixel_centers=True)

  def GetParams(self):
    return self.BuildParams(self.GraphFn, [[1, 32, 32, 1]], [[1, 64, 64, 1]])

  def ExpectedEnginesToBuild(self, run_params):
    if not run_params.dynamic_shape:
      return {}
    return {"TRTEngineOp_0": []}


class ResizeTestSizeConstAligned(ResizeTestBase):
  """Test bilinear resize with constant for target size."""

  def BuildParams(self, graph_fn, input_shapes, output_shapes):
    return self.BuildParamsStaticShape(graph_fn, input_shapes, output_shapes)

  def GraphFn(self, inp):
    return build_static_graph(inp, align_corners=True,
      half_pixel_centers=False)

  def GetParams(self):
    return self.BuildParams(self.GraphFn, [[1, 32, 32, 1]], [[1, 64, 64, 1]])

  def ExpectedEnginesToBuild(self, run_params):
    return {"TRTEngineOp_0": []}


class ResizeTestSizeConstWAR(ResizeTestBase):
  """Test bilinear resize using workaround algorithm."""

  def setUp(self):
    super().setUp()
    os.environ["TF_TRT_FORCE_BILINEAR_RESIZE_WAR"] = "1"

  def tearDown(self):
    super().tearDown()
    os.environ["TF_TRT_FORCE_BILINEAR_RESIZE_WAR"] = "0"

  def BuildParams(self, graph_fn, input_shapes, output_shapes):
    return self.BuildParamsStaticShape(graph_fn, input_shapes, output_shapes)

  def GraphFn(self, inp):
    return build_static_graph(inp, align_corners=False, 
      half_pixel_centers=False)

  def GetParams(self):
    return self.BuildParams(self.GraphFn, [[1, 32, 32, 1]], [[1, 64, 64, 1]])

  def ExpectedEnginesToBuild(self, run_params):
    return {"TRTEngineOp_0": []}

  def ExpectedAbsoluteTolerance(self, run_params):
    # Increase absolute error threshold for the workaround.
    # The Base test uses relatively small values.
    # Relative error threshold remains unchanged.
    return 1.e-05 if run_params.precision_mode == "FP32" else 1.e-01

class ResizeTestSizeConstNonIntegralWAR(ResizeTestBase):
  """Test bilinear resize using workaround algorithm, for non-integral size ratios."""

  def setUp(self):
    super().setUp()
    os.environ["TF_TRT_FORCE_BILINEAR_RESIZE_WAR"] = "1"

  def tearDown(self):
    super().tearDown()
    os.environ["TF_TRT_FORCE_BILINEAR_RESIZE_WAR"] = "0"

  def BuildParams(self, graph_fn, input_shapes, output_shapes):
    return self.BuildParamsStaticShape(graph_fn, input_shapes, output_shapes)

  def GraphFn(self, inp):
    return build_static_graph(inp, align_corners=False, 
      half_pixel_centers=False, target_shape=(33, 33))

  def GetParams(self):
    return self.BuildParams(self.GraphFn, [[1, 31, 31, 1]], [[1, 33, 33, 1]])

  def ExpectedEnginesToBuild(self, run_params):
    return {"TRTEngineOp_0": []}


if __name__ == "__main__":
  test.main()
