# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Script to test TF-TensorRT integration when disabling model freezing."""

import functools
import numpy as np

from tensorflow.python.compiler.tensorrt.test import test_utils
from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.platform import test
from tensorflow.python.saved_model import save, signature_constants


def run_without_freezing():
  """Execute test with model freezing disabled.
  Returns:
    Decorator which runs a test with model freezing disabled.
  """
  def decorator(f):
    @functools.wraps(f)
    def decorated(self, *args, **kwargs):
      with test_utils.experimental_feature_scope("disable_graph_freezing"):
        f(self, *args, **kwargs)
    return decorated
  return decorator


def run_all_without_freezing():
  """Execute all tests in a class with model freezing disabled."""
  return test_util.for_all_test_methods(run_without_freezing)


class TwoEnginesOneVarModel(module.Module):
  """Model with two engines using a common variable.
  The engines comprise ops that require constant inputs, so this checks that
  the variable is appropriately passed as weights to their converters in this
  situation.
  """
  # TODO: figure out why the two engines seem to have resource inputs

  def __init__(self):
    self.v1 = None

  @def_function.function
  def __call__(self, input_0):
    if self.v1 is None:
      self.v1 = variables.Variable(
          np.reshape(np.r_[:243].astype(np.float32), (3, 3, 3, 3, 3)),
          name="filter")
    conv0 = nn.conv3d(
        input=input_0,
        filter=self.v1,
        strides=[1, 1, 1, 1, 1],
        padding="SAME",
        name="conv0")
    bias0 = constant_op.constant([1., 2., 3.], name="bias0")
    added0 = nn.bias_add(conv0, bias0, name="bias_add0")
    relu0 = nn.relu(added0, "relu0")
    incompatible0 = math_ops.erfc(relu0, name="incompatible0")
    conv1 = nn.conv3d(
        input=incompatible0,
        filter=self.v1,
        strides=[1, 1, 1, 1, 1],
        padding="SAME",
        name="conv0")
    bias1 = constant_op.constant([4., 5., 6.], name="bias1")
    added1 = nn.bias_add(conv1, bias1, name="bias_add1")
    relu1 = nn.relu(added1, "relu1")
    return array_ops.squeeze(relu1, name="output_0")


@run_all_without_freezing()
class TwoEnginesOneVarTest(trt_test.TfTrtIntegrationTestBase):
  def _MakeSavedModelV2(self, run_params):
    my_model = TwoEnginesOneVarModel()  # TODO: same model as GetParams?
    cfunc = my_model.__call__.get_concrete_function(
        tensor_spec.TensorSpec([2, 32, 32, 32, 3], dtypes.float32))
    saved_model_dir = self._GetSavedModelDir(
        run_params, trt_test.GraphState.ORIGINAL)
    logging.info("Saving input SavedModel to %s", saved_model_dir)
    save.save(my_model, saved_model_dir,
              {signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: cfunc})
    return saved_model_dir

  def ShouldRunTest(self, run_params):
    # Variables are only supported in dynamic shape mode.
    return (run_params.dynamic_shape and
            run_params.is_v2, "test v2 dynamic shape")

  def GetParams(self):
    shapes = [[2, 32, 32, 32, 3]]
    my_model = TwoEnginesOneVarModel()
    return self.BuildParams(my_model.__call__, dtypes.float32,
                            input_shapes=shapes,
                            output_shapes=shapes)

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return {
        "TRTEngineOp_000": ["conv1", "bias1", "bias_add1", "relu1"],
        "TRTEngineOp_001": ["conv0", "bias0", "bias_add0", "relu0"]
    }


class BiasAddTwoConstInputsModel(module.Module):
  """Model with bias add taking only variable and constants inputs.
  If the model were frozen, this would be constant-folded. This checks that
  the BiasAdd converter handles cases with two constant inputs.
  """

  def __init__(self):
    self.v1 = None

  @def_function.function
  def __call__(self, input_0):
    if self.v1 is None:
      self.v1 = variables.Variable(
          np.reshape(np.r_[:48].astype(np.float32), (4, 4, 3)),
          name="v1")
    bias = constant_op.constant([1., 2., 3.], name="bias")
    bias_add = nn.bias_add(self.v1, bias, name="bias_add")
    add = math_ops.add(bias_add, input_0, name="add")
    relu = nn.relu(add, "relu")
    return array_ops.squeeze(relu, name="output_0")


@run_all_without_freezing()
class BiasAddTwoConstInputsTest(trt_test.TfTrtIntegrationTestBase):
  def _MakeSavedModelV2(self, run_params):
    my_model = BiasAddTwoConstInputsModel()
    cfunc = my_model.__call__.get_concrete_function(
        tensor_spec.TensorSpec([8, 4, 4, 3], dtypes.float32))
    saved_model_dir = self._GetSavedModelDir(
        run_params, trt_test.GraphState.ORIGINAL)
    logging.info("Saving input SavedModel to %s", saved_model_dir)
    save.save(my_model, saved_model_dir,
              {signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: cfunc})
    return saved_model_dir

  def ShouldRunTest(self, run_params):
    # Variables are only supported in dynamic shape mode.
    return (run_params.dynamic_shape and
            run_params.is_v2, "test v2 dynamic shape")

  def GetParams(self):
    shapes = [[8, 4, 4, 3]]
    my_model = BiasAddTwoConstInputsModel()
    return self.BuildParams(my_model.__call__, dtypes.float32,
                            input_shapes=shapes,
                            output_shapes=shapes)

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return {
        "TRTEngineOp_000": ["v1", "bias", "bias_add", "add", "relu"],
    }


class BatchMatMulTwoConstInputsModel(module.Module):
  """Model with batched matmul taking only variable and constants inputs.
  If the model were frozen, this would be constant-folded. This checks that
  the BatchMatMul converter handles cases with two constant inputs.
  """

  def __init__(self):
    self.v1 = None

  @def_function.function
  def __call__(self, input_0):
    if self.v1 is None:
      self.v1 = variables.Variable(
          np.reshape(np.r_[:80].astype(np.float32), (4, 4, 5)),
          name="v1")
    c1 = constant_op.constant(np.reshape(
        np.r_[:60].astype(np.float32), (4, 5, 3)), name="c1")
    mmul = math_ops.matmul(self.v1, c1, name="mmul")
    add = math_ops.add(mmul, input_0, name="add")
    relu = nn.relu(add, "relu")
    return array_ops.squeeze(relu, name="output_0")


@run_all_without_freezing()
class BatchMatMulTwoConstInputsTest(trt_test.TfTrtIntegrationTestBase):
  def _MakeSavedModelV2(self, run_params):
    my_model = BatchMatMulTwoConstInputsModel()
    cfunc = my_model.__call__.get_concrete_function(
        tensor_spec.TensorSpec([8, 4, 4, 3], dtypes.float32))
    saved_model_dir = self._GetSavedModelDir(
        run_params, trt_test.GraphState.ORIGINAL)
    logging.info("Saving input SavedModel to %s", saved_model_dir)
    save.save(my_model, saved_model_dir,
              {signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: cfunc})
    return saved_model_dir

  def ShouldRunTest(self, run_params):
    # Variables are only supported in dynamic shape mode.
    return (run_params.dynamic_shape and
            run_params.is_v2, "test v2 dynamic shape")

  def GetParams(self):
    shapes = [[8, 4, 4, 3]]
    my_model = BatchMatMulTwoConstInputsModel()
    return self.BuildParams(my_model.__call__, dtypes.float32,
                            input_shapes=shapes,
                            output_shapes=shapes)

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return {
        "TRTEngineOp_000": ["v1", "c1", "mmul", "add", "relu"],
    }


class EngineWithoutInputsModel(module.Module):
  """Model with an engine that doesn't take any non-resource inputs.
  This is not supported by TensorRT and the engine should be discarded.
  """

  def __init__(self):
    self.v1 = None

  @def_function.function
  def __call__(self, input_0):
    if self.v1 is None:
      self.v1 = variables.Variable(
          np.reshape(np.r_[:48].astype(np.float32), (4, 4, 3)),
          name="v1")
    c1 = constant_op.constant(np.reshape(
        np.r_[:48].astype(np.float32), (4, 4, 3)), name="c1")
    c2 = constant_op.constant(np.reshape(
        np.r_[48:96].astype(np.float32), (4, 4, 3)), name="c2")
    add1 = math_ops.add(self.v1, c1, name="add1")
    mul = math_ops.multiply(add1, c2, name="mul")
    relu = nn.relu(mul, "relu")
    incompatible = math_ops.erfc(relu, name="incompatible")
    add2 = math_ops.add(incompatible, input_0, name="add2")
    return array_ops.squeeze(add2, name="output_0")


@run_all_without_freezing()
class EngineWithoutInputsTest(trt_test.TfTrtIntegrationTestBase):
  def _MakeSavedModelV2(self, run_params):
    my_model = EngineWithoutInputsModel()
    cfunc = my_model.__call__.get_concrete_function(
        tensor_spec.TensorSpec([8, 4, 4, 3], dtypes.float32))
    saved_model_dir = self._GetSavedModelDir(
        run_params, trt_test.GraphState.ORIGINAL)
    logging.info("Saving input SavedModel to %s", saved_model_dir)
    save.save(my_model, saved_model_dir,
              {signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: cfunc})
    return saved_model_dir

  def ShouldRunTest(self, run_params):
    # Variables are only supported in dynamic shape mode.
    return (run_params.dynamic_shape and
            run_params.is_v2, "test v2 dynamic shape")

  def GetParams(self):
    shapes = [[8, 4, 4, 3]]
    my_model = EngineWithoutInputsModel()
    return self.BuildParams(my_model.__call__, dtypes.float32,
                            input_shapes=shapes,
                            output_shapes=shapes)

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    # Note: The engine with c1, c2, add1, mul, relu is discarded.
    return {
        "TRTEngineOp_000": ["add2"],
    }


class ResourceGatherModel(module.Module):
  """Model with a ResourceGather node.
  """

  def __init__(self):
    self.v1 = None

  @def_function.function
  def __call__(self, input_0):
    if self.v1 is None:
      self.v1 = variables.Variable(
          np.reshape(np.r_[:64].astype(np.float32), (16, 4)),
          name="v1")
    lookup = array_ops.gather(self.v1, input_0, name="lookup")
    c1 = constant_op.constant(np.reshape(
        np.r_[:20].astype(np.float32), (5, 4)), name="c1")
    add = math_ops.add(lookup, c1, name="add")
    relu = nn.relu(add, "relu")
    return array_ops.squeeze(relu, name="output_0")


@run_all_without_freezing()
class ResourceGatherTest(trt_test.TfTrtIntegrationTestBase):
  def _MakeSavedModelV2(self, run_params):
    my_model = ResourceGatherModel()
    cfunc = my_model.__call__.get_concrete_function(
        tensor_spec.TensorSpec([8, 5], dtypes.int32))
    saved_model_dir = self._GetSavedModelDir(
        run_params, trt_test.GraphState.ORIGINAL)
    logging.info("Saving input SavedModel to %s", saved_model_dir)
    save.save(my_model, saved_model_dir,
              {signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: cfunc})
    return saved_model_dir

  def ShouldRunTest(self, run_params):
    # Variables are only supported in dynamic shape mode.
    return (run_params.dynamic_shape and
            run_params.is_v2, "test v2 dynamic shape")

  def GetParams(self):
    in_shapes = [[8, 5]]
    out_shapes = [[8, 5, 4]]
    my_model = ResourceGatherModel()
    return self.BuildParams(my_model.__call__, dtypes.int32,
                            input_shapes=in_shapes,
                            output_shapes=out_shapes)

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return {
        "TRTEngineOp_000": ["v1", "lookup", "c1", "add", "relu"],
    }

if __name__ == "__main__":
  test.main()
