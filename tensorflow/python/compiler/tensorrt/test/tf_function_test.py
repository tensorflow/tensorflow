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
"""TF function conversion."""

import itertools
import os

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.compiler.tensorrt import trt_convert
from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.compiler.tensorrt.test.tf_trt_integration_test_base import GraphState
from tensorflow.python.compiler.tensorrt.test.tf_trt_integration_test_base import IsQuantizationWithCalibration
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.util import compat


class TfFunctionTest(trt_test.TfTrtIntegrationTestBase):

  def __init__(self, methodName):  # pylint: disable=invalid-name
    super(TfFunctionTest, self).__init__(methodName)
    self._profile_strategy = "Range"
    self._trt_engine_op_count_offset = 0
    self._test_conversion_params = {
        "_tftrt_convert_function": True,
        "_tftrt_trt_logger_name": "DefaultLogger",
        "_tftrt_max_batch_size": 10,
        "_tftrt_max_workspace_size_bytes":
            (trt_convert.DEFAULT_TRT_MAX_WORKSPACE_SIZE_BYTES),
        "_tftrt_precision_mode": "FP16",
        "_tftrt_minimum_segment_size": 2,
        "_tftrt_is_dyn_op": True,
        "_tftrt_max_cached_engines": 1,
        "_tftrt_use_calibration": False,
        "_tftrt_use_implicit_batch": True,
        "_tftrt_profile_strategy": self._profile_strategy,
        "_tftrt_allow_build_at_runtime": False
    }
    self._is_v2 = False

  def ShouldRunTest(self, run_params):
    should_run, reason_for_skipping = (
        trt_test.TfTrtIntegrationTestBase.ShouldRunTest(self, run_params))
    if not should_run:
      return should_run, reason_for_skipping
    else:
      # TODO(kyungtaek): Calibration currently does not run for nodes
      # nested within functions. If this gets fixed, this method should not
      # override the parent method.
      return (not IsQuantizationWithCalibration(run_params),
              "calibration is not supported for tf.functions")

  def _copy_test_attr_to_func_def(self, func_def, param_name, attr_value_type):
    test_value = self._test_conversion_params[param_name]
    if attr_value_type == "s":
      byte_value = compat.as_bytes(test_value)
      func_def.attr[param_name].CopyFrom(attr_value_pb2.AttrValue(s=byte_value))
    elif attr_value_type == "b":
      func_def.attr[param_name].CopyFrom(attr_value_pb2.AttrValue(b=test_value))
    elif attr_value_type == "i":
      func_def.attr[param_name].CopyFrom(attr_value_pb2.AttrValue(i=test_value))
    else:
      logging.info("Attr_value type %s is not supported", attr_value_type)

  def _ChainAllNodes(self, graph_def):
    return itertools.chain(
        graph_def.node,
        itertools.chain(
            *[func.node_def for func in graph_def.library.function]))

  def _VerifyTestAttrs(self, function_protos):
    if self._test_conversion_params["_tftrt_convert_function"]:
      for func_def in function_protos:
        if not func_def.signature.name.startswith("TRTEngine"):
          for key, value in self._test_conversion_params.items():
            self.assertIn(key, func_def.attr,
                          "key %s not found in func_def.attr" % key)
            if isinstance(value, str):
              self.assertEqual(func_def.attr[key].s, compat.as_bytes(value))
            elif isinstance(value, bool):
              self.assertEqual(func_def.attr[key].b, value)
            elif isinstance(value, int):
              self.assertEqual(func_def.attr[key].i, value)

  @def_function.function(input_signature=[
      tensor_spec.TensorSpec(shape=[None, 32, 32, 2], dtype=dtypes.float32)
  ])
  def _conv_and_pool_0(self, inp):
    dtype = inp.dtype
    conv_filter = constant_op.constant([[[[1., 0.5, 4.], [1., 0.5, 1.]]]],
                                       name="weights",
                                       dtype=dtype)
    conv = nn.conv2d(
        input=inp,
        filter=conv_filter,
        strides=[1, 2, 2, 1],
        padding="SAME",
        name="conv")
    bias = constant_op.constant([4., 1.5, 2.], name="bias", dtype=dtype)
    added = nn.bias_add(conv, bias, name="bias_add")
    relu = nn.relu(added, "relu")
    identity = array_ops.identity(relu, "identity")
    pool = nn_ops.max_pool(
        identity, [1, 2, 2, 1], [1, 2, 2, 1], "VALID", name="max_pool")
    return array_ops.squeeze(pool)

  def GraphFn(self, x):
    x = self._conv_and_pool_0(x)
    return array_ops.identity(x, name="output_0")

  def GetParams(self):
    return self.BuildParams(self.GraphFn, dtypes.float32, [[10, 32, 32, 2]],
                            [[10, 8, 8, 3]])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return {
        "TRTEngineOp_000": [
            "weights", "conv", "bias", "bias_add", "relu", "identity",
            "max_pool"
        ]
    }

  def _copy_test_attributes_to_func_def(self, func_def):
    self._copy_test_attr_to_func_def(
        func_def=func_def,
        param_name="_tftrt_convert_function",
        attr_value_type="b")
    self._copy_test_attr_to_func_def(
        func_def=func_def,
        param_name="_tftrt_trt_logger_name",
        attr_value_type="s")
    self._copy_test_attr_to_func_def(
        func_def=func_def,
        param_name="_tftrt_max_batch_size",
        attr_value_type="i")
    self._copy_test_attr_to_func_def(
        func_def=func_def,
        param_name="_tftrt_max_workspace_size_bytes",
        attr_value_type="i")
    self._copy_test_attr_to_func_def(
        func_def=func_def,
        param_name="_tftrt_precision_mode",
        attr_value_type="s")
    self._copy_test_attr_to_func_def(
        func_def=func_def,
        param_name="_tftrt_minimum_segment_size",
        attr_value_type="i")
    self._copy_test_attr_to_func_def(
        func_def=func_def, param_name="_tftrt_is_dyn_op", attr_value_type="b")
    self._copy_test_attr_to_func_def(
        func_def=func_def,
        param_name="_tftrt_max_cached_engines",
        attr_value_type="i")
    self._copy_test_attr_to_func_def(
        func_def=func_def,
        param_name="_tftrt_use_calibration",
        attr_value_type="b")
    self._copy_test_attr_to_func_def(
        func_def=func_def,
        param_name="_tftrt_use_implicit_batch",
        attr_value_type="b")
    self._copy_test_attr_to_func_def(
        func_def=func_def,
        param_name="_tftrt_profile_strategy",
        attr_value_type="s")
    self._copy_test_attr_to_func_def(
        func_def=func_def,
        param_name="_tftrt_allow_build_at_runtime",
        attr_value_type="b")

  def _MakeSavedModelV1(self, run_params):
    """Write the saved model as an input for testing.

    In addition to creating a SavedModel like its parent method, this method
    replaces this SavedModel by adding TF-TRT conversion parameters as function
    attributes to each function in the SavedModel.

    Args:
      run_params: The current test run parameters.

    Returns:
      The directory of the saved model.
    """
    saved_model_dir = trt_test.TfTrtIntegrationTestBase._MakeSavedModelV1(
        self, run_params)
    saved_model_proto = loader_impl.parse_saved_model(saved_model_dir)
    new_saved_model = saved_model_pb2.SavedModel()
    new_saved_model.CopyFrom(saved_model_proto)
    new_meta_graph_def = new_saved_model.meta_graphs[0]
    for func_def in new_meta_graph_def.graph_def.library.function:
      # Disable function inlining.
      func_def.attr["_noinline"].CopyFrom(attr_value_pb2.AttrValue(b=True))
      self._copy_test_attributes_to_func_def(func_def)
    old_saved_model_file = os.path.join(saved_model_dir,
                                        constants.SAVED_MODEL_FILENAME_PB)
    if os.path.exists(old_saved_model_file):
      os.remove(old_saved_model_file)
    path = os.path.join(
        compat.as_bytes(saved_model_dir),
        compat.as_bytes(constants.SAVED_MODEL_FILENAME_PB))
    file_io.write_string_to_file(
        path, new_saved_model.SerializeToString(deterministic=True))
    return saved_model_dir

  def _MakeSavedModelV2(self, run_params):
    """Write the saved model as an input for testing.

    In addition to creating a SavedModel like its parent method, this method
    replaces this SavedModel by adding TF-TRT conversion parameters as function
    attributes to each function in the SavedModel.

    Args:
      run_params: The current test run parameters.

    Returns:
      The directory of the saved model.
    """
    saved_model_dir = trt_test.TfTrtIntegrationTestBase._MakeSavedModelV2(
        self, run_params)
    saved_model_proto = loader_impl.parse_saved_model(saved_model_dir)
    new_saved_model = saved_model_pb2.SavedModel()
    new_saved_model.CopyFrom(saved_model_proto)
    new_meta_graph_def = new_saved_model.meta_graphs[0]
    prefix_len = len("__inference_")
    for func_def in new_meta_graph_def.graph_def.library.function:
      logging.info("_MakeSavedModelV2, func_def name: %s",
                   func_def.signature.name)
      func_name_without_prefix = func_def.signature.name[prefix_len:]
      if func_name_without_prefix.startswith(
          ("_conv_and_pool_0")):
        func_def.attr["_noinline"].CopyFrom(attr_value_pb2.AttrValue(b=True))
        self._copy_test_attributes_to_func_def(func_def)
    old_saved_model_file = os.path.join(saved_model_dir,
                                        constants.SAVED_MODEL_FILENAME_PB)
    if os.path.exists(old_saved_model_file):
      os.remove(old_saved_model_file)
    path = os.path.join(
        compat.as_bytes(saved_model_dir),
        compat.as_bytes(constants.SAVED_MODEL_FILENAME_PB))
    file_io.write_string_to_file(
        path, new_saved_model.SerializeToString(deterministic=True))
    return saved_model_dir

  def _VerifyGraphDefV1(self, run_params, original_gdef, gdef_to_verify,
                        graph_state):
    expected_engines = self.ExpectedEnginesToBuild(run_params)
    num_engines = 0
    functions = [f.signature.name for f in gdef_to_verify.library.function]
    all_nodes = list(self._ChainAllNodes(gdef_to_verify))
    all_nodes.sort(key=lambda x: x.name)

    for node in all_nodes:
      if node.op == "TRTEngineOp":
        logging.info("Found TRTEngineOp: " + node.name)
        num_engines += 1
        segment_funcdef_name = node.attr["segment_func"].func.name
        function_name = node.name + "_native_segment"
        is_dynamic_engine = not node.attr["static_engine"].b
        self.assertNotEmpty(segment_funcdef_name, node.name)
        self.assertIn(function_name, functions)
        if (not IsQuantizationWithCalibration(run_params) and
            not is_dynamic_engine):
          self.assertTrue(len(node.attr["serialized_segment"].s), node.name)
        self.assertIn(
            self._RemoveGraphSequenceNumber(node.name), expected_engines)
        self.assertEqual(
            self._ToBytes(run_params.precision_mode),
            node.attr["precision_mode"].s, node.name)

        self.assertEqual(run_params.dynamic_engine, is_dynamic_engine,
                         node.name)
        self.assertEqual(node.attr["use_calibration"].b,
                         run_params.use_calibration, node.name)

        has_calibration_data = len(node.attr["calibration_data"].s)
        if (IsQuantizationWithCalibration(run_params) and
            graph_state == GraphState.INFERENCE):
          self.assertTrue(has_calibration_data, node.name)
        else:
          self.assertFalse(has_calibration_data, node.name)
    if graph_state == GraphState.ORIGINAL:
      self.assertEqual(0, num_engines)
      self._VerifyTestAttrs(function_protos=gdef_to_verify.library.function)
    else:
      self.assertEqual(num_engines, len(expected_engines))
      expected_connections = self.ExpectedConnections(run_params)
      if expected_connections:
        self._VerifyConnections(expected_engines, expected_connections,
                                original_gdef, gdef_to_verify)
      self._VerifyMaxBatchSizeAnnotations(
          expected_engines=expected_engines,
          original_gdef=original_gdef,
          converted_gdef=gdef_to_verify,
          expected_max_batch_sizes=self.ExpectedMaxBatchSizes(run_params),
          default_max_batch_size=self.GetMaxBatchSize(run_params))
      self._VerifyTestAttrs(function_protos=gdef_to_verify.library.function)

  def _ShouldConverterBuild(self, run_params):
    return (run_params.is_v2 and not run_params.convert_online and
            run_params.dynamic_engine)

  def RunTest(self, run_params):
    self._test_conversion_params["_tftrt_precision_mode"] = (
        run_params.precision_mode)
    self._test_conversion_params["_tftrt_use_calibration"] = (
        run_params.use_calibration)
    self._test_conversion_params["_tftrt_is_dyn_op"] = (
        run_params.dynamic_engine)
    # When running with V1, using dynamic_engine and
    # allow_build_at_runtime==False at the same time do not work.
    if run_params.is_v2:
      self._test_conversion_params["_tftrt_allow_build_at_runtime"] = True
      self._is_v2 = True
    else:
      self._test_conversion_params["_tftrt_allow_build_at_runtime"] = (
          run_params.convert_online or run_params.dynamic_engine)
    self._test_conversion_params["_tftrt_use_implicit_batch"] = \
        not run_params.dynamic_shape
    self.DisableNonTrtOptimizers()
    trt_test.TfTrtIntegrationTestBase.RunTest(self, run_params)


if __name__ == "__main__":
  test.main()
