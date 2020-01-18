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
"""Utilities to test TF-TensorRT integration."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gc
import os
import tempfile

from absl.testing import parameterized
import numpy as np

from tensorflow.compiler.tf2tensorrt.wrap_py_utils import is_tensorrt_enabled
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.compiler.tensorrt import trt_convert
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.tools import saved_model_utils
from tensorflow.python.training.tracking import tracking
from tensorflow.python.util.lazy_loader import LazyLoader

_SAVED_MODEL_SIGNATURE_KEY = "mypredict"

gen_trt_ops = LazyLoader(
    "gen_trt_ops", globals(),
    "tensorflow.compiler.tf2tensorrt.ops.gen_trt_ops")


class TrtConvertTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  """Class to test Tensorflow-TensorRT integration python API."""

  # Use a small max_workspace_size for tests so they don't consume too much GPU
  # memory.
  _TRT_MAX_WORKSPACE_SIZE_BYTES = 2 << 20

  def mkdtemp(self):
    return tempfile.mkdtemp(dir=self.get_temp_dir())

  def testGetTensorrtRewriterConfig(self):
    """Test case for TrtGraphConverter.get_tensorrt_rewriter_config()."""
    if not is_tensorrt_enabled():
      return
    conversion_params = trt_convert.DEFAULT_TRT_CONVERSION_PARAMS._replace(
        max_batch_size=128,
        max_workspace_size_bytes=1234,
        precision_mode="INT8",
        minimum_segment_size=10,
        is_dynamic_op=True,
        maximum_cached_engines=2)
    rewriter_cfg = trt_convert.get_tensorrt_rewriter_config(
        conversion_params=conversion_params)
    self.assertEqual(["constfold", "layout", "constfold"],
                     rewriter_cfg.optimizers)
    self.assertEqual(rewriter_config_pb2.RewriterConfig.ONE,
                     rewriter_cfg.meta_optimizer_iterations)
    trt_optimizer = None
    for optimizer in rewriter_cfg.custom_optimizers:
      if optimizer.name == "TensorRTOptimizer":
        self.assertTrue(trt_optimizer is None)
        trt_optimizer = optimizer
    self.assertTrue(trt_optimizer is not None)
    for key in [
        "minimum_segment_size", "max_batch_size", "is_dynamic_op",
        "max_workspace_size_bytes", "precision_mode", "maximum_cached_engines"
    ]:
      self.assertTrue(key in trt_optimizer.parameter_map)
    self.assertEqual(10, trt_optimizer.parameter_map["minimum_segment_size"].i)
    self.assertEqual(128, trt_optimizer.parameter_map["max_batch_size"].i)
    self.assertEqual(True, trt_optimizer.parameter_map["is_dynamic_op"].b)
    self.assertEqual(1234,
                     trt_optimizer.parameter_map["max_workspace_size_bytes"].i)
    self.assertEqual(
        trt_convert._to_bytes("INT8"),
        trt_optimizer.parameter_map["precision_mode"].s)
    self.assertEqual(2, trt_optimizer.parameter_map["maximum_cached_engines"].i)

  def testGetTensorrtRewriterConfigTemplate(self):
    """Test case for TrtGraphConverter.get_tensorrt_rewriter_config()."""
    if not is_tensorrt_enabled():
      return

    rewriter_config_with_trt = rewriter_config_pb2.RewriterConfig()
    rewriter_config_with_trt.optimizers.extend(
        ["constfold", "layout", "constfold"])
    rewriter_config_with_trt.meta_optimizer_iterations = (
        rewriter_config_pb2.RewriterConfig.ONE)
    optimizer = rewriter_config_with_trt.custom_optimizers.add()
    rewriter_config_with_trt.custom_optimizers.add().name = "constfold"
    optimizer.name = "TensorRTOptimizer"
    optimizer.parameter_map["minimum_segment_size"].i = 10
    optimizer.parameter_map["max_batch_size"].i = 128
    optimizer.parameter_map["is_dynamic_op"].b = True
    optimizer.parameter_map["max_workspace_size_bytes"].i = 1234
    optimizer.parameter_map["precision_mode"].s = trt_convert._to_bytes(
        trt_convert.TrtPrecisionMode.INT8)
    optimizer.parameter_map["maximum_cached_engines"].i = 2
    optimizer.parameter_map["use_calibration"].b = False
    optimizer.parameter_map["use_implicit_batch"].b = True

    conversion_params = trt_convert.DEFAULT_TRT_CONVERSION_PARAMS._replace(
        rewriter_config_template=rewriter_config_with_trt)
    rewriter_cfg = trt_convert.get_tensorrt_rewriter_config(
        conversion_params=conversion_params)
    self.assertEqual(["constfold", "layout", "constfold"],
                     rewriter_cfg.optimizers)
    self.assertEqual(rewriter_config_pb2.RewriterConfig.ONE,
                     rewriter_cfg.meta_optimizer_iterations)
    trt_optimizer = None
    for optimizer in rewriter_cfg.custom_optimizers:
      if optimizer.name == "TensorRTOptimizer":
        self.assertIsNone(trt_optimizer)
        trt_optimizer = optimizer
    self.assertIsNotNone(trt_optimizer)
    for key in [
        "minimum_segment_size", "max_batch_size", "is_dynamic_op",
        "max_workspace_size_bytes", "precision_mode", "maximum_cached_engines"
    ]:
      self.assertIn(key, trt_optimizer.parameter_map)
    self.assertEqual(10, trt_optimizer.parameter_map["minimum_segment_size"].i)
    self.assertEqual(128, trt_optimizer.parameter_map["max_batch_size"].i)
    self.assertEqual(True, trt_optimizer.parameter_map["is_dynamic_op"].b)
    self.assertEqual(1234,
                     trt_optimizer.parameter_map["max_workspace_size_bytes"].i)
    self.assertEqual(
        trt_convert._to_bytes("INT8"),
        trt_optimizer.parameter_map["precision_mode"].s)
    self.assertEqual(2, trt_optimizer.parameter_map["maximum_cached_engines"].i)
    self.assertEqual(False, trt_optimizer.parameter_map["use_calibration"].b)
    self.assertEqual(True, trt_optimizer.parameter_map["use_implicit_batch"].b)

  def testOptimizationProfiles(self):
    """Test case for TrtGraphConverter.get_tensorrt_rewriter_config()."""
    if not is_tensorrt_enabled():
      return

    def _graph_fn(x):
      x = gen_math_ops.log(x)
      return x

    def _model():
      class SimpleModel(tracking.AutoTrackable):
        @def_function.function(input_signature=[
            tensor_spec.TensorSpec(shape=[None, None], dtype=dtypes.float32),
        ])
        def run(self, inp):
          return _graph_fn(inp)
      return SimpleModel()

    root = _model()
    input_saved_model_dir = self.mkdtemp()
    save.save(root, input_saved_model_dir,
              {signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: root.run})

    rewriter_config_with_trt = rewriter_config_pb2.RewriterConfig()
    rewriter_config_with_trt.optimizers.extend(
        ["constfold", "layout", "constfold"])
    rewriter_config_with_trt.meta_optimizer_iterations = (
        rewriter_config_pb2.RewriterConfig.ONE)
    optimizer = rewriter_config_with_trt.custom_optimizers.add()
    rewriter_config_with_trt.custom_optimizers.add().name = "constfold"
    optimizer.name = "TensorRTOptimizer"
    optimizer.parameter_map["minimum_segment_size"].i = 1
    optimizer.parameter_map["is_dynamic_op"].b = True
    optimizer.parameter_map["maximum_cached_engines"].i = 1
    optimizer.parameter_map["use_implicit_batch"].b = False

    conversion_params = trt_convert.DEFAULT_TRT_CONVERSION_PARAMS._replace(
        rewriter_config_template=rewriter_config_with_trt)
    converter = trt_convert.TrtGraphConverterV2(input_saved_model_dir=input_saved_model_dir,
                                    conversion_params=conversion_params)
    converter.convert()

    input_shapes = [(2, 4), (3, 9), (64, 128)]
    def my_input_fn():
      for x in input_shapes:
        yield (np.random.normal(size=x).astype(np.float32),)

    converter.build(input_fn=my_input_fn)

    output_saved_model_dir = self.mkdtemp()
    converter.save(output_saved_model_dir=output_saved_model_dir)

    saved_model_loaded = load.load(
        output_saved_model_dir, tags=[tag_constants.SERVING])
    graph_func = saved_model_loaded.signatures[
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    input_shapes = [(2, 4), (3, 9), (64, 128), (2, 6), (4, 9)]
    for x in input_shapes:
      input = ops.convert_to_tensor(
        np.random.normal(size=x).astype(np.float32))
      graph_func(input)

  def _GetConfigProto(self, rewriter_config=None):
    """Get ConfigProto for session creation."""
    config = config_pb2.ConfigProto(
        gpu_options=config_pb2.GPUOptions(allow_growth=True))
    if rewriter_config:
      config.graph_options.rewrite_options.CopyFrom(rewriter_config)
    return config

  @classmethod
  def _GetGraph(cls, inp1, inp2, var):
    """Get the graph for testing."""
    # The graph computes: inp1^2 + inp1*var + inp1 + inp2 + var
    add = inp1 + var
    mul = inp1 * add
    add = mul + add
    add = add + inp2
    out = array_ops.identity(add, name="output")
    return out

  def _GetModelForV2(self):

    class SimpleModel(tracking.AutoTrackable):

      def __init__(self):
        self.v = None

      @def_function.function(input_signature=[
          tensor_spec.TensorSpec(shape=[None, 1, 1], dtype=dtypes.float32),
          tensor_spec.TensorSpec(shape=[None, 1, 1], dtype=dtypes.float32)
      ])
      def run(self, inp1, inp2):
        if self.v is None:
          self.v = variables.Variable([[[1.0]]], dtype=dtypes.float32)
        return TrtConvertTest._GetGraph(inp1, inp2, self.v)

    return SimpleModel()

  def _GetGraphForV1(self, device):

    def _GraphFn():
      inp1 = array_ops.placeholder(
          dtype=dtypes.float32, shape=[None, 1, 1], name="input1")
      inp2 = array_ops.placeholder(
          dtype=dtypes.float32, shape=[None, 1, 1], name="input2")
      var = variables.Variable([[[1.0]]], dtype=dtypes.float32, name="v1")
      out = TrtConvertTest._GetGraph(inp1, inp2, var)
      return g, var, inp1, inp2, out

    g = ops.Graph()
    with g.as_default():
      if device:
        with g.device(device):
          return _GraphFn()
      return _GraphFn()

  def _GetGraphDefForV1(self, device):
    """Get the graph def for testing."""
    g, var, _, _, _ = self._GetGraphForV1(device)
    with self.session(graph=g, config=self._GetConfigProto()) as sess:
      sess.run(var.initializer)
      graph_def = graph_util.convert_variables_to_constants(
          sess, g.as_graph_def(add_shapes=True), ["output"])
    node_name_to_op = {node.name: node.op for node in graph_def.node}
    self.assertEqual(
        {
            "v1": "Const",
            "add/ReadVariableOp": "Identity",
            "input1": "Placeholder",
            "input2": "Placeholder",
            "add": "AddV2",
            "mul": "Mul",
            "add_1": "AddV2",
            "add_2": "AddV2",
            "output": "Identity"
        }, node_name_to_op)
    return graph_def

  def _WriteInputSavedModelForV1(self, input_saved_model_dir, device):
    """Write the saved model as an input for testing."""
    g, var, inp1, inp2, out = self._GetGraphForV1(device)
    signature_def = signature_def_utils.build_signature_def(
        inputs={
            "myinput1": utils.build_tensor_info(inp1),
            "myinput2": utils.build_tensor_info(inp2)
        },
        outputs={"myoutput": utils.build_tensor_info(out)},
        method_name=signature_constants.PREDICT_METHOD_NAME)
    saved_model_builder = builder.SavedModelBuilder(input_saved_model_dir)
    with self.session(graph=g, config=self._GetConfigProto()) as sess:
      sess.run(var.initializer)
      saved_model_builder.add_meta_graph_and_variables(
          sess, [tag_constants.SERVING],
          signature_def_map={_SAVED_MODEL_SIGNATURE_KEY: signature_def})
    saved_model_builder.save()

  def _ConvertGraphV1(self,
                      output_saved_model_dir=None,
                      need_calibration=False,
                      max_batch_size=1,
                      minimum_segment_size=3,
                      is_dynamic_op=False,
                      maximum_cached_engines=1,
                      device=None):
    """Helper method to convert a GraphDef or SavedModel using TF-TRT."""
    input_saved_model_dir = None
    if output_saved_model_dir:
      input_saved_model_dir = self.mkdtemp()
      self._WriteInputSavedModelForV1(input_saved_model_dir, device)

    converter = trt_convert.TrtGraphConverter(
        input_saved_model_dir=input_saved_model_dir,
        input_saved_model_signature_key=_SAVED_MODEL_SIGNATURE_KEY,
        input_graph_def=None
        if input_saved_model_dir else self._GetGraphDefForV1(device),
        nodes_blacklist=None if input_saved_model_dir else ["output"],
        session_config=self._GetConfigProto(),
        max_batch_size=max_batch_size,
        max_workspace_size_bytes=TrtConvertTest._TRT_MAX_WORKSPACE_SIZE_BYTES,
        precision_mode=(trt_convert.TrtPrecisionMode.INT8 if need_calibration
                        else trt_convert.TrtPrecisionMode.FP32),
        minimum_segment_size=minimum_segment_size,
        is_dynamic_op=is_dynamic_op,
        maximum_cached_engines=maximum_cached_engines)
    output_graph_def = converter.convert()

    if need_calibration:

      class CalibrationData(object):

        def __init__(self):
          self._data = 0

        def next(self):
          self._data += 1
          return {"input1:0": [[[self._data]]], "input2:0": [[[self._data]]]}

      output_graph_def = converter.calibrate(
          fetch_names=["output:0"],
          num_runs=10,
          feed_dict_fn=CalibrationData().next)

    if output_saved_model_dir is not None:
      converter.save(output_saved_model_dir=output_saved_model_dir)
    return output_graph_def

  def _TestTrtGraphConverter(self,
                             device,
                             output_saved_model_dir=None,
                             need_calibration=False,
                             is_dynamic_op=False):
    """General method to test trt_convert.TrtGraphConverter()."""
    output_graph_def = self._ConvertGraphV1(
        output_saved_model_dir=output_saved_model_dir,
        need_calibration=need_calibration,
        is_dynamic_op=is_dynamic_op,
        device=device)
    graph_defs_to_verify = [output_graph_def]

    if output_saved_model_dir:
      saved_model_graph_def = saved_model_utils.get_meta_graph_def(
          output_saved_model_dir, tag_constants.SERVING).graph_def
      self.assertIsInstance(saved_model_graph_def, graph_pb2.GraphDef)
      graph_defs_to_verify.append(saved_model_graph_def)

    for graph_def in graph_defs_to_verify:
      node_name_to_op = {node.name: node.op for node in graph_def.node}
      self.assertEqual(
          {
              "input1": "Placeholder",
              "input2": "Placeholder",
              "TRTEngineOp_0": "TRTEngineOp",
              "output": "Identity"
          }, node_name_to_op)

      if need_calibration:
        trt_engine_nodes = [
            node for node in graph_def.node if node.op == "TRTEngineOp"
        ]
        self.assertNotEmpty(trt_engine_nodes)
        for node in trt_engine_nodes:
          self.assertTrue(len(node.attr["calibration_data"].s))
        # Run the calibrated graph.
        # TODO(laigd): consider having some input where the answer is different.
        with ops.Graph().as_default():
          importer.import_graph_def(graph_def, name="")
          with self.session(config=self._GetConfigProto()) as sess:
            for test_data in range(10):
              self.assertEqual((test_data + 1.0)**2 + test_data,
                               sess.run(
                                   "output:0",
                                   feed_dict={
                                       "input1:0": [[[test_data]]],
                                       "input2:0": [[[test_data]]]
                                   }))

  @parameterized.named_parameters([
      ("NoDeviceAssignment", None),
      ("GPU", "/GPU:0"),
      ("CPU", "/CPU:0"),
  ])
  @test_util.deprecated_graph_mode_only
  def testTrtGraphConverter_OfflineConversion(self, device):
    """Test case for trt_convert.TrtGraphConverter()."""
    if not is_tensorrt_enabled():
      return

    for need_calibration in [False, True]:
      # Use GraphDef as input.
      self._TestTrtGraphConverter(device)

      # Use SavedModel as input.
      self._TestTrtGraphConverter(
          device,
          output_saved_model_dir=self.mkdtemp(),
          need_calibration=need_calibration)

  @parameterized.named_parameters([
      ("NoDeviceAssignment", None),
      ("GPU", "/device:GPU:0"),
      ("CPU", "/device:CPU:0"),
  ])
  @test_util.deprecated_graph_mode_only
  def testTrtGraphConverter_OnlineConversion(self, device):
    """Test case for TF-TRT conversion using Grappler directly."""
    if not is_tensorrt_enabled():
      return

    conversion_params = trt_convert.DEFAULT_TRT_CONVERSION_PARAMS._replace(
        precision_mode=trt_convert.TrtPrecisionMode.FP32, is_dynamic_op=True)
    config = self._GetConfigProto(
        rewriter_config=trt_convert.get_tensorrt_rewriter_config(
            conversion_params, is_v2=False))

    with ops.Graph().as_default():
      # Online conversion requires a frozen graph, so we reuse inp1 as the var
      # argument.
      inp1 = array_ops.placeholder(
          dtype=dtypes.float32, shape=[None, 1, 1], name="input1")
      inp2 = array_ops.placeholder(
          dtype=dtypes.float32, shape=[None, 1, 1], name="input2")
      if device:
        with ops.device(device):
          TrtConvertTest._GetGraph(inp1, inp2, inp1)
      else:
        TrtConvertTest._GetGraph(inp1, inp2, inp1)
      with self.session(config=config) as sess:
        self._TestRun(sess, batch_size=1)

  def _CreateConverterV2(
      self,
      input_saved_model_dir,
      input_saved_model_signature_key=_SAVED_MODEL_SIGNATURE_KEY,
      precision_mode=trt_convert.TrtPrecisionMode.FP32,
      is_dynamic_op=True,
      maximum_cached_engines=2):
    return trt_convert.TrtGraphConverterV2(
        input_saved_model_dir=input_saved_model_dir,
        input_saved_model_signature_key=input_saved_model_signature_key,
        conversion_params=trt_convert.DEFAULT_TRT_CONVERSION_PARAMS._replace(
            max_workspace_size_bytes=10 << 20,  # Use a smaller workspace.
            precision_mode=precision_mode,
            is_dynamic_op=is_dynamic_op,
            maximum_cached_engines=maximum_cached_engines))

  def _CheckTrtOps(self, concrete_func, check_fn=None):
    graph_def = concrete_func.graph.as_graph_def()
    trt_op_names = []
    for node in graph_def.node:
      if node.op == "TRTEngineOp":
        trt_op_names.append(node.name)
        if check_fn:
          check_fn(node)
    for func in graph_def.library.function:
      for node in func.node_def:
        if node.op == "TRTEngineOp":
          trt_op_names.append(node.name)
          if check_fn:
            check_fn(node)
    self.assertEqual(1, len(trt_op_names))
    self.assertIn("TRTEngineOp_0", trt_op_names[0])

  def _RandomInput(self, shape, dtype=np.float32):
    inp1 = np.random.random_sample(shape).astype(dtype)
    inp2 = np.random.random_sample(shape).astype(dtype)
    return inp1, inp2

  @test_util.run_v2_only
  def testTrtGraphConverter_DynamicConversion_v2(self):
    """Test case for trt_convert.TrtGraphConverter()."""
    if not is_tensorrt_enabled():
      return

    np_input1, np_input2 = self._RandomInput([4, 1, 1])

    # Create a model and save it.
    input_saved_model_dir = self.mkdtemp()
    root = self._GetModelForV2()
    expected_output = root.run(np_input1, np_input2)
    save.save(root, input_saved_model_dir,
              {_SAVED_MODEL_SIGNATURE_KEY: root.run})

    # Run TRT conversion.
    converter = self._CreateConverterV2(input_saved_model_dir)
    converter.convert()

    # Verify the converted GraphDef and ConcreteFunction.
    self._CheckTrtOps(converter._converted_func)  # pylint: disable=protected-access

    # Save the converted model without any TRT engine cache.
    output_saved_model_dir = self.mkdtemp()
    converter.save(output_saved_model_dir)
    unexpected_asset_file = os.path.join(
        output_saved_model_dir, "assets/trt-serialized-engine.TRTEngineOp_0")
    self.assertFalse(os.path.exists(unexpected_asset_file))

    # Run the converted function to populate the engine cache.
    def _InputFn():
      yield np_input1, np_input2

    converter.build(input_fn=_InputFn)

    # Save the converted model again with serialized engine cache.
    output_saved_model_dir = self.mkdtemp()
    converter.save(output_saved_model_dir)
    expected_asset_file = os.path.join(
        output_saved_model_dir, "assets/trt-serialized-engine.TRTEngineOp_0")
    self.assertTrue(os.path.exists(expected_asset_file))
    self.assertTrue(os.path.getsize(expected_asset_file))

    del converter
    gc.collect()  # Force GC to destroy the TRT engine cache.

    # Load and verify the converted model.
    #
    # TODO(laigd): the name of the new input_signature of the
    # `root_with_trt.run` function is empty string (originaly was None),
    # investigate why.
    root_with_trt = load.load(output_saved_model_dir)
    # TODO(laigd): `root_with_trt.run` is still using the original graph without
    # trt. Consider changing that.
    # self._CheckTrtOps(root_with_trt.run.get_concrete_function())
    converted_signature = root_with_trt.signatures[_SAVED_MODEL_SIGNATURE_KEY]
    self._CheckTrtOps(converted_signature)
    output_with_trt = converted_signature(
        inp1=ops.convert_to_tensor(np_input1),
        inp2=ops.convert_to_tensor(np_input2))
    # The output of running the converted signature is a dict due to
    # compatibility reasons with V1 SavedModel signature mechanism.
    self.assertAllClose(
        expected_output,
        list(output_with_trt.values())[0],
        atol=1e-6,
        rtol=1e-6)

    del root_with_trt
    gc.collect()  # Force GC to destroy the TRT engine cache.

  @test_util.run_v2_only
  def testTrtGraphConverter_StaticConversionNotSupportedInV2(self):
    """Test case for trt_convert.TrtGraphConverter() using static mode."""
    if not is_tensorrt_enabled():
      return

    # Create a model and save it.
    input_saved_model_dir = self.mkdtemp()
    root = self._GetModelForV2()
    save.save(root, input_saved_model_dir,
              {_SAVED_MODEL_SIGNATURE_KEY: root.run})

    # Run TRT conversion.
    with self.assertRaisesRegexp(
        ValueError, r"Option is_dynamic_op=False is not supported in TF 2.0, "
        "please set it to True instead."):
      self._CreateConverterV2(input_saved_model_dir, is_dynamic_op=False)

  @test_util.run_v2_only
  def testTrtGraphConverter_Int8Conversion_v2(self):
    if not is_tensorrt_enabled():
      return

    np_input1, np_input2 = self._RandomInput([4, 1, 1])

    # Create a model and save it.
    input_saved_model_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    root = self._GetModelForV2()
    expected_output = root.run(np_input1, np_input2)
    save.save(root, input_saved_model_dir,
              {_SAVED_MODEL_SIGNATURE_KEY: root.run})

    # Run TRT conversion.
    converter = self._CreateConverterV2(
        input_saved_model_dir,
        precision_mode=trt_convert.TrtPrecisionMode.INT8,
        maximum_cached_engines=3)

    # Convert and perform INT8 calibration
    def _CalibrationInputFn():
      yield np_input1, np_input2

    converter.convert(calibration_input_fn=_CalibrationInputFn)

    def _CheckFn(node):
      self.assertTrue(len(node.attr["calibration_data"].s), node.name)

    # Verify the converted GraphDef.
    self._CheckTrtOps(converter._converted_func, _CheckFn)  # pylint: disable=protected-access

    # Build another engine with different batch size.
    def _InputFn():
      yield self._RandomInput([5, 1, 1])

    converter.build(input_fn=_InputFn)

    # Save the converted model.
    # TODO(laigd): check that it should contain two engines.
    output_saved_model_dir = self.mkdtemp()
    converter.save(output_saved_model_dir)
    expected_asset_file = os.path.join(
        output_saved_model_dir, "assets/trt-serialized-engine.TRTEngineOp_0")
    self.assertTrue(os.path.exists(expected_asset_file))
    self.assertTrue(os.path.getsize(expected_asset_file))

    del converter
    gc.collect()  # Force GC to destroy the TRT engine cache.

    # Load and verify the converted model.
    root_with_trt = load.load(output_saved_model_dir)
    converted_signature = root_with_trt.signatures[_SAVED_MODEL_SIGNATURE_KEY]
    self._CheckTrtOps(converted_signature, _CheckFn)
    output_with_trt = converted_signature(
        inp1=ops.convert_to_tensor(np_input1),
        inp2=ops.convert_to_tensor(np_input2))
    self.assertEqual(1, len(output_with_trt))
    # The output of running the converted signature is a dict due to
    # compatibility reasons with V1 SavedModel signature mechanism.
    self.assertAllClose(
        expected_output,
        list(output_with_trt.values())[0],
        atol=1e-6,
        rtol=1e-6)

    # Run with an input of different batch size. It should build a new engine
    # using calibration table.
    # TODO(laigd): check that it should contain three engines.
    np_input1, np_input2 = self._RandomInput([6, 1, 1])
    converted_signature(
        inp1=ops.convert_to_tensor(np_input1),
        inp2=ops.convert_to_tensor(np_input2))

    del root_with_trt
    gc.collect()  # Force GC to destroy the TRT engine cache.

  @test_util.run_v2_only
  def testTrtGraphConverter_DestroyEngineCache(self):
    """Test case for trt_convert.TrtGraphConverter()."""
    if not is_tensorrt_enabled():
      return

    np_input1, np_input2 = self._RandomInput([4, 1, 1])

    # Create a model and save it.
    input_saved_model_dir = self.mkdtemp()
    root = self._GetModelForV2()
    save.save(root, input_saved_model_dir,
              {_SAVED_MODEL_SIGNATURE_KEY: root.run})

    # Run TRT conversion.
    converter = self._CreateConverterV2(input_saved_model_dir)
    converter.convert()

    def _InputFn():
      yield np_input1, np_input2

    converter.build(input_fn=_InputFn)  # Populate the TRT engine cache.
    output_saved_model_dir = self.mkdtemp()
    converter.save(output_saved_model_dir)

    def _DestroyCache():
      with ops.device("GPU:0"):
        handle = gen_trt_ops.create_trt_resource_handle(
            resource_name="TRTEngineOp_0")
        gen_resource_variable_ops.destroy_resource_op(
            handle, ignore_lookup_error=False)

    with self.assertRaisesRegexp(errors.NotFoundError,
                                 r"Resource .* does not exist."):
      _DestroyCache()

    # Load the converted model and make sure the engine cache is populated by
    # default.
    root = load.load(output_saved_model_dir)
    _DestroyCache()
    with self.assertRaisesRegexp(errors.NotFoundError,
                                 r"Resource .* does not exist."):
      _DestroyCache()

    # Load the converted model again and make sure the engine cache is destroyed
    # when the model goes out of scope.
    root = load.load(output_saved_model_dir)
    del root
    gc.collect()  # Force GC to destroy the TRT engine cache.
    with self.assertRaisesRegexp(errors.NotFoundError,
                                 r"Resource .* does not exist."):
      _DestroyCache()

  def _CompareSavedModel(self, model_class):
    signature_key = "serving_default"

    def _GetModelPaths(model_class):
      input_saved_model_dir = self.mkdtemp()
      root = model_class()
      save.save(root, input_saved_model_dir)

      converter = self._CreateConverterV2(
          input_saved_model_dir, input_saved_model_signature_key=signature_key)
      converter.convert()
      output_saved_model_dir = self.mkdtemp()
      converter.save(output_saved_model_dir)
      return input_saved_model_dir, output_saved_model_dir

    def _GetSignatureDef(export_dir):
      saved_model_proto = loader_impl.parse_saved_model(export_dir)
      self.assertEqual(1, len(saved_model_proto.meta_graphs))
      meta_graph = saved_model_proto.meta_graphs[0]
      self.assertIn(signature_key, meta_graph.signature_def)
      return meta_graph.signature_def[signature_key]

    def _CompareSignatureDef(original_def, converted_def, is_input):
      endpoints = original_def.inputs if is_input else original_def.outputs
      converted_endpoints = (
          converted_def.inputs if is_input else converted_def.outputs)
      self.assertEqual(set(endpoints.keys()), set(converted_endpoints.keys()))
      for key in endpoints:
        original_input = endpoints[key]
        converted_input = converted_endpoints[key]
        self.assertEqual(original_input.name, converted_input.name)
        self.assertEqual(original_input.dtype, converted_input.dtype)
        self.assertEqual(
            tensor_shape.TensorShape(original_input.tensor_shape).as_list(),
            tensor_shape.TensorShape(converted_input.tensor_shape).as_list())

    def _GetStructuredOutputs(export_dir):
      root = load.load(export_dir)
      return root.signatures[signature_key].structured_outputs

    saved_model_path, converted_saved_model_path = _GetModelPaths(model_class)
    original_def = _GetSignatureDef(saved_model_path)
    converted_def = _GetSignatureDef(converted_saved_model_path)
    self.assertEqual(original_def.method_name, converted_def.method_name)
    _CompareSignatureDef(original_def, converted_def, True)
    _CompareSignatureDef(original_def, converted_def, False)

    self.assertEqual(
        _GetStructuredOutputs(saved_model_path),
        _GetStructuredOutputs(converted_saved_model_path))

  @test_util.run_v2_only
  def testRetainSignatureInfo_NoInputs(self):
    if not is_tensorrt_enabled():
      return

    class _Model(tracking.AutoTrackable):

      @def_function.function(input_signature=[])
      def run(self):
        return array_ops.constant(1.0)

    self._CompareSavedModel(_Model)

  @test_util.run_v2_only
  def testRetainSignatureInfo_OneInput(self):
    if not is_tensorrt_enabled():
      return

    class _Model(tracking.AutoTrackable):

      @def_function.function(input_signature=[
          tensor_spec.TensorSpec(shape=[None, 1], dtype=dtypes.float32)
      ])
      def run(self, inp):
        return inp + inp * inp

    self._CompareSavedModel(_Model)

  @test_util.run_v2_only
  def testRetainSignatureInfo_TwoInputs(self):
    if not is_tensorrt_enabled():
      return

    class _Model(tracking.AutoTrackable):

      @def_function.function(input_signature=[
          tensor_spec.TensorSpec(shape=[None, 1], dtype=dtypes.float32),
          tensor_spec.TensorSpec(shape=[None, 2], dtype=dtypes.float32)
      ])
      def run(self, inp1, inp2):
        return inp1 + inp2 * inp2

    self._CompareSavedModel(_Model)

  @test_util.run_v2_only
  def testRetainSignatureInfo_OneOutputSignatureKey(self):
    if not is_tensorrt_enabled():
      return

    class _Model(tracking.AutoTrackable):

      @def_function.function(input_signature=[])
      def run(self):
        return {"my_output": array_ops.constant(1.0)}

    self._CompareSavedModel(_Model)

  @test_util.run_v2_only
  def testRetainSignatureInfo_TwoOutputSignatureKeys(self):
    if not is_tensorrt_enabled():
      return

    class _Model(tracking.AutoTrackable):

      @def_function.function(input_signature=[
          tensor_spec.TensorSpec(shape=[None, 1], dtype=dtypes.float32)
      ])
      def run(self, inp):
        # Here the keys are not ordered lexicographically on purpose.
        return {
            "output_b": array_ops.constant(1.0),
            "output_a": inp + inp * inp
        }

    self._CompareSavedModel(_Model)

  def _TestRun(self, sess, batch_size):
    result = sess.run(
        "output:0",
        feed_dict={
            "input1:0": [[[1.0]]] * batch_size,
            "input2:0": [[[1.0]]] * batch_size
        })
    self.assertAllEqual([[[5.0]]] * batch_size, result)

  @test_util.deprecated_graph_mode_only
  def testTrtGraphConverter_MinimumSegmentSize(self):
    if not is_tensorrt_enabled():
      return
    output_graph_def = self._ConvertGraphV1(minimum_segment_size=7)
    node_name_to_op = {node.name: node.op for node in output_graph_def.node}
    self.assertEqual(
        {
            "add/ReadVariableOp": "Const",
            "input1": "Placeholder",
            "input2": "Placeholder",
            "add": "AddV2",
            "mul": "Mul",
            "add_1": "AddV2",
            "add_2": "AddV2",
            "output": "Identity"
        }, node_name_to_op)

  @test_util.deprecated_graph_mode_only
  def testTrtGraphConverter_DynamicOp(self):
    if not is_tensorrt_enabled():
      return

    output_saved_model_dir = self.mkdtemp()
    output_graph_def = self._ConvertGraphV1(
        output_saved_model_dir=output_saved_model_dir,
        is_dynamic_op=True,
        maximum_cached_engines=2)

    # Test the output GraphDef.
    with ops.Graph().as_default():
      importer.import_graph_def(output_graph_def, name="")
      with self.session(config=self._GetConfigProto()) as sess:
        # Run with batch size 1, a new engine is created and cached.
        self._TestRun(sess, 1)
        # Run with batch size 2, a new engine is created and cached.
        self._TestRun(sess, 2)
        # Run with batch size 3, since the number of cached engines has reached
        # the max, it should evict an old engine and create a new one.
        self._TestRun(sess, 3)

    # Test the output SavedModel
    with ops.Graph().as_default():
      with self.session(config=self._GetConfigProto()) as sess:
        loader.load(sess, [tag_constants.SERVING], output_saved_model_dir)
        # Run with batch size 1, a new engine is created and cached.
        self._TestRun(sess, 1)
        # Run with batch size 2, a new engine is created and cached.
        self._TestRun(sess, 2)
        # Run with batch size 3, since the number of cached engines has reached
        # the max, it should evict an old engine and create a new one.
        self._TestRun(sess, 3)

  @test_util.deprecated_graph_mode_only
  def testTrtGraphConverter_StaticOp(self):
    if not is_tensorrt_enabled():
      return

    output_saved_model_dir = self.mkdtemp()
    output_graph_def = self._ConvertGraphV1(
        output_saved_model_dir=output_saved_model_dir, maximum_cached_engines=1)

    # Test the output GraphDef.
    with ops.Graph().as_default():
      importer.import_graph_def(output_graph_def, name="")
      with self.session(config=self._GetConfigProto()) as sess:
        # Run with batch size 1, the default engine embedded in the graphdef
        # will be used.
        self._TestRun(sess, 1)
        # Run with batch size 2, which exceed the max_batch_size, it should try
        # to fall back to TF function.
        self._TestRun(sess, 2)

    # Test the output SavedModel
    with ops.Graph().as_default():
      with self.session(config=self._GetConfigProto()) as sess:
        loader.load(sess, [tag_constants.SERVING], output_saved_model_dir)
        # Run with batch size 1, the default engine embedded in the graphdef
        # will be used.
        self._TestRun(sess, 1)
        # Run with batch size 2, which exceed the max_batch_size, it should try
        # to fall back to TF function.
        self._TestRun(sess, 2)

  @test_util.run_v2_only
  def testBackwardCompatibility(self):
    """Load and execute a model that was saved in TF2.0."""
    if not is_tensorrt_enabled():
      return

    model_dir = test.test_src_dir_path(
        "python/compiler/tensorrt/test/testdata/tftrt_2.0_saved_model")
    saved_model_loaded = load.load(model_dir, tags=[tag_constants.SERVING])
    graph_func = saved_model_loaded.signatures[
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

    np_input1 = ops.convert_to_tensor(np.ones([4, 1, 1]).astype(np.float32))
    np_input2 = ops.convert_to_tensor(np.ones([4, 1, 1]).astype(np.float32))
    output = graph_func(input1=np_input1, input2=np_input2)["output_0"]

    self.assertEqual(output.shape, (4, 1, 1))
    self.assertAllClose(
        np.asarray([5.0, 5.0, 5.0, 5.0]).reshape([4, 1, 1]), output)


if __name__ == "__main__":
  test.main()
