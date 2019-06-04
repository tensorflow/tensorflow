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

from collections import namedtuple
import itertools
import os
import tempfile
import warnings
import numpy as np
import six

from tensorflow.compiler.tf2tensorrt.wrap_py_utils import is_tensorrt_enabled
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.compiler.tensorrt import trt_convert
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.tools import saved_model_utils

TfTrtIntegrationTestParams = namedtuple(
    "TfTrtIntegrationTestParams",
    [
        # A function that creates the TF graph for testing.
        "graph_fn",
        # A list of specifications for input tensors.
        "input_specs",
        # A list of specifications for output tensors.
        "output_specs",
        # A list of list of input shapes. Each shape must match the
        # corresponding element in `input_specs`.
        "input_dims",
        # A list of list of expected output shapes. Each shape must match the
        # corresponding element in `output_specs`.
        "expected_output_dims"
    ])

RunParams = namedtuple(
    "RunParams",
    [
        # Whether to run the conversion online with RewriterConfig, or offline
        # with TrtGraphConverter.
        "convert_online",
        "precision_mode",
        "dynamic_engine",
        "use_calibration",
        "test_name",
    ])

FP32 = "FP32"
FP16 = "FP16"
INT8 = "INT8"
PRECISION_MODES = [FP32, FP16, INT8]


def IsQuantizationMode(mode):
  return mode == "INT8"


def IsQuantizationWithCalibration(params):
  return IsQuantizationMode(params.precision_mode) and params.use_calibration


class GraphState(object):
  ORIGINAL = 0
  CALIBRATE = 1
  INFERENCE = 2


def OptimizerDisabledRewriterConfig():
  """Returns a RewriterConfig with all default Grappler optimizers disabled."""
  rewriter_config = rewriter_config_pb2.RewriterConfig()

  # Turn off all default Grappler optimizers.
  off = rewriter_config_pb2.RewriterConfig.OFF
  rewriter_config.layout_optimizer = off
  rewriter_config.constant_folding = off
  rewriter_config.shape_optimization = off
  rewriter_config.remapping = off
  rewriter_config.arithmetic_optimization = off
  rewriter_config.dependency_optimization = off
  rewriter_config.loop_optimization = off
  rewriter_config.function_optimization = off
  rewriter_config.debug_stripper = off
  rewriter_config.disable_model_pruning = True
  rewriter_config.scoped_allocator_optimization = off
  rewriter_config.memory_optimization = (
      rewriter_config_pb2.RewriterConfig.NO_MEM_OPT)
  rewriter_config.pin_to_host_optimization = off
  rewriter_config.auto_parallel.enable = False

  # Run only once for each enabled optimizer.
  rewriter_config.meta_optimizer_iterations = (
      rewriter_config_pb2.RewriterConfig.ONE)
  return rewriter_config


class TfTrtIntegrationTestBase(test_util.TensorFlowTestCase):
  """Class to test Tensorflow-TensorRT integration."""

  @property
  def trt_incompatible_op(self):
    return math_ops.erf

  @property
  def precision_modes(self):
    return ["FP32", "FP16", "INT8"]

  # str is bytes in py2, but unicode in py3.
  def _ToUnicode(self, s):
    if six.PY2:
      if isinstance(s, unicode):
        return s
      return s.decode("utf-8")
    else:
      if isinstance(s, str):
        return s
      return s.decode("utf-8")

  def _ToBytes(self, s):
    if six.PY2:
      if isinstance(s, unicode):
        return s.encode("utf-8")
      return s
    else:
      if isinstance(s, str):
        return s.encode("utf-8")
      return s

  def _ToString(self, s):
    if six.PY2:
      if isinstance(s, unicode):
        return s.encode("utf-8")
      return s
    else:
      if isinstance(s, str):
        return s
      return s.decode("utf-8")

  def __init__(self, methodName="runTest"):  # pylint: disable=invalid-name
    super(TfTrtIntegrationTestBase, self).__init__(methodName)
    self._trt_test_params = None

  def setUp(self):
    """Setup method."""
    super(TfTrtIntegrationTestBase, self).setUp()
    warnings.simplefilter("always")

  def BuildParams(self, graph_fn, dtype, input_shapes, output_shapes):
    """Build test parameters when not considering dynamic shapes."""

    def _Validate(shapes):
      # Make sure all the shapes are fully specified.
      for shape in shapes:
        assert all(shape)

    _Validate(input_shapes)
    _Validate(output_shapes)

    return TfTrtIntegrationTestParams(
        graph_fn=graph_fn,
        # Unset the batch dim of the specs to make sure TRT can tolerate changes
        # on that.
        input_specs=[
            tensor_spec.TensorSpec([None] + shape[1:], dtype, "input_%d" % i)
            for i, shape in enumerate(input_shapes)
        ],
        output_specs=[
            tensor_spec.TensorSpec([None] + shape[1:], dtype, "output_%d" % i)
            for i, shape in enumerate(output_shapes)
        ],
        input_dims=[input_shapes],
        expected_output_dims=[output_shapes])

  def GetParams(self):
    """Return a TfTrtIntegrationTestParams for test, implemented by subclass."""
    raise NotImplementedError()

  def GetConversionParams(self, run_params):
    """Return a TrtConversionParams for test."""
    batch_list = []
    for dims_list in self._GetParamsCached().input_dims:
      assert dims_list
      # Each list of shapes should have same batch size.
      input_batches = [dims[0] for dims in dims_list]
      assert max(input_batches) == min(input_batches)
      batch_list.append(input_batches[0])
    conversion_params = trt_convert.TrtConversionParams(
        # We use the minimum of all the batch sizes, so when multiple different
        # input shapes are provided it'll always create new engines in the
        # cache, and we can therefore test the cache behavior.
        rewriter_config_template=None,
        max_workspace_size_bytes=1 << 25,
        precision_mode=run_params.precision_mode,
        minimum_segment_size=2,
        is_dynamic_op=run_params.dynamic_engine,
        maximum_cached_engines=1,
        use_calibration=run_params.use_calibration,
        use_function_backup=False,
        max_batch_size=min(batch_list),
        cached_engine_batches=None)
    return conversion_params._replace(
        use_function_backup=IsQuantizationWithCalibration(conversion_params))

  def ShouldRunTest(self, run_params):
    """Whether to run the test."""
    # This setting combination requires quantization nodes to be present in
    # order to build the engine.
    return (run_params.use_calibration or
            not IsQuantizationMode(run_params.precision_mode))

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build, implemented by subclass."""
    raise NotImplementedError()

  def ExpectedAbsoluteTolerance(self, run_params):
    """The absolute tolerance to compare floating point results."""
    return 1.e-05 if run_params.precision_mode == "FP32" else 1.e-02

  def ExpectedRelativeTolerance(self, run_params):
    """The relative tolerance to compare floating point results."""
    return 1.e-05 if run_params.precision_mode == "FP32" else 1.e-02

  def _GetParamsCached(self):
    if self._trt_test_params is None:
      self._trt_test_params = self.GetParams()
    return self._trt_test_params

  def _GetGPUOptions(self):
    gpu_options = config_pb2.GPUOptions()
    gpu_options.allow_growth = True
    return gpu_options

  def _GetConfigProto(self, run_params, graph_state):
    """Get config proto based on specific settings."""
    conversion_params = self.GetConversionParams(run_params)
    if graph_state == GraphState.INFERENCE and run_params.convert_online:
      rewriter_cfg = trt_convert.get_tensorrt_rewriter_config(conversion_params)
      graph_options = config_pb2.GraphOptions(rewrite_options=rewriter_cfg)
    else:
      graph_options = config_pb2.GraphOptions()
      if conversion_params.rewriter_config_template is not None:
        graph_options.rewrite_options.CopyFrom(
            conversion_params.rewriter_config_template)

    config = config_pb2.ConfigProto(
        gpu_options=self._GetGPUOptions(), graph_options=graph_options)
    return config

  def _GetFeedNames(self):
    params = self._GetParamsCached()
    # Construct the feeds tensor names by appending :0 to the node names.
    return [spec.name + ":0" for spec in params.input_specs]

  def _GetFetchNames(self):
    params = self._GetParamsCached()
    # Construct the fetches tensor names by appending :0 to the node names.
    return [spec.name + ":0" for spec in params.output_specs]

  def _GetFeedDict(self, inputs_data):
    return {name: data for name, data in zip(self._GetFeedNames(), inputs_data)}

  def _RunGraph(self,
                run_params,
                saved_model_dir,
                inputs_data,
                config,
                graph_state,
                num_runs=2):
    """Run given graphdef multiple times."""
    params = self._GetParamsCached()
    for data in inputs_data:
      assert len(params.input_specs) == len(data)

    fetches = self._GetFetchNames()
    g = ops.Graph()
    with g.as_default():
      with self.session(graph=g, config=config, use_gpu=True) as sess:
        loader.load(sess, [tag_constants.SERVING], saved_model_dir)
        vals = []
        # Run for each input(s) shape
        for expected_shapes, current_input_data in zip(
            params.expected_output_dims, inputs_data):
          val = None
          for _ in range(num_runs):
            new_val = sess.run(fetches, self._GetFeedDict(current_input_data))
            self.assertEqual(len(expected_shapes), len(new_val))
            for expected_shape, actual_val in zip(expected_shapes, new_val):
              self.assertEqual(list(expected_shape), list(actual_val.shape))
            if val is not None:
              self.assertAllClose(val, new_val, atol=1.e-06, rtol=1.e-06)
            val = new_val
          vals.append(val)
        return vals

  def _CreateConverter(self, saved_model_dir, session_config,
                       conversion_params):
    """Return a TrtGraphConverter."""
    converter = trt_convert.TrtGraphConverter(
        input_saved_model_dir=saved_model_dir,
        session_config=session_config,
        max_batch_size=conversion_params.max_batch_size,
        max_workspace_size_bytes=conversion_params.max_workspace_size_bytes,
        precision_mode=conversion_params.precision_mode,
        minimum_segment_size=conversion_params.minimum_segment_size,
        is_dynamic_op=conversion_params.is_dynamic_op,
        maximum_cached_engines=conversion_params.maximum_cached_engines,
        cached_engine_batches=conversion_params.cached_engine_batches,
        use_calibration=conversion_params.use_calibration,
        use_function_backup=conversion_params.use_function_backup)
    return converter

  def _GetCalibratedInferGraph(self, run_params, saved_model_dir, inputs_data):
    """Return trt converted graphdef in INT8 mode."""
    conversion_params = self.GetConversionParams(run_params)
    logging.info(conversion_params)
    assert conversion_params.precision_mode == "INT8"
    assert conversion_params.is_dynamic_op
    assert conversion_params.maximum_cached_engines == 1
    assert not conversion_params.cached_engine_batches
    assert conversion_params.use_calibration

    # We only support calibrating single engine.
    # TODO(aaroey): fix this.
    assert len(inputs_data) == 1

    session_config = self._GetConfigProto(run_params, GraphState.CALIBRATE)
    logging.info("Running calibration graph, config:\n%s", str(session_config))

    converter = self._CreateConverter(saved_model_dir, session_config,
                                      conversion_params)
    int8_gdef = converter.convert()
    self._VerifyGraphDef(run_params, saved_model_dir, int8_gdef,
                         GraphState.CALIBRATE)

    converter.calibrate(
        fetch_names=self._GetFetchNames(),
        num_runs=5,
        feed_dict_fn=lambda: self._GetFeedDict(inputs_data[0]))
    trt_saved_model_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    converter.save(trt_saved_model_dir)
    return trt_saved_model_dir

  def _GetInferGraph(self, run_params, saved_model_dir):
    """Return trt converted graphdef."""
    conversion_params = self.GetConversionParams(run_params)
    logging.info(conversion_params)

    session_config = self._GetConfigProto(run_params, GraphState.INFERENCE)
    logging.info("Creating TRT graph for inference, config\n%s",
                 str(session_config))
    converter = self._CreateConverter(saved_model_dir, session_config,
                                      conversion_params)
    converter.convert()
    trt_saved_model_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    converter.save(trt_saved_model_dir)
    return trt_saved_model_dir

  def _WriteGraph(self, run_params, gdef, graph_state):
    if graph_state == GraphState.ORIGINAL:
      label = "Original"
    elif graph_state == GraphState.CALIBRATE:
      label = "CalibEngine"
    elif graph_state == GraphState.INFERENCE:
      label = "InferEngine"
    graph_name = (
        self.__class__.__name__ + "_" + run_params.test_name + "_" + label +
        ".pbtxt")
    temp_dir = os.getenv("TRT_TEST_TMPDIR", self.get_temp_dir())
    if temp_dir:
      logging.info("Writing graph to %s/%s", temp_dir, graph_name)
      graph_io.write_graph(gdef, temp_dir, graph_name)

  def _VerifyConnections(self, expected_engines, original_gdef, converted_gdef):
    old_to_new_node_map = {
        self._ToString(node.name): self._ToString(node.name)
        for node in original_gdef.node
    }
    for engine_name, node_names in expected_engines.items():
      for node_name in node_names:
        old_to_new_node_map[node_name] = engine_name
    name_to_node_map = {
        self._ToString(node.name): node for node in original_gdef.node
    }

    def _InputName(inp):
      inp = self._ToString(inp)
      prefix = ""
      if inp[0] == "^":
        prefix = "^"
        inp = inp[1:]
      parts = inp.split(":")
      if len(parts) > 1 and parts[-1].isdigit():
        inp = inp[:-len(parts[-1]) - 1]
      return (prefix, inp)

    # Compute the expected mapping from each node to its input nodes.
    expected_input_map = {}
    for node in original_gdef.node:
      name_str = self._ToString(node.name)
      target_node_name = old_to_new_node_map[name_str]
      is_engine_op = (target_node_name != name_str)
      if target_node_name not in expected_input_map:
        expected_input_map[target_node_name] = set()
      input_set = expected_input_map[target_node_name]
      for inp in node.input:
        (prefix, inp_name) = _InputName(inp)
        # Add the input only if it's outside the segment (note that it could be
        # in a different engine).
        if (not is_engine_op or
            old_to_new_node_map[inp_name] != target_node_name):
          if is_engine_op and name_to_node_map[inp_name].op == "Const":
            # Const data input nodes to the segment has been copied to the
            # segment graphdef and the engine, and the dependency has been
            # converted to control dependendy.
            input_set.add("^" + old_to_new_node_map[inp_name])
          else:
            input_set.add(prefix + old_to_new_node_map[inp_name])

    # Compute the actual mapping from each node to its input nodes.
    actual_input_map = {}
    for node in converted_gdef.node:
      name_str = self._ToString(node.name)
      actual_input_map[name_str] = set()
      input_set = actual_input_map[name_str]
      for inp in node.input:
        (prefix, node_name) = _InputName(inp)
        input_set.add(prefix + node_name)

    self.assertEqual(
        expected_input_map,
        actual_input_map,
        msg="expected:\n%s\nvs actual:\n%s" %
        (sorted(expected_input_map.items()), sorted(actual_input_map.items())))

  def _GetGraphDef(self, gdef_or_saved_model_dir):
    if isinstance(gdef_or_saved_model_dir, str):
      return saved_model_utils.get_meta_graph_def(
          gdef_or_saved_model_dir, tag_constants.SERVING).graph_def
    assert isinstance(gdef_or_saved_model_dir, graph_pb2.GraphDef)
    return gdef_or_saved_model_dir

  def _VerifyGraphDef(self, run_params, original_gdef_or_saved_model_dir,
                      gdef_or_saved_model_dir_to_verify, graph_state):
    original_gdef = self._GetGraphDef(original_gdef_or_saved_model_dir)
    gdef_to_verify = self._GetGraphDef(gdef_or_saved_model_dir_to_verify)
    self._WriteGraph(run_params, gdef_to_verify, graph_state)

    expected_engines = self.ExpectedEnginesToBuild(run_params)
    num_engines = 0
    functions = [f.signature.name for f in gdef_to_verify.library.function]
    for node in gdef_to_verify.node:
      if node.op == "TRTEngineOp":
        logging.info("Found TRTEngineOp: " + node.name)
        num_engines += 1
        segment_funcdef_name = node.attr["segment_funcdef_name"].s
        function_name = node.name + "_native_segment"
        if IsQuantizationWithCalibration(run_params):
          self.assertNotEmpty(segment_funcdef_name, node.name)
          self.assertIn(function_name, functions)
        else:
          self.assertEmpty(segment_funcdef_name, node.name)
          self.assertNotIn(function_name, functions)
        self.assertIn(node.name, expected_engines)
        self.assertTrue(len(node.attr["serialized_segment"].s), node.name)
        self.assertEqual(
            self._ToBytes(run_params.precision_mode),
            node.attr["precision_mode"].s, node.name)

        is_dynamic_engine = not node.attr["static_engine"].b
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
    else:
      self.assertEqual(num_engines, len(expected_engines))
      if isinstance(expected_engines, dict):
        self._VerifyConnections(expected_engines, original_gdef, gdef_to_verify)
      # TODO(aaroey): consider verifying the corresponding TF function.

  def _MakeSavedModel(self, run_params):
    """Write the saved model as an input for testing."""
    params = self._GetParamsCached()
    g = ops.Graph()
    with g.as_default():
      inputs = []
      for spec in params.input_specs:
        inp = array_ops.placeholder(
            dtype=spec.dtype, shape=spec.shape, name=spec.name)
        inputs.append(inp)
      outputs = params.graph_fn(*inputs)
      if not isinstance(outputs, list) and not isinstance(outputs, tuple):
        outputs = [outputs]
      for spec, output in zip(params.output_specs, outputs):
        assert spec.name == output.name.split(":")[0]

    signature_def = signature_def_utils.build_signature_def(
        inputs={inp.op.name: utils.build_tensor_info(inp) for inp in inputs},
        outputs={out.op.name: utils.build_tensor_info(out) for out in outputs},
        method_name=signature_constants.PREDICT_METHOD_NAME)

    saved_model_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    saved_model_builder = builder.SavedModelBuilder(saved_model_dir)
    with self.session(
        graph=g, config=self._GetConfigProto(run_params,
                                             GraphState.ORIGINAL)) as sess:
      saved_model_builder.add_meta_graph_and_variables(
          sess, [tag_constants.SERVING],
          signature_def_map={
              signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                  signature_def
          })
    saved_model_builder.save()
    return saved_model_dir

  def RunTest(self, run_params):
    if not self.ShouldRunTest(run_params):
      return

    saved_model_dir = self._MakeSavedModel(run_params)

    np.random.seed(12345)  # Fix the seed so the test is deterministic.
    inputs_data = []
    input_specs = self._GetParamsCached().input_specs
    for dim_list in self._GetParamsCached().input_dims:
      assert len(input_specs) == len(dim_list)
      current_input_data = []
      for spec, np_shape in zip(input_specs, dim_list):
        np_dtype = spec.dtype.as_numpy_dtype()
        # Multiply the input by some constant to avoid all zeros input for
        # integer types.
        scale = 10.0 if np.issubdtype(np_dtype, np.integer) else 1.0
        # TODO(laigd): add debug options. E.g. we can set the input data to be
        # continuous natural numbers:
        # seq = np.arange(np.prod(np_shape))
        # seq.resize(np_shape)
        # inputs_data.append(scale * seq.astype(np_dtype))
        current_input_data.append(
            (scale * np.random.random_sample(np_shape)).astype(np_dtype))
      inputs_data.append(current_input_data)

    # Verify original graph.
    self._VerifyGraphDef(run_params, saved_model_dir, saved_model_dir,
                         GraphState.ORIGINAL)

    # Run original graph without trt to get reference result.
    config_no_trt = self._GetConfigProto(run_params, GraphState.ORIGINAL)
    logging.info("Running original graph w/o trt, config:\n%s",
                 str(config_no_trt))
    ref_result = self._RunGraph(run_params, saved_model_dir, inputs_data,
                                config_no_trt, GraphState.ORIGINAL)

    # Run calibration if necessary.
    if IsQuantizationWithCalibration(run_params):
      infer_saved_model_dir = self._GetCalibratedInferGraph(
          run_params, saved_model_dir, inputs_data)
      self._VerifyGraphDef(run_params, saved_model_dir, infer_saved_model_dir,
                           GraphState.INFERENCE)
    elif not run_params.convert_online:
      infer_saved_model_dir = self._GetInferGraph(run_params, saved_model_dir)
      self._VerifyGraphDef(run_params, saved_model_dir, infer_saved_model_dir,
                           GraphState.INFERENCE)
    else:
      infer_saved_model_dir = saved_model_dir

    # Run inference.
    infer_config = self._GetConfigProto(run_params, GraphState.INFERENCE)
    logging.info("Running final inference graph, config:\n%s",
                 str(infer_config))
    result = self._RunGraph(run_params, infer_saved_model_dir, inputs_data,
                            infer_config, GraphState.INFERENCE)
    self.assertAllClose(
        ref_result,
        result,
        atol=self.ExpectedAbsoluteTolerance(run_params),
        rtol=self.ExpectedRelativeTolerance(run_params))

  def testIdempotence(self):
    # Test that applying tensorrt optimizer or offline conversion tools multiple
    # times to the same graph will result in same graph.
    #
    # TODO(aaroey): implement this.
    pass


def _AddTests(test_class):
  """Adds test methods to TfTrtIntegrationTestBase."""

  def _GetTestConfigs():
    """Returns the config combinations to run the test."""
    convert_online, convert_offline = True, False
    dynamic_engine, static_engine = True, False
    use_calibration, no_calibration = True, False

    # Add all possible test cases and let the derived test class to decide
    # whether to run specific ones with ShouldRunTest().
    #
    # Note: INT8 without calibration behaves like FP32/FP16.
    opts = list(
        itertools.product([FP32, FP16, INT8], [convert_online, convert_offline],
                          [dynamic_engine, static_engine], [no_calibration]))
    # We always run calibration with offline tool.
    # TODO(aaroey): static calibration engine is not supported yet.
    opts.append((INT8, convert_offline, dynamic_engine, use_calibration))
    return opts

  def _GetTest(run_params):
    """Gets a single test method based on the parameters."""

    @test_util.deprecated_graph_mode_only
    def _Test(self):
      logging.info(
          "Running TFv1 test %s with parameters: convert_online=%s, "
          "precision_mode=%s, dynamic_engine=%s",
          "testTfTrt_" + run_params.test_name, run_params.convert_online,
          run_params.precision_mode, run_params.dynamic_engine)
      self.RunTest(run_params)

    return _Test

  opts = _GetTestConfigs()
  for (precision_mode, convert_online, dynamic_engine, use_calibration) in opts:
    conversion = "OnlineConversion" if convert_online else "OfflineConversion"
    engine_type = "DynamicEngine" if dynamic_engine else "StaticEngine"
    calibration_type = "UseCalibration" if use_calibration else "NoCalibration"
    test_name = "%s_%s_%s_%s" % (conversion, engine_type, precision_mode,
                                 calibration_type)
    run_params = RunParams(
        convert_online=convert_online,
        precision_mode=precision_mode,
        dynamic_engine=dynamic_engine,
        test_name=test_name,
        use_calibration=use_calibration)
    setattr(test_class, "testTfTrt_" + test_name, _GetTest(run_params))


if is_tensorrt_enabled():
  _AddTests(TfTrtIntegrationTestBase)
