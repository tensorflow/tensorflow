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
import warnings
import numpy as np
import six

from tensorflow.contrib.tensorrt.python import trt_convert
# pylint: disable=unused-import
from tensorflow.contrib.tensorrt.python.ops import trt_ops
# pylint: enable=unused-import
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging

TfTrtIntegrationTestParams = namedtuple(
    "TfTrtIntegrationTestParams",
    [
        "gdef",
        # A list of names of the input placeholder nodes.
        "input_names",
        # A list of list of output shapes of the input placeholder nodes.
        "input_dims",
        # A list of names of the output identity nodes.
        "output_names",
        # A list of list of expected output shapes of the output identity nodes.
        "expected_output_dims"
    ])

RunParams = namedtuple("RunParams", [
    "use_optimizer", "precision_mode", "dynamic_engine", "test_name",
    "use_calibration"
])

ConversionParams = namedtuple("ConversionParams", [
    "max_batch_size", "max_workspace_size_bytes", "precision_mode",
    "minimum_segment_size", "is_dynamic_op", "maximum_cached_engines",
    "cached_engine_batches", "rewriter_config", "use_calibration"
])

PRECISION_MODES = ["FP32", "FP16", "INT8"]


def IsQuantizationMode(mode):
  return mode == "INT8"


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
    return math_ops.sin

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

  @classmethod
  def setUpClass(cls):
    """Setup method for the module."""
    super(TfTrtIntegrationTestBase, cls).setUpClass()
    trt_convert.enable_test_value()

  def __init__(self, methodName="runTest"):  # pylint: disable=invalid-name
    super(TfTrtIntegrationTestBase, self).__init__(methodName)
    self._trt_test_params = None

  def setUp(self):
    """Setup method."""
    super(TfTrtIntegrationTestBase, self).setUp()
    warnings.simplefilter("always")
    trt_convert.clear_test_values("")

  def GetParams(self):
    """Return a TfTrtIntegrationTestParams for test, implemented by subclass."""
    raise NotImplementedError()

  def GetConversionParams(self, run_params):
    """Return a ConversionParams for test."""
    batch_list = []
    for dims_list in self._GetParamsCached().input_dims:
      assert dims_list
      # Each list of shapes should have same batch size.
      input_batches = [dims[0] for dims in dims_list]
      assert max(input_batches) == min(input_batches)
      batch_list.append(input_batches[0])
    return ConversionParams(
        # We use the minimum of all the batch sizes, so when multiple different
        # input shapes are provided it'll always create new engines in the
        # cache, and we can therefore test the cache behavior.
        max_batch_size=min(batch_list),
        max_workspace_size_bytes=1 << 25,
        precision_mode=run_params.precision_mode,
        minimum_segment_size=2,
        is_dynamic_op=run_params.dynamic_engine,
        maximum_cached_engines=1,
        cached_engine_batches=None,
        rewriter_config=None,
        use_calibration=run_params.use_calibration)

  def ShouldRunTest(self, run_params):
    """Whether to run the test."""
    # This setting combination requires quantization nodes to be present in
    # order to build the engine.
    return not (IsQuantizationMode(run_params.precision_mode) and
                not run_params.use_calibration)

  def VerifyRunForEngine(self, engine_name, graph_state, expect_run=True):
    """Verify the state of a particular engine after sess.run()."""
    if graph_state == GraphState.ORIGINAL:
      self._ExpectCalibration(engine_name, "")
      self._ExpectNativeSegment(engine_name, "")
      self._ExpectTrtEngine(engine_name, "")
    elif graph_state == GraphState.CALIBRATE:
      self._ExpectCalibration(engine_name, "done")
      self._ExpectNativeSegment(engine_name, "done")
      self._ExpectTrtEngine(engine_name, "")
    elif graph_state == GraphState.INFERENCE:
      self._ExpectCalibration(engine_name, "")
      if expect_run:
        self._ExpectNativeSegment(engine_name, "")
        self._ExpectTrtEngine(engine_name, "done")
      else:
        self._ExpectNativeSegment(engine_name, "done")
        self._ExpectTrtEngine(engine_name, "")

  def VerifyRun(self, run_params, graph_state):
    """Verify the state of all engines after sess.run()."""
    for engine_name in self.ExpectedEnginesToBuild(run_params):
      expect_run = (engine_name in self.ExpectedEnginesToRun(run_params))
      self.VerifyRunForEngine(engine_name, graph_state, expect_run)

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build, implemented by subclass."""
    raise NotImplementedError()

  def ExpectedEnginesToRun(self, run_params):
    """Return the expected engines to run."""
    return self.ExpectedEnginesToBuild(run_params)

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

  def _PrepareRun(self, graph_state):
    """Set up necessary testing environment before calling sess.run()."""
    # Clear test values added by TRTEngineOp.
    trt_convert.clear_test_values("TRTEngineOp_.*:ExecuteTrtEngine")
    trt_convert.clear_test_values("TRTEngineOp_.*:ExecuteCalibration")
    trt_convert.clear_test_values("TRTEngineOp_.*:ExecuteNativeSegment")

  def _GetGPUOptions(self):
    gpu_options = config_pb2.GPUOptions()
    gpu_options.allow_growth = True
    return gpu_options

  def _GetConfigProto(self, run_params, graph_state):
    """Get config proto based on specific settings."""
    conversion_params = self.GetConversionParams(run_params)
    if graph_state != GraphState.ORIGINAL and run_params.use_optimizer:
      rewriter_cfg = trt_convert.get_tensorrt_rewriter_config(
          conversion_params.rewriter_config, conversion_params.max_batch_size,
          conversion_params.max_workspace_size_bytes,
          conversion_params.precision_mode,
          conversion_params.minimum_segment_size,
          conversion_params.is_dynamic_op,
          conversion_params.maximum_cached_engines,
          conversion_params.cached_engine_batches,
          conversion_params.use_calibration)

      graph_options = config_pb2.GraphOptions(rewrite_options=rewriter_cfg)
    else:
      graph_options = config_pb2.GraphOptions()
      if conversion_params.rewriter_config is not None:
        graph_options.rewrite_options.CopyFrom(
            conversion_params.rewriter_config)

    config = config_pb2.ConfigProto(
        gpu_options=self._GetGPUOptions(), graph_options=graph_options)
    return config

  def _ExpectTestValue(self, engine_name, method, expected_value):
    label = "%s:%s" % (engine_name, method)
    actual_value = trt_convert.get_test_value(label)
    self.assertEqual(
        expected_value,
        actual_value,
        msg="Unexpected test value with label %s. Actual: %s; expected: %s" %
        (label, actual_value, expected_value))

  def _ExpectCalibration(self, engine_name, value):
    self._ExpectTestValue(engine_name, "ExecuteCalibration", value)

  def _ExpectTrtEngine(self, engine_name, value):
    self._ExpectTestValue(engine_name, "ExecuteTrtEngine", value)

  def _ExpectNativeSegment(self, engine_name, value):
    self._ExpectTestValue(engine_name, "ExecuteNativeSegment", value)

  def _RunGraph(self,
                run_params,
                gdef,
                inputs_data,
                config,
                graph_state,
                num_runs=2):
    """Run given graphdef multiple times."""
    params = self._GetParamsCached()
    for current_input_data in inputs_data:
      assert len(params.input_names) == len(current_input_data)

    vals = []
    g = ops.Graph()
    with g.as_default():
      io_ops = importer.import_graph_def(
          graph_def=gdef,
          return_elements=params.input_names + params.output_names,
          name="")
      inputs = [op.outputs[0] for op in io_ops[:len(params.input_names)]]
      for current_input_data in inputs_data:
        assert len(inputs) == len(current_input_data)
      outputs = [op.outputs[0] for op in io_ops[len(params.input_names):]]
      with self.test_session(
          graph=g, config=config, use_gpu=True, force_gpu=True) as sess:
        # Run for each input(s) shape
        for shape_index in range(len(inputs_data)):
          val = None
          # Defaults to 2 runs to verify result across multiple runs is same.
          for _ in range(num_runs):
            self._PrepareRun(graph_state)
            new_val = sess.run(outputs, {
                inputs[i]: inputs_data[shape_index][i]
                for i in range(len(inputs))
            })
            output_len = len(params.expected_output_dims[shape_index])
            self.assertEqual(output_len, len(new_val))
            for i in range(output_len):
              self.assertEqual(
                  list(params.expected_output_dims[shape_index][i]),
                  list(new_val[i].shape))
            if val is not None:
              self.assertAllClose(val, new_val, atol=1.e-06, rtol=1.e-06)
            val = new_val
            self.VerifyRun(run_params, graph_state)
          vals.append(val)
    return vals

  # Use real data that is representative of the inference dataset
  # for calibration. For this test script it is random data.
  def _RunCalibration(self, run_params, gdef, inputs_data, config):
    """Run calibration on given graph."""
    return self._RunGraph(
        run_params, gdef, inputs_data, config, GraphState.CALIBRATE, num_runs=5)

  def _GetTrtGraphDef(self, run_params, graph_state, gdef):
    """Return trt converted graphdef."""
    params = self._GetParamsCached()
    conversion_params = self.GetConversionParams(run_params)
    logging.info(conversion_params)

    config_for_trt = self._GetConfigProto(run_params, graph_state)
    return trt_convert.create_inference_graph(
        input_graph_def=gdef,
        outputs=params.input_names + params.output_names,
        max_batch_size=conversion_params.max_batch_size,
        max_workspace_size_bytes=conversion_params.max_workspace_size_bytes,
        precision_mode=conversion_params.precision_mode,
        minimum_segment_size=conversion_params.minimum_segment_size,
        is_dynamic_op=conversion_params.is_dynamic_op,
        maximum_cached_engines=conversion_params.maximum_cached_engines,
        cached_engine_batches=conversion_params.cached_engine_batches,
        use_calibration=conversion_params.use_calibration,
        session_config=config_for_trt)

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

  def _VerifyConnections(self, expected_engines, converted_gdef):
    params = self._GetParamsCached()
    old_to_new_node_map = {
        self._ToString(node.name): self._ToString(node.name)
        for node in params.gdef.node
    }
    for engine_name, node_names in expected_engines.items():
      for node_name in node_names:
        old_to_new_node_map[node_name] = engine_name
    name_to_node_map = {
        self._ToString(node.name): node for node in params.gdef.node
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

    expected_input_map = {}
    for node in params.gdef.node:
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
        msg="expected:\n%s\nvs actual:\n%s" % (sorted(
            expected_input_map.items()), sorted(actual_input_map.items())))

  def _VerifyGraphDef(self, run_params, gdef, graph_state):
    self._WriteGraph(run_params, gdef, graph_state)

    expected_engines = self.ExpectedEnginesToBuild(run_params)
    num_engines = 0
    for node in gdef.node:
      if node.op == "TRTEngineOp":
        logging.info("Found TRTEngineOp: " + node.name)
    for node in gdef.node:
      if node.op == "TRTEngineOp":
        num_engines += 1
        self.assertTrue(node.name in expected_engines, node.name)
        self.assertTrue(len(node.attr["serialized_segment"].s), node.name)
        self.assertTrue(len(node.attr["segment_funcdef_name"].s), node.name)
        self.assertEqual(
            self._ToBytes(run_params.precision_mode),
            node.attr["precision_mode"].s, node.name)

        is_dynamic_engine = not node.attr["static_engine"].b
        self.assertEqual(run_params.dynamic_engine, is_dynamic_engine,
                         node.name)
        self.assertEqual(node.attr["use_calibration"].b,
                         run_params.use_calibration, node.name)

        has_calibration_data = len(node.attr["calibration_data"].s)
        if (IsQuantizationMode(run_params.precision_mode) and
            run_params.use_calibration and graph_state == GraphState.INFERENCE):
          self.assertTrue(has_calibration_data, node.name)
        else:
          self.assertFalse(has_calibration_data, node.name)
    if graph_state == GraphState.ORIGINAL:
      self.assertEqual(0, num_engines)
    else:
      self.assertEqual(num_engines, len(expected_engines))
      if isinstance(expected_engines, dict):
        self._VerifyConnections(expected_engines, gdef)
      # TODO(aaroey): consider verifying the corresponding TF function.

  def RunTest(self, run_params):
    if not self.ShouldRunTest(run_params):
      return
    assert run_params.precision_mode in PRECISION_MODES
    np.random.seed(12345)

    params = self._GetParamsCached()
    input_gdef = params.gdef
    input_dtypes = {}
    for node in input_gdef.node:
      if self._ToString(node.name) in params.input_names:
        assert self._ToString(node.op) == "Placeholder"
        input_dtypes[self._ToString(node.name)] = (
            dtypes.as_dtype(node.attr["dtype"].type).as_numpy_dtype())
    assert len(params.input_names) == len(input_dtypes)

    inputs_data = []
    for inp in params.input_dims:
      current_input_data = []
      for i in range(len(params.input_names)):
        dtype = input_dtypes[params.input_names[i]]
        # Multiply the input by some constant to avoid all zeros input for
        # integer types.
        scale = 10.0 if np.issubdtype(dtype, np.integer) else 1.0
        dims = inp[i]
        # TODO(laigd): add debug options. E.g. we can set the input data to be
        # continuous natural numbers:
        # seq = np.arange(np.prod(dims))
        # seq.resize(dims)
        # input_data.append(scale * seq.astype(dtype))
        current_input_data.append(
            (scale * np.random.random_sample(dims)).astype(dtype))
      inputs_data.append(current_input_data)

    self._VerifyGraphDef(run_params, input_gdef, GraphState.ORIGINAL)

    # Get reference result without running trt.
    config_no_trt = self._GetConfigProto(run_params, GraphState.ORIGINAL)
    logging.info("Running original graph w/o trt, config:\n%s",
                 str(config_no_trt))
    ref_result = self._RunGraph(run_params, input_gdef, inputs_data,
                                config_no_trt, GraphState.ORIGINAL)

    # Run calibration if necessary.
    if (IsQuantizationMode(run_params.precision_mode) and
        run_params.use_calibration):

      calib_config = self._GetConfigProto(run_params, GraphState.CALIBRATE)
      logging.info("Running calibration graph, config:\n%s", str(calib_config))
      if run_params.use_optimizer:
        result = self._RunCalibration(run_params, input_gdef, inputs_data,
                                      calib_config)
      else:
        calib_gdef = self._GetTrtGraphDef(run_params, GraphState.CALIBRATE,
                                          input_gdef)
        self._VerifyGraphDef(run_params, calib_gdef, GraphState.CALIBRATE)
        result = self._RunCalibration(run_params, calib_gdef, inputs_data,
                                      calib_config)
      infer_gdef = trt_convert.calib_graph_to_infer_graph(
          calib_gdef, run_params.dynamic_engine)
      self._VerifyGraphDef(run_params, infer_gdef, GraphState.INFERENCE)

      self.assertAllClose(
          ref_result,
          result,
          atol=self.ExpectedAbsoluteTolerance(run_params),
          rtol=self.ExpectedRelativeTolerance(run_params))
    else:
      infer_gdef = input_gdef

    # Run inference.
    infer_config = self._GetConfigProto(run_params, GraphState.INFERENCE)
    logging.info("Running final inference graph, config:\n%s",
                 str(infer_config))
    if not run_params.use_optimizer:
      infer_gdef = self._GetTrtGraphDef(run_params, GraphState.INFERENCE,
                                        infer_gdef)
      self._VerifyGraphDef(run_params, infer_gdef, GraphState.INFERENCE)

    result = self._RunGraph(run_params, infer_gdef, inputs_data, infer_config,
                            GraphState.INFERENCE)
    self.assertAllClose(
        ref_result,
        result,
        atol=self.ExpectedAbsoluteTolerance(run_params),
        rtol=self.ExpectedRelativeTolerance(run_params))

  def testIdempotence(self):
    # Test that applying tensorrt optimizer or offline conversion tools multiple
    # times to the same graph will result in same graph.
    #
    # TODO(aaroey): currently the conversion is not deterministic, this is
    # mainly because during tensorflow::ConvertGraphDefToGraph(), the graph uses
    # EdgeSet which use a map keyed by Edge*, so the order of input/output edges
    # of a node is nondeterministic, thus the order for segmenter to contract
    # edges is nondeterministic. Need to evaluate whether we should fix this.
    pass


def _AddTests(test_class):
  """Adds test methods to TfTrtIntegrationTestBase."""

  def _GetTest(run_params):
    """Gets a single test method based on the parameters."""

    def _Test(self):
      logging.info(
          "Running test %s with parameters: use_optimizer=%s, "
          "precision_mode=%s, dynamic_engine=%s",
          "testTfTrt_" + run_params.test_name, run_params.use_optimizer,
          run_params.precision_mode, run_params.dynamic_engine)
      self.RunTest(run_params)

    return _Test

  use_optimizer_options = [False, True]
  dynamic_engine_options = [False, True]
  use_calibration_options = [False, True]
  opts = itertools.product(use_optimizer_options, PRECISION_MODES,
                           dynamic_engine_options, use_calibration_options)
  for (use_optimizer, precision_mode, dynamic_engine, use_calibration) in opts:
    if IsQuantizationMode(precision_mode):
      if use_optimizer:
        # TODO(aaroey): if use_optimizer is True we need to get the inference
        # graphdef using custom python wrapper class, which is not currently
        # supported yet.
        continue
      if use_calibration and not dynamic_engine:
        # Static engine with use_calibration=False will be static, so we want to
        # test that. If use_calibration=True, only dynamic op is supported.
        # TODO(aaroey): construction of static calibration engine is not
        # supported yet.
        continue
    else:
      if use_calibration:
        # Don't calibrate in FP32 or FP16 mode
        continue

    conversion = "OptimizerConversion" if use_optimizer else "ToolConversion"
    engine_type = "DynamicEngine" if dynamic_engine else "StaticEngine"
    calibration_type = "UseCalibration" if use_calibration else "NoCalibration"
    test_name = "%s_%s_%s_%s" % (conversion, engine_type, precision_mode,
                                 calibration_type)
    run_params = RunParams(
        use_optimizer=use_optimizer,
        precision_mode=precision_mode,
        dynamic_engine=dynamic_engine,
        test_name=test_name,
        use_calibration=use_calibration)
    setattr(test_class, "testTfTrt_" + test_name, _GetTest(run_params))


if trt_convert.is_tensorrt_enabled():
  _AddTests(TfTrtIntegrationTestBase)
