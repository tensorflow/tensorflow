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
from tensorflow.contrib.tensorrt.python.ops import trt_engine_op
# pylint: enable=unused-import
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging

TfTrtIntegrationTestParams = namedtuple("TfTrtIntegrationTestParams", [
    "gdef", "input_names", "input_dims", "expected_engines",
    "expected_output_dims", "allclose_atol", "allclose_rtol"
])

RunParams = namedtuple(
    "RunParams",
    ["use_optimizer", "precision_mode", "dynamic_engine", "test_name"])

PRECISION_MODES = ["FP32", "FP16", "INT8"]


def _IsQuantizationMode(mode):
  return mode == "INT8"


class GraphState(object):
  ORIGINAL = 0
  CALIBRATE = 1
  INFERENCE = 2


class TfTrtIntegrationTestBase(test_util.TensorFlowTestCase):
  """Class to test Tensorflow-TensorRT integration."""

  @property
  def output_name(self):
    return "output"

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

  def setUp(self):
    """Setup method."""
    super(TfTrtIntegrationTestBase, self).setUp()
    warnings.simplefilter("always")
    trt_convert.clear_test_values("")

  def GetParams(self):
    """Return a TfTrtIntegrationTestParams for test, implemented by subclass."""
    raise NotImplementedError()

  def _PrepareRun(self, params, graph_state):
    """Set up necessary testing environment before calling sess.run()."""
    # Clear test values added by TRTEngineOp.
    trt_convert.clear_test_values("my_trt_op_.*:ExecuteTrtEngine")
    trt_convert.clear_test_values("my_trt_op_.*:ExecuteCalibration")
    trt_convert.clear_test_values("my_trt_op_.*:ExecuteNativeSegment")

  def _VerifyRun(self, params, graph_state):
    """Verify the state after sess.run()."""
    for engine_name in params.expected_engines:
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
        self._ExpectNativeSegment(engine_name, "")
        self._ExpectTrtEngine(engine_name, "done")

  def _GetConfigProto(self, params, run_params, graph_state):
    """Get config proto based on specific settings."""
    if graph_state != GraphState.ORIGINAL and run_params.use_optimizer:
      rewriter_cfg = rewriter_config_pb2.RewriterConfig()
      rewriter_cfg.optimizers.extend(["constfold", "layout"])
      custom_op = rewriter_cfg.custom_optimizers.add()
      custom_op.name = "TensorRTOptimizer"
      custom_op.parameter_map["minimum_segment_size"].i = 2
      custom_op.parameter_map["max_batch_size"].i = max(
          [dims[0] for dims in params.input_dims])
      custom_op.parameter_map["is_dynamic_op"].b = run_params.dynamic_engine
      custom_op.parameter_map["max_workspace_size_bytes"].i = 1 << 25
      custom_op.parameter_map["precision_mode"].s = self._ToBytes(
          run_params.precision_mode)
      graph_options = config_pb2.GraphOptions(rewrite_options=rewriter_cfg)
    else:
      graph_options = config_pb2.GraphOptions()

    gpu_options = config_pb2.GPUOptions()
    gpu_options.allow_growth = True
    if trt_convert.get_linked_tensorrt_version()[0] == 3:
      gpu_options.per_process_gpu_memory_fraction = 0.50

    config = config_pb2.ConfigProto(
        gpu_options=gpu_options, graph_options=graph_options)
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

  def _RunGraph(self, params, gdef, input_data, config, graph_state,
                num_runs=2):
    """Run given graphdef multiple times."""
    assert len(params.input_names) == len(input_data)
    g = ops.Graph()
    with g.as_default():
      io_ops = importer.import_graph_def(
          graph_def=gdef,
          return_elements=params.input_names + [self.output_name],
          name="")
      inp = [i.outputs[0] for i in io_ops[:-1]]
      assert len(inp) == len(input_data)
      out = io_ops[-1].outputs[0]
    with self.test_session(
        graph=g, config=config, use_gpu=True, force_gpu=True) as sess:
      val = None
      # Defaults to 2 runs to verify result across multiple runs is same.
      for _ in range(num_runs):
        self._PrepareRun(params, graph_state)
        new_val = sess.run(out,
                           {inp[i]: input_data[i] for i in range(len(inp))})
        self.assertEqual(params.expected_output_dims, new_val.shape)
        if val is not None:
          self.assertAllEqual(val, new_val)
        val = new_val
        self._VerifyRun(params, graph_state)
    return val

  # Use real data that is representative of the inference dataset
  # for calibration. For this test script it is random data.
  def _RunCalibration(self, params, gdef, input_data, config):
    """Run calibration on given graph."""
    return self._RunGraph(
        params, gdef, input_data, config, GraphState.CALIBRATE, num_runs=5)

  def _GetTrtGraphDef(self, params, run_params, gdef):
    """Return trt converted graphdef."""
    return trt_convert.create_inference_graph(
        input_graph_def=gdef,
        outputs=[self.output_name],
        max_batch_size=max([dims[0] for dims in params.input_dims]),
        max_workspace_size_bytes=1 << 25,
        precision_mode=run_params.precision_mode,
        minimum_segment_size=2,
        is_dynamic_op=run_params.dynamic_engine)

  def _WriteGraph(self, params, run_params, gdef, graph_state):
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
    logging.info("Writing graph to %s/%s", temp_dir, graph_name)
    graph_io.write_graph(gdef, temp_dir, graph_name)

  def _VerifyConnections(self, params, converted_gdef):
    old_to_new_node_map = {
        self._ToString(node.name): self._ToString(node.name)
        for node in params.gdef.node
    }
    for engine_name, node_names in params.expected_engines.items():
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

  def _VerifyGraphDef(self, params, run_params, gdef, graph_state):
    self._WriteGraph(params, run_params, gdef, graph_state)

    num_engines = 0
    for node in gdef.node:
      if node.op == "TRTEngineOp":
        num_engines += 1
        self.assertTrue(node.name in params.expected_engines)
        self.assertTrue(len(node.attr["serialized_segment"].s))
        self.assertTrue(len(node.attr["segment_funcdef_name"].s))
        self.assertEqual(
            self._ToBytes(run_params.precision_mode),
            node.attr["precision_mode"].s)

        is_dynamic_engine = not node.attr["static_engine"].b
        self.assertEqual(run_params.dynamic_engine, is_dynamic_engine)

        has_calibration_data = len(node.attr["calibration_data"].s)
        if (_IsQuantizationMode(run_params.precision_mode) and
            graph_state == GraphState.INFERENCE):
          self.assertTrue(has_calibration_data)
        else:
          self.assertFalse(has_calibration_data)
    if graph_state == GraphState.ORIGINAL:
      self.assertEqual(0, num_engines)
    else:
      self.assertEqual(num_engines, len(params.expected_engines))
      if isinstance(params.expected_engines, dict):
        self._VerifyConnections(params, gdef)
      # TODO(aaroey): consider verifying the corresponding TF function.

  def RunTest(self, params, run_params):
    assert run_params.precision_mode in PRECISION_MODES
    input_data = [np.random.random_sample(dims) for dims in params.input_dims]
    input_gdef = params.gdef
    self._VerifyGraphDef(params, run_params, input_gdef, GraphState.ORIGINAL)

    # Get reference result without running trt.
    config_no_trt = self._GetConfigProto(params, run_params,
                                         GraphState.ORIGINAL)
    logging.info("Running original graph w/o trt, config:\n%s",
                 str(config_no_trt))
    ref_result = self._RunGraph(params, input_gdef, input_data, config_no_trt,
                                GraphState.ORIGINAL)

    # Run calibration if necessary.
    if _IsQuantizationMode(run_params.precision_mode):

      calib_config = self._GetConfigProto(params, run_params,
                                          GraphState.CALIBRATE)
      logging.info("Running calibration graph, config:\n%s", str(calib_config))
      if run_params.use_optimizer:
        result = self._RunCalibration(params, input_gdef, input_data,
                                      calib_config)
      else:
        calib_gdef = self._GetTrtGraphDef(params, run_params, input_gdef)
        self._VerifyGraphDef(params, run_params, calib_gdef,
                             GraphState.CALIBRATE)
        result = self._RunCalibration(params, calib_gdef, input_data,
                                      calib_config)
      infer_gdef = trt_convert.calib_graph_to_infer_graph(calib_gdef)
      self._VerifyGraphDef(params, run_params, infer_gdef, GraphState.INFERENCE)

      self.assertAllClose(
          ref_result,
          result,
          atol=params.allclose_atol,
          rtol=params.allclose_rtol)
    else:
      infer_gdef = input_gdef

    # Run inference.
    infer_config = self._GetConfigProto(params, run_params,
                                        GraphState.INFERENCE)
    logging.info("Running final inference graph, config:\n%s",
                 str(infer_config))
    if run_params.use_optimizer:
      result = self._RunGraph(params, infer_gdef, input_data, infer_config,
                              GraphState.INFERENCE)
    else:
      trt_infer_gdef = self._GetTrtGraphDef(params, run_params, infer_gdef)
      self._VerifyGraphDef(params, run_params, trt_infer_gdef,
                           GraphState.INFERENCE)
      result = self._RunGraph(params, trt_infer_gdef, input_data, infer_config,
                              GraphState.INFERENCE)

    self.assertAllClose(
        ref_result,
        result,
        atol=params.allclose_atol,
        rtol=params.allclose_rtol)

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
      params = self.GetParams()
      logging.info(
          "Running test %s with parameters: use_optimizer=%s, "
          "precision_mode=%s, dynamic_engine=%s",
          "testTfTrt_" + run_params.test_name, run_params.use_optimizer,
          run_params.precision_mode, run_params.dynamic_engine)
      self.RunTest(params, run_params)

    return _Test

  use_optimizer_options = [False, True]
  dynamic_engine_options = [False, True]
  for (use_optimizer, precision_mode, dynamic_engine) in itertools.product(
      use_optimizer_options, PRECISION_MODES, dynamic_engine_options):
    if _IsQuantizationMode(precision_mode):
      if use_optimizer:
        # TODO(aaroey): if use_optimizer is True we need to get the inference
        # graphdef using custom python wrapper class, which is not currently
        # supported yet.
        continue
      if not dynamic_engine:
        # TODO(aaroey): construction of static calibration engine is not
        # supported yet.
        continue

    conversion = "OptimizerConversion" if use_optimizer else "ToolConversion"
    engine_type = ("DynamicEngine" if dynamic_engine else "StaticEngine")
    test_name = "%s_%s_%s" % (conversion, precision_mode, engine_type)
    run_params = RunParams(
        use_optimizer=use_optimizer,
        precision_mode=precision_mode,
        dynamic_engine=dynamic_engine,
        test_name=test_name)
    setattr(test_class, "testTfTrt_" + test_name, _GetTest(run_params))


if trt_convert.is_tensorrt_enabled():
  _AddTests(TfTrtIntegrationTestBase)
