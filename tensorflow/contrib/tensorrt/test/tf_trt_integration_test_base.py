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
import warnings
import numpy as np
import six

from tensorflow.contrib.tensorrt.python import trt_convert
# pylint: disable=unused-import
from tensorflow.contrib.tensorrt.python.ops import trt_engine_op
# pylint: enable=unused-import
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging

TfTrtIntegrationTestParams = namedtuple("TfTrtIntegrationTestParams", [
    "gdef", "input_names", "input_dims", "num_expected_engines",
    "expected_output_dims", "allclose_atol", "allclose_rtol"
])

PRECISION_MODES = ["FP32", "FP16", "INT8"]


def _IsQuantizationMode(mode):
  return mode == "INT8"


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

  def _ToBytes(self, s):
    if six.PY2:
      return s
    else:
      return s.encode("utf-8")

  def _ToString(self, s):
    if six.PY2:
      return s
    else:
      return s.decode("utf-8")

  def setUp(self):
    """Setup method."""
    super(TfTrtIntegrationTestBase, self).setUp()
    warnings.simplefilter("always")

  def GetParams(self):
    """Return a TfTrtIntegrationTestParams for test, implemented by subclass."""
    raise NotImplementedError()

  def _GetConfigProto(self,
                      params,
                      use_optimizer,
                      precision_mode=None,
                      is_dynamic_op=None):
    """Get config proto based on specific settings."""
    if use_optimizer:
      rewriter_cfg = rewriter_config_pb2.RewriterConfig()
      rewriter_cfg.optimizers.extend(["constfold", "layout"])
      custom_op = rewriter_cfg.custom_optimizers.add()
      custom_op.name = "TensorRTOptimizer"
      custom_op.parameter_map["minimum_segment_size"].i = 3
      custom_op.parameter_map["max_batch_size"].i = max(
          [dims[0] for dims in params.input_dims])
      custom_op.parameter_map["is_dynamic_op"].b = is_dynamic_op
      custom_op.parameter_map["max_workspace_size_bytes"].i = 1 << 25
      custom_op.parameter_map["precision_mode"].s = self._ToBytes(
          precision_mode)
      graph_options = config_pb2.GraphOptions(rewrite_options=rewriter_cfg)
    else:
      graph_options = config_pb2.GraphOptions()

    gpu_options = config_pb2.GPUOptions()
    if trt_convert.get_linked_tensorrt_version()[0] == 3:
      gpu_options.per_process_gpu_memory_fraction = 0.50

    config = config_pb2.ConfigProto(
        gpu_options=gpu_options, graph_options=graph_options)
    return config

  def _RunGraph(self, params, gdef, input_data, config, num_runs=2):
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
        new_val = sess.run(out,
                           {inp[i]: input_data[i] for i in range(len(inp))})
        self.assertEqual(params.expected_output_dims, new_val.shape)
        if val is not None:
          self.assertAllEqual(val, new_val)
        val = new_val
    return val

  # Use real data that is representative of the inference dataset
  # for calibration. For this test script it is random data.
  def _RunCalibration(self, params, gdef, input_data, config):
    """Run calibration on given graph."""
    return self._RunGraph(params, gdef, input_data, config, 30)

  def _GetTrtGraphDef(self, params, gdef, precision_mode, is_dynamic_op):
    """Return trt converted graphdef."""
    return trt_convert.create_inference_graph(
        input_graph_def=gdef,
        outputs=[self.output_name],
        max_batch_size=max([dims[0] for dims in params.input_dims]),
        max_workspace_size_bytes=1 << 25,
        precision_mode=precision_mode,
        minimum_segment_size=2,
        is_dynamic_op=is_dynamic_op)

  def _VerifyGraphDef(self,
                      params,
                      gdef,
                      precision_mode=None,
                      is_calibrated=None,
                      dynamic_engine=None):
    num_engines = 0
    for n in gdef.node:
      # TODO(jie): we should have coverage for failed conversion (TF fallback).
      # where the conversion will fail and we shouldn't count this engine as the
      # converted engines.
      if n.op == "TRTEngineOp":
        num_engines += 1
        self.assertNotEqual(self._ToBytes(""), n.attr["serialized_segment"].s)
        self.assertNotEqual(self._ToBytes(""), n.attr["segment_funcdef_name"].s)
        self.assertEqual(
            self._ToBytes(precision_mode), n.attr["precision_mode"].s)
        self.assertEqual(not dynamic_engine, n.attr["static_engine"].b)
        if _IsQuantizationMode(precision_mode) and is_calibrated:
          self.assertNotEqual(self._ToBytes(""), n.attr["calibration_data"].s)
        else:
          self.assertEqual(self._ToBytes(""), n.attr["calibration_data"].s)
    if precision_mode is None:  # This means gdef is the original GraphDef.
      self.assertEqual(0, num_engines)
    else:
      self.assertEqual(num_engines, params.num_expected_engines)

  def RunTest(self, params, use_optimizer, precision_mode,
              dynamic_infer_engine, dynamic_calib_engine):
    assert precision_mode in PRECISION_MODES
    input_data = [np.random.random_sample(dims) for dims in params.input_dims]
    input_gdef = params.gdef
    self._VerifyGraphDef(params, input_gdef)

    # Get reference result without running trt.
    config_no_trt = self._GetConfigProto(params, False)
    logging.info("Running original graph w/o trt, config:\n%s",
                 str(config_no_trt))
    ref_result = self._RunGraph(params, input_gdef, input_data, config_no_trt)

    # Run calibration if necessary.
    if _IsQuantizationMode(precision_mode):

      calib_config = self._GetConfigProto(params, use_optimizer, precision_mode,
                                          dynamic_calib_engine)
      logging.info("Running calibration graph, config:\n%s", str(calib_config))
      if use_optimizer:
        self.assertTrue(False)
        # TODO(aaroey): uncomment this and get infer_gdef when this mode is
        # supported.
        # result = self._RunCalibration(params, input_gdef, input_data,
        #                               calib_config)
      else:
        calib_gdef = self._GetTrtGraphDef(params, input_gdef, precision_mode,
                                          dynamic_calib_engine)
        self._VerifyGraphDef(params, calib_gdef, precision_mode, False,
                             dynamic_calib_engine)
        result = self._RunCalibration(params, calib_gdef, input_data,
                                      calib_config)
        infer_gdef = trt_convert.calib_graph_to_infer_graph(calib_gdef)
        self._VerifyGraphDef(params, infer_gdef, precision_mode, True,
                             dynamic_calib_engine)

      self.assertAllClose(
          ref_result,
          result,
          atol=params.allclose_atol,
          rtol=params.allclose_rtol)
    else:
      infer_gdef = input_gdef

    # Run inference.
    infer_config = self._GetConfigProto(params, use_optimizer, precision_mode,
                                        dynamic_infer_engine)
    logging.info("Running final inference graph, config:\n%s",
                 str(infer_config))
    if use_optimizer:
      result = self._RunGraph(params, infer_gdef, input_data, infer_config)
    else:
      trt_infer_gdef = self._GetTrtGraphDef(params, infer_gdef, precision_mode,
                                            dynamic_infer_engine)
      self._VerifyGraphDef(params, trt_infer_gdef, precision_mode, True,
                           dynamic_infer_engine)
      result = self._RunGraph(params, trt_infer_gdef, input_data, infer_config)

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

  def _GetTest(use_optimizer, precision_mode, dynamic_infer_engine,
               dynamic_calib_engine):
    """Gets a single test method based on the parameters."""

    def _Test(self):
      params = self.GetParams()
      logging.info(
          "Running test with parameters: use_optimizer=%s, precision_mode=%s, "
          "dynamic_infer_engine=%s, dynamic_calib_engine=%s", use_optimizer,
          precision_mode, dynamic_infer_engine, dynamic_calib_engine)
      self.RunTest(params, use_optimizer, precision_mode, dynamic_infer_engine,
                   dynamic_calib_engine)

    return _Test

  use_optimizer_options = [False, True]
  dynamic_infer_engine_options = [False, True]
  dynamic_calib_engine_options = [False, True]
  for (use_optimizer, precision_mode,
       dynamic_infer_engine, dynamic_calib_engine) in itertools.product(
           use_optimizer_options, PRECISION_MODES, dynamic_infer_engine_options,
           dynamic_calib_engine_options):
    if _IsQuantizationMode(precision_mode):
      if not dynamic_calib_engine and dynamic_infer_engine:
        # TODO(aaroey): test this case, the conversion from static calibration
        # engine to dynamic inference engine should be a noop.
        continue
      if use_optimizer:
        # TODO(aaroey): if use_optimizer is True we need to get the inference
        # graphdef using custom python wrapper class, which is not currently
        # supported yet.
        continue
      if not dynamic_calib_engine:
        # TODO(aaroey): construction of static calibration engine is not
        # supported yet.
        continue
      if dynamic_calib_engine and not dynamic_infer_engine:
        # TODO(aaroey): construction of static inference engine using dynamic
        # calibration engine is not supported yet.
        continue
    else:  # In non int8 mode.
      if dynamic_calib_engine:
        # dynamic_calib_engine doesn't affect non-int8 modes, so just let
        # related tests run once on dynamic_calib_engine=False.
        continue

    conversion = "OptimizerConversion" if use_optimizer else "ToolConversion"
    infer_engine_type = ("DynamicInferEngine"
                         if dynamic_infer_engine else "StaticInferEngine")
    calib_engine_type = ""
    if precision_mode == "INT8":
      calib_engine_type = ("DynamicCalibEngine"
                           if dynamic_calib_engine else "StaticCalibEngine")
    test_name = "%s_%s_%s%s" % (conversion, precision_mode, infer_engine_type,
                                ("_" + calib_engine_type)
                                if len(calib_engine_type) else "")
    setattr(
        test_class, "testTfTRT_" + test_name,
        _GetTest(use_optimizer, precision_mode, dynamic_infer_engine,
                 dynamic_calib_engine))


if trt_convert.is_tensorrt_enabled():
  _AddTests(TfTrtIntegrationTestBase)
