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
"""Script to test TF-TensorRT integration."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import itertools
import warnings
import numpy as np
import six

from tensorflow.contrib import tensorrt as trt
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test

INPUT_NAME = "input"
OUTPUT_NAME = "output"
INPUT_DIMS = [100, 24, 24, 2]
MODE_FP32 = "FP32"
MODE_FP16 = "FP16"
MODE_INT8 = "INT8"

if six.PY2:
  to_bytes = lambda s: s
  to_string = lambda s: s
else:
  to_bytes = lambda s: s.encode("utf-8", errors="surrogateescape")
  to_string = lambda s: s.decode("utf-8")


# TODO(aaroey): test graph with different dtypes.
def GetSingleEngineGraphDef(dtype=dtypes.float32):
  """Create a graph containing single segment."""
  g = ops.Graph()
  with g.as_default():
    inp = array_ops.placeholder(
        dtype=dtype, shape=[None] + INPUT_DIMS[1:], name=INPUT_NAME)
    with g.device("/GPU:0"):
      conv_filter = constant_op.constant(
          [[[[1., 0.5, 4., 6., 0.5, 1.], [1., 0.5, 1., 1., 0.5, 1.]]]],
          name="weights",
          dtype=dtype)
      conv = nn.conv2d(
          input=inp,
          filter=conv_filter,
          strides=[1, 2, 2, 1],
          padding="SAME",
          name="conv")
      bias = constant_op.constant(
          [4., 1.5, 2., 3., 5., 7.], name="bias", dtype=dtype)
      added = nn.bias_add(conv, bias, name="bias_add")
      relu = nn.relu(added, "relu")
      identity = array_ops.identity(relu, "identity")
      pool = nn_ops.max_pool(
          identity, [1, 2, 2, 1], [1, 2, 2, 1], "VALID", name="max_pool")
    array_ops.squeeze(pool, name=OUTPUT_NAME)
  return g.as_graph_def()


# TODO(aaroey): test graph with different dtypes.
def GetMultiEngineGraphDef(dtype=dtypes.float32):
  """Create a graph containing multiple segment."""
  g = ops.Graph()
  with g.as_default():
    inp = array_ops.placeholder(
        dtype=dtype, shape=[None] + INPUT_DIMS[1:], name=INPUT_NAME)
    with g.device("/GPU:0"):
      conv_filter = constant_op.constant(
          [[[[1., 0.5, 4., 6., 0.5, 1.], [1., 0.5, 1., 1., 0.5, 1.]]]],
          name="weights",
          dtype=dtype)
      conv = nn.conv2d(
          input=inp,
          filter=conv_filter,
          strides=[1, 2, 2, 1],
          padding="SAME",
          name="conv")
      c1 = constant_op.constant(
          np.random.randn(INPUT_DIMS[0], 12, 12, 6), dtype=dtype)
      p = conv * c1
      c2 = constant_op.constant(
          np.random.randn(INPUT_DIMS[0], 12, 12, 6), dtype=dtype)
      q = conv / c2

      edge = math_ops.sin(q)
      edge /= edge
      r = edge + edge

      p -= edge
      q *= edge
      s = p + q
      s -= r
    array_ops.squeeze(s, name=OUTPUT_NAME)
  return g.as_graph_def()


TestGraph = namedtuple("TestGraph",
                       ["gdef", "num_expected_engines", "expected_output_dims"])

TEST_GRAPHS = {
    "SingleEngineGraph":
        TestGraph(
            gdef=GetSingleEngineGraphDef(),
            num_expected_engines=1,
            expected_output_dims=(100, 6, 6, 6)),
    "MultiEngineGraph":
        TestGraph(
            gdef=GetMultiEngineGraphDef(),
            num_expected_engines=2,
            expected_output_dims=(100, 12, 12, 6)),
    # TODO(aaroey): add a large complex graph to test.
}


class TfTrtIntegrationTest(test_util.TensorFlowTestCase):
  """Class to test Tensorflow-TensorRT integration."""

  def setUp(self):
    """Setup method."""
    super(TfTrtIntegrationTest, self).setUp()
    warnings.simplefilter("always")
    self._input = np.random.random_sample(INPUT_DIMS)

  def _GetConfigProto(self,
                      use_optimizer,
                      precision_mode=None,
                      is_dynamic_op=None):
    if use_optimizer:
      rewriter_cfg = rewriter_config_pb2.RewriterConfig()
      rewriter_cfg.optimizers.extend(["constfold", "layout"])
      custom_op = rewriter_cfg.custom_optimizers.add()
      custom_op.name = "TensorRTOptimizer"
      custom_op.parameter_map["minimum_segment_size"].i = 3
      custom_op.parameter_map["max_batch_size"].i = self._input.shape[0]
      custom_op.parameter_map["is_dynamic_op"].b = is_dynamic_op
      custom_op.parameter_map["max_workspace_size_bytes"].i = 1 << 25
      custom_op.parameter_map["precision_mode"].s = to_bytes(precision_mode)
      graph_options = config_pb2.GraphOptions(rewrite_options=rewriter_cfg)
    else:
      graph_options = config_pb2.GraphOptions()

    gpu_options = config_pb2.GPUOptions()
    if trt.trt_convert.get_linked_tensorrt_version()[0] == 3:
      gpu_options.per_process_gpu_memory_fraction = 0.50

    config = config_pb2.ConfigProto(
        gpu_options=gpu_options, graph_options=graph_options)
    return config

  def _RunGraph(self, graph_key, gdef, input_data, config, num_runs=2):
    """Run given graphdef multiple times."""
    g = ops.Graph()
    with g.as_default():
      inp, out = importer.import_graph_def(
          graph_def=gdef, return_elements=[INPUT_NAME, OUTPUT_NAME], name="")
      inp = inp.outputs[0]
      out = out.outputs[0]
    with self.test_session(
        graph=g, config=config, use_gpu=True, force_gpu=True) as sess:
      val = None
      # Defaults to 2 runs to verify result across multiple runs is same.
      for _ in range(num_runs):
        new_val = sess.run(out, {inp: input_data})
        self.assertEqual(TEST_GRAPHS[graph_key].expected_output_dims,
                         new_val.shape)
        if val is not None:
          self.assertAllEqual(new_val, val)
        val = new_val
    return val

  # Use real data that is representative of the inference dataset
  # for calibration. For this test script it is random data.
  def _RunCalibration(self, graph_key, gdef, input_data, config):
    """Run calibration on given graph."""
    return self._RunGraph(graph_key, gdef, input_data, config, 30)

  def _GetTrtGraph(self, gdef, precision_mode, is_dynamic_op):
    """Return trt converted graph."""
    return trt.create_inference_graph(
        input_graph_def=gdef,
        outputs=[OUTPUT_NAME],
        max_batch_size=self._input.shape[0],
        max_workspace_size_bytes=1 << 25,
        precision_mode=precision_mode,
        minimum_segment_size=2,
        is_dynamic_op=is_dynamic_op)

  def _VerifyGraphDef(self,
                      graph_key,
                      gdef,
                      precision_mode=None,
                      is_calibrated=None,
                      dynamic_engine=None):
    num_engines = 0
    for n in gdef.node:
      if n.op == "TRTEngineOp":
        num_engines += 1
        self.assertNotEqual(to_bytes(""), n.attr["serialized_segment"].s)
        self.assertNotEqual(to_bytes(""), n.attr["segment_funcdef_name"].s)
        self.assertEqual(n.attr["precision_mode"].s, to_bytes(precision_mode))
        self.assertEqual(n.attr["static_engine"].b, not dynamic_engine)
        if precision_mode == MODE_INT8 and is_calibrated:
          self.assertNotEqual(to_bytes(""), n.attr["calibration_data"].s)
        else:
          self.assertEqual(to_bytes(""), n.attr["calibration_data"].s)
    if precision_mode is None:
      self.assertEqual(num_engines, 0)
    else:
      self.assertEqual(num_engines,
                       TEST_GRAPHS[graph_key].num_expected_engines)

  def _RunTest(self, graph_key, use_optimizer, precision_mode,
               dynamic_infer_engine, dynamic_calib_engine):
    assert precision_mode in [MODE_FP32, MODE_FP16, MODE_INT8]
    input_gdef = TEST_GRAPHS[graph_key].gdef
    self._VerifyGraphDef(graph_key, input_gdef)

    # Get reference result without running trt.
    config_no_trt = self._GetConfigProto(False)
    print("Running original graph w/o trt, config:\n%s" % str(config_no_trt))
    ref_result = self._RunGraph(graph_key, input_gdef, self._input,
                                config_no_trt)

    # Run calibration if necessary.
    if precision_mode == MODE_INT8:

      calib_config = self._GetConfigProto(use_optimizer, precision_mode,
                                          dynamic_calib_engine)
      print("Running calibration graph, config:\n%s" % str(calib_config))
      if use_optimizer:
        self.assertTrue(False)
        # TODO(aaroey): uncomment this and get infer_gdef when this mode is
        # supported.
        # result = self._RunCalibration(graph_key, input_gdef, self._input,
        #                               calib_config)
      else:
        calib_gdef = self._GetTrtGraph(input_gdef, precision_mode,
                                       dynamic_calib_engine)
        self._VerifyGraphDef(graph_key, calib_gdef, precision_mode, False,
                             dynamic_calib_engine)
        result = self._RunCalibration(graph_key, calib_gdef, self._input,
                                      calib_config)
        infer_gdef = trt.calib_graph_to_infer_graph(calib_gdef)
        self._VerifyGraphDef(graph_key, infer_gdef, precision_mode, True,
                             dynamic_calib_engine)
      self.assertAllClose(ref_result, result, rtol=1.e-03)
    else:
      infer_gdef = input_gdef

    # Run inference.
    infer_config = self._GetConfigProto(use_optimizer, precision_mode,
                                        dynamic_infer_engine)
    print("Running final inference graph, config:\n%s" % str(infer_config))
    if use_optimizer:
      result = self._RunGraph(graph_key, infer_gdef, self._input, infer_config)
    else:
      trt_infer_gdef = self._GetTrtGraph(infer_gdef, precision_mode,
                                         dynamic_infer_engine)
      self._VerifyGraphDef(graph_key, trt_infer_gdef, precision_mode, True,
                           dynamic_infer_engine)
      result = self._RunGraph(graph_key, trt_infer_gdef, self._input,
                              infer_config)
    self.assertAllClose(ref_result, result, rtol=1.e-03)

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


def GetTests():

  def _GetTest(g, u, p, i, c):

    def _Test(self):
      print("Running test with parameters: graph_key=%s, use_optimizer=%s, "
            "precision_mode=%s, dynamic_infer_engine=%s, "
            "dynamic_calib_engine=%s" % (g, u, p, i, c))
      self._RunTest(g, u, p, i, c)

    return _Test

  use_optimizer_options = [False, True]
  precision_mode_options = [MODE_FP32, MODE_FP16, MODE_INT8]
  dynamic_infer_engine_options = [False, True]
  dynamic_calib_engine_options = [False, True]
  for (graph_key, use_optimizer, precision_mode,
       dynamic_infer_engine, dynamic_calib_engine) in itertools.product(
           TEST_GRAPHS, use_optimizer_options, precision_mode_options,
           dynamic_infer_engine_options, dynamic_calib_engine_options):
    if precision_mode == MODE_INT8:
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
    yield _GetTest(graph_key, use_optimizer, precision_mode,
                   dynamic_infer_engine, dynamic_calib_engine)


if __name__ == "__main__":
  if trt.is_tensorrt_enabled():
    for index, t in enumerate(GetTests()):
      setattr(TfTrtIntegrationTest, "testTfTRT_" + str(index), t)
  test.main()
