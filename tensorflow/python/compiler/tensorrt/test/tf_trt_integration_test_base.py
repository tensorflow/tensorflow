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

import collections
import errno
import gc
import itertools
import os
import re
import shutil
import tempfile
import warnings

import numpy as np

from tensorflow.compiler.tf2tensorrt._pywrap_py_utils import is_tensorrt_enabled
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.compiler.tensorrt import trt_convert
from tensorflow.python.compiler.tensorrt import utils as trt_utils
from tensorflow.python.eager import def_function
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.tools import saved_model_utils
from tensorflow.python.trackable import autotrackable
from tensorflow.python.util import nest

logging.get_logger().propagate = False

TfTrtIntegrationTestParams = collections.namedtuple(
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

RunParams = collections.namedtuple(
    "RunParams",
    [
        # Whether to run the conversion online with RewriterConfig, or offline
        # with TrtGraphConverter.
        "convert_online",
        "precision_mode",
        "dynamic_engine",
        "use_calibration",
        "test_name",
        # Is this test for TF 2.0?
        "is_v2",
        "dynamic_shape",
    ])

FP32 = "FP32"
FP16 = "FP16"
INT8 = "INT8"
PRECISION_MODES = [FP32, FP16, INT8]


def IsQuantizationMode(mode):
  return mode == "INT8"


def IsQuantizationWithCalibration(params):
  return IsQuantizationMode(params.precision_mode) and params.use_calibration


def IsQuantizationWithoutCalibration(params):
  return IsQuantizationMode(
      params.precision_mode) and not params.use_calibration


class GraphState(object):
  ORIGINAL = 0
  CALIBRATE = 1
  INFERENCE = 2


class TfTrtIntegrationTestBase(test_util.TensorFlowTestCase):
  """Class to test Tensorflow-TensorRT integration."""

  @property
  def trt_incompatible_op(self):
    return math_ops.erfc

  @property
  def trt_incompatible_binary_op(self):
    return math_ops.igamma

  @property
  def precision_modes(self):
    return ["FP32", "FP16", "INT8"]

  # str is bytes in py2, but unicode in py3.
  def _ToUnicode(self, s):
    if isinstance(s, str):
      return s
    return s.decode("utf-8")

  def _ToBytes(self, s):
    if isinstance(s, str):
      return s.encode("utf-8")
    return s

  def _ToString(self, s):
    if isinstance(s, str):
      return s
    return s.decode("utf-8")

  def __init__(self, methodName="runTest"):  # pylint: disable=invalid-name
    super(TfTrtIntegrationTestBase, self).__init__(methodName)
    self._trt_test_params = None
    self._disable_non_trt_optimizers = False
    self._profile_strategy = "ImplicitBatchModeCompatible"

  def setUp(self):
    """Setup method."""
    super().setUp()
    warnings.simplefilter("always")

    if not is_tensorrt_enabled():
      self.skipTest("Test requires TensorRT")

  def tearDown(self):
    """Making sure to clean artifact."""
    idx = 0
    while gc.garbage:
      gc.collect()  # Force GC to destroy the TRT engine cache.
      idx += 1
      if idx >= 10:  # After 10 iterations, break to avoid infinite collect.
        break

  def _GetTensorSpec(self, shape, mask, dtype, name):
    # Set dimension i to None if mask[i] == False
    assert len(shape) == len(mask), (
        f"len(shape): {len(shape)} == len(mask): {len(mask)}")

    new_shape = [s if m else None for s, m in zip(shape, mask)]
    return tensor_spec.TensorSpec(new_shape, dtype, name)

  def BuildParams(self, graph_fn, dtype, input_shapes, output_shapes):
    """Build test parameters.

    The input_shapes and output_shapes arguments are known (static) shapes that
    can be used to generate test data. To define the model, we also specify
    corresponding input/output TensorSpecs. These are defined using the shape
    arguments. For each input tensor we define:

    input_spec = [None] + input_shape[1:]

    and similarly for output shapes. This means that we leave the first (batch)
    dimension unknown, the rest is just copied from the shapes arg.

    Args:
      graph_fn: The function to build the graph.
      dtype: The element type.
      input_shapes: The input shapes.
      output_shapes: The output shapes.

    Returns:
      The test parameters.
    """

    input_mask = [[False] + [True] * (len(shape) - 1) for shape in input_shapes]
    output_mask = [[False] + [True] * (len(shape) - 1) if shape else []
                   for shape in output_shapes]

    return self.BuildParamsWithMask(graph_fn, dtype, input_shapes,
                                    output_shapes, input_mask, output_mask, [],
                                    [])

  def BuildParamsWithMask(self, graph_fn, dtype, input_shapes, output_shapes,
                          input_mask, output_mask, extra_inputs, extra_outputs):
    """Build test parameters with static or dynamic input shapes.

    To define dynamic shapes give a boolean mask that describes which
    dimensions to treat as known. The values in input_mask are interpreted the
    following way:
    - True: known dim (use the corresponding value from input_shapes)
    - False: unknown dim (replace the corresponding value from input_shapes
             with None)
    For example, to define the first two dimension with unknown size use
    input_shapes=[[1,2,1,8]], input_mask=[[False, False, True, True]].

    Args:
      graph_fn: The function to build the graph.
      dtype: The element type.
      input_shapes: The input shapes.
      output_shapes: The output shapes.
      input_mask: The input shape masks.
      output_mask: the output shape masks.
      extra_inputs: list of additional input shapes
      extra_outputs: list of additional outputs shapes

    Returns:
      The test parameters.
    """

    def _ValidateShapes(shapes):
      # Make sure all the shapes are fully specified.
      for shape in shapes:
        assert all(shape), f"Shape unspecified: {shape}"

    _ValidateShapes(input_shapes)
    _ValidateShapes(output_shapes)

    assert len(input_mask) == len(input_shapes), (
        f"Inconsistent input_mask and input_shapes: len({input_mask}) != "
        f"len({input_shapes}).")
    assert len(output_mask) == len(output_shapes), (
        f"Inconsistent output_mask and output_shapes: len({output_mask}) != "
        f"len({output_shapes}).")
    for extra_in_shape, extra_out_shape in zip(extra_inputs, extra_outputs):
      assert len(input_shapes) == len(extra_in_shape), (
          f"Inconsistent input_shapes and extra_in_shape: len({input_shapes}) "
          f"!= len({extra_in_shape}).")
      assert len(output_shapes) == len(extra_out_shape), (
          f"Inconsistent output_shapes and extra_out_shape: "
          f"len({output_shapes}) != len({extra_out_shape}).")

    return TfTrtIntegrationTestParams(
        graph_fn=graph_fn,
        input_specs=[
            self._GetTensorSpec(shape, mask, dtype, "input_%d" % i)
            for i, (shape, mask) in enumerate(zip(input_shapes, input_mask))
        ],
        output_specs=[
            self._GetTensorSpec(shape, mask, dtype, "output_%d" % i)
            for i, (shape, mask) in enumerate(zip(output_shapes, output_mask))
        ],
        input_dims=[input_shapes] + extra_inputs,
        expected_output_dims=[output_shapes] + extra_outputs)

  def DisableNonTrtOptimizers(self):
    self._disable_non_trt_optimizers = True

  def GetParams(self):
    """Returns a TfTrtIntegrationTestParams for the test."""
    raise NotImplementedError()

  def GetConversionParams(self, run_params):
    """Returns a TrtConversionParams for test."""
    conversion_params = trt_convert.TrtConversionParams(
        # We use the minimum of all the batch sizes, so when multiple different
        # input shapes are provided it'll always create new engines in the
        # cache, and we can therefore test the cache behavior.
        max_workspace_size_bytes=(
            trt_convert.DEFAULT_TRT_MAX_WORKSPACE_SIZE_BYTES),
        precision_mode=run_params.precision_mode,
        minimum_segment_size=2,
        maximum_cached_engines=1,
        use_calibration=run_params.use_calibration)
    return conversion_params

  def GetMaxBatchSize(self, run_params):
    """Returns the max_batch_size that the converter should use for tests."""
    if run_params.dynamic_engine:
      return None
    batch_list = []
    for dims_list in self._GetParamsCached().input_dims:
      assert dims_list, f"Expect non-empty `dim_list` but got: {dims_list}"
      # Each list of shapes should have same batch size.
      input_batches = [dims[0] for dims in dims_list]
      assert max(input_batches) == min(input_batches), (
          f"Inconsistent batch_size: max({input_batches}) != "
          f"min({input_batches}).")
      batch_list.append(input_batches[0])
    return max(batch_list)

  def ShouldRunTest(self, run_params):
    """Whether to run the test."""
    # Ensure use_calibration=True in case of INT8 precision
    return (run_params.use_calibration or not IsQuantizationMode(
        run_params.precision_mode)), "test either calibration or non-INT8"

  def ExpectedEnginesToBuild(self, run_params):
    """Returns the expected engines to build, implemented by subclass."""
    raise NotImplementedError()

  def ExpectedConnections(self, run_params):
    """Returns the expected edges or an empty dict to skip the check."""
    return {}

  def ExpectedMaxBatchSizes(self, run_params):
    """Returns the expected maximum batch sizes of the build engines."""
    return None

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
    max_batch_size = self.GetMaxBatchSize(run_params)

    if graph_state == GraphState.INFERENCE and run_params.convert_online:
      rewriter_cfg = trt_convert.get_tensorrt_rewriter_config(
          conversion_params,
          is_dynamic_op=run_params.dynamic_engine,
          max_batch_size=max_batch_size,
          disable_non_trt_optimizers=self._disable_non_trt_optimizers)
    else:
      rewriter_cfg = rewriter_config_pb2.RewriterConfig()
      if self._disable_non_trt_optimizers:
        trt_utils.disable_non_trt_optimizers_in_rewriter_config(rewriter_cfg)

    config = config_pb2.ConfigProto(
        gpu_options=self._GetGPUOptions(),
        graph_options=config_pb2.GraphOptions(rewrite_options=rewriter_cfg))
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

  def _RunGraphV1(self, saved_model_dir, inputs_data, config, num_runs=2):
    """Run given graphdef multiple times using TF 1.x runtime."""
    params = self._GetParamsCached()
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
              # Some ops may have nondeterministic output. E.g. Conv2D may use
              # winograd algorithm. So we set atol/rtol be larger than 1.e-06.
              self.assertAllClose(val, new_val, atol=1.e-05, rtol=1.e-05)
            val = new_val
          vals.append(val)
        return vals

  def _RunGraphV2(self, saved_model_dir, inputs_data, graph_state, num_runs=2):
    """Run given graphdef multiple times using TF 2.0 runtime."""
    params = self._GetParamsCached()
    root = load.load(saved_model_dir)
    func = root.signatures[
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    results = []
    for expected_shapes, current_input_data in zip(params.expected_output_dims,
                                                   inputs_data):
      val = None
      for _ in range(num_runs):
        feed_dict = {
            params.input_specs[i].name: current_input_data[i]
            for i in range(len(params.input_specs))
        }
        new_val = func(**feed_dict)
        assert isinstance(
            new_val, dict), (f"Invalid type for `new_val`, expected `dict`. "
                             f"Got: {type(new_val)}.")
        # The key of the output map is always like output_i.
        new_val = [new_val[key] for key in sorted(new_val)]
        # Each element is an eager Tensor, and accessing individual elements is
        # very expensive, so we convert them to a numpy array first.
        new_val = [v.numpy() for v in new_val]
        self.assertEqual(len(expected_shapes), len(new_val))
        for expected_shape, actual_val in zip(expected_shapes, new_val):
          self.assertEqual(list(expected_shape), list(actual_val.shape))
        if val is not None:
          # Some ops may have nondeterministic output. E.g. Conv2D may use
          # winograd algorithm. So we set atol/rtol be larger than 1.e-06.
          self.assertAllClose(val, new_val, atol=1.e-05, rtol=1.e-05)
        val = new_val
      results.append(val)

    return results

  def _RunGraph(self,
                run_params,
                saved_model_dir,
                inputs_data,
                graph_state,
                num_runs=2):
    params = self._GetParamsCached()
    for data in inputs_data:
      assert len(params.input_specs) == len(data), (
          f"Inconsistent params.input_specs and data: "
          f"len({params.input_specs}) != len({data}).")

    if run_params.is_v2:
      results = self._RunGraphV2(saved_model_dir, inputs_data, graph_state,
                                 num_runs)
      gc.collect()  # Force GC to destroy the TRT engine cache.
      return results

    # The default config for tf.session is None. Create a config with
    # TensorRTOptimizer enabled to support convert_online for inference.
    config = None
    # TODO(b/170220818): use the default session config to run inferenence
    #   graphs for the offline conversion case after fixing the bug.
    if graph_state == GraphState.INFERENCE:
      config = self._GetConfigProto(run_params, GraphState.INFERENCE)
    return self._RunGraphV1(saved_model_dir, inputs_data, config, num_runs)

  def _CreateConverter(self, run_params, saved_model_dir, conversion_params):
    """Returns a TrtGraphConverter."""
    if run_params.is_v2:
      converter_v2 = trt_convert.TrtGraphConverterV2(
          input_saved_model_dir=saved_model_dir,
          use_dynamic_shape=run_params.dynamic_shape,
          dynamic_shape_profile_strategy=self._profile_strategy,
          **conversion_params._asdict())
      if self._disable_non_trt_optimizers:
        converter_v2._test_only_disable_non_trt_optimizers = True  # pylint: disable=protected-access
      return converter_v2

    converter_v1 = trt_convert.TrtGraphConverter(
        input_saved_model_dir=saved_model_dir,
        max_batch_size=self.GetMaxBatchSize(run_params),
        max_workspace_size_bytes=conversion_params.max_workspace_size_bytes,
        precision_mode=conversion_params.precision_mode,
        minimum_segment_size=conversion_params.minimum_segment_size,
        is_dynamic_op=run_params.dynamic_engine,
        maximum_cached_engines=conversion_params.maximum_cached_engines,
        use_calibration=conversion_params.use_calibration)
    if self._disable_non_trt_optimizers:
      converter_v1._test_only_disable_non_trt_optimizers = True  # pylint: disable=protected-access
    return converter_v1

  def _GetCalibratedInferGraph(self, run_params, saved_model_dir, inputs_data):
    """Return trt converted graphdef in INT8 mode."""
    conversion_params = self.GetConversionParams(run_params)
    logging.info(conversion_params)
    assert conversion_params.precision_mode == "INT8", (
        f"Incorrect precision mode, expected INT8 but got: "
        f"{conversion_params.precision_mode}.")
    assert run_params.dynamic_engine, "dynamic_engine parameter must be True."
    assert conversion_params.maximum_cached_engines == 1, (
        f"maximum_cached_engines: {conversion_params.maximum_cached_engines} "
        f"== 1")
    assert conversion_params.use_calibration, "use_calibration must be True."

    # We only support calibrating single engine.
    # TODO(aaroey): fix this.
    assert len(inputs_data) == 1, (f"len(inputs_data): {len(inputs_data)} == 1")

    converter = self._CreateConverter(run_params, saved_model_dir,
                                      conversion_params)
    if run_params.is_v2:

      def CalibrationInputFn():
        for data_tensors in inputs_data:
          yield data_tensors

      converter.convert(calibration_input_fn=CalibrationInputFn)
    else:
      int8_gdef = converter.convert()
      self._VerifyGraphDef(run_params, saved_model_dir, int8_gdef,
                           GraphState.CALIBRATE)

      converter.calibrate(
          fetch_names=self._GetFetchNames(),
          num_runs=5,
          feed_dict_fn=lambda: self._GetFeedDict(inputs_data[0]))

    if run_params.dynamic_shape and self._ShouldConverterBuild(run_params):
      logging.info("Using build mode")

      def _BuildInputFn():
        for shapes in self._GetParamsCached().input_dims:
          yield [
              array_ops.zeros(x, dtype=spec.dtype)
              for (x, spec) in zip(shapes,
                                   self._GetParamsCached().input_specs)
          ]

      converter.build(input_fn=_BuildInputFn)

    trt_saved_model_dir = self._GetSavedModelDir(run_params,
                                                 GraphState.CALIBRATE)
    converter.save(trt_saved_model_dir)
    return trt_saved_model_dir

  def _ShouldConverterBuild(self, run_params):
    return True

  def _GetInferGraph(self, run_params, saved_model_dir):
    """Return trt converted graphdef."""
    conversion_params = self.GetConversionParams(run_params)
    logging.info(conversion_params)

    converter = self._CreateConverter(run_params, saved_model_dir,
                                      conversion_params)
    converter.convert()

    if run_params.is_v2:
      try:
        line_length = max(160, os.get_terminal_size().columns)
      except OSError:
        line_length = 160
      converter.summary(line_length=line_length, detailed=True)

    if run_params.dynamic_shape and self._ShouldConverterBuild(run_params):
      logging.info("Using build mode")

      def _BuildInputFn():
        for shapes in self._GetParamsCached().input_dims:
          yield [
              array_ops.zeros(x, dtype=spec.dtype)
              for (x, spec) in zip(shapes,
                                   self._GetParamsCached().input_specs)
          ]

      converter.build(input_fn=_BuildInputFn)

    trt_saved_model_dir = self._GetSavedModelDir(run_params,
                                                 GraphState.INFERENCE)
    converter.save(trt_saved_model_dir)
    return trt_saved_model_dir

  def _GetGraphStateLabel(self, graph_state):
    if graph_state == GraphState.ORIGINAL:
      return "Original"
    elif graph_state == GraphState.CALIBRATE:
      return "CalibEngine"
    elif graph_state == GraphState.INFERENCE:
      return "InferEngine"
    else:
      return "UnknownState"

  def _WriteGraph(self, run_params, gdef, graph_state):
    temp_dir = os.getenv("TRT_TEST_TMPDIR")
    if not temp_dir:
      return

    graph_name = (
        self.__class__.__name__ + "_" + run_params.test_name + "_" +
        self._GetGraphStateLabel(graph_state) + ".pbtxt")
    logging.info("Writing graph to %s/%s", temp_dir, graph_name)
    graph_io.write_graph(gdef, temp_dir, graph_name)

  # Removes the prefix(s) of function name(s).
  # The input value can be a string or a sequence of string.
  def _Canonicalize(self, value):
    if isinstance(value, str):
      return self._ToString(value.split("/")[-1])
    elif isinstance(value, collections.abc.Iterable):
      return set(self._Canonicalize(nm) for nm in value)
    else:
      raise TypeError(
          "'_Canonicalize' can only be used on strings or sequence of strings!")

  # Removes the graph sequence number prefix from the name(s) only if the
  # name(s) has a prefix TRTEngineOp_n_. When expecting_prefix is true, asserts
  # such a prefix exists.
  # The input value can be a string or a sequence of string.
  def _RemoveGraphSequenceNumberImpl(self, value, expecting_prefix):
    if isinstance(value, str):
      match = re.search(r"TRTEngineOp_\d{3,}_", value)
      has_prefix = match and value.startswith(match.group(0))
      assert (not expecting_prefix) or has_prefix, (
          f"Expect (not expecting_prefix) or has_prefix but got: "
          f"- expecting_prefix = {expecting_prefix}\n"
          f"- has_prefix = {has_prefix}")
      if has_prefix:
        parts = value.split("_", maxsplit=2)
        assert len(parts) == 3, (
            f"Incorrect `parts` of length == 3, but got: len({parts}).")
        return parts[0] + "_" + parts[2]
      return value
    elif isinstance(value, collections.abc.Iterable):
      return set(
          self._RemoveGraphSequenceNumberImpl(nm, expecting_prefix)
          for nm in value)
    else:
      raise TypeError(
          "'_RemoveGraphSequenceNumberImpl' can only be used on strings "
          "or sequence of strings!")

  def _RemoveGraphSequenceNumber(self, name):
    return self._RemoveGraphSequenceNumberImpl(name, True)

  def _MayRemoveGraphSequenceNumber(self, name):
    return self._RemoveGraphSequenceNumberImpl(name, False)

  def _VerifyConnections(self, expected_engines, expected_input_map,
                         original_gdef, converted_gdef):
    """Checks that the converted graph contains the expected connections."""
    old_to_new_node_map = {
        self._ToString(node.name): self._ToString(node.name)
        for node in original_gdef.node
    }
    for engine_name, node_names in expected_engines.items():
      for node_name in node_names:
        old_to_new_node_map[node_name] = engine_name

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

    # Compute the actual mapping from each node to its input nodes. If a cast
    # op doesn't exist in the original graph, we replace the use of the cast op
    # with the input of the op. This allows the verification to handle the case
    # where the TF-TRT bridge splits a cast op into a chain of two cast ops.
    new_cast_op_name_to_node_map = {
        node.name: node
        for node in converted_gdef.node
        if (node.name not in old_to_new_node_map and node.op == "Cast")
    }
    actual_input_map = {}
    for node in converted_gdef.node:
      name_str = node.name
      # Only nodes from the original graph or TRTEngineOp nodes are added as
      # keys to the map.
      if node.op == "TRTEngineOp":
        name_str = self._RemoveGraphSequenceNumber(name_str)
      elif name_str not in old_to_new_node_map:
        continue
      actual_input_map[name_str] = set()
      input_set = actual_input_map[name_str]
      for inp in node.input:
        (prefix, node_name) = _InputName(inp)
        node_name = self._MayRemoveGraphSequenceNumber(node_name)
        if node_name in new_cast_op_name_to_node_map:
          (prefix, node_name) = _InputName(
              new_cast_op_name_to_node_map[node_name].input[0])
        input_set.add(prefix + node_name)

    self.assertEqual(
        expected_input_map,
        actual_input_map,
        msg="\nexpected:\n%s\nvs actual:\n%s" %
        (sorted(expected_input_map.items()), sorted(actual_input_map.items())))

  def _VerifyMaxBatchSizeAnnotations(
      self,
      expected_engines,
      original_gdef,
      converted_gdef,
      default_max_batch_size,
      expected_max_batch_sizes=None,
  ):
    """Verifies the max batch size annotations in the original and converted GraphDef.

    Args:
      expected_engines: A sequence of engines names.
      original_gdef: GraphDef. The graph def before TensorRT conversion.
      converted_gdef: GraphDef. The graph def after TensorRT conversion.
      default_max_batch_size: The default maximum batch size to use if no node
        inside a segment is annoted with a customized max batch size. This value
        is None when the graph is converted to TF-TRT with dynamic engines.
      expected_max_batch_sizes: Optional. A sequence of max batch sizes for all
        the engines. `None` if does not check enforce max batch sizes.
    """
    if isinstance(expected_max_batch_sizes, collections.abc.Collection):
      self.assertEqual(len(expected_max_batch_sizes), len(expected_engines))
    else:
      self.assertIsNone(
          expected_max_batch_sizes,
          "'expected_max_batch_sizes' shall only be a sequence "
          "of integers or `None`.")

    def _ChainAllNodes(graph_def):
      return itertools.chain(
          graph_def.node,
          itertools.chain(
              *[func.node_def for func in graph_def.library.function]))

    old_name_to_node_map = {
        self._ToString(node.name): node
        for node in _ChainAllNodes(original_gdef)
    }
    new_name_to_func_map = {
        self._ToString(func.signature.name): func
        for func in converted_gdef.library.function
    }

    def _DetectStaticBatchSize(node_def):
      """Returns the static batch size of an operation or None.

      It is incorrect to use the output shapes to find the batch size of an
      operation, as the segmenter actually uses the input shapes. However, it is
      a simplication and works for most of the cases for the test purposes.

      Args:
        node_def: `tf.NodeDef`. The target node for analysis.

      Returns:
        If all the outputs of the node have the same static batch size, returns
        the int value for the batch size. Otherwise returns None.
      """
      shapes = node_def.attr["_output_shapes"].list.shape
      batch_size = set(
          list(s.dim)[0].size if len(s.dim) >= 2 else None for s in shapes)
      if len(batch_size) == 1 and list(batch_size)[0] >= 1:
        return list(batch_size)[0]
      return None

    name_to_engines_map = {}
    actual_max_batch_sizes = []
    for node in _ChainAllNodes(converted_gdef):
      if node.op == "TRTEngineOp":
        engine = node
        engine_name = self._RemoveGraphSequenceNumber(
            self._Canonicalize(self._ToString(engine.name)))
        self.assertIn(engine_name, expected_engines)
        name_to_engines_map[engine_name] = engine
        # The input nodes shall not have the conflicting annotation (no
        # annotation or the same annotation) with the maximum batch size
        # annotation. If the engine has maximum batch size annotation as the
        # non-default maximum batch size, then at least one input node shall
        # have the same annotation to be the source.
        self.assertIn("max_batch_size", node.attr)
        engine_max_batch_size = node.attr["max_batch_size"].i
        self.assertIsInstance(engine_max_batch_size, int)
        actual_max_batch_sizes.append(engine_max_batch_size)
        seg_func = node.attr["segment_func"].func
        self.assertIsNotNone(seg_func)
        self.assertIn(seg_func.name, new_name_to_func_map)
        seg_func_def = new_name_to_func_map[seg_func.name]
        logging.info("Segment function name: %s. Including %d nodes.",
                     seg_func.name, len(seg_func_def.node_def))
        node_max_batch_size_all_none = True
        # Use the native segment to search for replaced nodes
        for alternative_node in seg_func_def.node_def:
          node_name = self._Canonicalize(self._ToString(alternative_node.name))
          if node_name not in old_name_to_node_map:
            continue
          original_node = old_name_to_node_map[node_name]
          node_max_batch_size = None
          if "_tftrt_op_max_batch_size" in original_node.attr:
            node_max_batch_size = original_node.attr[
                "_tftrt_op_max_batch_size"].i
          elif (original_node.op != "Const" and
                alternative_node.op != "Const" and
                "_output_shapes" in original_node.attr):
            node_max_batch_size = _DetectStaticBatchSize(original_node)
          logging.info(
              "'{%s}(%s)'s max batch size annotation is %s. "
              "'{%s}'s max batch size is %s.", node_name, original_node.op,
              str(node_max_batch_size), engine_name, str(engine_max_batch_size))
          node_max_batch_size_all_none &= node_max_batch_size is None
          self.assertTrue(engine_max_batch_size == node_max_batch_size or
                          node_max_batch_size is None)
        logging.info("'{%s}'s max batch size is %d.", engine_name,
                     engine_max_batch_size)
        self.assertTrue(default_max_batch_size is None or
                        engine_max_batch_size == default_max_batch_size or
                        not node_max_batch_size_all_none)

    self.assertCountEqual(expected_engines, tuple(name_to_engines_map.keys()))
    if expected_max_batch_sizes is not None:
      self.assertCountEqual(expected_max_batch_sizes, actual_max_batch_sizes)

  def _GetGraphDef(self, run_params, gdef_or_saved_model_dir):
    if isinstance(gdef_or_saved_model_dir, str):
      if run_params.is_v2:
        root = load.load(gdef_or_saved_model_dir)
        func = root.signatures[
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        gdef = func.graph.as_graph_def()
        # Manually unref the loaded saved model and force GC to destroy the TRT
        # engine cache after load(). There is currently a reference cycle in 2.0
        # which prevents auto deletion of the resource.
        # TODO(laigd): fix this.
        del func
        del root
        gc.collect()
        return gdef
      return saved_model_utils.get_meta_graph_def(
          gdef_or_saved_model_dir, tag_constants.SERVING).graph_def
    assert isinstance(gdef_or_saved_model_dir, graph_pb2.GraphDef), (
        f"Incorrect `gdef_or_saved_model_dir` type, expected "
        f"`graph_pb2.GraphDef`, but got: {type(gdef_or_saved_model_dir)}.")
    return gdef_or_saved_model_dir

  def _VerifyGraphDefV1(self, run_params, original_gdef, gdef_to_verify,
                        graph_state):
    expected_engines = self.ExpectedEnginesToBuild(run_params)
    num_engines = 0
    functions = [f.signature.name for f in gdef_to_verify.library.function]
    for node in gdef_to_verify.node:
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
        if IsQuantizationWithoutCalibration(run_params):
          # TODO(bixia): Refine this check by inspecting nodes in the engine.
          if self._ToBytes("INT8") != node.attr["precision_mode"].s:
            self.assertEqual(
                self._ToBytes("FP16"), node.attr["precision_mode"].s, node.name)
        else:
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

  def _VerifyGraphDefV2(self, run_params, original_gdef, gdef_to_verify,
                        graph_state):
    if graph_state == GraphState.ORIGINAL:
      return
    expected_engines = self.ExpectedEnginesToBuild(run_params)
    all_op_names = [node.name for node in gdef_to_verify.node]
    trt_op_names = []
    for func in gdef_to_verify.library.function:
      if not re.search(r"TRTEngineOp_\d{3,}_\d{3,}_native_segment",
                       func.signature.name):
        for node in func.node_def:
          all_op_names.append(node.name)
          if node.op == "TRTEngineOp":
            trt_op_names.append(node.name)
            if run_params.dynamic_shape:
              self.assertEqual(
                  self._ToString(node.attr["profile_strategy"].s).lower(),
                  self._profile_strategy.lower())

    all_op_names = self._Canonicalize(all_op_names)
    trt_op_names = self._RemoveGraphSequenceNumber(
        self._Canonicalize(trt_op_names))

    if isinstance(expected_engines, dict):
      # For simplicity we don't verify the connections inside the engine in
      # 2.0, but we still make sure that the converted ops are gone from the
      # graph.
      unexpected_names = set(nest.flatten(expected_engines.values()))
      self.assertEmpty(
          [name for name in unexpected_names if name in all_op_names])
      expected_engines = set(expected_engines.keys())

    self.assertEqual(set(expected_engines), trt_op_names)

  def _VerifyGraphDef(self, run_params, original_gdef_or_saved_model_dir,
                      gdef_or_saved_model_dir_to_verify, graph_state):
    original_gdef = self._GetGraphDef(run_params,
                                      original_gdef_or_saved_model_dir)
    gdef_to_verify = self._GetGraphDef(run_params,
                                       gdef_or_saved_model_dir_to_verify)
    self._WriteGraph(run_params, gdef_to_verify, graph_state)
    if run_params.is_v2:
      self._VerifyGraphDefV2(run_params, original_gdef, gdef_to_verify,
                             graph_state)
    else:
      self._VerifyGraphDefV1(run_params, original_gdef, gdef_to_verify,
                             graph_state)

  def _GetSavedModelDir(self, run_params, graph_state):
    test_tmpdir = os.getenv("TRT_TEST_TMPDIR")
    if test_tmpdir:
      saved_model_dir = os.path.join(
          test_tmpdir, self.__class__.__name__ + "_" + run_params.test_name +
          "_" + self._GetGraphStateLabel(graph_state))
      try:
        # For TF 1.x we need to make sure the output directory doesn't exist
        # before exporting the saved model.
        shutil.rmtree(saved_model_dir)
      except OSError as e:
        if e.errno != errno.ENOENT:
          raise
      return saved_model_dir
    return tempfile.mkdtemp(dir=self.get_temp_dir())

  def _MakeSavedModelV1(self, run_params):
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

    signature_def = signature_def_utils.build_signature_def(
        inputs={inp.op.name: utils.build_tensor_info(inp) for inp in inputs},
        outputs={out.op.name: utils.build_tensor_info(out) for out in outputs},
        method_name=signature_constants.PREDICT_METHOD_NAME)

    saved_model_dir = self._GetSavedModelDir(run_params, GraphState.ORIGINAL)
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

  def _MakeSavedModelV2(self, run_params):
    params = self._GetParamsCached()
    root = autotrackable.AutoTrackable()
    root.run = def_function.function(
        params.graph_fn, input_signature=params.input_specs)
    saved_model_dir = self._GetSavedModelDir(run_params, GraphState.ORIGINAL)
    logging.info("Saving input SavedModel to %s", saved_model_dir)
    save.save(root, saved_model_dir,
              {signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: root.run})
    return saved_model_dir

  def _MakeSavedModel(self, run_params):
    if run_params.is_v2:
      return self._MakeSavedModelV2(run_params)
    return self._MakeSavedModelV1(run_params)

  def RunTest(self, run_params):
    with trace.Trace(run_params.test_name):
      should_run, reason_for_skipping = self.ShouldRunTest(run_params)
      if not should_run:
        return self.skipTest(reason_for_skipping)

      saved_model_dir = self._MakeSavedModel(run_params)

      np.random.seed(12345)  # Fix the seed so the test is deterministic.
      inputs_data = []
      input_specs = self._GetParamsCached().input_specs
      for dim_list in self._GetParamsCached().input_dims:
        assert len(input_specs) == len(dim_list), (
            f"Inconsistent input_specs and dim_list: len({input_specs}) != "
            f"len({dim_list}).")
        current_input_data = []
        for spec, np_shape in zip(input_specs, dim_list):
          np_dtype = spec.dtype.as_numpy_dtype()
          if not np.issubdtype(np_dtype, np.bool_):
            # Multiply the input by some constant to avoid all zeros input for
            # integer types.
            scale = 10.0 if np.issubdtype(np_dtype, np.integer) else 1.0
            data = (scale * np.random.random_sample(np_shape)).astype(np_dtype)
          else:
            data = np.random.choice(a=[False, True], size=np_shape)

          if run_params.is_v2:
            with ops.device("/GPU:0"):
              data = ops.convert_to_tensor(data)
          current_input_data.append(data)
        inputs_data.append(current_input_data)

      # Verify the original graph.
      self._VerifyGraphDef(run_params, saved_model_dir, saved_model_dir,
                           GraphState.ORIGINAL)

      # Run the original graph without TensorRT to get the reference result.
      logging.info("Running original graph w/o TensorRT\n")
      ref_result = self._RunGraph(
          run_params,
          saved_model_dir,
          inputs_data,
          GraphState.ORIGINAL,
          num_runs=1)

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

      # Run the inference graph, either using the converted graph or the
      # original graph with convert_online == True.
      logging.info("Running final inference graph\n")
      result = self._RunGraph(run_params, infer_saved_model_dir, inputs_data,
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
    # TODO(aaroey): implement this.
    pass


def _GetTestConfigsV1():
  """Returns the config combinations to run the test."""
  convert_online, convert_offline = True, False
  dynamic_engine, static_engine = True, False
  use_calibration, no_calibration = True, False
  implicit_batch = False

  # Add all possible test cases and let the derived test class to decide
  # whether to run specific ones with ShouldRunTest().
  #
  # Note: INT8 without calibration behaves like FP32/FP16.
  opts = list(
      itertools.product([FP32, FP16, INT8], [convert_online, convert_offline],
                        [dynamic_engine, static_engine], [no_calibration],
                        [implicit_batch]))
  # We always run calibration with offline tool.
  # TODO(aaroey): static calibration engine is not supported yet.
  opts.append(
      (INT8, convert_offline, dynamic_engine, use_calibration, implicit_batch))
  return opts


def _GetTestConfigsV2():
  """Returns the config combinations to run the test."""
  convert_offline = False
  # TODO(laigd): add support for static_engine.
  dynamic_engine = True
  # TODO(laigd): add support for calibration.
  no_calibration = False
  use_calibration = True

  # Add all possible test cases and let the derived test class to decide
  # whether to run specific ones with ShouldRunTest().
  #
  # Note:
  # - In TF2.0 the conversion always produce dynamic engine, and we don't test
  #   the offline mode here.
  # - For simplicity we don't test online conversion which requires setting the
  #   Grappler config in default eager context.
  # - INT8 without calibration behaves like FP32/FP16.
  opts = list(
      itertools.product([FP32, FP16], [convert_offline], [dynamic_engine],
                        [no_calibration], [False, True]))
  # We always run calibration with offline tool.
  opts.append((INT8, convert_offline, dynamic_engine, use_calibration, False))
  opts.append((INT8, convert_offline, dynamic_engine, use_calibration, True))
  return opts


def _GetTest(run_params):
  """Gets a single test method based on the parameters."""

  def _Test(self):
    logging.info(f"Running test `{run_params.test_name}` with parameters: "
                 f"convert_online={run_params.convert_online}, "
                 f"precision_mode={run_params.precision_mode}, "
                 f"dynamic_engine={run_params.dynamic_engine}, "
                 f"dynamic_shape={run_params.dynamic_shape}")
    self.RunTest(run_params)

  return _Test


def _AddTestsFor(test_class, is_v2):
  """Adds test methods to TfTrtIntegrationTestBase for specific TF version."""
  opts = _GetTestConfigsV2() if is_v2 else _GetTestConfigsV1()
  for (precision_mode, convert_online, dynamic_engine, use_calibration,
       dynamic_shape) in opts:
    conversion = "OnlineConversion" if convert_online else "OfflineConversion"
    engine_type = "DynamicEngine" if dynamic_engine else "StaticEngine"
    calibration_type = "UseCalibration" if use_calibration else "NoCalibration"
    dynamic_shape_type = "DynamicShape" if dynamic_shape else "ImplicitBatch"
    test_name = "%s_%s_%s_%s_%s_%s" % ("testTfTrtV2" if is_v2 else "testTfTrt",
                                       conversion, engine_type, precision_mode,
                                       calibration_type, dynamic_shape_type)
    run_params = RunParams(
        convert_online=convert_online,
        precision_mode=precision_mode,
        dynamic_engine=dynamic_engine,
        test_name=test_name,
        use_calibration=use_calibration,
        is_v2=is_v2,
        dynamic_shape=dynamic_shape)
    if is_v2:
      setattr(test_class, test_name,
              test_util.run_v2_only(_GetTest(run_params)))
    else:
      setattr(test_class, test_name,
              test_util.run_v1_only("", _GetTest(run_params)))


def _AddTests(test_class):
  """Adds test methods to TfTrtIntegrationTestBase."""
  _AddTestsFor(test_class, is_v2=False)
  _AddTestsFor(test_class, is_v2=True)


if is_tensorrt_enabled():
  os.environ["TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT_EXECUTION"] = "False"
  _AddTests(TfTrtIntegrationTestBase)
