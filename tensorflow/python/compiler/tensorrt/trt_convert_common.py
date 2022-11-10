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
# =============================================================================
"""Exposes the Python wrapper conversion to trt_graph."""

import collections
import platform
import sys

import six as _six

from tensorflow.core.framework import variable_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.compiler.tensorrt import utils as trt_utils
from tensorflow.python.compiler.tensorrt.lazy_utils import LazyObj
from tensorflow.python.compiler.tensorrt.types import TrtVersion
from tensorflow.python.eager import context
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import resource
from tensorflow.python.training import saver
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.lazy_loader import LazyLoader
from tensorflow.python.util.tf_export import tf_export

# Lazily load the op, since it's not available in cpu-only builds. Importing
# this at top will cause tests that imports TF-TRT fail when they're built
# and run without CUDA/GPU.
gen_trt_ops = LazyLoader(
    "gen_trt_ops", globals(),
    "tensorflow.compiler.tf2tensorrt.ops.gen_trt_ops")

_pywrap_py_utils = LazyLoader(
    "_pywrap_py_utils", globals(),
    "tensorflow.compiler.tf2tensorrt._pywrap_py_utils")

# Register TRT ops in python, so that when users import this module they can
# execute a TRT-converted graph without calling any of the methods in this
# module.
#
# This will call register_op_list() in
# tensorflow/python/framework/op_def_registry.py, but it doesn't register
# the op or the op kernel in C++ runtime.
try:
  gen_trt_ops.trt_engine_op  # pylint: disable=pointless-statement
except AttributeError:
  pass


def _to_bytes(s):
  """Encode s if it is a sequence of chars."""
  if isinstance(s, _six.text_type):
    return s.encode("utf-8", errors="surrogateescape")
  return s


def _to_string(s):
  """Decode s if it is a sequence of bytes."""
  if isinstance(s, _six.binary_type):
    return s.decode("utf-8")
  return s


class TrtPrecisionMode(object):
  FP32 = "FP32"
  FP16 = "FP16"
  INT8 = "INT8"

  @staticmethod
  def supported_precision_modes():
    precisions = [
        TrtPrecisionMode.FP32, TrtPrecisionMode.FP16, TrtPrecisionMode.INT8
    ]
    return precisions + [p.lower() for p in precisions]


def _get_default_max_trt_workspace_size():
  # Use a large enough number as the default max_workspace_size for TRT engines,
  # so it can produce reasonable performance results with the default.
  # For TRT >= 8.4, the recommendation is MAX_INT.
  if (
      _pywrap_py_utils.is_tensorrt_enabled() and
      trt_utils.is_loaded_tensorrt_version_greater_equal(8, 4, 0)
  ):
    # We must use `sys.maxsize - 512` to avoid overflow during casting.
    return sys.maxsize - 512
  else:
    return 1 << 30  # 1,073,741,824


DEFAULT_TRT_MAX_WORKSPACE_SIZE_BYTES = LazyObj(int)(
    _get_default_max_trt_workspace_size)


PROFILE_STRATEGY_RANGE = "Range"
PROFILE_STRATEGY_OPTIMAL = "Optimal"
PROFILE_STRATEGY_RANGE_OPTIMAL = "Range+Optimal"
PROFILE_STRATEGY_IMPLICIT_BATCH_MODE_COMPATIBLE = "ImplicitBatchModeCompatible"


def supported_profile_strategies():
  return [
      PROFILE_STRATEGY_RANGE, PROFILE_STRATEGY_OPTIMAL,
      PROFILE_STRATEGY_RANGE_OPTIMAL,
      PROFILE_STRATEGY_IMPLICIT_BATCH_MODE_COMPATIBLE
  ]


@tf_export("experimental.tensorrt.ConversionParams", v1=[])
class TrtConversionParams(
    collections.namedtuple("TrtConversionParams", [
        "max_workspace_size_bytes", "precision_mode", "minimum_segment_size",
        "maximum_cached_engines", "use_calibration", "allow_build_at_runtime"
    ])):
  """Parameters that are used for TF-TRT conversion.

  Fields:
    max_workspace_size_bytes: the maximum GPU temporary memory that the TRT
      engine can use at execution time. This corresponds to the
      'workspaceSize' parameter of nvinfer1::IBuilder::setMaxWorkspaceSize().
    precision_mode: one of the strings in
      TrtPrecisionMode.supported_precision_modes().
    minimum_segment_size: the minimum number of nodes required for a subgraph
      to be replaced by TRTEngineOp.
    maximum_cached_engines: max number of cached TRT engines for dynamic TRT
      ops. Created TRT engines for a dynamic dimension are cached. If the
      number of cached engines is already at max but none of them supports the
      input shapes, the TRTEngineOp will fall back to run the original TF
      subgraph that corresponds to the TRTEngineOp.
    use_calibration: this argument is ignored if precision_mode is not INT8.
      If set to True, a calibration graph will be created to calibrate the
      missing ranges. The calibration graph must be converted to an inference
      graph by running calibration with calibrate(). If set to False,
      quantization nodes will be expected for every tensor in the graph
      (excluding those which will be fused). If a range is missing, an error
      will occur. Please note that accuracy may be negatively affected if
      there is a mismatch between which tensors TRT quantizes and which
      tensors were trained with fake quantization.
    allow_build_at_runtime: whether to allow building TensorRT engines during
      runtime if no prebuilt TensorRT engine can be found that can handle the
      given inputs during runtime, then a new TensorRT engine is built at
      runtime if allow_build_at_runtime=True, and otherwise native TF is used.
  """

  def __new__(cls,
              max_workspace_size_bytes=DEFAULT_TRT_MAX_WORKSPACE_SIZE_BYTES,
              precision_mode=TrtPrecisionMode.FP32,
              minimum_segment_size=3,
              maximum_cached_engines=1,
              use_calibration=True,
              allow_build_at_runtime=True):
    return super(TrtConversionParams,
                 cls).__new__(cls, max_workspace_size_bytes, precision_mode,
                              minimum_segment_size, maximum_cached_engines,
                              use_calibration, allow_build_at_runtime)


DEFAULT_TRT_CONVERSION_PARAMS = TrtConversionParams()

_TRT_ENGINE_OP_NAME = "TRTEngineOp"


def _check_conversion_params(conversion_params, is_v2=False):
  """Validate the provided TrtConversionParams.

  Args:
    conversion_params: a TrtConversionParams instance.
    is_v2: whether we're getting a RewriterConfig for TF 2.0.

  Raises:
    TypeError: if any of the parameters are of unexpected type.
    ValueError: if any of the parameters are of unexpected value.
  """
  supported_precision_modes = TrtPrecisionMode.supported_precision_modes()
  if conversion_params.precision_mode not in supported_precision_modes:
    raise ValueError(
        ("precision mode '{}' is not supported."
         "It should be one of {}").format(conversion_params.precision_mode,
                                          supported_precision_modes))
  if (conversion_params.minimum_segment_size <= 0 and
      conversion_params.minimum_segment_size != -1):
    raise ValueError("minimum segment size should be positive or -1 "
                     "(to disable main graph conversion).")


def _check_trt_version_compatibility():
  """Check compatibility of TensorRT version.

  Raises:
    RuntimeError: if the TensorRT library version is incompatible.
  """

  if not _pywrap_py_utils.is_tensorrt_enabled():
    logging.error(
        "Tensorflow needs to be built with TensorRT support enabled to allow "
        "TF-TRT to operate.")

    raise RuntimeError("Tensorflow has not been built with TensorRT support.")

  if platform.system() == "Windows":
    logging.warn(
        "Windows support is provided experimentally. No guarantee is made "
        "regarding functionality or engineering support. Use at your own risk.")

  linked_version = TrtVersion(_pywrap_py_utils.get_linked_tensorrt_version())
  loaded_version = TrtVersion(_pywrap_py_utils.get_loaded_tensorrt_version())

  logging.info("Linked TensorRT version: %s", str(linked_version))
  logging.info("Loaded TensorRT version: %s", str(loaded_version))

  def raise_trt_version_deprecated(version_type, trt_version):
    assert version_type in ["linked", "loaded"], (
        f"Incorrect value received for version_type: {version_type}. "
        "Accepted: ['linked', 'loaded']"
    )

    logging.error(
        f"The {version_type} version of TensorRT: `{trt_version}` has now "
        "been removed. Please upgrade to TensorRT 7 or more recent.")

    raise RuntimeError("Incompatible %s TensorRT versions" % version_type)

  if not trt_utils.is_linked_tensorrt_version_greater_equal(7, 0, 0):
    raise_trt_version_deprecated("linked", linked_version)

  if not trt_utils.is_loaded_tensorrt_version_greater_equal(7, 0, 0):
    raise_trt_version_deprecated("loaded", loaded_version)

  if (loaded_version.major != linked_version.major or
      linked_version > loaded_version):
    logging.error(
        "Loaded TensorRT %s but linked TensorFlow against TensorRT %s. A few "
        "requirements must be met:\n"
        "\t-It is required to use the same major version of TensorRT during "
        "compilation and runtime.\n"
        "\t-TensorRT does not support forward compatibility. The loaded "
        "version has to be equal or more recent than the linked version.",
        loaded_version,
        linked_version)
    raise RuntimeError("Incompatible TensorRT major version")

  elif loaded_version != linked_version:
    logging.info(
        "Loaded TensorRT %s and linked TensorFlow against TensorRT %s. This is "
        "supported because TensorRT minor/patch upgrades are backward "
        "compatible.", loaded_version, linked_version)


def _get_tensorrt_rewriter_config(conversion_params,
                                  is_dynamic_op=None,
                                  max_batch_size=None,
                                  is_v2=False,
                                  disable_non_trt_optimizers=False,
                                  use_implicit_batch=True,
                                  profile_strategy=PROFILE_STRATEGY_RANGE):
  """Returns a RewriterConfig proto for TRT transformation.

  Args:
    conversion_params: a TrtConversionParams instance.
    is_dynamic_op: whether to use dynamic engines.
    max_batch_size: maximum batch size for static engines.
    is_v2: whether we're getting a RewriterConfig for TF 2.0.
    disable_non_trt_optimizers: Turn off all default Grappler optimizers.
    use_implicit_batch: Whether to use implicit batch or explicit batch.
    profile_strategy: dynamic shape optimization profile strategy.

  Returns:
    A RewriterConfig proto which sets a TensorRTOptimizer to run Grappler.

  Raises:
    TypeError: if any of the parameters are of unexpected type.
    ValueError: if any of the parameters are of unexpected value.
  """
  _check_conversion_params(conversion_params, is_v2=is_v2)
  if is_v2 and is_dynamic_op is not None and not is_dynamic_op:
    raise ValueError("is_dynamic_op is either None or True for TF2")
  if not is_v2 and is_dynamic_op is None:
    raise ValueError("is_dynamic_op can't be None for TF1")

  if (is_dynamic_op is None or is_dynamic_op) and max_batch_size is not None:
    raise ValueError("max_batch_size has to be None for TF2"
                     " or when is_dynamic_op == True in TF1")
  if is_dynamic_op is not None and not is_dynamic_op and not isinstance(
      max_batch_size, int):
    raise ValueError(
        "max_batch_size has to be an integer for is_dynamic_op==False in TF1")
  rewriter_config_with_trt = rewriter_config_pb2.RewriterConfig()
  # Disable Grappler Remapper to avoid that fused OPs that may not be
  # beneficial to TF-TRT and are not supported by TF-TRT.
  rewriter_config_with_trt.remapping = False

  # Prevent folding of Const->QDQ chains.
  rewriter_config_with_trt. \
    experimental_disable_folding_quantization_emulation = (
      trt_utils.is_linked_tensorrt_version_greater_equal(8, 0, 0) or
      trt_utils.is_loaded_tensorrt_version_greater_equal(8, 0, 0))

  if not disable_non_trt_optimizers:
    rewriter_config_with_trt.optimizers.extend([
        "pruning", "debug_stripper", "layout", "dependency", "constfold",
        "common_subgraph_elimination"
    ])

  rewriter_config_with_trt.meta_optimizer_iterations = (
      rewriter_config_pb2.RewriterConfig.ONE)
  optimizer = rewriter_config_with_trt.custom_optimizers.add()

  if not disable_non_trt_optimizers:
    # Add a constfold optimizer to cleanup the unused Const nodes.
    rewriter_config_with_trt.custom_optimizers.add().name = "constfold"

  optimizer.name = "TensorRTOptimizer"
  optimizer.parameter_map[
      "minimum_segment_size"].i = conversion_params.minimum_segment_size
  optimizer.parameter_map["max_workspace_size_bytes"].i = (
      conversion_params.max_workspace_size_bytes)
  optimizer.parameter_map["precision_mode"].s = _to_bytes(
      conversion_params.precision_mode)
  optimizer.parameter_map[
      "maximum_cached_engines"].i = conversion_params.maximum_cached_engines
  optimizer.parameter_map[
      "use_calibration"].b = conversion_params.use_calibration
  optimizer.parameter_map["is_dynamic_op"].b = is_dynamic_op
  optimizer.parameter_map[
      "allow_build_at_runtime"].b = conversion_params.allow_build_at_runtime
  if max_batch_size is not None:
    optimizer.parameter_map["max_batch_size"].i = max_batch_size
  optimizer.parameter_map["use_implicit_batch"].b = use_implicit_batch
  # While we accept case insensitive strings from the users, we only pass the
  # strings in lower cases to TF-TRT converter.
  if not use_implicit_batch:
    optimizer.parameter_map["profile_strategy"].s = _to_bytes(
        profile_strategy.lower())

  # Disabling optimizers should happen after defining the TF-TRT grappler pass
  # otherwise the template can overwrite the disablement.
  if disable_non_trt_optimizers:
    trt_utils.disable_non_trt_optimizers_in_rewriter_config(
        rewriter_config_with_trt)

  return rewriter_config_with_trt


@deprecation.deprecated(
    None, "You shouldn't need a rewriter_config with the current TF-TRT APIs.")
def get_tensorrt_rewriter_config(conversion_params,
                                 is_dynamic_op=None,
                                 max_batch_size=None,
                                 is_v2=False,
                                 disable_non_trt_optimizers=False):
  return _get_tensorrt_rewriter_config(conversion_params, is_dynamic_op,
                                       max_batch_size, is_v2,
                                       disable_non_trt_optimizers)


# Remove all scope prefixes in the node name. In TF 2.0, the same concrete
# function can be initialized multiple times with different prefixes, and
# this will result in the same TRTEngineOp being initialized multiple times
# with different cache and duplicate TRT engines.
# TODO(laigd): this may be caused by the fact that TRTEngineOp is not
# stateful, need to investigate.
# TODO(laigd): we rely on the fact that all functions are fully inlined
# before TF-TRT optimizer is called, as otherwise it may generate the same
# name when optimizing a different function graph. Fix this.
def _get_canonical_engine_name(name):
  return name.split("/")[-1]


def _get_resource_handle(name, device):
  with ops.device(device):
    return gen_trt_ops.create_trt_resource_handle(resource_name=name)


class _TRTEngineResource(resource.TrackableResource):
  """Class to track the serialized engines resource."""

  def __init__(self,
               resource_name,
               filename,
               maximum_cached_engines,
               device="GPU"):
    super(_TRTEngineResource, self).__init__(device=device)
    self._resource_name = resource_name
    # Track the serialized engine file in the SavedModel.
    self._filename = self._track_trackable(
        asset.Asset(filename), "_serialized_trt_resource_filename")
    self._maximum_cached_engines = maximum_cached_engines

  def _create_resource(self):
    return _get_resource_handle(self._resource_name, self._resource_device)

  def _initialize(self):
    gen_trt_ops.initialize_trt_resource(
        self.resource_handle,
        self._filename,
        max_cached_engines_count=self._maximum_cached_engines)

  def _destroy_resource(self):
    handle = _get_resource_handle(self._resource_name, self._resource_device)
    with ops.device(self._resource_device):
      gen_resource_variable_ops.destroy_resource_op(
          handle, ignore_lookup_error=True)


def _print_row(fields, positions, print_fn):
  """Prints a row."""
  line = ""
  for i, field in enumerate(fields):
    field = str(field)
    end_line_pos = positions[i]
    if i > 0:
      line = line + " "
    line = "{0:{min_length}}".format(line + field, min_length=end_line_pos)

    if len(line) > end_line_pos:
      line = line[:(end_line_pos - 4)] + " ..."

  print_fn(line)


def _construct_function_from_graph_def(func, graph_def, frozen_func=None):
  """Rebuild function from graph_def."""
  if frozen_func is None:
    frozen_func = func

  # If a function is converted, then the TF context contains the original
  # function while the converted_graph_def contains the converted function.
  # Remove the original function from the TF context in this case.
  for f in graph_def.library.function:
    while context.context().has_function(f.signature.name):
      context.context().remove_function(f.signature.name)

  # pylint: disable = protected-access
  captures = {
      t2.name.split(":")[0]: t1
      for _, (t1, t2) in frozen_func.graph._captures.items()
  }
  new_func = wrap_function.function_from_graph_def(
      graph_def, [tensor.name for tensor in frozen_func.inputs],
      [tensor.name for tensor in frozen_func.outputs], captures)
  new_func.graph.structured_outputs = nest.pack_sequence_as(
      func.graph.structured_outputs, new_func.graph.structured_outputs)

  # Copy structured input signature from original function (used during
  # serialization)
  new_func.graph.structured_input_signature = (func.structured_input_signature)

  return new_func


def _apply_inlining(func):
  """Apply an inlining optimization to the function's graph definition."""
  graph_def = func.graph.as_graph_def()

  # In some cases, a secondary implementation of the function (e.g. for GPU) is
  # written to the "api_implements" attribute. (e.g. `tf.keras.layers.LSTM` in
  # TF2 produces a CuDNN-based RNN for GPU).
  # This function suppose to inline all functions calls, but "api_implements"
  # prevents this from happening. Removing the attribute solves the problem.
  # To learn more about "api_implements", see:
  #   tensorflow/core/grappler/optimizers/implementation_selector.h
  for function in graph_def.library.function:
    if "api_implements" in function.attr:
      del function.attr["api_implements"]

  meta_graph = saver.export_meta_graph(graph_def=graph_def, graph=func.graph)

  # Clear the initializer_name for the variables collections, since they are not
  # needed after saved to saved_model.
  for name in [
      "variables", "model_variables", "trainable_variables", "local_variables"
  ]:
    raw_list = []
    for raw in meta_graph.collection_def["variables"].bytes_list.value:
      variable = variable_pb2.VariableDef()
      variable.ParseFromString(raw)
      variable.ClearField("initializer_name")
      raw_list.append(variable.SerializeToString())
    meta_graph.collection_def[name].bytes_list.value[:] = raw_list

  # Add a collection 'train_op' so that Grappler knows the outputs.
  fetch_collection = meta_graph_pb2.CollectionDef()
  for array in func.inputs + func.outputs:
    fetch_collection.node_list.value.append(array.name)
  meta_graph.collection_def["train_op"].CopyFrom(fetch_collection)

  # Initialize RewriterConfig with everything disabled except function inlining.
  config = config_pb2.ConfigProto()
  rewrite_options = config.graph_options.rewrite_options
  rewrite_options.min_graph_nodes = -1  # do not skip small graphs
  rewrite_options.optimizers.append("function")

  new_graph_def = tf_optimizer.OptimizeGraph(config, meta_graph)

  return new_graph_def


def _annotate_variable_ops(func, graph_def):
  """Annotates variable operations with custom `_shape` attribute.

  This is required for the converters and shape inference. The graph
  definition is modified in-place.

  Args:
    func: Function represented by the graph definition.
    graph_def: Graph definition to be annotated in-place.

  Raises:
    RuntimeError: if some shapes cannot be annotated.
  """
  ph_shape_map = {}
  for ph, var in zip(func.graph.internal_captures, func.variables):
    ph_shape_map[ph.name] = var.shape
  # Construct a mapping of node names to nodes
  name_to_node = {node.name: node for node in graph_def.node}
  # Go through all the ReadVariableOp nodes in the graph def
  for node in graph_def.node:
    if node.op == "ReadVariableOp" or node.op == "ResourceGather":
      node_ = node
      # Go up the chain of identities to find a placeholder
      while name_to_node[node_.input[0]].op == "Identity":
        node_ = name_to_node[node_.input[0]]
      ph_name = node_.input[0] + ":0"
      if ph_name in ph_shape_map:
        shape = ph_shape_map[ph_name]
        node.attr["_shape"].shape.CopyFrom(shape.as_proto())
      else:
        raise RuntimeError(
            "Not found in the function captures: {}".format(ph_name))


def _save_calibration_table(node):
  try:
    calibration_table = gen_trt_ops.get_calibration_data_op(
        _get_canonical_engine_name(node.name))
    node.attr["calibration_data"].s = calibration_table.numpy()
  except (errors.UnknownError, errors.NotFoundError):
    logging.warning("Warning calibration error for %s", node.name)


def _convert_to_tensor(inp):
  try:
    if isinstance(inp, dict):
      args = []
      kwargs = {k: ops.convert_to_tensor(v) for k, v in inp.items()}
    else:
      kwargs = {}
      if isinstance(inp, (list, tuple)):
        args = map(ops.convert_to_tensor, inp)
      else:
        args = [ops.convert_to_tensor(inp)]
  except:
    error_msg = "Failed to convert input to tensor."
    logging.error(error_msg + "\ninp = `{0}`\n".format(inp))
    raise RuntimeError(error_msg)

  return args, kwargs
