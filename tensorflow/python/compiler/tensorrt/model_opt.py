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
# =============================================================================

from tensorflow.core.framework import variable_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.compiler.tensorrt import trt_utils
from tensorflow.python.compiler.tensorrt.constants import TrtVersionEnv
from tensorflow.python.compiler.tensorrt.constants import TrtProfileStrategy
from tensorflow.python.compiler.tensorrt.types import TrtVersion
from tensorflow.python.eager import context
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.training import saver
from tensorflow.python.util import nest


def disable_non_trt_optimizers_in_rewriter_config(rewriter_config):
  """Modifies rewriter_config to disable all non-TRT optimizations."""
  off = rewriter_config_pb2.RewriterConfig.OFF

  rewriter_config.arithmetic_optimization = off
  rewriter_config.auto_mixed_precision = off
  rewriter_config.auto_parallel.enable = False
  rewriter_config.constant_folding = off
  rewriter_config.debug_stripper = off
  rewriter_config.dependency_optimization = off
  # This one needs to be ON to allow TF-TRT
  rewriter_config.disable_meta_optimizer = False
  rewriter_config.disable_model_pruning = True
  rewriter_config.function_optimization = off
  rewriter_config.implementation_selector = off
  rewriter_config.layout_optimizer = off
  rewriter_config.loop_optimization = off
  rewriter_config.memory_optimization = (
      rewriter_config_pb2.RewriterConfig.NO_MEM_OPT)
  rewriter_config.min_graph_nodes = -1
  rewriter_config.pin_to_host_optimization = off
  rewriter_config.remapping = off
  rewriter_config.scoped_allocator_optimization = off
  rewriter_config.shape_optimization = off


def apply_inlining(func):
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


def annotate_variable_ops(func, graph_def):
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
        raise RuntimeError(f"Not found in the function captures: {ph_name}")


def construct_function_from_graph_def(func, graph_def, frozen_func=None):
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


def prepare_func_for_inference(func, freeze):

  if freeze:
    frozen_func = convert_to_constants.convert_variables_to_constants_v2(func)
    frozen_graphdef = frozen_graph_def = frozen_func.graph.as_graph_def(
        add_shapes=True
    )
  else:
    inlined_graph_def = _apply_inlining(func)

  rewriter_config = rewriter_config_pb2.RewriterConfig()
  rewriter_config.optimizers.extend(["constfold"])
  # rewriter_config.optimizers.extend([
  #     "pruning", "debug_stripper", "layout", "dependency", "constfold",
  #     "common_subgraph_elimination"
  # ])

  grappler_conf = config_pb2.ConfigProto()
  grappler_conf.graph_options.rewrite_options.CopyFrom(rewriter_config)

  graph = ops.Graph()
  with graph.as_default():
    importer.import_graph_def(frozen_graphdef, name="")

    meta_graph = saver.export_meta_graph(
        graph_def=graph.as_graph_def(add_shapes=True),
        graph=graph)

    fetch_collection = meta_graph_pb2.CollectionDef()
    for array in frozen_fx.inputs + frozen_fx.outputs:
      fetch_collection.node_list.value.append(array.name)

    meta_graph.collection_def["train_op"].CopyFrom(fetch_collection)

    constfold_frozen_graphdef = tf_optimizer.OptimizeGraph(
        grappler_conf,
        meta_graph
    )

    return constfold_frozen_graphdef


def get_tensorrt_rewriter_config(conversion_params,
                                  is_dynamic_op=None,
                                  max_batch_size=None,
                                  is_v2=False,
                                  disable_non_trt_optimizers=False,
                                  use_implicit_batch=True,
                                  profile_strategy=TrtProfileStrategy.RANGE):
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

  if is_v2 and is_dynamic_op is not None and not is_dynamic_op:
    raise ValueError("is_dynamic_op is either None or True for TF2")

  if not is_v2 and is_dynamic_op is None:
    raise ValueError("is_dynamic_op can't be None for TF1")

  if (is_dynamic_op is None or is_dynamic_op) and max_batch_size is not None:
    raise ValueError("max_batch_size has to be None for TF2"
                     " or when is_dynamic_op == True in TF1")
  if (
      is_dynamic_op is not None and
      not is_dynamic_op and
      not isinstance(max_batch_size, int)
  ):
    raise ValueError(
        "max_batch_size has to be an integer for is_dynamic_op==False in TF1")

  rewriter_config_with_trt = rewriter_config_pb2.RewriterConfig()
  # Disable Grappler Remapper to avoid that fused OPs that may not be
  # beneficial to TF-TRT and are not supported by TF-TRT.
  rewriter_config_with_trt.remapping = False

  # Prevent folding of Const->QDQ chains.
  rewriter_config_with_trt. \
    experimental_disable_folding_quantization_emulation = \
        TrtVersionEnv() >= TrtVersion("8.0.0")

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
  optimizer.parameter_map["precision_mode"].s = trt_utils.cast_to_bytes(
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
    optimizer.parameter_map["profile_strategy"].s = trt_utils.cast_to_bytes(
        profile_strategy.lower())

  # Disabling optimizers should happen after defining the TF-TRT grappler pass
  # otherwise the template can overwrite the disablement.
  if disable_non_trt_optimizers:
    disable_non_trt_optimizers_in_rewriter_config(rewriter_config_with_trt)

  return rewriter_config_with_trt
