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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six as _six
# pylint: disable=unused-import,line-too-long
from tensorflow.contrib.tensorrt.wrap_conversion import add_test_value
from tensorflow.contrib.tensorrt.wrap_conversion import calib_convert
from tensorflow.contrib.tensorrt.wrap_conversion import clear_test_values
from tensorflow.contrib.tensorrt.wrap_conversion import enable_test_value
from tensorflow.contrib.tensorrt.wrap_conversion import get_linked_tensorrt_version
from tensorflow.contrib.tensorrt.wrap_conversion import get_loaded_tensorrt_version
from tensorflow.contrib.tensorrt.wrap_conversion import get_test_value
from tensorflow.contrib.tensorrt.wrap_conversion import is_tensorrt_enabled
# pylint: enable=unused-import,line-too-long
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import errors_impl as _impl
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.platform import tf_logging
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.training import saver

if _six.PY2:
  _to_bytes = lambda s: s
  _to_string = lambda s: s
else:
  _to_bytes = lambda s: s.encode("utf-8", errors="surrogateescape")
  _to_string = lambda s: s.decode("utf-8")


class TrtPrecisionMode(object):
  FP32 = "FP32"
  FP16 = "FP16"
  INT8 = "INT8"

  @staticmethod
  def supported_precision_modes():
    return [TrtPrecisionMode.FP32, TrtPrecisionMode.FP16, TrtPrecisionMode.INT8]


def get_tensorrt_rewriter_config(rewriter_config=None,
                                 max_batch_size=1,
                                 max_workspace_size_bytes=2 << 20,
                                 precision_mode=TrtPrecisionMode.FP32,
                                 minimum_segment_size=3,
                                 is_dynamic_op=False,
                                 maximum_cached_engines=1,
                                 cached_engine_batch_sizes=None):
  """Returns a RewriterConfig proto for TRT transformation.

  Args:
    rewriter_config: a template RewriterConfig proto used to create a
      TRT-enabled RewriterConfig. If None, it will use a default one.
    max_batch_size: max size for the input batch
    max_workspace_size_bytes: the maximum GPU temporary memory which the TRT
      engine can use at execution time. This corresponds to the 'workspaceSize'
      parameter of nvinfer1::IBuilder::setMaxWorkspaceSize().
    precision_mode: one of TrtPrecisionMode.supported_precision_modes().
    minimum_segment_size: the minimum number of nodes required for a subgraph to
      be replaced by TRTEngineOp.
    is_dynamic_op: whether to generate dynamic TRT ops which will build the TRT
      network and engine at run time.
    maximum_cached_engines: max number of cached TRT engines in dynamic TRT ops.
      If the number of cached engines is already at max but none of them can
      serve the input, the TRTEngineOp will fall back to run the TF function
      based on which the TRTEngineOp is created.
    cached_engine_batch_sizes: a list of batch sizes used to create cached
      engines, only used when is_dynamic_op is True. The length of the list
      should be smaller than maximum_cached_engines, and the dynamic TRT op will
      use this list to determine the batch sizes of the cached engines, instead
      of making the decision on the fly. This is useful when we know the most
      common batch size(s) the application is going to generate.

  Returns:
    A RewriterConfig proto which sets a TensorRTOptimizer to run Grappler.

  Raises:
    TypeError: if any of the parameters are of unexpected type.
    ValueError: if any of the parameters are of unexpected value.
  """
  if rewriter_config is not None and not isinstance(
      rewriter_config, rewriter_config_pb2.RewriterConfig):
    raise TypeError("rewriter_config should be a RewriterConfig proto.")

  rewriter_config_with_trt = rewriter_config_pb2.RewriterConfig()
  if rewriter_config is None:
    # Layout optimizer may add Const nodes followed by Reshape nodes, thus we
    # need to run constant folding again.
    rewriter_config_with_trt.optimizers.extend(
        ["constfold", "layout", "constfold"])
    rewriter_config_with_trt.meta_optimizer_iterations = (
        rewriter_config_pb2.RewriterConfig.ONE)
  else:
    rewriter_config_with_trt.CopyFrom(rewriter_config)

  if precision_mode.upper() not in TrtPrecisionMode.supported_precision_modes():
    raise ValueError(("precision mode '{}' is not supported."
                      "It should be one of {}").format(
                          precision_mode,
                          TrtPrecisionMode.supported_precision_modes))

  optimizer = rewriter_config_with_trt.custom_optimizers.add()
  optimizer.name = "TensorRTOptimizer"
  optimizer.parameter_map["minimum_segment_size"].i = minimum_segment_size
  optimizer.parameter_map["max_batch_size"].i = max_batch_size
  optimizer.parameter_map["is_dynamic_op"].b = is_dynamic_op
  optimizer.parameter_map[
      "max_workspace_size_bytes"].i = max_workspace_size_bytes
  optimizer.parameter_map["precision_mode"].s = _to_bytes(precision_mode)
  optimizer.parameter_map["maximum_cached_engines"].i = maximum_cached_engines
  if cached_engine_batch_sizes:
    if not isinstance(cached_engine_batch_sizes, list):
      raise TypeError("cached_engine_batch_sizes should be a list.")
    if len(cached_engine_batch_sizes) > maximum_cached_engines:
      raise ValueError("cached_engine_batch_sizes should not contain more than "
                       "maximum_cached_engines items.")
    optimizer.parameter_map["cached_engine_batches"].list.i.extend(
        cached_engine_batch_sizes)
  return rewriter_config_with_trt


def create_inference_graph(input_graph_def,
                           outputs,
                           max_batch_size=1,
                           max_workspace_size_bytes=2 << 20,
                           precision_mode=TrtPrecisionMode.FP32,
                           minimum_segment_size=3,
                           is_dynamic_op=False,
                           maximum_cached_engines=1,
                           cached_engine_batch_sizes=None,
                           input_saved_model_dir=None,
                           input_saved_model_tags=None,
                           output_saved_model_dir=None,
                           session_config=None):
  """Python wrapper for the TRT transformation.

  Args:
    input_graph_def: a GraphDef object containing a model to be transformed. If
      set to None, the graph will be read from the SavedModel loaded from
      input_saved_model_dir.
    outputs: list of tensors or node names for the model outputs. Only used when
      input_graph_def is not None.
    max_batch_size: max size for the input batch.
    max_workspace_size_bytes: the maximum GPU temporary memory which the TRT
      engine can use at execution time. This corresponds to the 'workspaceSize'
      parameter of nvinfer1::IBuilder::setMaxWorkspaceSize().
    precision_mode: one of TrtPrecisionMode.supported_precision_modes().
    minimum_segment_size: the minimum number of nodes required for a subgraph to
      be replaced by TRTEngineOp.
    is_dynamic_op: whether to generate dynamic TRT ops which will build the TRT
      network and engine at run time.
    maximum_cached_engines: max number of cached TRT engines in dynamic TRT ops.
      If the number of cached engines is already at max but none of them can
      serve the input, the TRTEngineOp will fall back to run the TF function
      based on which the TRTEngineOp is created.
    cached_engine_batch_sizes: a list of batch sizes used to create cached
      engines, only used when is_dynamic_op is True. The length of the list
      should be smaller than maximum_cached_engines, and the dynamic TRT op will
      use this list to determine the batch sizes of the cached engines, instead
      of making the decision on the fly. This is useful when we know the most
      common batch size(s) the application is going to generate.
    input_saved_model_dir: the directory to load the SavedModel which contains
      the input graph to transforms. Used only when input_graph_def is None.
    input_saved_model_tags: list of tags to load the SavedModel.
    output_saved_model_dir: if not None, construct a SavedModel using the
      returned GraphDef and save it to the specified directory. This option only
      works when the input graph is loaded from a SavedModel, i.e. when
      input_saved_model_dir is specified and input_graph_def is None.
    session_config: the ConfigProto used to create a Session. It's also used as
      a template to create a TRT-enabled ConfigProto for conversion. If not
      specified, a default ConfigProto will be used.

  Returns:
    A GraphDef transformed from input_graph_def (or the SavedModel graph def
    loaded from input_saved_model_dir, if input_graph_def is not present), where
    all TRT compatible subgraphs are replaced with TRTEngineOps, and a TF
    function is added for each of the subgraphs.

    If is_dynamic_op is True, each TRTEngineOp will contain a serialized
    subgraph GraphDef, which will be converted to a TRT engine at execution time
    and the TRT engine will be cached for future usage. A new TRT engine will be
    created each time when none of the cached engines match the input shapes. If
    it fails to execute the TRT engine or the number of cached engines reaches
    maximum_cached_engines, the op will fall back to call the corresponding TF
    function.

    If is_dynamic_op is False, each TRTEngineOp will contain a serialized TRT
    engine created from the corresponding subgraph. No more engines will be
    created on the fly, and the op will fall back to call the corresponding TF
    function when it fails to execute the engine.

  Raises:
    ValueError: if the combination of the parameters is invalid.
    RuntimeError: if the TensorRT library version is incompatible.
  """
  compiled_version = get_linked_tensorrt_version()
  loaded_version = get_loaded_tensorrt_version()
  version_mismatch = False
  if loaded_version[0] < compiled_version[0]:
    tf_logging.error(
        "TensorRT version mismatch. Tensorflow was compiled against " +
        "TensorRT %s but library loaded from environment is TensorRT %s" %
        (".".join([str(x) for x in compiled_version]),
         ".".join([str(x) for x in loaded_version])) +
        ". Please make sure that correct version of TensorRT " +
        "is available in the system and added to ldconfig or LD_LIBRARY_PATH")
    raise RuntimeError("Incompatible TensorRT library version")
  for i in zip(loaded_version, compiled_version):
    if i[0] != i[1]:
      tf_logging.warn("TensorRT mismatch. Compiled against version " +
                      "%s, but loaded %s. Things may not work" %
                      (".".join([str(x) for x in compiled_version]),
                       ".".join([str(x) for x in loaded_version])))
      version_mismatch = True
      break
  if not version_mismatch:
    tf_logging.info("Running against TensorRT version %s" % ".".join(
        [str(x) for x in loaded_version]))

  if session_config is None:
    session_config = config_pb2.ConfigProto()

  if input_saved_model_tags is None:
    input_saved_model_tags = [tag_constants.SERVING]
  saved_model_loader = None
  grappler_meta_graph_def = None

  if input_graph_def is None:
    # Read from SavedModel and freeze the graph if necessary.
    if input_saved_model_dir is None:
      raise ValueError("input_graph_def and input_saved_model_dir cannot be "
                       "both None")
    with ops.Graph().as_default():
      with session.Session(config=session_config) as sess:
        saved_model_loader = loader_impl.SavedModelLoader(input_saved_model_dir)
        input_meta_graph_def = saved_model_loader.load(sess,
                                                       input_saved_model_tags)
        output_node_names = set()

        def _gather_names(tensor_info):
          """Get the node names from a TensorInfo."""
          return set(
              [tensor_info[key].name.split(":")[0] for key in tensor_info])

        # Get input and outputs from all SignatureDef.
        for key in input_meta_graph_def.signature_def:
          signature_def = input_meta_graph_def.signature_def[key]
          output_node_names.update(_gather_names(signature_def.inputs))
          output_node_names.update(_gather_names(signature_def.outputs))

        # Freeze the variables in the SavedModel graph and copy the frozen
        # graph over.
        frozen_graph_def = graph_util.convert_variables_to_constants(
            sess, sess.graph.as_graph_def(add_shapes=True),
            list(output_node_names))
        grappler_meta_graph_def = meta_graph_pb2.MetaGraphDef()
        grappler_meta_graph_def.graph_def.CopyFrom(frozen_graph_def)

        # Copy the collections that are not variables.
        for key in input_meta_graph_def.collection_def:
          # TODO(laigd): currently we use the collection key to filter out
          # collections that depend on variable ops, but this may miss some
          # other user-defined collections. A better way would be to use
          # CollectionDef::NodeList for the filtering.
          if key not in [
              "variables", "local_variables", "model_variables",
              "trainable_variables", "train_op", "table_initializer"
          ]:
            grappler_meta_graph_def.collection_def[key].CopyFrom(
                input_meta_graph_def.collection_def[key])

        # Copy other information.
        grappler_meta_graph_def.meta_info_def.CopyFrom(
            input_meta_graph_def.meta_info_def)
        for key in input_meta_graph_def.signature_def:
          grappler_meta_graph_def.signature_def[key].CopyFrom(
              input_meta_graph_def.signature_def[key])
        # TODO(laigd): maybe add back AssetFileDef.
  else:
    if output_saved_model_dir is not None:
      raise ValueError("output_saved_model_dir cannot be set when "
                       "input_graph_def is set")
    # Create MetaGraphDef from input graph.
    graph = ops.Graph()
    with graph.as_default():
      importer.import_graph_def(input_graph_def, name="")
    grappler_meta_graph_def = saver.export_meta_graph(
        graph_def=graph.as_graph_def(add_shapes=True), graph=graph)
    if outputs:
      output_collection = meta_graph_pb2.CollectionDef()
      output_list = output_collection.node_list.value
      for i in outputs:
        if isinstance(i, ops.Tensor):
          output_list.append(_to_bytes(i.name))
        else:
          output_list.append(_to_bytes(i))
      # TODO(laigd): use another key as the outputs are really not train_op.
      grappler_meta_graph_def.collection_def["train_op"].CopyFrom(
          output_collection)

  # Create TRT-enabled ConfigProto.
  session_config_with_trt = config_pb2.ConfigProto()
  session_config_with_trt.CopyFrom(session_config)
  rewriter_config = None
  if (session_config_with_trt.HasField("graph_options") and
      session_config_with_trt.graph_options.HasField("rewrite_options")):
    rewriter_config = session_config_with_trt.graph_options.rewrite_options
  rewriter_config_with_trt = get_tensorrt_rewriter_config(
      rewriter_config, max_batch_size, max_workspace_size_bytes, precision_mode,
      minimum_segment_size, is_dynamic_op, maximum_cached_engines,
      cached_engine_batch_sizes)
  session_config_with_trt.graph_options.rewrite_options.CopyFrom(
      rewriter_config_with_trt)

  # Run Grappler.
  transformed_graph_def = tf_optimizer.OptimizeGraph(
      session_config_with_trt, grappler_meta_graph_def, graph_id=b"tf_graph")

  # Optionally write the transformed graphdef as SavedModel.
  if output_saved_model_dir is not None:
    saved_model_builder = builder.SavedModelBuilder(output_saved_model_dir)
    with ops.Graph().as_default():
      importer.import_graph_def(transformed_graph_def, name="")
      # We don't use TRT here.
      with session.Session(config=session_config) as sess:
        saved_model_builder.add_meta_graph_and_variables(
            sess,
            input_saved_model_tags,
            signature_def_map=grappler_meta_graph_def.signature_def)
    # Ignore other meta graphs from the input SavedModel.
    saved_model_builder.save()

  return transformed_graph_def


def calib_graph_to_infer_graph(calibration_graph_def, is_dynamic_op=False):
  """Convert an existing calibration graph to inference graph.

  Args:
    calibration_graph_def: the calibration GraphDef object with calibration data
    is_dynamic_op: whether to create dynamic static engines from calibration

  Returns:
    New GraphDef with TRTEngineOps placed in graph replacing calibration nodes.
  Raises:
    RuntimeError: if the returned status message is malformed.
  """

  is_calib_graph = False
  for n in calibration_graph_def.node:
    if n.op == "TRTEngineOp":
      is_calib_graph = is_calib_graph or not n.attr["calibration_data"].s
  if not is_calib_graph:
    tf_logging.error(
        "Not a calib graph. Doesn't seem to contain any calibration nodes.")
    return None
  graph_str = calibration_graph_def.SerializeToString()
  out = calib_convert(graph_str, is_dynamic_op)
  status = _to_string(out[0])
  output_graph_def_string = out[1]
  del graph_str  # Save some memory
  if len(status) < 2:
    raise _impl.UnknownError(None, None, status)
  if status[:2] != "OK":
    msg = status.split(";")
    if len(msg) == 1:
      raise RuntimeError("Status message is malformed {}".format(status))
    # pylint: disable=protected-access
    raise _impl._make_specific_exception(None, None, ";".join(msg[1:]),
                                         int(msg[0]))
    # pylint: enable=protected-access
  output_graph_def = graph_pb2.GraphDef()
  output_graph_def.ParseFromString(output_graph_def_string)
  del output_graph_def_string  # Save some memory
  return output_graph_def
