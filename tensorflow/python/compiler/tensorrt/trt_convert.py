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
from tensorflow.compiler.tf2tensorrt.python.ops import trt_ops
from tensorflow.python.compiler.tensorrt.wrap_conversion import add_test_value
from tensorflow.python.compiler.tensorrt.wrap_conversion import calib_convert
from tensorflow.python.compiler.tensorrt.wrap_conversion import clear_test_values
from tensorflow.python.compiler.tensorrt.wrap_conversion import enable_test_value
from tensorflow.python.compiler.tensorrt.wrap_conversion import get_linked_tensorrt_version
from tensorflow.python.compiler.tensorrt.wrap_conversion import get_loaded_tensorrt_version
from tensorflow.python.compiler.tensorrt.wrap_conversion import get_test_value
from tensorflow.python.compiler.tensorrt.wrap_conversion import is_tensorrt_enabled
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
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.training import saver
import ctypes


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


class GraphConverter(object):
  """Base class for offline converters to optimize SavedModels/GraphDefs.

  A `GraphConverter` object encapsulates the environment to convert (optimize) a
  TensorFlow SavedModel or GraphDef.

  To create a custom GraphConverter:

  ```python
  class MyGraphConverter(GraphConverter):
    ...

    def get_rewriter_config(self, rewriter_config_template=None):
      my_rewriter_config = ...
      return my_rewriter_config
  ```

  Then to run the conversion without quantization calibration:

  ```python
  my_converter = MyGraphConverter(input_saved_model_dir="my_dir")
  converted_graph_def = my_converter.convert()
  my_converter.save(output_saved_model_dir)  # Optional
  ```

  TODO(laigd): add calibration support.
  """

  def __init__(self,
               input_saved_model_dir=None,
               input_saved_model_tags=None,
               input_graph_def=None,
               nodes_blacklist=None,
               session_config=None):
    """Initialize the converter.

    Args:
      input_saved_model_dir: the directory to load the SavedModel which contains
        the input graph to transforms. Used only when input_graph_def is None.
      input_saved_model_tags: list of tags to load the SavedModel.
      input_graph_def: a GraphDef object containing a model to be transformed.
        If set to None, the graph will be read from the SavedModel loaded from
        input_saved_model_dir.
      nodes_blacklist: list of node names to prevent the converter from
        touching. Only used when input_graph_def is not None.
      session_config: the ConfigProto used to create a Session. It's also used
        as a template to create a RewriterConfig for conversion. If not
        specified, a default ConfigProto will be used.

    Raises:
      ValueError: if the combination of the parameters is invalid.
    """
    if input_graph_def and input_saved_model_dir:
      raise ValueError(
          "Can only specify one of input_graph_def and input_saved_model_dir")
    if not input_graph_def and not input_saved_model_dir:
      raise ValueError("Must specify one of input_graph_def and "
                       "input_saved_model_dir")

    self._input_graph_def = input_graph_def
    self._nodes_blacklist = nodes_blacklist
    self._input_saved_model_dir = input_saved_model_dir
    self._converted = False
    self._grappler_meta_graph_def = None

    self._input_saved_model_tags = (
        input_saved_model_tags or [tag_constants.SERVING])
    self._session_config = session_config or config_pb2.ConfigProto()

  def get_rewriter_config(self, rewriter_config_template=None):
    """Returns a RewriterConfig proto for TRT transformation.

    Args:
      rewriter_config_template: a template RewriterConfig proto used to create a
        RewriterConfig for the conversion. The implementation should not modify
        the template. If None, it will use a default one.

    Returns:
      A RewriterConfig proto which will be used to run the conversion using
      Grappler.
    """
    raise NotImplementedError("get_rewriter_config")

  def _run_conversion(self):
    """Run Grappler's OptimizeGraph() tool to convert the graph."""
    # Create custom ConfigProto for Grappler.
    grappler_session_config = config_pb2.ConfigProto()
    grappler_session_config.CopyFrom(self._session_config)
    rewriter_config = None
    if (grappler_session_config.HasField("graph_options") and
        grappler_session_config.graph_options.HasField("rewrite_options")):
      rewriter_config = grappler_session_config.graph_options.rewrite_options
    custom_rewriter_config = self.get_rewriter_config(rewriter_config)
    grappler_session_config.graph_options.rewrite_options.CopyFrom(
        custom_rewriter_config)

    # Run Grappler.
    self._converted_graph_def = tf_optimizer.OptimizeGraph(
        grappler_session_config,
        self._grappler_meta_graph_def,
        graph_id=b"tf_graph")
    self._converted = True

  def _convert_graph_def(self):
    """Convert the input GraphDef."""
    graph = ops.Graph()
    with graph.as_default():
      importer.import_graph_def(self._input_graph_def, name="")
    self._grappler_meta_graph_def = saver.export_meta_graph(
        graph_def=graph.as_graph_def(add_shapes=True), graph=graph)
    if self._nodes_blacklist:
      output_collection = meta_graph_pb2.CollectionDef()
      output_list = output_collection.node_list.value
      for i in self._nodes_blacklist:
        if isinstance(i, ops.Tensor):
          output_list.append(_to_bytes(i.name))
        else:
          output_list.append(_to_bytes(i))
      # TODO(laigd): use another key as the self._nodes_blacklist are really
      # not train_op.
      self._grappler_meta_graph_def.collection_def["train_op"].CopyFrom(
          output_collection)

    self._run_conversion()

  def _convert_saved_model(self):
    """Convert the input SavedModel."""
    graph = ops.Graph()
    with session.Session(graph=graph, config=self._session_config) as sess:
      input_meta_graph_def = loader.load(sess, self._input_saved_model_tags,
                                         self._input_saved_model_dir)

      def _gather_names(tensor_info):
        """Get the node names from a TensorInfo."""
        return set([tensor_info[key].name.split(":")[0] for key in tensor_info])

      # Get input and outputs from all SignatureDef.
      output_node_names = set()
      for key in input_meta_graph_def.signature_def:
        signature_def = input_meta_graph_def.signature_def[key]
        output_node_names.update(_gather_names(signature_def.inputs))
        output_node_names.update(_gather_names(signature_def.outputs))

      # Freeze the variables in the SavedModel graph and copy the frozen
      # graph over.
      frozen_graph_def = graph_util.convert_variables_to_constants(
          sess, sess.graph.as_graph_def(add_shapes=True),
          list(output_node_names))
      self._grappler_meta_graph_def = meta_graph_pb2.MetaGraphDef()
      self._grappler_meta_graph_def.graph_def.CopyFrom(frozen_graph_def)

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
          self._grappler_meta_graph_def.collection_def[key].CopyFrom(
              input_meta_graph_def.collection_def[key])

      # Copy other information.
      self._grappler_meta_graph_def.meta_info_def.CopyFrom(
          input_meta_graph_def.meta_info_def)
      for key in input_meta_graph_def.signature_def:
        self._grappler_meta_graph_def.signature_def[key].CopyFrom(
            input_meta_graph_def.signature_def[key])
      # TODO(laigd): maybe add back AssetFileDef.

    self._run_conversion()

  def convert(self):
    """Run the conversion.

    Returns:
      The converted GraphDef.
    """
    assert not self._converted

    if self._input_graph_def:
      self._convert_graph_def()
    else:
      self._convert_saved_model()
    return self._converted_graph_def

  def save(self, output_saved_model_dir):
    """Save the converted graph as a SavedModel.

    Args:
      output_saved_model_dir: construct a SavedModel using the converted
        GraphDef and save it to the specified directory. This option only works
        when the input graph is loaded from a SavedModel, i.e. when
        input_saved_model_dir is specified and input_graph_def is None in
        __init__().

    Raises:
      ValueError: if the input to the converter is a GraphDef instead of a
      SavedModel.
    """
    assert self._converted

    if self._input_graph_def:
      raise ValueError(
          "Not able to save to a SavedModel since input is a GraphDef")

    # Write the transformed graphdef as SavedModel.
    saved_model_builder = builder.SavedModelBuilder(output_saved_model_dir)
    with ops.Graph().as_default():
      importer.import_graph_def(self._converted_graph_def, name="")
      # We don't use any specific converter here.
      with session.Session(config=self._session_config) as sess:
        saved_model_builder.add_meta_graph_and_variables(
            sess,
            self._input_saved_model_tags,
            signature_def_map=self._grappler_meta_graph_def.signature_def)
    # Ignore other meta graphs from the input SavedModel.
    saved_model_builder.save()


class TrtPrecisionMode(object):
  FP32 = "FP32"
  FP16 = "FP16"
  INT8 = "INT8"

  @staticmethod
  def supported_precision_modes():
    return [TrtPrecisionMode.FP32, TrtPrecisionMode.FP16, TrtPrecisionMode.INT8]


# Use a large enough number as the default max_workspace_size for TRT engines,
# so it can produce reasonable performance results with the default.
DEFAULT_TRT_MAX_WORKSPACE_SIZE_BYTES = 1 << 30


class TrtGraphConverter(GraphConverter):
  """A GraphConverter for TRT transformation."""

  _TRT_CALIBRATION_RESOURCE_CONTAINER_NAME = "TF_TRT_Calibration"

  @classmethod
  def get_tensorrt_rewriter_config(
      cls,
      rewriter_config_template=None,
      max_batch_size=1,
      max_workspace_size_bytes=DEFAULT_TRT_MAX_WORKSPACE_SIZE_BYTES,
      precision_mode=TrtPrecisionMode.FP32,
      minimum_segment_size=3,
      is_dynamic_op=False,
      maximum_cached_engines=1,
      cached_engine_batches=None,
      use_calibration=True):
    """Returns a RewriterConfig proto for TRT transformation.

    Args:
      rewriter_config_template: a template RewriterConfig proto used to create a
        TRT-enabled RewriterConfig. If None, it will use a default one.
      max_batch_size: max size for the input batch
      max_workspace_size_bytes: the maximum GPU temporary memory which the TRT
        engine can use at execution time. This corresponds to the
        'workspaceSize'
        parameter of nvinfer1::IBuilder::setMaxWorkspaceSize().
      precision_mode: one of TrtPrecisionMode.supported_precision_modes().
      minimum_segment_size: the minimum number of nodes required for a subgraph
        to be replaced by TRTEngineOp.
      is_dynamic_op: whether to generate dynamic TRT ops which will build the
        TRT network and engine at run time.
      maximum_cached_engines: max number of cached TRT engines in dynamic TRT
        ops. If the number of cached engines is already at max but none of them
        can serve the input, the TRTEngineOp will fall back to run the TF
        function based on which the TRTEngineOp is created.
      cached_engine_batches: a list of batch sizes used to create cached
        engines, only used when is_dynamic_op is True. The length of the list
        should be <= maximum_cached_engines, and the dynamic TRT op will use
        this list to determine the batch sizes of the cached engines, instead of
        making the decision on the fly. This is useful when we know the most
        common batch size(s) the application is going to generate.
      use_calibration: this argument is ignored if precision_mode is not INT8.
        If set to True, a calibration graph will be created to calibrate the
        missing ranges. The calibration graph must be converted to an inference
        graph using calib_graph_to_infer_graph() after running calibration. if
        set to False, quantization nodes will be expected for every tensor in
        the graph (exlcuding those which will be fused). If a range is missing,
        an error will occur. Please note that accuracy may be negatively
        affected if there is a mismatch between which tensors TRT quantizes and
        which tensors were trained with fake quantization.

    Returns:
      A RewriterConfig proto which sets a TensorRTOptimizer to run Grappler.

    Raises:
      TypeError: if any of the parameters are of unexpected type.
      ValueError: if any of the parameters are of unexpected value.
    """
    if rewriter_config_template is not None and not isinstance(
        rewriter_config_template, rewriter_config_pb2.RewriterConfig):
      raise TypeError(
          "rewriter_config_template should be a RewriterConfig proto.")

    rewriter_config_with_trt = rewriter_config_pb2.RewriterConfig()
    if rewriter_config_template is None:
      # Layout optimizer may add Const nodes followed by Reshape nodes, thus we
      # need to run constant folding again.
      rewriter_config_with_trt.optimizers.extend(
          ["constfold", "layout", "constfold"])
      rewriter_config_with_trt.meta_optimizer_iterations = (
          rewriter_config_pb2.RewriterConfig.ONE)
    else:
      rewriter_config_with_trt.CopyFrom(rewriter_config_template)

    optimizer = rewriter_config_with_trt.custom_optimizers.add()
    optimizer.name = "TensorRTOptimizer"
    optimizer.parameter_map["minimum_segment_size"].i = minimum_segment_size
    optimizer.parameter_map["max_batch_size"].i = max_batch_size
    optimizer.parameter_map["is_dynamic_op"].b = is_dynamic_op
    optimizer.parameter_map[
        "max_workspace_size_bytes"].i = max_workspace_size_bytes
    optimizer.parameter_map["precision_mode"].s = _to_bytes(precision_mode)
    optimizer.parameter_map["maximum_cached_engines"].i = maximum_cached_engines
    if cached_engine_batches:
      optimizer.parameter_map["cached_engine_batches"].list.i.extend(
          cached_engine_batches)
    optimizer.parameter_map["use_calibration"].b = use_calibration
    return rewriter_config_with_trt

  def __init__(self,
               input_saved_model_dir=None,
               input_saved_model_tags=None,
               input_graph_def=None,
               nodes_blacklist=None,
               session_config=None,
               max_batch_size=1,
               max_workspace_size_bytes=DEFAULT_TRT_MAX_WORKSPACE_SIZE_BYTES,
               precision_mode=TrtPrecisionMode.FP32,
               minimum_segment_size=3,
               is_dynamic_op=False,
               maximum_cached_engines=1,
               cached_engine_batches=None,
               use_calibration=True):
    """Initialize the converter.

    Args:
      input_saved_model_dir: the directory to load the SavedModel which contains
        the input graph to transforms. Used only when input_graph_def is None.
      input_saved_model_tags: list of tags to load the SavedModel.
      input_graph_def: a GraphDef object containing a model to be transformed.
        If set to None, the graph will be read from the SavedModel loaded from
        input_saved_model_dir.
      nodes_blacklist: list of node names to prevent the converter from
        touching. Only used when input_graph_def is not None.
      session_config: the ConfigProto used to create a Session. It's also used
        as a template to create a TRT-enabled ConfigProto for conversion. If not
        specified, a default ConfigProto will be used.
      max_batch_size: max size for the input batch.
      max_workspace_size_bytes: the maximum GPU temporary memory which the TRT
        engine can use at execution time. This corresponds to the
        'workspaceSize'
        parameter of nvinfer1::IBuilder::setMaxWorkspaceSize().
      precision_mode: one of TrtPrecisionMode.supported_precision_modes().
      minimum_segment_size: the minimum number of nodes required for a subgraph
        to be replaced by TRTEngineOp.
      is_dynamic_op: whether to generate dynamic TRT ops which will build the
        TRT network and engine at run time.
      maximum_cached_engines: max number of cached TRT engines in dynamic TRT
        ops. If the number of cached engines is already at max but none of them
        can serve the input, the TRTEngineOp will fall back to run the TF
        function based on which the TRTEngineOp is created.
      cached_engine_batches: a list of batch sizes used to create cached
        engines, only used when is_dynamic_op is True. The length of the list
        should be <= maximum_cached_engines, and the dynamic TRT op will use
        this list to determine the batch sizes of the cached engines, instead of
        making the decision on the fly. This is useful when we know the most
        common batch size(s) the application is going to generate.
      use_calibration: this argument is ignored if precision_mode is not INT8.
        If set to True, a calibration graph will be created to calibrate the
        missing ranges. The calibration graph must be converted to an inference
        graph using calib_graph_to_infer_graph() after running calibration. if
        set to False, quantization nodes will be expected for every tensor in
        the graph (exlcuding those which will be fused). If a range is missing,
        an error will occur. Please note that accuracy may be negatively
        affected if there is a mismatch between which tensors TRT quantizes and
        which tensors were trained with fake quantization.

    Raises:
      ValueError: if the combination of the parameters is invalid.
      RuntimeError: if the TensorRT library version is incompatible.
    """
    super(TrtGraphConverter, self).__init__(
        input_saved_model_dir=input_saved_model_dir,
        input_saved_model_tags=input_saved_model_tags,
        input_graph_def=input_graph_def,
        nodes_blacklist=nodes_blacklist,
        session_config=session_config)

    # Check compatibility of TensorRT version.
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

    # Load TensorRT plugins.
    try:
      plugin_lib = ctypes.CDLL("libnvinfer_plugin.so")
    except Exception as e:
      tf_logging.warn("Failed to load libnvinfer_plugin.so" +  str(e))
    else:
      # Initialize and register TensorRT plugins.
      plugin_lib_registered = plugin_lib.initLibNvInferPlugins(None, "")
      if plugin_lib_registered != 1:
        tf_logging.warn("Failed to initialize and register TensorRT plugins "
          "with initLibNvInferPlugins")

    # Check input arguments.
    if precision_mode.upper() not in TrtPrecisionMode.supported_precision_modes(
    ):
      raise ValueError(("precision mode '{}' is not supported."
                        "It should be one of {}").format(
                            precision_mode,
                            TrtPrecisionMode.supported_precision_modes))

    if cached_engine_batches:
      if not isinstance(cached_engine_batches, list):
        raise TypeError("cached_engine_batches should be a list.")
      if len(cached_engine_batches) > maximum_cached_engines:
        raise ValueError("cached_engine_batches should not contain more than "
                         "maximum_cached_engines items.")

    # TODO(laigd):
    # - Get rid of is_dynamic_op option, it should always be True, and it should
    #   accept N shapes as input.
    # - Verify in int8 mode that maximum_cached_engines and
    #   cached_engine_batches are set appropriately.
    # - If it fails to build the int8 engine it should return error.
    self._max_batch_size = max_batch_size
    self._max_workspace_size_bytes = max_workspace_size_bytes
    self._precision_mode = precision_mode
    self._minimum_segment_size = minimum_segment_size
    self._is_dynamic_op = is_dynamic_op
    self._maximum_cached_engines = maximum_cached_engines
    self._cached_engine_batches = cached_engine_batches
    self._use_calibration = use_calibration

  def get_rewriter_config(self, rewriter_config_template=None):
    return TrtGraphConverter.get_tensorrt_rewriter_config(
        rewriter_config_template,
        max_batch_size=self._max_batch_size,
        max_workspace_size_bytes=self._max_workspace_size_bytes,
        precision_mode=self._precision_mode,
        minimum_segment_size=self._minimum_segment_size,
        is_dynamic_op=self._is_dynamic_op,
        maximum_cached_engines=self._maximum_cached_engines,
        cached_engine_batches=self._cached_engine_batches,
        use_calibration=self._use_calibration)


def create_inference_graph(
    input_graph_def,
    outputs,
    max_batch_size=1,
    max_workspace_size_bytes=DEFAULT_TRT_MAX_WORKSPACE_SIZE_BYTES,
    precision_mode=TrtPrecisionMode.FP32,
    minimum_segment_size=3,
    is_dynamic_op=False,
    maximum_cached_engines=1,
    cached_engine_batches=None,
    use_calibration=True,
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
    cached_engine_batches: a list of batch sizes used to create cached engines,
      only used when is_dynamic_op is True. The length of the list should be <=
      maximum_cached_engines, and the dynamic TRT op will use this list to
      determine the batch sizes of the cached engines, instead of making the
      decision on the fly. This is useful when we know the most common batch
      size(s) the application is going to generate.
    use_calibration: this argument is ignored if precision_mode is not INT8. If
      set to True, a calibration graph will be created to calibrate the missing
      ranges. The calibration graph must be converted to an inference graph
      using calib_graph_to_infer_graph() after running calibration. if set to
      False, quantization nodes will be expected for every tensor in the graph
      (exlcuding those which will be fused). If a range is missing, an error
      will occur. Please note that accuracy may be negatively affected if there
      is a mismatch between which tensors TRT quantizes and which tensors were
      trained with fake quantization.
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
  trt_converter = TrtGraphConverter(
      input_saved_model_dir=input_saved_model_dir,
      input_saved_model_tags=input_saved_model_tags,
      input_graph_def=input_graph_def,
      nodes_blacklist=outputs,
      session_config=session_config,
      max_batch_size=max_batch_size,
      max_workspace_size_bytes=max_workspace_size_bytes,
      precision_mode=precision_mode,
      minimum_segment_size=minimum_segment_size,
      is_dynamic_op=is_dynamic_op,
      maximum_cached_engines=maximum_cached_engines,
      cached_engine_batches=cached_engine_batches,
      use_calibration=use_calibration)
  converted_graph_def = trt_converter.convert()
  if output_saved_model_dir:
    trt_converter.save(output_saved_model_dir)
  return converted_graph_def


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
