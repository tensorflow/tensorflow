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

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.compiler.tensorrt import utils as trt_utils
from tensorflow.python.compiler.tensorrt.trt_convert_common import DEFAULT_TRT_MAX_WORKSPACE_SIZE_BYTES
from tensorflow.python.compiler.tensorrt.trt_convert_common import TrtConversionParams
from tensorflow.python.compiler.tensorrt.trt_convert_common import TrtPrecisionMode
from tensorflow.python.compiler.tensorrt.trt_convert_common import _TRT_ENGINE_OP_NAME
from tensorflow.python.compiler.tensorrt.trt_convert_common import _check_conversion_params
from tensorflow.python.compiler.tensorrt.trt_convert_common import _check_trt_version_compatibility
from tensorflow.python.compiler.tensorrt.trt_convert_common import _get_canonical_engine_name
from tensorflow.python.compiler.tensorrt.trt_convert_common import _get_tensorrt_rewriter_config
from tensorflow.python.compiler.tensorrt.trt_convert_common import _to_bytes
from tensorflow.python.compiler.tensorrt.trt_convert_common import gen_trt_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.training import saver
from tensorflow.python.util import deprecation


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


class TrtGraphConverter(object):
  """A converter for TF-TRT transformation for TF 1.x GraphDef/SavedModels.

  To run the conversion without quantization calibration (e.g. for FP32/FP16
  precision modes):

  ```python
  converter = TrtGraphConverter(
      input_saved_model_dir="my_dir",
      precision_mode=TrtPrecisionMode.FP16)
  converted_graph_def = converter.convert()
  converter.save(output_saved_model_dir)
  ```

  To run the conversion with quantization calibration:

  ```python
  converter = TrtGraphConverter(
      input_saved_model_dir="my_dir",
      precision_mode=TrtPrecisionMode.INT8)
  converter.convert()

  # Run calibration 10 times.
  converted_graph_def = converter.calibrate(
      fetch_names=['output:0'],
      num_runs=10,
      feed_dict_fn=lambda: {'input:0': my_next_data()})

  converter.save(output_saved_model_dir)
  ```
  """

  def __init__(self,
               input_saved_model_dir=None,
               input_saved_model_tags=None,
               input_saved_model_signature_key=None,
               input_graph_def=None,
               nodes_denylist=None,
               max_batch_size=1,
               max_workspace_size_bytes=DEFAULT_TRT_MAX_WORKSPACE_SIZE_BYTES,
               precision_mode=TrtPrecisionMode.FP32,
               minimum_segment_size=3,
               is_dynamic_op=False,
               maximum_cached_engines=1,
               use_calibration=True):
    """Initializes the converter.

    Args:
      input_saved_model_dir: the directory to load the SavedModel which contains
        the input graph to transforms. Used only when input_graph_def is None.
      input_saved_model_tags: list of tags to load the SavedModel.
      input_saved_model_signature_key: the key of the signature to optimize the
        graph for.
      input_graph_def: a GraphDef object containing a model to be transformed.
        If set to None, the graph will be read from the SavedModel loaded from
        input_saved_model_dir.
      nodes_denylist: list of node names to prevent the converter from touching.
      max_batch_size: max size for the input batch.
      max_workspace_size_bytes: the maximum GPU temporary memory which the TRT
        engine can use at execution time. This corresponds to the
        'workspaceSize' parameter of nvinfer1::IBuilder::setMaxWorkspaceSize().
      precision_mode: one of TrtPrecisionMode.supported_precision_modes().
      minimum_segment_size: the minimum number of nodes required for a subgraph
        to be replaced by TRTEngineOp.
      is_dynamic_op: whether to generate dynamic TRT ops which will build the
        TRT network and engine at run time.
      maximum_cached_engines: max number of cached TRT engines in dynamic TRT
        ops. If the number of cached engines is already at max but none of them
        can serve the input, the TRTEngineOp will fall back to run the TF
        function based on which the TRTEngineOp is created.
      use_calibration: this argument is ignored if precision_mode is not INT8.
        If set to True, a calibration graph will be created to calibrate the
        missing ranges. The calibration graph must be converted to an inference
        graph by running calibration with calibrate(). If set to False,
        quantization nodes will be expected for every tensor in the graph
        (excluding those which will be fused). If a range is missing, an error
        will occur. Please note that accuracy may be negatively affected if
        there is a mismatch between which tensors TRT quantizes and which
        tensors were trained with fake quantization.

    Raises:
      ValueError: if the combination of the parameters is invalid.
      RuntimeError: if this class is used in TF 2.0.
    """
    if context.executing_eagerly():
      raise RuntimeError(
          "Please use tf.experimental.tensorrt.Converter in TF 2.0.")

    if input_graph_def and input_saved_model_dir:
      raise ValueError(
          "Can only specify one of input_graph_def and input_saved_model_dir")
    if not input_graph_def and not input_saved_model_dir:
      raise ValueError("Must specify one of input_graph_def and "
                       "input_saved_model_dir")
    _check_trt_version_compatibility()

    self._input_graph_def = input_graph_def
    self._nodes_denylist = nodes_denylist

    self._input_saved_model_dir = input_saved_model_dir
    self._converted = False
    self._grappler_meta_graph_def = None

    self._input_saved_model_tags = (
        input_saved_model_tags or [tag_constants.SERVING])
    self._input_saved_model_signature_key = (
        input_saved_model_signature_key or
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY)

    # For calibration usage.
    self._calibration_graph = None
    self._calibration_data_collected = False
    self._need_calibration = (
        ((precision_mode == TrtPrecisionMode.INT8) or
         (precision_mode == TrtPrecisionMode.INT8.lower())) and use_calibration)
    if self._need_calibration and not is_dynamic_op:
      logging.warn(
          "INT8 precision mode with calibration is supported with "
          "dynamic TRT ops only. Disregarding is_dynamic_op parameter.")
      is_dynamic_op = True

    self._is_dynamic_op = is_dynamic_op
    if is_dynamic_op:
      self._max_batch_size = None
      if max_batch_size is not None:
        logging.warn("When is_dynamic_op==True max_batch_size should be None")
    else:
      if not isinstance(max_batch_size, int):
        raise ValueError("When is_dynamic_op==False max_batch_size should be "
                         "an integer")
      self._max_batch_size = max_batch_size

    self._conversion_params = TrtConversionParams(
        max_workspace_size_bytes=max_workspace_size_bytes,
        precision_mode=precision_mode,
        minimum_segment_size=minimum_segment_size,
        maximum_cached_engines=maximum_cached_engines,
        use_calibration=use_calibration,
        allow_build_at_runtime=True)
    _check_conversion_params(self._conversion_params)

    self._test_only_disable_non_trt_optimizers = False

  def _run_conversion(self):
    """Run Grappler's OptimizeGraph() tool to convert the graph."""
    # Create custom ConfigProto for Grappler.
    grappler_session_config = config_pb2.ConfigProto()
    custom_rewriter_config = _get_tensorrt_rewriter_config(
        conversion_params=self._conversion_params,
        is_dynamic_op=self._is_dynamic_op,
        max_batch_size=self._max_batch_size,
        disable_non_trt_optimizers=self._test_only_disable_non_trt_optimizers,
        use_implicit_batch=True)
    grappler_session_config.graph_options.rewrite_options.CopyFrom(
        custom_rewriter_config)

    # Run Grappler.
    self._converted_graph_def = tf_optimizer.OptimizeGraph(
        grappler_session_config,
        self._grappler_meta_graph_def,
        graph_id=b"tf_graph")
    self._converted = True

  def _add_nodes_denylist(self):
    if self._nodes_denylist:
      collection_def = self._grappler_meta_graph_def.collection_def["train_op"]
      denylist = collection_def.node_list.value
      for i in self._nodes_denylist:
        if isinstance(i, ops.Tensor):
          denylist.append(_to_bytes(i.name))
        else:
          denylist.append(_to_bytes(i))

  def _convert_graph_def(self):
    """Convert the input GraphDef."""
    graph = ops.Graph()
    with graph.as_default():
      importer.import_graph_def(self._input_graph_def, name="")
    self._grappler_meta_graph_def = saver.export_meta_graph(
        graph_def=graph.as_graph_def(add_shapes=True), graph=graph)
    self._add_nodes_denylist()

    self._run_conversion()

  def _collections_to_keep(self, collection_keys):
    # TODO(laigd): currently we use the collection key to filter out
    # collections that depend on variable ops, but this may miss some
    # other user-defined collections. A better way would be to use
    # CollectionDef::NodeList for the filtering.
    collections_to_remove = (
        ops.GraphKeys._VARIABLE_COLLECTIONS + [
            ops.GraphKeys.TRAIN_OP, ops.GraphKeys.WHILE_CONTEXT,
            ops.GraphKeys.COND_CONTEXT
        ])
    return [key for key in collection_keys if key not in collections_to_remove]

  def _convert_saved_model(self):
    """Convert the input SavedModel."""
    graph = ops.Graph()
    with session.Session(graph=graph) as sess:
      input_meta_graph_def = loader.load(sess, self._input_saved_model_tags,
                                         self._input_saved_model_dir)
      input_signature_def = input_meta_graph_def.signature_def[
          self._input_saved_model_signature_key]

      def _gather_names(tensor_info):
        """Get the node names from a TensorInfo."""
        return {tensor_info[key].name.split(":")[0] for key in tensor_info}

      # Get input and outputs from all SignatureDef.
      output_node_names = _gather_names(input_signature_def.inputs).union(
          _gather_names(input_signature_def.outputs))

      # Preserve nodes in collection
      for collection_key in self._collections_to_keep(
          input_meta_graph_def.collection_def):
        for op in sess.graph.get_collection(collection_key):
          if isinstance(op, ops.Operation):
            output_node_names.add(op.name.split(":")[0])

      # Freeze the variables in the SavedModel graph and copy the frozen
      # graph over.
      frozen_graph_def = graph_util.convert_variables_to_constants(
          sess, sess.graph.as_graph_def(add_shapes=True),
          list(output_node_names))
      self._grappler_meta_graph_def = meta_graph_pb2.MetaGraphDef()
      self._grappler_meta_graph_def.graph_def.CopyFrom(frozen_graph_def)

      # Copy the collections that are not variables.
      for collection_key in self._collections_to_keep(
          input_meta_graph_def.collection_def):
        self._grappler_meta_graph_def.collection_def[collection_key].CopyFrom(
            input_meta_graph_def.collection_def[collection_key])

      self._add_nodes_denylist()

      # Copy other information.
      self._grappler_meta_graph_def.meta_info_def.CopyFrom(
          input_meta_graph_def.meta_info_def)
      self._grappler_meta_graph_def.signature_def[
          self._input_saved_model_signature_key].CopyFrom(input_signature_def)
      # TODO(laigd): maybe add back AssetFileDef.

    self._run_conversion()

  def convert(self):
    """Run the TF-TRT conversion.

    Returns:
      The converted GraphDef for TF 1.x.
    """
    assert not self._converted
    if self._input_graph_def:
      self._convert_graph_def()
    else:
      self._convert_saved_model()
    return self._converted_graph_def

  def calibrate(self,
                fetch_names,
                num_runs,
                feed_dict_fn=None,
                input_map_fn=None):
    """Run the calibration and return the calibrated GraphDef.

    Args:
      fetch_names: a list of output tensor name to fetch during calibration.
      num_runs: number of runs of the graph during calibration.
      feed_dict_fn: a function that returns a dictionary mapping input names (as
        strings) in the GraphDef to be calibrated to values (e.g. Python list,
        numpy arrays, etc). One and only one of `feed_dict_fn` and
        `input_map_fn` should be specified.
      input_map_fn: a function that returns a dictionary mapping input names (as
        strings) in the GraphDef to be calibrated to Tensor objects. The values
        of the named input tensors in the GraphDef to be calibrated will be
        re-mapped to the respective `Tensor` values during calibration. One and
        only one of `feed_dict_fn` and `input_map_fn` should be specified.

    Raises:
      ValueError: if the input combination is invalid.
      RuntimeError: if this method is called in eager mode.

    Returns:
      The GraphDef after the calibration.
    """
    assert self._converted
    assert self._need_calibration
    assert not self._calibration_data_collected

    if (feed_dict_fn and input_map_fn) or (not feed_dict_fn and
                                           not input_map_fn):
      raise ValueError(
          "Should specify one and only one of feed_dict_fn and input_map_fn.")

    if input_map_fn:
      for k, v in input_map_fn().items():
        if not isinstance(k, str):
          raise ValueError("Keys of input_map_fn must be of type str")
        if not isinstance(v, ops.Tensor):
          raise ValueError("Values of input_map_fn must be of type tf.Tensor")

    self._calibration_graph = ops.Graph()
    with self._calibration_graph.as_default():
      fetches = importer.import_graph_def(
          self._converted_graph_def,
          input_map=input_map_fn() if input_map_fn else None,
          return_elements=fetch_names,
          name="")

    calibrate_rewriter_cfg = rewriter_config_pb2.RewriterConfig()
    if self._test_only_disable_non_trt_optimizers:
      trt_utils.disable_non_trt_optimizers_in_rewriter_config(
          calibrate_rewriter_cfg)

    # Set allow_soft_placement=True to run the graph for calibration so that
    # OPs supported by TensorRT but don't have a GPU implementation are allowed
    # to execute on CPU.
    calibrate_config = config_pb2.ConfigProto(
        allow_soft_placement=True,
        graph_options=config_pb2.GraphOptions(
            rewrite_options=calibrate_rewriter_cfg))

    with session.Session(
        graph=self._calibration_graph,
        config=calibrate_config) as calibration_sess:
      for _ in range(num_runs):
        calibration_sess.run(
            fetches, feed_dict=feed_dict_fn() if feed_dict_fn else None)

      # Maps device name to the corresponding get_calibration_data.
      #
      # TODO(laigd): a better way would be to use calibration_sess to list
      # all the devices, add one get_calibration_data for each device, and
      # fetch each such op for every resource until its found. This can work
      # even when the device of the TRTEngineOp is empty or not fully specified.
      device_to_get_resource_op_map = {}

      with self._calibration_graph.as_default():
        resource_name_input = array_ops.placeholder(dtypes.string)

        for node in self._converted_graph_def.node:
          if node.op == _TRT_ENGINE_OP_NAME:
            # Adds the get_calibration_data op for the device if not done
            # before. We only add one such op for each device.
            # TODO(laigd): What if the device is empty?????
            if node.device not in device_to_get_resource_op_map:
              with self._calibration_graph.device(node.device):
                serialized_resources_output = (
                    gen_trt_ops.get_calibration_data_op(resource_name_input))
              device_to_get_resource_op_map[node.device] = (
                  serialized_resources_output)

            # Get the calibration resource.
            calibration_result = calibration_sess.run(
                device_to_get_resource_op_map[node.device],
                feed_dict={
                    resource_name_input: _get_canonical_engine_name(node.name)
                })
            node.attr["calibration_data"].s = calibration_result

      self._calibration_data_collected = True

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
    if self._need_calibration:
      assert self._calibration_data_collected
    if self._input_graph_def:
      raise ValueError(
          "Not able to save to a SavedModel since input is a GraphDef")

    def _restore_collections(dest_graph, src_meta_graph_def, collection_keys):
      """Restores collections that we need to keep."""
      scope = ""
      for key in collection_keys:
        collection_def = src_meta_graph_def.collection_def[key]
        kind = collection_def.WhichOneof("kind")
        if kind is None:
          logging.error(
              "Cannot identify data type for collection %s. Skipping.", key)
          continue
        from_proto = ops.get_from_proto_function(key)
        if from_proto and kind == "bytes_list":
          proto_type = ops.get_collection_proto_type(key)
          # It is assumed that there are no Variables Keys in collections
          for value in collection_def.bytes_list.value:
            proto = proto_type()
            proto.ParseFromString(value)
            try:
              new_value = from_proto(proto, import_scope=scope)
            except:
              continue
            dest_graph.add_to_collection(key, new_value)
        else:
          field = getattr(collection_def, kind)
          if kind == "node_list":
            for value in field.value:
              name = ops.prepend_name_scope(value, scope)
              # Since the graph has been optimized, the node may no longer
              # exists
              try:
                col_op = dest_graph.as_graph_element(name)
              except (TypeError, ValueError, KeyError):
                continue
              dest_graph.add_to_collection(key, col_op)
          elif kind == "int64_list":
            # NOTE(opensource): This force conversion is to work around the
            # fact that Python2 distinguishes between int and long, while
            # Python3 has only int.
            for value in field.value:
              dest_graph.add_to_collection(key, int(value))
          else:
            for value in field.value:
              dest_graph.add_to_collection(key,
                                           ops.prepend_name_scope(value, scope))

    # Write the transformed graphdef as SavedModel.
    saved_model_builder = builder.SavedModelBuilder(output_saved_model_dir)
    with ops.Graph().as_default():
      importer.import_graph_def(self._converted_graph_def, name="")
      _restore_collections(
          ops.get_default_graph(), self._grappler_meta_graph_def,
          self._collections_to_keep(
              self._grappler_meta_graph_def.collection_def))
      # We don't use any specific converter here.
      with session.Session() as sess:
        saved_model_builder.add_meta_graph_and_variables(
            sess,
            self._input_saved_model_tags,
            signature_def_map=self._grappler_meta_graph_def.signature_def)
    # Ignore other meta graphs from the input SavedModel.
    saved_model_builder.save()


# TODO(laigd): use TrtConversionParams here.
def create_inference_graph(
    input_graph_def,
    outputs,
    max_batch_size=1,
    max_workspace_size_bytes=DEFAULT_TRT_MAX_WORKSPACE_SIZE_BYTES,
    precision_mode=TrtPrecisionMode.FP32,
    minimum_segment_size=3,
    is_dynamic_op=False,
    maximum_cached_engines=1,
    input_saved_model_dir=None,
    input_saved_model_tags=None,
    input_saved_model_signature_key=None,
    output_saved_model_dir=None):
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
    input_saved_model_dir: the directory to load the SavedModel which contains
      the input graph to transforms. Used only when input_graph_def is None.
    input_saved_model_tags: list of tags to load the SavedModel.
    input_saved_model_signature_key: the key of the signature to optimize the
      graph for.
    output_saved_model_dir: if not None, construct a SavedModel using the
      returned GraphDef and save it to the specified directory. This option only
      works when the input graph is loaded from a SavedModel, i.e. when
      input_saved_model_dir is specified and input_graph_def is None.

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
  """
  trt_converter = TrtGraphConverter(
      input_saved_model_dir=input_saved_model_dir,
      input_saved_model_tags=input_saved_model_tags,
      input_saved_model_signature_key=input_saved_model_signature_key,
      input_graph_def=input_graph_def,
      nodes_denylist=outputs,
      max_batch_size=max_batch_size,
      max_workspace_size_bytes=max_workspace_size_bytes,
      precision_mode=precision_mode,
      minimum_segment_size=minimum_segment_size,
      is_dynamic_op=is_dynamic_op,
      maximum_cached_engines=maximum_cached_engines,
      use_calibration=False)
  converted_graph_def = trt_converter.convert()
  if output_saved_model_dir:
    trt_converter.save(output_saved_model_dir)
  return converted_graph_def
