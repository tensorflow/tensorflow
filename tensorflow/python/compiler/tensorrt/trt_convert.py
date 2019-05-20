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

import collections
import os
import tempfile

import six as _six

from tensorflow.compiler.tf2tensorrt.python.ops import trt_ops
from tensorflow.compiler.tf2tensorrt.wrap_py_utils import get_linked_tensorrt_version
from tensorflow.compiler.tf2tensorrt.wrap_py_utils import get_loaded_tensorrt_version
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import tf_logging
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.training import saver
from tensorflow.python.training.tracking import tracking
from tensorflow.python.util.lazy_loader import LazyLoader

# Import TRT library. This is fine since we don't import TF-TRT in
# tensorflow/python/compiler/__init__.py, and `import tensorflow` won't trigger
# importing of TF-TRT. Note that TF-TRT is still included in GPU build since
# tensorflow/python/BUILD depends on it.
#
# We need this import so that when users import this module, they can execute a
# TRT-converted graph without calling any of the methods in this module.
trt_ops.load_trt_ops()

# Lazily load the op, since it's not available in cpu-only builds. Importing
# this at top will cause tests that imports TF-TRT fail when they're built
# and run without CUDA/GPU.
gen_trt_ops = LazyLoader(
    "gen_trt_ops", globals(),
    "tensorflow.compiler.tf2tensorrt.ops.gen_trt_ops")


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

    def get_rewriter_config(self):
      my_rewriter_config = ...
      return my_rewriter_config
  ```

  Then to run the conversion without quantization calibration:

  ```python
  my_converter = MyGraphConverter(input_saved_model_dir="my_dir")
  converted_graph_def = my_converter.convert()
  my_converter.save(output_saved_model_dir)  # Optional
  ```

  To run the conversion with quantization calibration:

  ```python
  my_converter = MyGraphConverter(input_saved_model_dir="my_dir")
  my_converter.convert()

  # Run calibration 10 times.
  converted_graph_def = my_converter.calibrate(
      fetch_names=['output:0'],
      num_runs=10,
      feed_dict_fn=lambda: {'input:0': my_next_data()})

  my_converter.save(output_saved_model_dir)  # Optional
  ```
  """

  # TODO(laigd): clean up the parameters.
  def __init__(self,
               input_saved_model_dir=None,
               input_saved_model_tags=None,
               input_saved_model_signature_key=None,
               input_graph_def=None,
               nodes_blacklist=None,
               session_config=None):
    """Initialize the converter.

    Args:
      input_saved_model_dir: the directory to load the SavedModel which contains
        the input graph to transforms. Used only when input_graph_def is None.
      input_saved_model_tags: list of tags to load the SavedModel.
      input_saved_model_signature_key: the key of the signature to optimize the
        graph for.
      input_graph_def: a GraphDef object containing a model to be transformed.
        If set to None, the graph will be read from the SavedModel loaded from
        input_saved_model_dir.
      nodes_blacklist: list of node names to prevent the converter from
        touching.
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
    self._input_saved_model_signature_key = (
        input_saved_model_signature_key or
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY)
    self._session_config = session_config or config_pb2.ConfigProto()

    # For calibration usage.
    self._calibration_graph = None
    self._calibration_sess = None
    self._calibration_data_collected = False

  def get_rewriter_config(self):
    """Returns a RewriterConfig proto for TRT transformation.

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
    custom_rewriter_config = self.get_rewriter_config()
    grappler_session_config.graph_options.rewrite_options.CopyFrom(
        custom_rewriter_config)

    # Run Grappler.
    self._converted_graph_def = tf_optimizer.OptimizeGraph(
        grappler_session_config,
        self._grappler_meta_graph_def,
        graph_id=b"tf_graph")
    self._converted = True

  def _add_nodes_blacklist(self):
    if self._nodes_blacklist:
      collection_def = self._grappler_meta_graph_def.collection_def["train_op"]
      blacklist = collection_def.node_list.value
      for i in self._nodes_blacklist:
        if isinstance(i, ops.Tensor):
          blacklist.append(_to_bytes(i.name))
        else:
          blacklist.append(_to_bytes(i))

  def _convert_graph_def(self):
    """Convert the input GraphDef."""
    graph = ops.Graph()
    with graph.as_default():
      importer.import_graph_def(self._input_graph_def, name="")
    self._grappler_meta_graph_def = saver.export_meta_graph(
        graph_def=graph.as_graph_def(add_shapes=True), graph=graph)
    self._add_nodes_blacklist()

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
    with session.Session(graph=graph, config=self._session_config) as sess:
      input_meta_graph_def = loader.load(sess, self._input_saved_model_tags,
                                         self._input_saved_model_dir)
      input_signature_def = input_meta_graph_def.signature_def[
          self._input_saved_model_signature_key]

      def _gather_names(tensor_info):
        """Get the node names from a TensorInfo."""
        return set([tensor_info[key].name.split(":")[0] for key in tensor_info])

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

      self._add_nodes_blacklist()

      # Copy other information.
      self._grappler_meta_graph_def.meta_info_def.CopyFrom(
          input_meta_graph_def.meta_info_def)
      self._grappler_meta_graph_def.signature_def[
          self._input_saved_model_signature_key].CopyFrom(input_signature_def)
      # TODO(laigd): maybe add back AssetFileDef.

    self._run_conversion()

  def convert(self):
    """Run the conversion.

    Returns:
      The converted GraphDef for TF 1.x, or the converted ConcreteFunction in TF
      2.0+.
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
    assert not self._calibration_sess

    if context.executing_eagerly():
      raise RuntimeError("Calibration for TF 2.0 is not supported yet.")

    if (feed_dict_fn and input_map_fn) or (not feed_dict_fn and
                                           not input_map_fn):
      raise ValueError(
          "Should specify one and only one of feed_dict_fn and input_map_fn.")

    self._calibration_graph = ops.Graph()
    with self._calibration_graph.as_default():
      fetches = importer.import_graph_def(
          self._converted_graph_def,
          input_map=input_map_fn() if input_map_fn else None,
          return_elements=fetch_names,
          name="")
    self._calibration_sess = session.Session(
        graph=self._calibration_graph, config=self._session_config)

    for _ in range(num_runs):
      self._calibration_sess.run(
          fetches, feed_dict=feed_dict_fn() if feed_dict_fn else None)

    self.finalize_calibration()
    return self._converted_graph_def

  def finalize_calibration(self):
    """Clean up calibration resources and finalize the calibration.

    Implementations need to close self._calibration_sess before returning.
    """
    raise NotImplementedError("finalize_calibration")

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

    def _restore_collections(dest_graph, src_meta_graph_def, collection_keys):
      """Restores collections that we need to keep."""
      scope = ""
      for key in collection_keys:
        collection_def = src_meta_graph_def.collection_def[key]
        kind = collection_def.WhichOneof("kind")
        if kind is None:
          tf_logging.error(
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
              except (TypeError, ValueError, KeyError) as e:
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

# TrtConversionParams encapsulates the parameters that are used for TF-TRT
# conversion.
TrtConversionParams = collections.namedtuple(
    "TrtConversionParams",
    [

        # A template RewriterConfig proto used to create a TRT-enabled
        # RewriterConfig. If None, it will use a default one.
        "rewriter_config_template",

        # The maximum GPU temporary memory which the TRT engine can use at
        # execution time. This corresponds to the 'workspaceSize' parameter of
        # nvinfer1::IBuilder::setMaxWorkspaceSize().
        "max_workspace_size_bytes",

        # One of TrtPrecisionMode.supported_precision_modes().
        "precision_mode",

        # The minimum number of nodes required for a subgraph to be replaced by
        # TRTEngineOp.
        "minimum_segment_size",

        # Whether to generate dynamic TRT ops which will build the TRT network
        # and engine at run time.
        #
        # TODO(laigd): In TF 2.0, this options should only affect INT8 mode.
        "is_dynamic_op",

        # Max number of cached TRT engines in dynamic TRT ops. If the number of
        # cached engines is already at max but none of them can serve the input,
        # the TRTEngineOp will fall back to run the TF function based on which
        # the TRTEngineOp is created.
        "maximum_cached_engines",

        # This argument is ignored if precision_mode is not INT8. If set to
        # True, a calibration graph will be created to calibrate the missing
        # ranges. The calibration graph must be converted to an inference graph
        # by running calibration with calibrate(). If set to False, quantization
        # nodes will be expected for every tensor in the graph (exlcuding those
        # which will be fused). If a range is missing, an error will occur.
        # Please note that accuracy may be negatively affected if there is a
        # mismatch between which tensors TRT quantizes and which tensors were
        # trained with fake quantization.
        "use_calibration",

        # If set to True, it will create a FunctionDef for each subgraph that is
        # converted to TRT op, and if TRT ops fail to execute at runtime, it'll
        # invoke that function as a fallback.
        "use_function_backup",

        # Max size for the input batch.
        # This option is deprecated in TF 2.0.
        "max_batch_size",

        # A list of batch sizes used to create cached engines, only used when
        # is_dynamic_op is True. The length of the list should be <=
        # maximum_cached_engines, and the dynamic TRT op will use this list to
        # determine the batch sizes of the cached engines, instead of making the
        # decision on the fly. This is useful when we know the most common batch
        # size(s) the application is going to generate.
        # This option is deprecated in TF 2.0.
        "cached_engine_batches",
    ])

DEFAULT_TRT_CONVERSION_PARAMS = TrtConversionParams(
    rewriter_config_template=None,
    max_workspace_size_bytes=DEFAULT_TRT_MAX_WORKSPACE_SIZE_BYTES,
    precision_mode=TrtPrecisionMode.FP32,
    minimum_segment_size=3,
    is_dynamic_op=False,
    maximum_cached_engines=1,
    use_calibration=True,
    use_function_backup=True,
    max_batch_size=1,
    cached_engine_batches=None)

_TRT_CALIBRATION_RESOURCE_CONTAINER_NAME = "TF-TRT-Calibration"
_TRT_ENGINE_CACHE_CONTAINER_NAME = "TF-TRT-Engine-Cache"
_TRT_ENGINE_OP_NAME = "TRTEngineOp"


def _check_conversion_params(conversion_params):
  """Validate the provided TrtConversionParams.

  Args:
    conversion_params: a TrtConversionParams instance.

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
  if conversion_params.cached_engine_batches:
    if not isinstance(conversion_params.cached_engine_batches, list):
      raise TypeError("cached_engine_batches should be a list.")
    if len(conversion_params.cached_engine_batches
          ) > conversion_params.maximum_cached_engines:
      raise ValueError("cached_engine_batches should not contain more than "
                       "maximum_cached_engines items.")


def _check_trt_version_compatibility():
  """Check compatibility of TensorRT version.

  Raises:
    RuntimeError: if the TensorRT library version is incompatible.
  """
  compiled_version = get_linked_tensorrt_version()
  loaded_version = get_loaded_tensorrt_version()
  tf_logging.info("Linked TensorRT version: %s" % str(compiled_version))
  tf_logging.info("Loaded TensorRT version: %s" % str(loaded_version))
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
    tf_logging.info("Running against TensorRT version %s" %
                    ".".join([str(x) for x in loaded_version]))


def get_tensorrt_rewriter_config(
    conversion_params=DEFAULT_TRT_CONVERSION_PARAMS, is_v2=False):
  """Returns a RewriterConfig proto for TRT transformation.

  Args:
    conversion_params: a TrtConversionParams instance.
    is_v2: whether we're getting a RewriterConfig for TF 2.0.

  Returns:
    A RewriterConfig proto which sets a TensorRTOptimizer to run Grappler.

  Raises:
    TypeError: if any of the parameters are of unexpected type.
    ValueError: if any of the parameters are of unexpected value.
  """
  if conversion_params.rewriter_config_template is not None and not isinstance(
      conversion_params.rewriter_config_template,
      rewriter_config_pb2.RewriterConfig):
    raise TypeError(
        "rewriter_config_template should be a RewriterConfig proto.")
  _check_conversion_params(conversion_params)

  rewriter_config_with_trt = rewriter_config_pb2.RewriterConfig()
  if conversion_params.rewriter_config_template is None:
    # Layout optimizer may add Const nodes followed by Reshape nodes, thus we
    # need to run constant folding again.
    rewriter_config_with_trt.optimizers.extend(
        ["constfold", "layout", "constfold"])
    rewriter_config_with_trt.meta_optimizer_iterations = (
        rewriter_config_pb2.RewriterConfig.ONE)
  else:
    rewriter_config_with_trt.CopyFrom(
        conversion_params.rewriter_config_template)

  optimizer = rewriter_config_with_trt.custom_optimizers.add()
  optimizer.name = "TensorRTOptimizer"
  optimizer.parameter_map[
      "minimum_segment_size"].i = conversion_params.minimum_segment_size
  optimizer.parameter_map[
      "max_workspace_size_bytes"].i = conversion_params.max_workspace_size_bytes
  optimizer.parameter_map["precision_mode"].s = _to_bytes(
      conversion_params.precision_mode)
  optimizer.parameter_map[
      "maximum_cached_engines"].i = conversion_params.maximum_cached_engines
  optimizer.parameter_map[
      "use_calibration"].b = conversion_params.use_calibration
  optimizer.parameter_map[
      "use_function_backup"].b = conversion_params.use_function_backup

  if is_v2:
    # Static mode (a.k.a pre-generating TRT engines and make them node
    # attributes) is deprecated in TF 2.0.
    optimizer.parameter_map["is_dynamic_op"].b = True
  else:
    optimizer.parameter_map[
        "max_batch_size"].i = conversion_params.max_batch_size
    optimizer.parameter_map["is_dynamic_op"].b = conversion_params.is_dynamic_op
    if conversion_params.cached_engine_batches:
      optimizer.parameter_map["cached_engine_batches"].list.i.extend(
          conversion_params.cached_engine_batches)
  return rewriter_config_with_trt


class TrtGraphConverter(GraphConverter):
  """A GraphConverter for TRT transformation."""

  # TODO(laigd): use TrtConversionParams here.
  def __init__(self,
               input_saved_model_dir=None,
               input_saved_model_tags=None,
               input_saved_model_signature_key=None,
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
               use_calibration=True,
               use_function_backup=True):
    """Initialize the converter.

    Args:
      input_saved_model_dir: the directory to load the SavedModel which contains
        the input graph to transforms. Used only when input_graph_def is None.
      input_saved_model_tags: list of tags to load the SavedModel.
      input_saved_model_signature_key: the key of the signature to optimize the
        graph for.
      input_graph_def: a GraphDef object containing a model to be transformed.
        If set to None, the graph will be read from the SavedModel loaded from
        input_saved_model_dir.
      nodes_blacklist: list of node names to prevent the converter from
        touching.
      session_config: the ConfigProto used to create a Session. It's also used
        as a template to create a TRT-enabled ConfigProto for conversion. If not
        specified, a default ConfigProto will be used.
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
      cached_engine_batches: a list of batch sizes used to create cached
        engines, only used when is_dynamic_op is True. The length of the list
        should be <= maximum_cached_engines, and the dynamic TRT op will use
        this list to determine the batch sizes of the cached engines, instead of
        making the decision on the fly. This is useful when we know the most
        common batch size(s) the application is going to generate.
      use_calibration: this argument is ignored if precision_mode is not INT8.
        If set to True, a calibration graph will be created to calibrate the
        missing ranges. The calibration graph must be converted to an inference
        graph by running calibration with calibrate(). If set to False,
        quantization nodes will be expected for every tensor in the graph
        (exlcuding those which will be fused). If a range is missing, an error
        will occur. Please note that accuracy may be negatively affected if
        there is a mismatch between which tensors TRT quantizes and which
        tensors were trained with fake quantization.
      use_function_backup: if set to True, it will create a FunctionDef for each
        subgraph that is converted to TRT op, and if TRT ops fail to execute at
        runtime, it'll invoke that function as a fallback.

    Raises:
      ValueError: if the combination of the parameters is invalid.
    """
    super(TrtGraphConverter, self).__init__(
        input_saved_model_dir=input_saved_model_dir,
        input_saved_model_tags=input_saved_model_tags,
        input_saved_model_signature_key=input_saved_model_signature_key,
        input_graph_def=input_graph_def,
        nodes_blacklist=nodes_blacklist,
        session_config=session_config)
    _check_trt_version_compatibility()

    self._need_calibration = (
        precision_mode == TrtPrecisionMode.INT8 and use_calibration)

    # TODO(laigd): consider provide a mechanism to remove the fallback path
    # after calibration is done.
    if self._need_calibration and not use_function_backup:
      raise ValueError(
          "Calibration requires enabling fallback to TF function execution.")

    # TODO(laigd):
    # - Verify in int8 mode that maximum_cached_engines and
    #   cached_engine_batches are set appropriately.
    # - If it fails to build the int8 engine it should return error.
    rewriter_config_template = None
    if (session_config and session_config.HasField("graph_options") and
        session_config.graph_options.HasField("rewrite_options")):
      rewriter_config_template = session_config.graph_options.rewrite_options

    self._conversion_params = TrtConversionParams(
        rewriter_config_template=rewriter_config_template,
        max_workspace_size_bytes=max_workspace_size_bytes,
        precision_mode=precision_mode,
        minimum_segment_size=minimum_segment_size,
        is_dynamic_op=is_dynamic_op,
        maximum_cached_engines=maximum_cached_engines,
        use_calibration=use_calibration,
        use_function_backup=use_function_backup,
        max_batch_size=max_batch_size,
        cached_engine_batches=cached_engine_batches)
    _check_conversion_params(self._conversion_params)

  def get_rewriter_config(self):
    return get_tensorrt_rewriter_config(
        conversion_params=self._conversion_params)

  def finalize_calibration(self):
    assert self._need_calibration
    assert self._converted
    assert not self._calibration_data_collected

    # TODO(laigd): a better way would be to use self._calibration_sess to list
    # all the devices, add one get_serialized_resource_op for each device, and
    # fetch each such op for every resource until its found. This can work
    # even when the device of the TRTEngineOp is empty or not fully specified.

    # Maps device name to the corresponding get_serialized_resource_op.
    device_to_get_resource_op_map = {}

    with self._calibration_graph.as_default():
      container_input = array_ops.placeholder(dtypes.string)
      resource_name_input = array_ops.placeholder(dtypes.string)

      for node in self._converted_graph_def.node:
        if node.op == _TRT_ENGINE_OP_NAME:
          # Adds the get_serialized_resource_op for the device if not done
          # before. We only add one such op for each device.
          # TODO(laigd): What if the device is empty?????
          if node.device not in device_to_get_resource_op_map:
            with self._calibration_graph.device(node.device):
              serialized_resources_output = (
                  gen_trt_ops.get_serialized_resource_op(
                      container_input, resource_name_input))
            device_to_get_resource_op_map[node.device] = (
                serialized_resources_output)

          # Get the calibration resource.
          calibration_result = self._calibration_sess.run(
              device_to_get_resource_op_map[node.device],
              feed_dict={
                  container_input: _TRT_CALIBRATION_RESOURCE_CONTAINER_NAME,
                  resource_name_input: node.name
              })
          node.attr["calibration_data"].s = calibration_result

    self._calibration_data_collected = True
    self._calibration_sess.close()

  def save(self, output_saved_model_dir):
    """Save the converted graph as a SavedModel."""
    if self._need_calibration:
      assert self._calibration_data_collected

    super(TrtGraphConverter, self).save(output_saved_model_dir)


class TRTEngineResource(tracking.TrackableResource):
  """Class to track the serialized engines resource."""

  def __init__(self, resource_name, filename, maximum_cached_engines):
    super(TRTEngineResource, self).__init__()
    self._resource_name = resource_name
    # Track the serialized engine file in the SavedModel.
    self._filename = self._track_trackable(
        tracking.TrackableAsset(filename), "_serialized_trt_engine_filename")
    self._maximum_cached_engines = maximum_cached_engines

  def _create_resource(self):
    return gen_trt_ops.create_trt_engine_cache(
        container=_TRT_ENGINE_CACHE_CONTAINER_NAME,
        resource_name=self._resource_name,
        max_cached_engines_count=self._maximum_cached_engines)

  def _initialize(self):
    gen_trt_ops.populate_trt_engine_cache(self.resource_handle, self._filename)


class TrtGraphConverterV2(object):
  """An offline converter for TF-TRT transformation for TF 2.0 SavedModels.

  To run the conversion without quantization calibration (e.g. for FP32/FP16
  precision modes):

  ```python
  params = DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode='FP16')
  converter = TrtGraphConverterV2(
      input_saved_model_dir="my_dir", conversion_params=params)
  converter.convert()
  converter.save(output_saved_model_dir)
  ```

  As a result, a TF-TRT converted SavedModel will be generated and saved to
  `output_saved_model_dir`. The SavedModel will have TRT compatible subgraph
  replaced by TRTEngineOps, but no TRT engines will be pre-built until execution
  time. We can also build the TRT engines offline by running the converted
  function with some input data:

  ```python
  params = DEFAULT_TRT_CONVERSION_PARAMS._replace(
      precision_mode='FP16',
      # Set this to a large enough number so it can cache all the TRT engines.
      maximum_cached_engines=16)
  converter = TrtGraphConverterV2(
      input_saved_model_dir="my_dir", conversion_params=params)
  converted_func = converter.convert()
  for data in my_input_data:
    converted_func(my_input_data)
  converter.save(output_saved_model_dir)
  ```

  In this way, for each unique shapes of the inputs to the TRTEngineOp, if it
  cannot be handled by any previously generated TRT engine, a new engine will be
  generated and serialized to the output SavedModel in `output_saved_model_dir`.
  This is good for applications that cannot afford building TRT engines at
  runtime but have access to input data that is similar to the one used in
  production (for example, that will result in the same input shapes to the
  TRTEngineOps). Also, the generated TRT engines is platform dependent, so we
  need to run `converted_func` in an environment that is similar to production
  (at least with same type of GPU).

  TODO(laigd/hinsu): running conversion with calibration in INT8 mode should
  follow exactly the same steps.
  """

  def __init__(self,
               input_saved_model_dir=None,
               input_saved_model_tags=None,
               input_saved_model_signature_key=None,
               conversion_params=DEFAULT_TRT_CONVERSION_PARAMS):
    """Initialize the converter.

    Args:
      input_saved_model_dir: the directory to load the SavedModel which contains
        the input graph to transforms. Used only when input_graph_def is None.
      input_saved_model_tags: list of tags to load the SavedModel.
      input_saved_model_signature_key: the key of the signature to optimize the
        graph for.
      conversion_params: a TrtConversionParams instance.
    """
    assert context.executing_eagerly()
    _check_trt_version_compatibility()

    self._input_saved_model_dir = input_saved_model_dir
    self._input_saved_model_tags = (
        input_saved_model_tags or [tag_constants.SERVING])
    self._input_saved_model_signature_key = (
        input_saved_model_signature_key or
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY)

    self._need_calibration = (
        conversion_params.precision_mode == TrtPrecisionMode.INT8 and
        conversion_params.use_calibration)
    self._conversion_params = conversion_params
    _check_conversion_params(self._conversion_params)
    self._converted = False

  def _run_conversion(self, meta_graph_def):
    """Run Grappler's OptimizeGraph() tool to convert the graph.

    Args:
      meta_graph_def: the MetaGraphDef instance to run the optimizations on.

    Returns:
      The optimized GraphDef.
    """
    rewriter_config = get_tensorrt_rewriter_config(
        conversion_params=self._conversion_params, is_v2=True)
    grappler_session_config = config_pb2.ConfigProto()
    grappler_session_config.graph_options.rewrite_options.CopyFrom(
        rewriter_config)
    return tf_optimizer.OptimizeGraph(
        grappler_session_config, meta_graph_def, graph_id=b"tf_graph")

  # TODO(laigd): provide a utility function to optimize a ConcreteFunction and
  # use it here (b/124792963).
  def convert(self):
    """Convert the input SavedModel in 2.0 format.

    Returns:
      The TF-TRT converted Function.
    """
    assert not self._converted
    self._saved_model = load.load(self._input_saved_model_dir,
                                  self._input_saved_model_tags)
    func = self._saved_model.signatures[self._input_saved_model_signature_key]
    frozen_func = convert_to_constants.convert_variables_to_constants_v2(func)
    grappler_meta_graph_def = saver.export_meta_graph(
        graph_def=frozen_func.graph.as_graph_def(), graph=frozen_func.graph)

    # Add a collection 'train_op' so that Grappler knows the outputs.
    fetch_collection = meta_graph_pb2.CollectionDef()
    for array in frozen_func.inputs + frozen_func.outputs:
      fetch_collection.node_list.value.append(array.name)
    grappler_meta_graph_def.collection_def["train_op"].CopyFrom(
        fetch_collection)

    # Run TRT optimizer in Grappler to convert the graph.
    self._converted_graph_def = self._run_conversion(grappler_meta_graph_def)
    self._converted_func = wrap_function.function_from_graph_def(
        self._converted_graph_def,
        [tensor.name for tensor in frozen_func.inputs],
        [tensor.name for tensor in frozen_func.outputs])

    self._converted = True

    # Wrap the converted ConcreteFunction in a Function so it can accept numpy
    # arrays as input.
    @def_function.function
    def wrapper_func(*args, **kwargs):
      return self._converted_func(*args, **kwargs)

    return wrapper_func

  def save(self, output_saved_model_dir):
    """Save the converted SavedModel.

    Args:
      output_saved_model_dir: directory to saved the converted SavedModel.
    """
    assert self._converted

    @def_function.function
    def _dump_trt_cache(resource_name, filename):
      gen_trt_ops.dump_trt_engine_cache(
          container=_TRT_ENGINE_CACHE_CONTAINER_NAME,
          resource_name=resource_name,
          filename=filename,
          delete_cache_after_dump=True)

    # Serialize the TRT engines in the cache if any, and create trackable
    # resource to track them.
    engine_asset_dir = tempfile.mkdtemp()
    resource_map = {}

    def _serialize_and_track_engine(canonical_engine_name):
      """Serialize TRT engines in the cache and track them."""
      # Don't dump the same cache twice.
      if canonical_engine_name in resource_map:
        return

      filename = os.path.join(engine_asset_dir,
                              "trt-serialized-engine." + canonical_engine_name)
      try:
        _dump_trt_cache(canonical_engine_name, filename)
      except errors.NotFoundError:
        # If user haven't run the function to populate the engine, it's fine,
        # and we don't need to track any serialized TRT engines.
        return

      resource_map[canonical_engine_name] = TRTEngineResource(
          canonical_engine_name, filename,
          self._conversion_params.maximum_cached_engines)

    # Remove all scope prefixes in the node name. In TF 2.0, the same concrete
    # function can be initialized multiple times with different prefixes, and
    # this will result in the same TRTEngineOp being initialized multiple times
    # with different cache and duplicate TRT engines.
    # TODO(laigd): this may be caused by the fact that TRTEngineOp is not
    # stataful, need to investigate.
    # TODO(laigd): we rely on the fact that all functions are fully inlined
    # before TF-TRT optimizer is called, as otherwise it may generate the same
    # name when optimizing a different function graph. Fix this.
    canonical_engine_name = lambda node: node.name.split("/")[-1]
    for node in self._converted_graph_def.node:
      if node.op == _TRT_ENGINE_OP_NAME:
        _serialize_and_track_engine(canonical_engine_name(node))
    for func in self._converted_graph_def.library.function:
      for node in func.node_def:
        if node.op == _TRT_ENGINE_OP_NAME:
          _serialize_and_track_engine(canonical_engine_name(node))

    self._saved_model.trt_engine_resources = resource_map

    # Rewrite the signature map using the optimized ConcreteFunction.
    signatures = {
        key: value for key, value in self._saved_model.signatures.items()
    }
    signatures[self._input_saved_model_signature_key] = self._converted_func
    save.save(self._saved_model, output_saved_model_dir, signatures)


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
    cached_engine_batches=None,
    input_saved_model_dir=None,
    input_saved_model_tags=None,
    input_saved_model_signature_key=None,
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
    input_saved_model_dir: the directory to load the SavedModel which contains
      the input graph to transforms. Used only when input_graph_def is None.
    input_saved_model_tags: list of tags to load the SavedModel.
    input_saved_model_signature_key: the key of the signature to optimize the
      graph for.
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
  """
  trt_converter = TrtGraphConverter(
      input_saved_model_dir=input_saved_model_dir,
      input_saved_model_tags=input_saved_model_tags,
      input_saved_model_signature_key=input_saved_model_signature_key,
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
      use_calibration=False)
  converted_graph_def = trt_converter.convert()
  if output_saved_model_dir:
    trt_converter.save(output_saved_model_dir)
  return converted_graph_def
