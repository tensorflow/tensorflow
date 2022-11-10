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

from functools import partial  # pylint: disable=g-importing-member
import os
import tempfile

import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.compiler.tensorrt import utils as trt_utils
from tensorflow.python.compiler.tensorrt.trt_convert_common import DEFAULT_TRT_MAX_WORKSPACE_SIZE_BYTES
from tensorflow.python.compiler.tensorrt.trt_convert_common import PROFILE_STRATEGY_RANGE
from tensorflow.python.compiler.tensorrt.trt_convert_common import TrtConversionParams
from tensorflow.python.compiler.tensorrt.trt_convert_common import TrtPrecisionMode
from tensorflow.python.compiler.tensorrt.trt_convert_common import _TRTEngineResource
from tensorflow.python.compiler.tensorrt.trt_convert_common import _TRT_ENGINE_OP_NAME
from tensorflow.python.compiler.tensorrt.trt_convert_common import _annotate_variable_ops
from tensorflow.python.compiler.tensorrt.trt_convert_common import _apply_inlining
from tensorflow.python.compiler.tensorrt.trt_convert_common import _check_conversion_params
from tensorflow.python.compiler.tensorrt.trt_convert_common import _check_trt_version_compatibility
from tensorflow.python.compiler.tensorrt.trt_convert_common import _construct_function_from_graph_def
from tensorflow.python.compiler.tensorrt.trt_convert_common import _convert_to_tensor
from tensorflow.python.compiler.tensorrt.trt_convert_common import _get_canonical_engine_name
from tensorflow.python.compiler.tensorrt.trt_convert_common import _get_tensorrt_rewriter_config
from tensorflow.python.compiler.tensorrt.trt_convert_common import _print_row
from tensorflow.python.compiler.tensorrt.trt_convert_common import _save_calibration_table
from tensorflow.python.compiler.tensorrt.trt_convert_common import gen_trt_ops
from tensorflow.python.compiler.tensorrt.trt_convert_common import supported_profile_strategies
from tensorflow.python.eager import context
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.framework import errors
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.trackable import autotrackable
from tensorflow.python.training import saver
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export


@tf_export("experimental.tensorrt.Converter", v1=[])
class TrtGraphConverterV2(object):
  """An offline converter for TF-TRT transformation for TF 2.0 SavedModels.

  Windows support is provided experimentally. No guarantee is made regarding
  functionality or engineering support. Use at your own risk.

  There are several ways to run the conversion:

  1. FP32/FP16 precision

     ```python
     params = tf.experimental.tensorrt.ConversionParams(
         precision_mode='FP16')
     converter = tf.experimental.tensorrt.Converter(
         input_saved_model_dir="my_dir", conversion_params=params)
     converter.convert()
     converter.save(output_saved_model_dir)
     ```

     In this case, no TRT engines will be built or saved in the converted
     SavedModel. But if input data is available during conversion, we can still
     build and save the TRT engines to reduce the cost during inference (see
     option 2 below).

  2. FP32/FP16 precision with pre-built engines

     ```python
     params = tf.experimental.tensorrt.ConversionParams(
         precision_mode='FP16',
         # Set this to a large enough number so it can cache all the engines.
         maximum_cached_engines=16)
     converter = tf.experimental.tensorrt.Converter(
         input_saved_model_dir="my_dir", conversion_params=params)
     converter.convert()

     # Define a generator function that yields input data, and use it to execute
     # the graph to build TRT engines.
     def my_input_fn():
       for _ in range(num_runs):
         inp1, inp2 = ...
         yield inp1, inp2

     converter.build(input_fn=my_input_fn)  # Generate corresponding TRT engines
     converter.save(output_saved_model_dir)  # Generated engines will be saved.
     ```

     In this way, one engine will be built/saved for each unique input shapes of
     the TRTEngineOp. This is good for applications that cannot afford building
     engines during inference but have access to input data that is similar to
     the one used in production (for example, that has the same input shapes).
     Also, the generated TRT engines is platform dependent, so we need to run
     `build()` in an environment that is similar to production (e.g. with
     same type of GPU).

  3. INT8 precision and calibration with pre-built engines

     ```python
     params = tf.experimental.tensorrt.ConversionParams(
         precision_mode='INT8',
         # Currently only one INT8 engine is supported in this mode.
         maximum_cached_engines=1,
         use_calibration=True)
     converter = tf.experimental.tensorrt.Converter(
         input_saved_model_dir="my_dir", conversion_params=params)

     # Define a generator function that yields input data, and run INT8
     # calibration with the data. All input data should have the same shape.
     # At the end of convert(), the calibration stats (e.g. range information)
     # will be saved and can be used to generate more TRT engines with different
     # shapes. Also, one TRT engine will be generated (with the same shape as
     # the calibration data) for save later.
     def my_calibration_input_fn():
       for _ in range(num_runs):
         inp1, inp2 = ...
         yield inp1, inp2

     converter.convert(calibration_input_fn=my_calibration_input_fn)

     # (Optional) Generate more TRT engines offline (same as the previous
     # option), to avoid the cost of generating them during inference.
     def my_input_fn():
       for _ in range(num_runs):
         inp1, inp2 = ...
         yield inp1, inp2
     converter.build(input_fn=my_input_fn)

     # Save the TRT engine and the engines.
     converter.save(output_saved_model_dir)
     ```
  4. To use dynamic shape, we need to call the build method with an input
     function to generate profiles. This step is similar to the INT8 calibration
     step described above. The converter also needs to be created with
     use_dynamic_shape=True and one of the following profile_strategies for
     creating profiles based on the inputs produced by the input function:
     * `Range`: create one profile that works for inputs with dimension values
       in the range of [min_dims, max_dims] where min_dims and max_dims are
       derived from the provided inputs.
     * `Optimal`: create one profile for each input. The profile only works for
       inputs with the same dimensions as the input it is created for. The GPU
       engine will be run with optimal performance with such inputs.
     * `Range+Optimal`: create the profiles for both `Range` and `Optimal`.
  """

  def _verify_profile_strategy(self, strategy):
    supported_strategies = [s.lower() for s in supported_profile_strategies()]
    if strategy.lower() not in supported_strategies:
      raise ValueError(
          ("profile_strategy '{}' is not supported. It should be one of {}"
          ).format(strategy, supported_profile_strategies()))
    if strategy == "ImplicitBatchModeCompatible":
      logging.warn(
          "ImplicitBatchModeCompatible strategy is deprecated, and"
          " using it may result in errors during engine building. Please"
          " consider using a different profile strategy.")

  @deprecation.deprecated_args(None,
                               "Use individual converter parameters instead",
                               "conversion_params")
  def __init__(self,
               input_saved_model_dir=None,
               input_saved_model_tags=None,
               input_saved_model_signature_key=None,
               use_dynamic_shape=None,
               dynamic_shape_profile_strategy=None,
               max_workspace_size_bytes=DEFAULT_TRT_MAX_WORKSPACE_SIZE_BYTES,
               precision_mode=TrtPrecisionMode.FP32,
               minimum_segment_size=3,
               maximum_cached_engines=1,
               use_calibration=True,
               allow_build_at_runtime=True,
               conversion_params=None):
    """Initialize the converter.

    Args:
      input_saved_model_dir: the directory to load the SavedModel which contains
        the input graph to transforms. Required.
      input_saved_model_tags: list of tags to load the SavedModel.
      input_saved_model_signature_key: the key of the signature to optimize the
        graph for.
      use_dynamic_shape: whether to enable dynamic shape support. None is
        equivalent to False in the current implementation.
      dynamic_shape_profile_strategy: one of the strings in
        supported_profile_strategies(). None is equivalent to Range in the
        current implementation.
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
      conversion_params: a TrtConversionParams instance (deprecated).

    Raises:
      ValueError: if the combination of the parameters is invalid.
    """
    assert context.executing_eagerly()
    if conversion_params is None:
      conversion_params = TrtConversionParams(
          max_workspace_size_bytes=max_workspace_size_bytes,
          precision_mode=precision_mode,
          minimum_segment_size=minimum_segment_size,
          maximum_cached_engines=maximum_cached_engines,
          use_calibration=use_calibration,
          allow_build_at_runtime=allow_build_at_runtime)

    _check_trt_version_compatibility()
    _check_conversion_params(conversion_params, is_v2=True)

    self._conversion_params = conversion_params
    self._input_saved_model_dir = input_saved_model_dir
    self._input_saved_model_tags = (
        input_saved_model_tags or [tag_constants.SERVING])
    self._input_saved_model_signature_key = (
        input_saved_model_signature_key or
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY)
    self.freeze = not trt_utils.is_experimental_feature_activated(
        "disable_graph_freezing")

    self._need_calibration = ((
        (conversion_params.precision_mode == TrtPrecisionMode.INT8) or
        (conversion_params.precision_mode == TrtPrecisionMode.INT8.lower())) and
                              conversion_params.use_calibration)

    self._calibration_input_fn = None

    self._converted = False
    self._device = None
    self._build_called_once = False
    self._calibrated = False

    if use_dynamic_shape is None:
      self._use_dynamic_shape = False
    else:
      self._use_dynamic_shape = use_dynamic_shape

    if not self.freeze and not self._use_dynamic_shape:
      logging.warn(
          "Disabling graph freezing is only possible in dynamic shape mode."
          " The graph will be frozen.")
      self.freeze = True

    self._profile_strategy = "Unknown"
    if self._use_dynamic_shape:
      if dynamic_shape_profile_strategy is None:
        self._profile_strategy = PROFILE_STRATEGY_RANGE
      else:
        self._verify_profile_strategy(dynamic_shape_profile_strategy)
        self._profile_strategy = dynamic_shape_profile_strategy

    # Fields to support TF-TRT testing and shouldn't be used for other purpose.
    self._test_only_disable_non_trt_optimizers = False

  def _need_trt_profiles(self):
    return self._use_dynamic_shape

  def _run_conversion(self, meta_graph_def):
    """Run Grappler's OptimizeGraph() tool to convert the graph.

    Args:
      meta_graph_def: the MetaGraphDef instance to run the optimizations on.

    Returns:
      The optimized GraphDef.
    """
    grappler_session_config = config_pb2.ConfigProto()
    # Always set `allow_build_at_runtime` for offline TensorRT engine building.
    custom_rewriter_config = _get_tensorrt_rewriter_config(
        conversion_params=self._conversion_params._replace(
            allow_build_at_runtime=True),
        is_dynamic_op=True,
        max_batch_size=None,
        disable_non_trt_optimizers=self._test_only_disable_non_trt_optimizers,
        use_implicit_batch=not self._use_dynamic_shape,
        profile_strategy=self._profile_strategy)
    grappler_session_config.graph_options.rewrite_options.CopyFrom(
        custom_rewriter_config)
    return tf_optimizer.OptimizeGraph(
        grappler_session_config, meta_graph_def, graph_id=b"tf_graph")

  def _for_each_trt_node(self, graph_def, fn):
    """Helper method to manipulate all TRTEngineOps in a GraphDef."""
    for node in graph_def.node:
      if node.op == _TRT_ENGINE_OP_NAME:
        fn(node)
    for func in graph_def.library.function:
      for node in func.node_def:
        if node.op == _TRT_ENGINE_OP_NAME:
          fn(node)

  def _execute_calibration(self, calibration_input_fn):
    """Run INT8 calibration with the provided input generator function."""
    for inp in calibration_input_fn():
      args, kwargs = _convert_to_tensor(inp)
      self._converted_func(*args, **kwargs)

    self._for_each_trt_node(self._converted_graph_def, _save_calibration_table)

    # Rebuild the function since calibration has changed the graph.
    self._converted_func = _construct_function_from_graph_def(
        self._converted_func, self._converted_graph_def)
    self._calibrated = True

  # TODO(laigd): provide a utility function to optimize a ConcreteFunction and
  # use it here (b/124792963).
  def convert(self, calibration_input_fn=None):
    """Convert the input SavedModel in 2.0 format.

    Args:
      calibration_input_fn: a generator function that yields input data as a
        list or tuple or dict, which will be used to execute the converted
        signature for calibration. All the returned input data should have the
        same shape. Example: `def input_fn(): yield input1, input2, input3`

        If dynamic_shape_mode==False, (or if the graph has static input shapes)
        then we run calibration and build the calibrated engine during
        conversion.

        If dynamic_shape_mode==True (and the graph has any unknown input
        shape), then the reference to calibration_input_fn is stored, and the
        calibration is actually performed when we build the engine (see
        build()).

    Raises:
      ValueError: if the input combination is invalid.

    Returns:
      The TF-TRT converted Function.
    """
    assert not self._converted

    # Creating an empty tensor to fetch queried device
    device_requested = array_ops.zeros([]).device

    if "gpu" not in device_requested.lower():
      raise ValueError(f"Specified device is not a GPU: {device_requested}")

    if "gpu:0" not in device_requested.lower():
      self._device = device_requested
      logging.info(f"Placing imported graph from "
                   f"`{self._input_saved_model_dir}` on device: {self._device}")

    if (self._need_calibration and not calibration_input_fn):
      raise ValueError("Should specify calibration_input_fn because INT8 "
                       "calibration is needed")
    if (not self._need_calibration and calibration_input_fn):
      raise ValueError("Should not specify calibration_input_fn because INT8 "
                       "calibration is not needed")

    self._saved_model = load.load(self._input_saved_model_dir,
                                  self._input_saved_model_tags)
    func = self._saved_model.signatures[self._input_saved_model_signature_key]
    if self.freeze:
      frozen_func = convert_to_constants.convert_variables_to_constants_v2(func)
    else:
      inlined_graph_def = _apply_inlining(func)
      _annotate_variable_ops(func, inlined_graph_def)
      frozen_func = _construct_function_from_graph_def(func, inlined_graph_def)
    frozen_graph_def = frozen_func.graph.as_graph_def()

    # Clear any prior device assignments
    logging.info("Clearing prior device assignments in loaded saved model")
    for node in frozen_graph_def.node:
      node.device = ""

    if self._device is None:
      grappler_meta_graph_def = saver.export_meta_graph(
          graph_def=frozen_graph_def, graph=frozen_func.graph)
    else:
      with ops.Graph().as_default() as graph, ops.device(self._device):
        importer.import_graph_def(frozen_graph_def, name="")
        grappler_meta_graph_def = saver.export_meta_graph(
            graph_def=graph.as_graph_def(), graph=graph)

    # Add a collection 'train_op' so that Grappler knows the outputs.
    fetch_collection = meta_graph_pb2.CollectionDef()
    for array in frozen_func.inputs + frozen_func.outputs:
      fetch_collection.node_list.value.append(array.name)
    grappler_meta_graph_def.collection_def["train_op"].CopyFrom(
        fetch_collection)

    # Run TRT optimizer in Grappler to convert the graph.
    self._converted_graph_def = self._run_conversion(grappler_meta_graph_def)
    self._converted_func = _construct_function_from_graph_def(
        func, self._converted_graph_def, frozen_func)

    if self._need_calibration:
      # Execute calibration here only if not in dynamic shape mode.
      if not self._need_trt_profiles():
        self._execute_calibration(calibration_input_fn)
      else:
        self._calibration_input_fn = calibration_input_fn

    self._converted = True

    graphviz_path = os.environ.get("TF_TRT_EXPORT_GRAPH_VIZ_PATH", default=None)
    if graphviz_path is not None:
      try:
        trt_utils.draw_graphdef_as_graphviz(
            graphdef=self._converted_func.graph.as_graph_def(add_shapes=True),
            dot_output_filename=graphviz_path)
      except Exception as e:
        logging.error("An Exception occured during the export of the graph "
                      f"visualization: {e}")

    return self._converted_func

  def build(self, input_fn):
    """Run inference with converted graph in order to build TensorRT engines.

    If the conversion requires INT8 calibration, then a reference to the
    calibration function was stored during the call to convert(). Calibration
    will be performed while we build the TensorRT engines.

    Args:
      input_fn: a generator function that provides the input data as a single
        array, OR a list or tuple of the arrays OR a dict, which will be used
        to execute the converted signature to generate TRT engines.
        Example 1:
        `def input_fn():
             # Let's assume a network with 1 input tensor.
             # We generate 2 sets of dummy input data:
             input_shapes = [(1, 16),    # 1st shape
                             (2, 32)]    # 2nd shape
             for shapes in input_shapes:
                 # return an input tensor
                 yield np.zeros(shape).astype(np.float32)'

        Example 2:
        `def input_fn():
             # Let's assume a network with 2 input tensors.
             # We generate 3 sets of dummy input data:
             input_shapes = [[(1, 16), (2, 16)], # 1st input list
                             [(2, 32), (4, 32)], # 2nd list of two tensors
                             [(4, 32), (8, 32)]] # 3rd input list
             for shapes in input_shapes:
                 # return a list of input tensors
                 yield [np.zeros(x).astype(np.float32) for x in shapes]`

    Raises:
      NotImplementedError: build() is already called.
      RuntimeError: the input_fx is None.
    """
    if self._build_called_once:
      raise NotImplementedError("build() is already called. It is not "
                                "supported to call build() more than once.")
    if not input_fn:
      raise RuntimeError("input_fn is None. Method build() needs input_fn "
                         "to be specified in order to build TensorRT engines")
    if not self._converted:
      raise RuntimeError("Need to call convert() before build()")
    if (self._need_calibration and not self._calibrated and
        self._calibration_input_fn is None):
      raise RuntimeError("Need to provide the calibration_input_fn arg while "
                         "calling convert().")

    def _set_profile_generation_mode(value, node):
      node.attr["_profile_generation_mode"].b = value

    if self._need_trt_profiles():
      # Enable profile generation.
      self._for_each_trt_node(self._converted_graph_def,
                              partial(_set_profile_generation_mode, True))
      # Profile generation is enabled using the _profile_generation_mode
      # attribute of the TRTEngineOps. We need to rebuild the function to
      # change this attribute.
      func = _construct_function_from_graph_def(self._converted_func,
                                                self._converted_graph_def)
    else:
      func = self._converted_func

    first_input = None
    # Run inference:
    #   Builds TRT engines if self._need_trt_profiles is False.
    #   Builds TRT optimization profiles if self._need_trt_profiles is True.
    for inp in input_fn():
      if first_input is None:
        first_input = inp
      args, kwargs = _convert_to_tensor(inp)
      func(*args, **kwargs)

    if self._need_trt_profiles():
      # Disable profile generation.
      self._for_each_trt_node(self._converted_graph_def,
                              partial(_set_profile_generation_mode, False))

    # Run calibration if required, this would have been skipped in
    # the convert step
    if self._need_calibration and not self._calibrated:
      self._execute_calibration(self._calibration_input_fn)
      # calibration also builds the engine
    else:
      # Use the first input in explicit batch mode to build TensorRT engines
      # after generating all the profiles. The first input is used but any of
      # the inputs can be used because the shape of this input does not
      # determine the engine and instead the shapes collected in profiles
      # determine the engine.
      args, kwargs = _convert_to_tensor(first_input)
      self._converted_func(*args, **kwargs)

    self._build_called_once = True

  def save(self,
           output_saved_model_dir,
           save_gpu_specific_engines=True,
           options=None):
    """Save the converted SavedModel.

    Args:
      output_saved_model_dir: directory to saved the converted SavedModel.
      save_gpu_specific_engines: whether to save TRT engines that have been
        built. When True, all engines are saved and when False, the engines
        are not saved and will be rebuilt at inference time. By using
        save_gpu_specific_engines=False after doing INT8 calibration, inference
        can be done on different GPUs than the GPU that the model was calibrated
        and saved on.
      options: `tf.saved_model.SaveOptions` object for configuring save options.
    Raises:
      RuntimeError: if the needed calibration hasn't been done.
    """
    assert self._converted
    if self._need_calibration and not self._calibrated:
      raise RuntimeError("A model that requires INT8 calibration has to be "
                         "built before saving it. Call build() to build and "
                         "calibrate the TensorRT engines.")
    # Serialize the TRT engines in the cache if any, and create trackable
    # resource to track them.
    engine_asset_dir = tempfile.mkdtemp()
    resource_map = {}

    def _serialize_and_track_engine(node):
      """Serialize TRT engines in the cache and track them."""
      # Don't dump the same cache twice.
      canonical_engine_name = _get_canonical_engine_name(node.name)
      if canonical_engine_name in resource_map:
        return

      filename = os.path.join(engine_asset_dir,
                              "trt-serialized-engine." + canonical_engine_name)

      try:
        gen_trt_ops.serialize_trt_resource(
            resource_name=canonical_engine_name,
            filename=filename,
            delete_resource=True,
            save_gpu_specific_engines=save_gpu_specific_engines)
      except errors.NotFoundError:
        logging.info(
            "Could not find %s in TF-TRT cache. "
            "This can happen if build() is not called, "
            "which means TensorRT engines will be built "
            "and cached at runtime.", canonical_engine_name)
        return

      # TODO(laigd): add an option for the user to choose the device.
      resource_map[canonical_engine_name] = _TRTEngineResource(
          canonical_engine_name, filename,
          self._conversion_params.maximum_cached_engines)

    self._for_each_trt_node(self._converted_graph_def,
                            _serialize_and_track_engine)
    # If the graph is frozen, tracked variables are not needed by the converted model.
    trackable = autotrackable.AutoTrackable(
    ) if self.freeze else self._saved_model
    trackable.trt_engine_resources = resource_map

    # Set allow_build_at_runtime=False if asked by user.
    #
    # This attribute is set here because build() needs it to be True in order to
    # build engines.
    if not self._conversion_params.allow_build_at_runtime:

      def _reset_allow_build_at_runtime(node):
        node.attr["_allow_build_at_runtime"].b = False

      self._for_each_trt_node(self._converted_graph_def,
                              _reset_allow_build_at_runtime)
      # Rebuild the function since a node attribute changed above
      reset_converted_func = wrap_function.function_from_graph_def(
          self._converted_graph_def,
          [tensor.name for tensor in self._converted_func.inputs],
          [tensor.name for tensor in self._converted_func.outputs])
      reset_converted_func.graph.structured_outputs = nest.pack_sequence_as(
          self._converted_func.graph.structured_outputs,
          reset_converted_func.graph.structured_outputs)
      reset_converted_func.graph.structured_input_signature = (
          self._converted_func.structured_input_signature)
      self._converted_func = reset_converted_func

    # Rewrite the signature map using the optimized ConcreteFunction.
    signatures = {self._input_saved_model_signature_key: self._converted_func}
    save.save(trackable, output_saved_model_dir, signatures, options=options)

  def summary(self, line_length=160, detailed=True, print_fn=None):
    """This method describes the results of the conversion by TF-TRT.

    It includes information such as the name of the engine, the number of nodes
    per engine, the input and output dtype, along with the input shape of each
    TRTEngineOp.

    Args:
      line_length: Default line length when printing on the console. Minimum 160
        characters long.
      detailed: Whether or not to show the nodes inside each TRTEngineOp.
      print_fn: Print function to use. Defaults to `print`. It will be called on
        each line of the summary. You can set it to a custom function in order
        to capture the string summary.

    Raises:
      RuntimeError: if the graph is not converted.
    """
    if not self._converted:
      raise RuntimeError(
          f"Impossible to call `{self.__class__.__name__}.summary()` before "
          f"calling {self.__class__.__name__}.convert()`.")

    if line_length < 160:
      raise ValueError(f"Invalid `line_length` value has been received: "
                       f"{line_length}. Minimum: 160.")

    if print_fn is None:
      print_fn = print

    # positions are percentage of `line_length`. positions[i]+1 is the starting
    # position for (i+1)th field. We also make sure that the last char printed
    # for each field is a space.
    columns = [
        # (column name, column size in % of line)
        ("TRTEngineOP Name", .20),  # 20%
        ("Device", .09),  # 29%
        ("# Nodes", .05),  # 34%
        ("# Inputs", .09),  # 43%
        ("# Outputs", .09),  # 52%
        ("Input DTypes", .12),  # 64%
        ("Output Dtypes", .12),  # 76%
        ("Input Shapes", .12),  # 88%
        ("Output Shapes", .12)  # 100%
    ]

    positions = [int(line_length * p) for _, p in columns]
    positions = np.cumsum(positions).tolist()
    headers = [h for h, _ in columns]

    _print_row(headers, positions, print_fn=print_fn)
    print_fn("=" * line_length)

    n_engines = 0
    n_ops_converted = 0
    n_ops_not_converted = 0

    graphdef = self._converted_func.graph.as_graph_def(add_shapes=True)

    trtengineops_dict = dict()
    for node in graphdef.node:
      if node.op != "TRTEngineOp":
        n_ops_not_converted += 1
        continue
      else:
        trtengineops_dict[node.name] = node
        n_engines += 1

    for name, node in sorted(trtengineops_dict.items()):
      node_device = node.device.split("/")[-1]
      in_shapes = trt_utils.get_node_io_shapes(node, "input_shapes")
      out_shapes = trt_utils.get_node_io_shapes(node, "_output_shapes")
      in_dtypes = trt_utils.get_trtengineop_io_dtypes(node, "InT")
      out_dtypes = trt_utils.get_trtengineop_io_dtypes(node, "OutT")
      in_nodes_count = trt_utils.get_trtengineop_io_nodes_count(node, "InT")
      out_nodes_count = trt_utils.get_trtengineop_io_nodes_count(node, "OutT")
      node_count, converted_ops_dict = trt_utils.get_trtengineop_node_op_count(
          graphdef, name)

      n_ops_converted += node_count

      if n_engines != 1:
        print_fn(f"\n{'-'*40}\n")

      _print_row(
          fields=[
              name, node_device, node_count, in_nodes_count, out_nodes_count,
              in_dtypes, out_dtypes, in_shapes, out_shapes
          ],
          positions=positions,
          print_fn=print_fn)

      if detailed:
        print_fn()
        for key, value in sorted(dict(converted_ops_dict).items()):
          print_fn(f"\t- {key}: {value}x")

    print_fn(f"\n{'='*line_length}")
    print_fn(f"[*] Total number of TensorRT engines: {n_engines}")
    total_ops = n_ops_not_converted + n_ops_converted
    conversion_ratio = n_ops_converted / total_ops * 100
    print_fn(f"[*] % of OPs Converted: {conversion_ratio:.2f}% "
             f"[{n_ops_converted}/{total_ops}]\n")
