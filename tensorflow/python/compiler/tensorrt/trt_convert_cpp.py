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

import os
import tempfile

from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.compiler.tensorrt import _pywrap_trt_convert
from tensorflow.python.compiler.tensorrt import model_opt
from tensorflow.python.compiler.tensorrt import trt_utils
from tensorflow.python.compiler.tensorrt.constants import TRT_ENGINE_OP_NAME
from tensorflow.python.compiler.tensorrt.constants import TrtPrecisionMode
from tensorflow.python.compiler.tensorrt.constants import TrtProfileStrategy
from tensorflow.python.compiler.tensorrt.types import TRTEngineResource
from tensorflow.python.eager import context
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import ops
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.tools import saved_model_utils
from tensorflow.python.trackable import autotrackable
from tensorflow.python.util import nest


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
     * `ImplicitBatchModeCompatible`: create the profiles that will produce the
       same GPU engines as the implicit_batch_mode would produce.
  """

  # Indicates that this is the experimental API; used for tests.
  _experimental = True

  def __init__(self,
               input_saved_model_dir=None,
               input_saved_model_tags=None,
               input_saved_model_signature_key=None,
               use_dynamic_shape=None,
               dynamic_shape_profile_strategy=None,
               max_workspace_size_bytes=None,
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

    # Get _pywrap_trt_convert enums from constants wrappers
    if conversion_params is not None:
      precision_mode = conversion_params.precision_mode
    if isinstance(precision_mode, TrtPrecisionMode):
      precision_mode = precision_mode.value
    if isinstance(dynamic_shape_profile_strategy, TrtProfileStrategy):
      dynamic_shape_profile_strategy = dynamic_shape_profile_strategy.value

    # Get _pywrap_trt_convert.TrtConversionParams from constants.TrtConversionParams
    if conversion_params is not None:
      conversion_params = _pywrap_trt_convert.TrtConversionParams(
          max_workspace_size_bytes=conversion_params.max_workspace_size_bytes,
          precision_mode=precision_mode,
          minimum_segment_size=conversion_params.minimum_segment_size,
          maximum_cached_engines=conversion_params.maximum_cached_engines,
          use_calibration=conversion_params.use_calibration,
          allow_build_at_runtime=conversion_params.allow_build_at_runtime)
    else:
      conversion_params = _pywrap_trt_convert.TrtConversionParams(
          precision_mode=precision_mode,
          minimum_segment_size=minimum_segment_size,
          maximum_cached_engines=maximum_cached_engines,
          use_calibration=use_calibration,
          allow_build_at_runtime=allow_build_at_runtime)

    if max_workspace_size_bytes is not None:
      conversion_params.max_workspace_size_bytes = max_workspace_size_bytes
    if use_dynamic_shape is not None:
      conversion_params.use_dynamic_shape = use_dynamic_shape
    if dynamic_shape_profile_strategy is not None:
      conversion_params.dynamic_shape_profile_strategy = (
          dynamic_shape_profile_strategy)

    trt_utils.validate_environment()

    self._test_only_disable_non_trt_optimizers = False
    self._conversion_params = conversion_params
    self._input_saved_model_dir = input_saved_model_dir
    self._input_saved_model_tags = (
        input_saved_model_tags or [tag_constants.SERVING])
    self._input_saved_model_signature_key = (
        input_saved_model_signature_key or
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY)
    self.freeze = not trt_utils.is_experimental_feature_activated(
        "disable_graph_freezing")

    if not self.freeze and not conversion_params.use_dynamic_shape:
      logging.warn(
          "Disabling graph freezing is only possible in dynamic shape mode."
          " The graph will be frozen.")
      self.freeze = True

    # Load input model
    self._saved_model = load.load(self._input_saved_model_dir,
                                  self._input_saved_model_tags)
    self._func = self._saved_model.signatures[
        self._input_saved_model_signature_key]
    if self.freeze:
      self._frozen_func = convert_variables_to_constants_v2(self._func)
      frozen_graph_def = self._frozen_func.graph.as_graph_def()
      # Create converter
      frozen_graph_def_str = frozen_graph_def.SerializeToString()
      self._converter = _pywrap_trt_convert.TrtGraphConverter(
          frozen_graph_def_str,
          [tensor.name for tensor in self._frozen_func.inputs],
          [tensor.name for tensor in self._frozen_func.outputs],
          self._conversion_params)
    else:
      # Create converter
      self._frozen_func = None
      self._converter = _pywrap_trt_convert.TrtGraphConverter(
          self._input_saved_model_dir,
          self._input_saved_model_signature_key,
          set(self._input_saved_model_tags),
          self._conversion_params)
      meta_graph_def = meta_graph_pb2.MetaGraphDef()
      input_graph_def = meta_graph_def.graph_def
      input_graph_def.ParseFromString(self._converter._get_input_graph_def())

      # Wrap the loaded GraphDef
      # TODO: To avoid 2GB limit, do not pass GraphDef between C++ and Python
      meta_graph_def = saved_model_utils.get_meta_graph_def(
          self._input_saved_model_dir,
          ','.join(self._input_saved_model_tags))
      signature = meta_graph_def.signature_def[
          self._input_saved_model_signature_key]
      inputs_info, outputs_info = signature.inputs, signature.outputs
      output_names = [tensor_info.name for tensor_info in outputs_info.values()]
      input_names = [name + ":0" for name in inputs_info.keys()]
      inlined_func = wrap_function.function_from_graph_def(
          input_graph_def, input_names, output_names)
      inlined_func.graph.structured_outputs = nest.pack_sequence_as(
          self._func.graph.structured_outputs,
          inlined_func.graph.structured_outputs)
      inlined_func.graph.structured_input_signature = (
          self._func.structured_input_signature)
      self._func = inlined_func


  def _for_each_trt_node(self, graph_def, fn):
    """Helper method to manipulate all TRTEngineOps in a GraphDef."""
    for node in graph_def.node:
      if node.op == TRT_ENGINE_OP_NAME:
        fn(node)
    for func in graph_def.library.function:
      for node in func.node_def:
        if node.op == TRT_ENGINE_OP_NAME:
          fn(node)

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
    # Creating an empty tensor to fetch queried device
    device_requested = array_ops.zeros([]).device

    converted_graph_def_str = self._converter.convert(
      calibration_input_fn()
      if calibration_input_fn is not None
      else None,
      self._test_only_disable_non_trt_optimizers,
      device_requested
    )
    meta_graph_def = meta_graph_pb2.MetaGraphDef()
    self._converted_graph_def = meta_graph_def.graph_def
    self._converted_graph_def.ParseFromString(converted_graph_def_str)
    self._converted_func = model_opt.construct_function_from_graph_def(
      self._func, self._converted_graph_def, self._frozen_func)

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
    func = self._frozen_func if self.freeze else self._func
    input_names = [tensor.name.split(":")[0] for tensor in func.inputs]
    def convert_to_tensor(input_fn):
      for inp in input_fn():
        try:
          if isinstance(inp, dict):
            args = [ops.convert_to_tensor(inp[name]) for name in input_names]
          else:
            if isinstance(inp, (list, tuple)):
              args = list(map(ops.convert_to_tensor, inp))
            else:
              args = [ops.convert_to_tensor(inp)]
        except:
          error_msg = "Failed to convert input to tensor."
          logging.error(f"{error_msg}\ninp = `{inp}`\n")
          raise RuntimeError(error_msg)
        yield args

    static_graph_def_str = self._converter.build(convert_to_tensor(input_fn))
    self._converted_graph_def.ParseFromString(static_graph_def_str)
    self._converted_func = model_opt.construct_function_from_graph_def(
        self._func, self._converted_graph_def, self._frozen_func)

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
    engine_asset_dir = tempfile.mkdtemp()
    path_map = self._converter._serialize_engines(
        engine_asset_dir, save_gpu_specific_engines)

    # Track engines
    resource_map = {engine: TRTEngineResource(
      engine, path, self._conversion_params.maximum_cached_engines
    ) for engine, path in path_map.items()}

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
      self._converted_func = model_opt.construct_function_from_graph_def(
          self._func, self._converted_graph_def, self._frozen_func)

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
    # TODO
    pass