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
# ==============================================================================
"""Converts a frozen graph into a TFLite FlatBuffer."""

import distutils.spawn
import enum
import hashlib
import os as _os
import platform as _platform
import subprocess as _subprocess
import tempfile as _tempfile
from typing import Optional
import warnings

from tensorflow.compiler.mlir.quantization.stablehlo import quantization_options_pb2 as quant_opts_pb2
from tensorflow.lite.python import lite_constants
from tensorflow.lite.python import util
from tensorflow.lite.python import wrap_toco
from tensorflow.lite.python.convert_phase import Component
from tensorflow.lite.python.convert_phase import convert_phase
from tensorflow.lite.python.convert_phase import ConverterError
from tensorflow.lite.python.convert_phase import SubComponent
from tensorflow.lite.python.metrics.wrapper import metrics_wrapper as _metrics_wrapper
from tensorflow.lite.toco import model_flags_pb2 as _model_flags_pb2
from tensorflow.lite.toco import toco_flags_pb2 as _conversion_flags_pb2
from tensorflow.lite.toco import types_pb2 as _types_pb2
from tensorflow.lite.tools import flatbuffer_utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import resource_loader as _resource_loader
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export as _tf_export


def _is_quantized_input_stats_required(
    conversion_flags: _conversion_flags_pb2.TocoFlags,
) -> bool:
  """Checks if the `quantized_input_stats` flag is required for conversion.

  Args:
    conversion_flags: A protocol buffer describing the conversion process.

  Returns:
    True, if the `inference_type` or the `inference_input_type` is a quantized
    type and it is not post training quantization, else False.
  """
  quantized_inference_types = [
      _types_pb2.QUANTIZED_UINT8,
      _types_pb2.QUANTIZED_INT8,
  ]
  return (
      conversion_flags.inference_type in quantized_inference_types
      or conversion_flags.inference_input_type in quantized_inference_types
  ) and not conversion_flags.post_training_quantize


def convert_tensor_tf_type_to_tflite_type(
    tf_type: dtypes.DType, usage: str = ""
) -> _types_pb2.IODataType:
  """Convert tensor type from tf type to tflite type.

  Args:
    tf_type: TensorFlow type.
    usage: Text describing the reason for invoking this function.

  Raises:
    ValueError: If `tf_type` is unsupported.

  Returns:
    tflite_type: TFLite type. Refer to lite/toco/types.proto.
  """
  mapping = {
      dtypes.float16: _types_pb2.FLOAT16,
      dtypes.float32: _types_pb2.FLOAT,
      dtypes.float64: _types_pb2.FLOAT64,
      dtypes.int8: _types_pb2.INT8,
      dtypes.int16: _types_pb2.INT16,
      dtypes.uint16: _types_pb2.UINT16,
      dtypes.int32: _types_pb2.INT32,
      dtypes.int64: _types_pb2.INT64,
      dtypes.uint8: _types_pb2.UINT8,
      dtypes.uint32: _types_pb2.UINT32,
      dtypes.uint64: _types_pb2.UINT64,
      dtypes.string: _types_pb2.STRING,
      dtypes.bool: _types_pb2.BOOL,
      dtypes.complex64: _types_pb2.COMPLEX64,
      dtypes.complex128: _types_pb2.COMPLEX128,
  }
  tflite_type = mapping.get(tf_type)
  if tflite_type is None:
    raise ValueError(
        "Unsupported TensorFlow type `{0}` provided for the {1}".format(
            tf_type, usage
        )
    )
  return tflite_type


# Only a few restricted tensor types are allowed for explicitly setting
# inference/input/output types.
def convert_inference_tf_type_to_tflite_type(
    tf_type: dtypes.DType, usage: str = ""
) -> _types_pb2.IODataType:
  """Convert inference type from tf type to tflite type.

  Args:
    tf_type: TensorFlow type.
    usage: Text describing the reason for invoking this function.

  Raises:
    ValueError: If `tf_type` is unsupported.

  Returns:
    tflite_type: TFLite type. Refer to lite/toco/types.proto.
  """
  mapping = {
      dtypes.float32: _types_pb2.FLOAT,
      dtypes.uint8: _types_pb2.QUANTIZED_UINT8,
      dtypes.int8: _types_pb2.QUANTIZED_INT8,
      dtypes.int16: _types_pb2.QUANTIZED_INT16,
  }
  tflite_type = mapping.get(tf_type)
  if tflite_type is None:
    raise ValueError(
        "Unsupported TensorFlow type `{0}` provided for the {1}".format(
            tf_type, usage
        )
    )
  return tflite_type


# Find the deprecated conversion binary using the resource loader if using from
# bazel, otherwise we are in a pip where console_scripts already has the tool.
if lite_constants.EXPERIMENTAL_USE_TOCO_API_DIRECTLY:
  _deprecated_conversion_binary = ""
else:
  _deprecated_conversion_binary = _resource_loader.get_path_to_datafile(
      "../toco/python/toco_from_protos"
  )
  if not _os.path.exists(_deprecated_conversion_binary):
    _deprecated_conversion_binary = "toco_from_protos"


def _try_convert_to_unicode(output):
  if output is None:
    return ""

  if isinstance(output, bytes):
    try:
      return output.decode("utf-8")
    except UnicodeDecodeError:
      pass
  return output


@_tf_export("lite.OpsSet")
class OpsSet(enum.Enum):
  """Enum class defining the sets of ops available to generate TFLite models.

  WARNING: Experimental interface, subject to change.
  """

  # Convert model using TensorFlow Lite builtin ops.
  TFLITE_BUILTINS = "TFLITE_BUILTINS"

  # Convert model using TensorFlow ops. Not all TensorFlow ops are available.
  # WARNING: Experimental interface, subject to change.
  SELECT_TF_OPS = "SELECT_TF_OPS"

  # Convert model using only TensorFlow Lite quantized int8 operations.
  # Specifying this will throw an error for operations that do not yet have
  # quantized implementations.
  TFLITE_BUILTINS_INT8 = "TFLITE_BUILTINS_INT8"

  # Convert model using only TensorFlow Lite operations with quantized int8
  # weights, int16 activations and int64 bias.
  # Specifying this will throw an error for operations that do not yet have
  # quantized implementations.
  # This quantization mode may be used in models for super-resolution,
  # audio signal processing or image de-noising. It improves accuracy
  # significantly, but only slightly increases the model size.
  # WARNING: These ops are currently experimental and have not yet been
  # finalized.
  # They are only compatible with CPU execution, and have not been optimized for
  # production.
  EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8 = (
      "EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8"
  )

  # Convert model using only stablehlo ops.
  # This option can not be combined with other OpsSets.
  # The feature is in early development.
  # The code to execute StableHLO ops in the runtime is to be implemented
  # and the serialization format is not stabilized yet.

  EXPERIMENTAL_STABLEHLO_OPS = "EXPERIMENTAL_STABLEHLO_OPS"

  def __str__(self):
    return str(self.value)

  @staticmethod
  def get_options():
    """Returns a list of OpsSet options as a list of strings."""
    return [str(option) for option in list(OpsSet)]


@convert_phase(Component.OPTIMIZE_TFLITE_MODEL, SubComponent.QUANTIZE)
def mlir_quantize(
    input_data_str,
    disable_per_channel=False,
    fully_quantize=False,
    inference_type=_types_pb2.QUANTIZED_INT8,
    input_data_type=dtypes.float32,
    output_data_type=dtypes.float32,
    enable_numeric_verify=False,
    enable_whole_model_verify=False,
    denylisted_ops=None,
    denylisted_nodes=None,
    enable_variable_quantization=False,
):
  """Quantize `input_data_str` with calibration results.

  Args:
    input_data_str: Input data in serialized form (e.g. a TFLITE model with
      calibration results).
    disable_per_channel: Bool indicating whether to do per-channel or per-tensor
      quantization
    fully_quantize: Bool indicating whether to fully quantize the model. Besides
      model body, the input/output will be quantized as well.
    inference_type: Data type for the activations. The default value is int8.
    input_data_type: Data type for the inputs. The default value is float32.
    output_data_type: Data type for the outputs. The default value is float32.
    enable_numeric_verify: Experimental. Subject to change. Bool indicating
      whether to add NumericVerify ops into the debug mode quantized model.
    enable_whole_model_verify: Experimental. Subject to change. Bool indicating
      whether to add verification for layer by layer, or on whole model. When
      disabled (per-layer) float and quantized ops will be run from same input
      (output of previous quantized layer). When enabled, float and quantized
      ops will run with respective float and quantized output of previous ops.
    denylisted_ops: Experimental. Subject to change. Set of ops to denylist.
    denylisted_nodes: Experimental. Subject to change. Set of notes to denylist.
    enable_variable_quantization: Experimental. Subject to change. Bool
      indicating whether to enable quantization of the residual variables
      remaining after the variable freezing pass.

  Returns:
    Quantized model in serialized form (e.g. a TFLITE model) with floating-point
    inputs and outputs.
  """
  return wrap_toco.wrapped_experimental_mlir_quantize(
      input_data_str,
      disable_per_channel,
      fully_quantize,
      inference_type,
      convert_tensor_tf_type_to_tflite_type(input_data_type),
      convert_tensor_tf_type_to_tflite_type(output_data_type),
      enable_numeric_verify,
      enable_whole_model_verify,
      denylisted_ops,
      denylisted_nodes,
      enable_variable_quantization,
  )


@convert_phase(Component.OPTIMIZE_TFLITE_MODEL, SubComponent.SPARSIFY)
def mlir_sparsify(input_data_str):
  """Sparsify `input_data_str` to encode sparse tensor with proper format.

  Args:
    input_data_str: Input data in serialized form (e.g. a TFLITE model).

  Returns:
    Sparsified model in serialized form (e.g. a TFLITE model).
  """
  return wrap_toco.wrapped_experimental_mlir_sparsify(input_data_str)


def register_custom_opdefs(custom_opdefs_list):
  """Register the given custom opdefs to the TensorFlow global op registry.

  Args:
    custom_opdefs_list: String representing the custom ops OpDefs that are
      included in the GraphDef.

  Returns:
    True if the registration is successfully completed.
  """
  return wrap_toco.wrapped_register_custom_opdefs(custom_opdefs_list)


def convert(
    model_flags: _model_flags_pb2.ModelFlags,
    conversion_flags: _conversion_flags_pb2.TocoFlags,
    input_data_str: Optional[str] = None,
    debug_info_str: Optional[str] = None,
    enable_mlir_converter: bool = True,
):
  """Converts `input_data_str` to a TFLite model.

  Args:
    model_flags: Proto describing model properties, see `model_flags.proto`.
    conversion_flags: Proto describing conversion properties, see
      `toco/toco_flags.proto`.
    input_data_str: Input data in serialized form (e.g. a graphdef is common, or
      it can be hlo text or proto)
    debug_info_str: Serialized `GraphDebugInfo` proto describing logging
      information.
    enable_mlir_converter: Enables MLIR-based conversion.

  Returns:
    Converted model in serialized form (e.g. a TFLITE model is common).
  Raises:
    ConverterError: When conversion fails in TFLiteConverter, usually due to
      ops not being supported.
    RuntimeError: When conversion fails, an exception is raised with the error
      message embedded.
  """
  # Historically, deprecated conversion failures would trigger a crash, so we
  # attempt to run the converter out-of-process. The current MLIR conversion
  # pipeline surfaces errors instead, and can be safely run in-process.
  if enable_mlir_converter or not _deprecated_conversion_binary:
    try:
      model_str = wrap_toco.wrapped_toco_convert(
          model_flags.SerializeToString(),
          conversion_flags.SerializeToString(),
          input_data_str,
          debug_info_str,
          enable_mlir_converter,
      )
      return model_str
    except Exception as e:
      converter_error = ConverterError(str(e))

      for error_data in _metrics_wrapper.retrieve_collected_errors():
        converter_error.append_error(error_data)
      raise converter_error

  return _run_deprecated_conversion_binary(
      model_flags.SerializeToString(),
      conversion_flags.SerializeToString(),
      input_data_str,
      debug_info_str,
  )


@convert_phase(
    Component.CONVERT_TF_TO_TFLITE_MODEL,
    SubComponent.CONVERT_GRAPHDEF_USING_DEPRECATED_CONVERTER,
)
def _run_deprecated_conversion_binary(
    model_flags_str, conversion_flags_str, input_data_str, debug_info_str=None
):
  """Convert `input_data_str` using deprecated conversion binary.

  Args:
    model_flags_str: Serialized proto describing model properties, see
      `model_flags.proto`.
    conversion_flags_str: Serialized proto describing TFLite converter
      properties, see `toco/toco_flags.proto`.
    input_data_str: Input data in serialized form (e.g. a graphdef is common)
    debug_info_str: Serialized `GraphDebugInfo` proto describing logging
      information. (default None)

  Returns:
    Converted model in serialized form (e.g. a TFLITE model is common).
  Raises:
    ConverterError: When cannot find the deprecated conversion binary.
    RuntimeError: When conversion fails, an exception is raised with the error
      message embedded.
  """
  if distutils.spawn.find_executable(_deprecated_conversion_binary) is None:
    raise ConverterError("""Could not find `toco_from_protos` binary, make sure
your virtualenv bin directory or pip local bin directory is in your path.
In particular, if you have installed TensorFlow with --user, make sure you
add the install directory to your path.

For example:
Linux: export PATH=$PATH:~/.local/bin/
Mac: export PATH=$PATH:~/Library/Python/<version#>/bin

Alternative, use virtualenv.""")
  # Windows and TemporaryFile are not that useful together,
  # since you cannot have two readers/writers. So we have to
  # make the temporaries and close and delete them explicitly.
  conversion_filename, model_filename, input_filename, output_filename = (
      None,
      None,
      None,
      None,
  )
  try:
    # Build all input files
    with _tempfile.NamedTemporaryFile(
        delete=False
    ) as fp_conversion, _tempfile.NamedTemporaryFile(
        delete=False
    ) as fp_model, _tempfile.NamedTemporaryFile(
        delete=False
    ) as fp_input, _tempfile.NamedTemporaryFile(
        delete=False
    ) as fp_debug:
      conversion_filename = fp_conversion.name
      input_filename = fp_input.name
      model_filename = fp_model.name
      debug_filename = fp_debug.name

      fp_model.write(model_flags_str)
      fp_conversion.write(conversion_flags_str)
      fp_input.write(input_data_str)
      debug_info_str = debug_info_str if debug_info_str else ""
      # if debug_info_str contains a "string value", then the call to
      # fp_debug.write(debug_info_str) will fail with the following error
      #
      # TypeError: a bytes-like object is required, not 'str'
      #
      # Some of the subtests within the "convert_test" unit-test fail
      # with the error shown above. So watch out for that scenario and
      # convert debug_info_str to bytes where needed
      if not isinstance(debug_info_str, bytes):
        fp_debug.write(debug_info_str.encode("utf-8"))
      else:
        fp_debug.write(debug_info_str)

    # Reserve an output file
    with _tempfile.NamedTemporaryFile(delete=False) as fp:
      output_filename = fp.name

    # Run
    cmd = [
        _deprecated_conversion_binary,
        model_filename,
        conversion_filename,
        input_filename,
        output_filename,
        "--debug_proto_file={}".format(debug_filename),
    ]
    cmdline = " ".join(cmd)
    is_windows = _platform.system() == "Windows"
    proc = _subprocess.Popen(
        cmdline,
        shell=True,
        stdout=_subprocess.PIPE,
        stderr=_subprocess.STDOUT,
        close_fds=not is_windows,
    )
    stdout, stderr = proc.communicate()
    exitcode = proc.returncode
    if exitcode == 0:
      with open(output_filename, "rb") as fp:
        return fp.read()
    else:
      stdout = _try_convert_to_unicode(stdout)
      stderr = _try_convert_to_unicode(stderr)
      raise ConverterError("See console for info.\n%s\n%s\n" % (stdout, stderr))
  finally:
    # Must manually cleanup files.
    for filename in [
        conversion_filename,
        input_filename,
        model_filename,
        output_filename,
    ]:
      try:
        _os.unlink(filename)
      except (OSError, TypeError):
        pass


def build_model_flags(
    change_concat_input_ranges=False,
    allow_nonexistent_arrays=False,
    saved_model_dir=None,
    saved_model_version=0,
    saved_model_tags=None,
    saved_model_exported_names=None,
    **_
):
  """Builds the model flags object from params.

  Args:
    change_concat_input_ranges: Boolean to change behavior of min/max ranges for
      inputs and outputs of the concat operator for quantized models. Changes
      the ranges of concat operator overlap when true. (default False)
    allow_nonexistent_arrays: Allow specifying array names that don't exist or
      are unused in the final graph. (default False)
    saved_model_dir: Filepath of the saved model to be converted. This value
      will be non-empty only when the saved model import path will be used.
      Otherwises, the graph def-based conversion will be processed.
    saved_model_version: SavedModel file format version of The saved model file
      to be converted. This value will be set only when the SavedModel import
      path will be used.
    saved_model_tags: Set of string saved model tags, formatted in the
      comma-separated value. This value will be set only when the SavedModel
      import path will be used.
    saved_model_exported_names: Names to be exported (default: export all) when
      the saved model import path is on. This value will be set only when the
      SavedModel import path will be used.

  Returns:
    model_flags: protocol buffer describing the model.
  """
  model_flags = _model_flags_pb2.ModelFlags()
  model_flags.change_concat_input_ranges = change_concat_input_ranges
  model_flags.allow_nonexistent_arrays = allow_nonexistent_arrays
  if saved_model_dir:
    model_flags.saved_model_dir = saved_model_dir
  model_flags.saved_model_version = saved_model_version
  if saved_model_tags:
    model_flags.saved_model_tags.extend(saved_model_tags)
  if saved_model_exported_names:
    model_flags.saved_model_exported_names.extend(saved_model_exported_names)
  return model_flags


def build_conversion_flags(
    inference_type=dtypes.float32,
    inference_input_type=None,
    input_format=lite_constants.TENSORFLOW_GRAPHDEF,
    output_format=lite_constants.TFLITE,
    default_ranges_stats=None,
    drop_control_dependency=True,
    reorder_across_fake_quant=False,
    allow_custom_ops=False,
    post_training_quantize=False,
    quantize_to_float16=False,
    dump_graphviz_dir=None,
    dump_graphviz_video=False,
    target_ops=None,
    conversion_summary_dir=None,
    select_user_tf_ops=None,
    allow_all_select_tf_ops=False,
    enable_tflite_resource_variables=True,
    unfold_batchmatmul=True,
    lower_tensor_list_ops=True,
    default_to_single_batch_in_tensor_list_ops=False,
    accumulation_type=None,
    allow_bfloat16=False,
    unfold_large_splat_constant=False,
    supported_backends=None,
    disable_per_channel_quantization=False,
    enable_mlir_dynamic_range_quantizer=False,
    tf_quantization_mode=None,
    disable_infer_tensor_range=False,
    use_fake_quant_num_bits=False,
    enable_dynamic_update_slice=False,
    preserve_assert_op=False,
    guarantee_all_funcs_one_use=False,
    enable_mlir_variable_quantization=False,
    disable_fuse_mul_and_fc=False,
    quantization_options: Optional[quant_opts_pb2.QuantizationOptions] = None,
    **_
):
  """Builds protocol buffer describing a conversion of a model.

  Typically this is to convert from TensorFlow GraphDef to TFLite, in which
  case the default `input_format` and `output_format` are sufficient.

  Args:
    inference_type: Data type of numeric arrays, excluding the input layer.
      (default tf.float32, must be in {tf.float32, tf.int8, tf.uint8})
    inference_input_type: Data type of the numeric arrays in the input layer. If
      `inference_input_type` is in {tf.int8, tf.uint8}, then
      `quantized_input_stats` must be provided. (default is the value assigned
      to `inference_type`, must be in {tf.float32, tf.int8, tf.uint8})
    input_format: Type of data to read. (default TENSORFLOW_GRAPHDEF, must be in
      {TENSORFLOW_GRAPHDEF})
    output_format: Output file format. (default TFLITE, must be in {TFLITE,
      GRAPHVIZ_DOT})
    default_ranges_stats: Tuple of integers representing (min, max) range values
      for all arrays without a specified range. Intended for experimenting with
      quantization via "dummy quantization". (default None)
    drop_control_dependency: Boolean indicating whether to drop control
      dependencies silently. This is due to TFLite not supporting control
      dependencies. (default True)
    reorder_across_fake_quant: Boolean indicating whether to reorder FakeQuant
      nodes in unexpected locations. Used when the location of the FakeQuant
      nodes is preventing graph transformations necessary to convert the graph.
      Results in a graph that differs from the quantized training graph,
      potentially causing differing arithmetic behavior. (default False)
    allow_custom_ops: Boolean indicating whether to allow custom operations.
      When false any unknown operation is an error. When true, custom ops are
      created for any op that is unknown. The developer will need to provide
      these to the TensorFlow Lite runtime with a custom resolver. (default
      False)
    post_training_quantize: Boolean indicating whether to quantize the weights
      of the converted float model. Model size will be reduced and there will be
      latency improvements (at the cost of accuracy). (default False) If
      quantization_options is set, all quantization arg will be ignored.
    quantize_to_float16: Boolean indicating whether to convert float buffers to
      float16. (default False)
    dump_graphviz_dir: Full filepath of folder to dump the graphs at various
      stages of processing GraphViz .dot files. Preferred over
      --output_format=GRAPHVIZ_DOT in order to keep the requirements of the
      output file. (default None)
    dump_graphviz_video: Boolean indicating whether to dump the graph after
      every graph transformation. (default False)
    target_ops: Experimental flag, subject to change. Set of OpsSet options
      indicating which converter to use. (default set([OpsSet.TFLITE_BUILTINS]))
    conversion_summary_dir: A string, the path to the generated conversion logs.
    select_user_tf_ops: List of user's defined TensorFlow ops need to be
      supported in the TensorFlow Lite runtime. These ops will be supported as
      select TensorFlow ops.
    allow_all_select_tf_ops: If True, automatically add all TF ops (including
      custom TF ops) to the converted model as flex ops.
    enable_tflite_resource_variables: Experimental flag, subject to change.
      Enables conversion of resource variables. (default False)
    unfold_batchmatmul: Whether to unfold tf.BatchMatMul to a set of
      tfl.fully_connected ops. If not, translate to tfl.batch_matmul.
    lower_tensor_list_ops: Whether to lower tensor list ops to builtin ops. If
      not, use Flex tensor list ops.
    default_to_single_batch_in_tensor_list_ops: Whether to force to use batch
      size one when the tensor list ops has the unspecified batch size.
    accumulation_type: Data type of the accumulators in quantized inference.
      Typically used for float16 quantization and is either fp16 or fp32.
    allow_bfloat16: Whether the converted model supports reduced precision
      inference with the bfloat16 type.
    unfold_large_splat_constant: Whether to unfold large splat constant tensors
      in the flatbuffer model to reduce size.
    supported_backends: List of TFLite backends which needs to check
      compatibility.
    disable_per_channel_quantization: Disable per-channel quantized weights for
      dynamic range quantization. Only per-tensor quantization will be used.
    enable_mlir_dynamic_range_quantizer: Enable MLIR dynamic range quantization.
      If False, the old converter dynamic range quantizer is used.
    tf_quantization_mode: Indicates the mode of TF Quantization when the output
      model is used for TF Quantization.
    disable_infer_tensor_range: Disable infering tensor ranges.
    use_fake_quant_num_bits: Allow quantization parameters to be calculated from
      num_bits attribute.
    enable_dynamic_update_slice: Enable to convert to DynamicUpdateSlice op.
      (default: False).
    preserve_assert_op: Whether to preserve `TF::AssertOp` (default: False).
    guarantee_all_funcs_one_use: Whether to clone functions so that each
      function only has a single use. This option will be helpful if the
      conversion fails when the `PartitionedCall` or `StatefulPartitionedCall`
      can't be properly inlined (default: False).
    enable_mlir_variable_quantization: Enable MLIR variable quantization. There
      is a variable freezing pass, but some variables may not be fully frozen by
      it. This flag enables quantization of those residual variables in the MLIR
      graph.
    disable_fuse_mul_and_fc: Disable fusing input multiplication with
      fullyconnected operations. Useful when quantizing weights.
    quantization_options: Config to indicate quantization options of each
      components (ex: weight, bias, activation). This can be a preset method or
      a custom method, and allows finer, modular control. This option will
      override any other existing quantization flags. We plan on gradually
      migrating all quantization-related specs into this option.

  Returns:
    conversion_flags: protocol buffer describing the conversion process.
  Raises:
    ValueError, if the input tensor type is unknown.
  """
  conversion_flags = _conversion_flags_pb2.TocoFlags()
  conversion_flags.inference_type = convert_inference_tf_type_to_tflite_type(
      inference_type, usage="inference_type flag"
  )
  if inference_input_type:
    conversion_flags.inference_input_type = (
        convert_inference_tf_type_to_tflite_type(
            inference_input_type, usage="inference_input_type flag"
        )
    )
  else:
    conversion_flags.inference_input_type = conversion_flags.inference_type
  conversion_flags.input_format = input_format
  conversion_flags.output_format = output_format
  if default_ranges_stats:
    conversion_flags.default_ranges_min = default_ranges_stats[0]
    conversion_flags.default_ranges_max = default_ranges_stats[1]
  conversion_flags.drop_control_dependency = drop_control_dependency
  conversion_flags.reorder_across_fake_quant = reorder_across_fake_quant
  conversion_flags.allow_custom_ops = allow_custom_ops
  conversion_flags.post_training_quantize = post_training_quantize
  conversion_flags.quantize_to_float16 = quantize_to_float16
  if dump_graphviz_dir:
    conversion_flags.dump_graphviz_dir = dump_graphviz_dir
  conversion_flags.dump_graphviz_include_video = dump_graphviz_video
  if target_ops:
    if OpsSet.SELECT_TF_OPS in target_ops:
      conversion_flags.enable_select_tf_ops = True
    if set(target_ops) == {OpsSet.SELECT_TF_OPS}:
      conversion_flags.force_select_tf_ops = True
    if OpsSet.EXPERIMENTAL_STABLEHLO_OPS in target_ops:
      conversion_flags.convert_to_stablehlo = True
    if OpsSet.EXPERIMENTAL_STABLEHLO_OPS in target_ops and len(target_ops) > 1:
      raise ValueError(
          "StableHLO Ops set can not be specified with other Ops set together"
      )
  if conversion_summary_dir:
    conversion_flags.conversion_summary_dir = conversion_summary_dir
  if select_user_tf_ops:
    conversion_flags.select_user_tf_ops.extend(select_user_tf_ops)
  conversion_flags.allow_all_select_tf_ops = allow_all_select_tf_ops
  conversion_flags.enable_tflite_resource_variables = (
      enable_tflite_resource_variables
  )
  conversion_flags.unfold_batchmatmul = unfold_batchmatmul
  conversion_flags.lower_tensor_list_ops = lower_tensor_list_ops
  conversion_flags.default_to_single_batch_in_tensor_list_ops = (
      default_to_single_batch_in_tensor_list_ops
  )
  if accumulation_type:
    conversion_flags.accumulation_type = convert_tensor_tf_type_to_tflite_type(
        accumulation_type, usage="accumulation_type flag"
    )
  conversion_flags.allow_bfloat16 = allow_bfloat16
  conversion_flags.unfold_large_splat_constant = unfold_large_splat_constant
  if supported_backends:
    conversion_flags.supported_backends.extend(supported_backends)
  conversion_flags.disable_per_channel_quantization = (
      disable_per_channel_quantization
  )
  conversion_flags.enable_mlir_dynamic_range_quantizer = (
      enable_mlir_dynamic_range_quantizer
  )
  conversion_flags.enable_dynamic_update_slice = enable_dynamic_update_slice
  conversion_flags.preserve_assert_op = preserve_assert_op
  conversion_flags.guarantee_all_funcs_one_use = guarantee_all_funcs_one_use
  if tf_quantization_mode:
    conversion_flags.tf_quantization_mode = tf_quantization_mode
  conversion_flags.disable_infer_tensor_range = disable_infer_tensor_range
  conversion_flags.use_fake_quant_num_bits = use_fake_quant_num_bits
  conversion_flags.enable_mlir_variable_quantization = (
      enable_mlir_variable_quantization
  )
  conversion_flags.disable_fuse_mul_and_fc = disable_fuse_mul_and_fc
  if quantization_options:
    conversion_flags.quantization_options.CopyFrom(quantization_options)
  return conversion_flags


@convert_phase(
    Component.CONVERT_TF_TO_TFLITE_MODEL, SubComponent.CONVERT_GRAPHDEF
)
def convert_graphdef_with_arrays(
    input_data,
    input_arrays_with_shape,
    output_arrays,
    control_output_arrays,
    **kwargs
):
  """Convert a frozen GraphDef that can't be loaded in TF.

  Conversion can be customized by providing arguments that are forwarded to
  `build_model_flags` and `build_conversion_flags` (see documentation).

  Args:
    input_data: Input data (i.e. often `sess.graph_def`),
    input_arrays_with_shape: Tuple of strings representing input tensor names
      and list of integers representing input shapes (e.g., [("foo" : [1, 16,
      16, 3])]). Use only when graph cannot be loaded into TensorFlow and when
      `input_tensors` is None.
    output_arrays: List of output tensors to freeze graph with. Use only when
      graph cannot be loaded into TensorFlow and when `output_tensors` is None.
    control_output_arrays: Control output node names. This is used when
      converting a Graph with no output tensors. For example, if the graph's
      last operation is a Print op, just specify that op's name in this field.
      This can be used together with the `output_arrays` parameter.
    **kwargs: See `build_model_flags` and `build_conversion_flags`.

  Returns:
    The converted data. For example if TFLite was the destination, then
    this will be a tflite flatbuffer in a bytes array.

  Raises:
    Defined in `build_conversion_flags`.
  """
  model_flags = build_model_flags(**kwargs)
  conversion_flags = build_conversion_flags(**kwargs)
  enable_mlir_converter = kwargs.get("enable_mlir_converter", True)
  quantized_input_stats = kwargs.get("quantized_input_stats", None)

  for idx, (name, shape) in enumerate(input_arrays_with_shape):
    input_array = model_flags.input_arrays.add()
    if _is_quantized_input_stats_required(conversion_flags):
      if quantized_input_stats:
        input_array.mean_value, input_array.std_value = quantized_input_stats[
            idx
        ]
      else:
        raise ValueError(
            "The `quantized_input_stats` flag must be defined when either "
            "`inference_type` flag or `inference_input_type` flag is set to "
            "tf.int8 or tf.uint8."
        )
    input_array.name = name
    input_array.shape.dims.extend(list(map(int, shape)))

  if output_arrays:
    for name in output_arrays:
      model_flags.output_arrays.append(name)
  if control_output_arrays:
    for name in control_output_arrays:
      model_flags.control_output_arrays.append(name)

  data = convert(
      model_flags,
      conversion_flags,
      input_data.SerializeToString(),
      debug_info_str=None,
      enable_mlir_converter=enable_mlir_converter,
  )
  return data


@convert_phase(
    Component.CONVERT_TF_TO_TFLITE_MODEL, SubComponent.CONVERT_GRAPHDEF
)
def convert_graphdef(input_data, input_tensors, output_tensors, **kwargs):
  """Convert a frozen GraphDef model using the TF Lite converter.

  Conversion can be customized by providing arguments that are forwarded to
  `build_model_flags` and `build_conversion_flags` (see documentation).

  Args:
    input_data: Input data (i.e. often `sess.graph_def`),
   input_tensors: List of input tensors. Type and shape are computed using
     `foo.shape` and `foo.dtype`.
    output_tensors: List of output tensors (only .name is used from this).
    **kwargs: See `build_model_flags` and `build_conversion_flags`.

  Returns:
    The converted data. For example if TFLite was the destination, then
    this will be a tflite flatbuffer in a bytes array.

  Raises:
    Defined in `build_conversion_flags`.
  """
  model_flags = build_model_flags(**kwargs)
  conversion_flags = build_conversion_flags(**kwargs)
  saved_model_dir = kwargs.get("saved_model_dir", None)
  input_shapes = kwargs.get("input_shapes", None)
  enable_mlir_converter = kwargs.get("enable_mlir_converter", True)
  quantized_input_stats = kwargs.get("quantized_input_stats", None)
  debug_info = kwargs.get("debug_info", None)

  for idx, input_tensor in enumerate(input_tensors):
    input_array = model_flags.input_arrays.add()
    if saved_model_dir:
      input_array.name = input_tensor.name
    else:
      input_array.name = util.get_tensor_name(input_tensor)
    input_array.data_type = convert_tensor_tf_type_to_tflite_type(
        input_tensor.dtype, usage="input type of the TensorFlow model"
    )

    if _is_quantized_input_stats_required(conversion_flags):
      if quantized_input_stats:
        input_array.mean_value, input_array.std_value = quantized_input_stats[
            idx
        ]
      else:
        # We should ideally raise an error here, but we don't as it would break
        # several models/projects that depend on this workflow.
        warnings.warn(
            "Statistics for quantized inputs were expected, but not "
            "specified; continuing anyway."
        )

    if input_shapes is None:
      shape = input_tensor.shape
    else:
      shape = input_shapes[idx]

    if shape.rank is not None:
      # Create shapes with -1 for unknown dimensions.
      dims = []
      for dim in shape:
        if dim is None or (
            isinstance(dim, tensor_shape.Dimension) and dim.value is None
        ):
          dims.append(-1)
        else:
          dims.append(int(dim))
      input_array.shape.dims.extend(dims)
      input_array.shape.unknown_rank = False
    else:
      input_array.shape.unknown_rank = True

  for output_tensor in output_tensors:
    if saved_model_dir:
      model_flags.output_arrays.append(output_tensor.name)
    else:
      model_flags.output_arrays.append(util.get_tensor_name(output_tensor))

  data = convert(
      model_flags,
      conversion_flags,
      input_data.SerializeToString(),
      debug_info_str=debug_info.SerializeToString() if debug_info else None,
      enable_mlir_converter=enable_mlir_converter,
  )
  return data


@convert_phase(
    Component.CONVERT_TF_TO_TFLITE_MODEL, SubComponent.CONVERT_SAVED_MODEL
)
def convert_saved_model(**kwargs):
  """Converts a SavedModel using TF Lite converter."""
  model_flags = build_model_flags(**kwargs)
  conversion_flags = build_conversion_flags(**kwargs)
  data = convert(
      model_flags,
      conversion_flags,
      input_data_str=None,
      debug_info_str=None,
      enable_mlir_converter=True,
  )
  return data


@convert_phase(
    Component.CONVERT_TF_TO_TFLITE_MODEL, SubComponent.CONVERT_JAX_HLO
)
def convert_jax_hlo(input_content, input_names, is_proto_format, **kwargs):
  """Converts a Jax hlo-based model using TFLite converter."""
  model_flags = _model_flags_pb2.ModelFlags()
  model_flags.use_hlo_import = True
  if is_proto_format:
    model_flags.hlo_file_type = _model_flags_pb2.ModelFlags.HLO_PROTO
  else:
    model_flags.hlo_file_type = _model_flags_pb2.ModelFlags.HLO_TEXT

  # Build input names.
  for input_name in input_names:
    input_array = model_flags.input_arrays.add()
    input_array.name = input_name

  conversion_flags = build_conversion_flags(**kwargs)
  data = convert(
      model_flags,
      conversion_flags,
      input_data_str=input_content,
      debug_info_str=None,
      enable_mlir_converter=True,
  )
  return data


@_tf_export(v1=["lite.toco_convert"])
@deprecation.deprecated(None, "Use `lite.TFLiteConverter` instead.")
def toco_convert(input_data, input_tensors, output_tensors, *args, **kwargs):
  """Convert a TensorFlow GraphDef to TFLite.

  This function is deprecated. Please use `tf.lite.TFLiteConverter` API instead.
  Conversion can be customized by providing arguments that are forwarded to
  `build_model_flags` and `build_conversion_flags` (see documentation for
  details).
  Args:
    input_data: Input data (i.e. often `sess.graph_def`).
    input_tensors: List of input tensors. Type and shape are computed using
      `foo.shape` and `foo.dtype`.
    output_tensors: List of output tensors (only .name is used from this).
    *args: See `build_model_flags` and `build_conversion_flags`.
    **kwargs: See `build_model_flags` and `build_conversion_flags`.

  Returns:
    The converted TensorFlow Lite model in a bytes array.

  Raises:
    Defined in `convert`.
  """
  kwargs["enable_mlir_converter"] = kwargs.get("enable_mlir_converter", False)
  return convert_graphdef(
      input_data, input_tensors, output_tensors, *args, **kwargs
  )


def deduplicate_readonly_buffers(tflite_model):
  """Generates a new model byte array after deduplicating readonly buffers.

  This function should be invoked after the model optimization toolkit. The
  model optimization toolkit assumes that each tensor object owns its each
  buffer separately.

  Args:
    tflite_model: TFLite flatbuffer in a byte array to be deduplicated.

  Returns:
    TFLite flatbuffer in a bytes array, processed with the deduplication method.
  """
  # Load TFLite Flatbuffer byte array into an object.
  model = flatbuffer_utils.convert_bytearray_to_object(tflite_model)

  # Get all the read-only buffers, which can be modified without causing any
  # issue in the graph invocation stage.
  read_only_buffer_indices = set()
  for subgraph in model.subgraphs:
    # To get all the read-only buffers:
    # (1) Get all read-only input tensors.
    # (2) Discard intermediate or output tensors.
    # (3) Discard the subgraph's input/output tensors.
    # (4) Gather the buffers of the read-only input tensors.

    # (1) Get read-only input tensors.
    read_only_input_tensor_indices = set()
    for op in subgraph.operators:
      if op.inputs is None:
        continue
      for i, input_tensor_idx in enumerate(op.inputs):
        # Ignore mutable tensors.
        if op.mutatingVariableInputs is not None:
          # Ignore invalid tensors.
          if (
              i < len(op.mutatingVariableInputs)
              and op.mutatingVariableInputs[i]
          ):
            continue
        # Ignore variable tensors.
        if subgraph.tensors[input_tensor_idx].isVariable:
          continue
        read_only_input_tensor_indices.add(input_tensor_idx)

    # (2) Discard intermediate or output tensors.
    for op in subgraph.operators:
      if op.outputs is not None:
        for output_tensor_idx in op.outputs:
          read_only_input_tensor_indices.discard(output_tensor_idx)
      if op.intermediates is not None:
        for intermediate_tensor_idx in op.intermediates:
          read_only_input_tensor_indices.discard(intermediate_tensor_idx)

    # (3) Discard the subgraph's input and output tensors.
    if subgraph.inputs is not None:
      for input_tensor_idx in subgraph.inputs:
        read_only_input_tensor_indices.discard(input_tensor_idx)
    if subgraph.outputs is not None:
      for output_tensor_idx in subgraph.outputs:
        read_only_input_tensor_indices.discard(output_tensor_idx)

    # (4) Gather the buffers of the read-only input tensors.
    for tensor_idx in read_only_input_tensor_indices:
      read_only_buffer_indices.add(subgraph.tensors[tensor_idx].buffer)

  # Ignore invalid negative index or zero-sized buffers.
  for buffer_idx in read_only_buffer_indices.copy():
    if buffer_idx < 0 or (
        model.buffers[buffer_idx].data is None
        or isinstance(model.buffers[buffer_idx].data, list)
        or model.buffers[buffer_idx].data.size == 0
    ):
      read_only_buffer_indices.discard(buffer_idx)

  class BufferIndex:
    """A class to store index, size, hash of the buffers in TFLite model."""

    def __init__(self, idx, size, hash_value):
      self.idx = idx
      self.size = size
      self.hash_value = hash_value

  read_only_buffers = list(
      map(
          lambda index: BufferIndex(  # pylint: disable=g-long-lambda
              index,
              model.buffers[index].data.size,
              hashlib.md5(model.buffers[index].data.data.tobytes()).hexdigest(),
          ),
          read_only_buffer_indices,
      )
  )

  # Sort read_only_buffers by buffer size & hash in descending order.
  read_only_buffers = sorted(
      read_only_buffers,
      key=lambda buffer: (buffer.size, buffer.hash_value),
      reverse=True,
  )

  # Create a map of duplicate buffers (same size and same type).
  # eg: In [1, 2, 3, 4, 5, 6] if (1, 4, 6) and (2, 5) are each, groups of buffer
  # indices of the same size and type, then the map would be {4:1, 6:1, 5:2}
  duplicate_buffer_map = {}
  for i, buffer_i in enumerate(read_only_buffers):
    # This buffer is a duplicate.
    if buffer_i.idx in duplicate_buffer_map:
      continue
    # This buffer is unique. Scan rest of the list to find duplicates
    # of this buffer and mark them accordingly.
    for buffer_j in read_only_buffers[i + 1 :]:
      if buffer_j.idx in duplicate_buffer_map:
        continue
      if buffer_i.size != buffer_j.size:
        break
      if buffer_i.hash_value != buffer_j.hash_value:
        continue
      # Found duplicate. Nullify j-th buffer and use i-th buffer instead.
      duplicate_buffer_map[buffer_j.idx] = buffer_i.idx

  # Make the duplicated tensors use the single shared buffer index.
  for subgraph in model.subgraphs:
    for op in subgraph.operators:
      if op.inputs is None:
        continue
      for input_tensor in op.inputs:
        buffer_idx = subgraph.tensors[input_tensor].buffer
        if buffer_idx in duplicate_buffer_map:
          subgraph.tensors[input_tensor].buffer = duplicate_buffer_map[
              buffer_idx
          ]

  # Nullify the unused buffers.
  for idx in duplicate_buffer_map:
    model.buffers[idx].data = None

  # Return a TFLite flatbuffer as a byte array.
  return flatbuffer_utils.convert_object_to_bytearray(model)
