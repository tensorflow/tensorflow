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
"""Defines TF Quantization API from SavedModel to SavedModel."""

import tempfile
from typing import Mapping, Optional

from absl import logging

from tensorflow.compiler.mlir.quantization.tensorflow import quantization_options_pb2 as quant_opts_pb2
from tensorflow.compiler.mlir.quantization.tensorflow.python import py_function_lib
from tensorflow.compiler.mlir.quantization.tensorflow.python import pywrap_quantize_model
from tensorflow.compiler.mlir.quantization.tensorflow.python import representative_dataset as repr_dataset
from tensorflow.compiler.mlir.quantization.tensorflow.python import save_model
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.lib.io import file_io
from tensorflow.python.saved_model import load as saved_model_load
from tensorflow.python.saved_model import loader_impl as saved_model_loader
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.trackable import autotrackable
from tensorflow.python.util import tf_export

# Type aliases for quant_opts_pb2 messages.
_QuantizationOptions = tf_export.tf_export(
    'quantization.experimental.QuantizationOptions'
)(quant_opts_pb2.QuantizationOptions)

_QuantizationMethod = tf_export.tf_export(
    'quantization.experimental.QuantizationMethod'
)(quant_opts_pb2.QuantizationMethod)

_QuantizationComponentSpec = tf_export.tf_export(
    'quantization.experimental.QuantizationComponentSpec'
)(quant_opts_pb2.QuantizationComponentSpec)

_UnitWiseQuantizationSpec = tf_export.tf_export(
    'quantization.experimental.UnitWiseQuantizationSpec'
)(quant_opts_pb2.UnitWiseQuantizationSpec)

_PresetMethod = _QuantizationMethod.PresetMethod
_CalibrationMethod = quant_opts_pb2.CalibrationOptions.CalibrationMethod

_QuantizationComponent = _QuantizationComponentSpec.QuantizationComponent
_TensorType = _QuantizationComponentSpec.TensorType

# Mapping of signature def key -> SignatureDef.
_SignatureDefMap = Mapping[str, meta_graph_pb2.SignatureDef]

# Default minimum number of elements in the weights for them to be quantized
# during dynamic range quantization (DRQ) and weight-only quantization.
_DYNAMIC_RANGE_DEFAULT_MIN_NUM_ELEMENTS_FOR_WEIGHTS = 1024


def _is_qat_saved_model(saved_model_path: str):
  """Checks if the SavedModel is QAT-enabled by looking for 'FakeQuant' ops."""
  saved_model_proto = saved_model_loader.parse_saved_model(saved_model_path)
  for meta_graph in saved_model_proto.meta_graphs:
    if any(
        node.op.startswith('FakeQuant') for node in meta_graph.graph_def.node
    ):
      return True
    for function in meta_graph.graph_def.library.function:
      if any(node.op.startswith('FakeQuant') for node in function.node_def):
        return True
  return False


def _serialize_signature_def_map(
    signature_def_map: _SignatureDefMap,
) -> dict[str, bytes]:
  """Serializes SignatureDef values in `signature_def_map`.

  Args:
    signature_def_map: Signature key -> SignatureDef mapping.

  Returns:
    Signature def map where the values (`SignatureDef`) are serialized.
  """
  signature_def_map_serialized = {}
  for key, signature_def in signature_def_map.items():
    signature_def_map_serialized[key] = signature_def.SerializeToString()

  return signature_def_map_serialized


def _run_static_range_qat(
    src_saved_model_path: str,
    dst_saved_model_path: str,
    quant_opts: _QuantizationOptions,
    signature_def_map: _SignatureDefMap,
) -> None:
  """Runs static-range quantization for a Quantization-Aware Trained model.

  Runs the quantization for a model trained using QAT.

  Args:
    src_saved_model_path: Path to the source SavedModel directory.
    dst_saved_model_path: Path to the destination SavedModel directory.
    quant_opts: Quantization options.
    signature_def_map: Signature def key -> SignatureDef mapping.
  """
  logging.info('Running static-range quantization for QAT model.')

  loader = saved_model_loader.SavedModelLoader(src_saved_model_path)
  function_aliases = loader.get_meta_graph_def_from_tags(
      quant_opts.tags
  ).meta_info_def.function_aliases

  pywrap_quantize_model.quantize_qat_model(
      src_saved_model_path,
      dst_saved_model_path,
      quantization_options_serialized=quant_opts.SerializeToString(),
      signature_keys=list(quant_opts.signature_keys),
      signature_def_map_serialized=_serialize_signature_def_map(
          signature_def_map
      ),
      function_aliases=dict(function_aliases),
      py_function_library=py_function_lib.PyFunctionLibrary(),
  )


def _run_static_range_ptq(
    src_saved_model_path: str,
    dst_saved_model_path: str,
    quant_opts: _QuantizationOptions,
    representative_dataset: repr_dataset.RepresentativeDatasetOrMapping,
    signature_def_map: _SignatureDefMap,
) -> None:
  """Runs static-range Post-Training Quantization.

  Runs static-range PTQ for the model. Runs the calibration step with
  `representative_dataset` to collect statistics required for quantization. This
  produces the quantized GraphDef along with the SignatureDefs which might have
  been modified according to the changes in the graph.

  Args:
    src_saved_model_path: Path to the source SavedModel directory.
    dst_saved_model_path: Path to the destination SavedModel directory.
    quant_opts: Quantization options.
    representative_dataset: Representative dataset used for the calibration
      step. Representative datasets should exist for each signature def key in
      `signature_def_keys`.
    signature_def_map: Signature def key -> SignatureDef mapping.

  Raises:
    ValueError if the graph doesn't contain a valid signature.
  """
  logging.info('Running static-range post-training quantization.')

  loader = saved_model_loader.SavedModelLoader(src_saved_model_path)
  function_aliases = loader.get_meta_graph_def_from_tags(
      quant_opts.tags
  ).meta_info_def.function_aliases

  signature_def_map_serialized = _serialize_signature_def_map(signature_def_map)
  pywrap_quantize_model.quantize_ptq_static_range(
      src_saved_model_path,
      dst_saved_model_path,
      quantization_options_serialized=quant_opts.SerializeToString(),
      signature_keys=list(quant_opts.signature_keys),
      signature_def_map_serialized=signature_def_map_serialized,
      function_aliases=dict(function_aliases),
      py_function_library=py_function_lib.PyFunctionLibrary(),
      representative_dataset=representative_dataset,
  )


def _static_range_quantize(
    src_saved_model_path: str,
    dst_saved_model_path: str,
    quantization_options: _QuantizationOptions,
    representative_dataset: Optional[
        repr_dataset.RepresentativeDatasetOrMapping
    ] = None,
) -> autotrackable.AutoTrackable:
  """Quantizes the given SavedModel via static range quantization.

  If the model is not trained with Quantization-Aware Training (QAT) technique,
  it requires `representative_dataset` to collect statistics required for
  quantization. If non-None `representative_dataset` is provided with a QAT
  model input, `representative_dataset` will be ignored.

  Args:
    src_saved_model_path: Path to the saved model. When representative_dataset
      is not provided, this should be a model trained with QAT.
    dst_saved_model_path: The path to save the output SavedModel. The directory
      will be overwritten if not empty.
    quantization_options: QuantizationOptions proto describing quantization
      related config.
    representative_dataset: a generator that returns a dictionary in {input_key:
      input_value} format or a tuple with signature key and a dictionary in
      {input_key: input_value} format that feeds calibration data for quantizing
      model. This should be provided when the model is not a QAT model.

  Returns:
    A SavedModel object with TF quantization applied.

  Raises:
    ValueError: when representative_dataset is not provided for non-QAT model.
    RuntimeError: When a MetaGraphDef could not be found associated with `tags`
      in the SavedModel.
  """
  logging.info(
      'Running static range quantization on model: %s', src_saved_model_path
  )
  logging.info('QuantizationOptions: \n%s', quantization_options)

  is_qat_saved_model_or_method_no_quantize = _is_qat_saved_model(
      src_saved_model_path
  ) or (
      quantization_options.quantization_method.preset_method
      == _QuantizationMethod.METHOD_NO_QUANTIZE
  )
  signature_def_map = save_model.get_signatures_from_saved_model(
      src_saved_model_path,
      quantization_options.signature_keys,
      set(quantization_options.tags),
  )

  # Checks if the model is from QAT or method is METHOD_NO_QUANTIZE.
  if (
      representative_dataset is None
      and not is_qat_saved_model_or_method_no_quantize
  ):
    raise ValueError(
        'When `representative_dataset` is not provided, the model should be '
        'trained with quantization-aware training (QAT).'
    )
  if quantization_options.min_num_elements_for_weights > 0:
    logging.warn(
        'min_num_elements_for_weights is set but is not supported for the '
        'Post-training static range quantization. '
        'The flag is ignored.'
    )

  if is_qat_saved_model_or_method_no_quantize:
    _run_static_range_qat(
        src_saved_model_path,
        dst_saved_model_path,
        quantization_options,
        signature_def_map,
    )
  else:
    _run_static_range_ptq(
        src_saved_model_path,
        dst_saved_model_path,
        quantization_options,
        representative_dataset,
        signature_def_map,
    )

  return saved_model_load.load(dst_saved_model_path)


def _dynamic_range_quantize(
    src_saved_model_path: str,
    dst_saved_model_path: str,
    quantization_options: _QuantizationOptions,
) -> autotrackable.AutoTrackable:
  """Quantizes the given SavedModel via post-training dynamic range quantization.

  Args:
    src_saved_model_path: Path to the saved model.
    dst_saved_model_path: The path to save the output SavedModel. The directory
      will be overwritten if not empty.
    quantization_options: QuantizationOptions proto describing quantization
      related config.

  Returns:
    A SavedModel object with TF quantization applied.

  Raises:
    ValueError: when the model is QAT model.
  """
  mode_str = 'dynamic-range quantization'
  if _is_qat_saved_model(src_saved_model_path):
    raise ValueError(
        'The models trained with quantization-aware training (QAT) is not '
        'supported for %s.' % mode_str
    )

  logging.info(
      'Running post-training %s on model: %s', mode_str, src_saved_model_path
  )
  logging.info('QuantizationOptions: \n%s', quantization_options)

  loader = saved_model_loader.SavedModelLoader(src_saved_model_path)

  function_aliases = loader.get_meta_graph_def_from_tags(
      quantization_options.tags
  ).meta_info_def.function_aliases

  signature_def_map = save_model.get_signatures_from_saved_model(
      src_saved_model_path,
      quantization_options.signature_keys,
      quantization_options.tags,
  )

  # Apply post-training dynamic range quantization to the model.
  pywrap_quantize_model.quantize_ptq_dynamic_range(
      src_saved_model_path,
      dst_saved_model_path,
      quantization_options_serialized=quantization_options.SerializeToString(),
      signature_keys=list(quantization_options.signature_keys),
      signature_def_map_serialized=_serialize_signature_def_map(
          signature_def_map
      ),
      function_aliases=dict(function_aliases),
      py_function_library=py_function_lib.PyFunctionLibrary(),
  )

  return saved_model_load.load(dst_saved_model_path)


def _weight_only_quantize(
    src_saved_model_path: str,
    dst_saved_model_path: str,
    quantization_options: quant_opts_pb2.QuantizationOptions,
) -> autotrackable.AutoTrackable:
  """Quantizes the given SavedModel via weight-only quantization.

  Args:
    src_saved_model_path: Path to the saved model.
    dst_saved_model_path: The path to save the output SavedModel. The directory
      will be overwritten if not empty.
    quantization_options: QuantizationOptions proto describing quantization
      related config.

  Returns:
    A SavedModel object with TF quantization applied.

  Raises:
    ValueError: when the model is QAT model.
  """
  mode_str = 'weight-only quantization'

  # QAT weight-only is not supported yet.
  if _is_qat_saved_model(src_saved_model_path):
    raise ValueError(
        'The models trained with quantization-aware training (QAT) is not '
        'supported for %s.' % mode_str
    )

  logging.info(
      'Running post-training %s on model: %s', mode_str, src_saved_model_path
  )
  logging.info('QuantizationOptions: \n%s', quantization_options)

  loader = saved_model_loader.SavedModelLoader(src_saved_model_path)

  function_aliases = loader.get_meta_graph_def_from_tags(
      quantization_options.tags
  ).meta_info_def.function_aliases

  signature_def_map = save_model.get_signatures_from_saved_model(
      src_saved_model_path,
      list(quantization_options.signature_keys),
      set(quantization_options.tags),
  )

  pywrap_quantize_model.quantize_weight_only(
      src_saved_model_path,
      dst_saved_model_path,
      quantization_options_serialized=quantization_options.SerializeToString(),
      signature_def_map_serialized=_serialize_signature_def_map(
          signature_def_map
      ),
      function_aliases=dict(function_aliases),
      py_function_library=py_function_lib.PyFunctionLibrary(),
  )

  return saved_model_load.load(dst_saved_model_path)


def _verify_output_dir(output_dir: Optional[str], overwrite: bool) -> None:
  """Verifies the output directory.

  Raises an error if `output_dir` is not suitable for writing the output saved
  model.

  Args:
    output_dir: Output directory.
    overwrite: An option allowing to overwrite the existing output directory if
      set to true. Does not actually create or modify the `output_dir` in this
      function.

  Raises:
    FileExistsError: Iff `output_dir` is not empty and `overwrite` is false.
  """
  dir_not_empty = (
      output_dir is not None
      and file_io.file_exists_v2(output_dir)
      and file_io.list_directory_v2(output_dir)
  )

  if dir_not_empty and not overwrite:
    raise FileExistsError(
        f'Output directory already exists: {output_dir} . '
        'Please set overwrite_output_directory to true to '
        'overwrite the existing directory.'
    )


def _populate_quantization_component_spec(
    quant_method: _QuantizationMethod,
) -> None:
  """Populates default values for QuantizationComponentSpec.

  Args:
    quant_method: The quantization method to be updated.
  """
  # Make sure creating one spec per component.
  updated_component_spec = dict()

  # Populate default configuration.
  if (
      quant_method.preset_method == _PresetMethod.METHOD_STATIC_RANGE_INT8
      or quant_method.preset_method == _PresetMethod.METHOD_DYNAMIC_RANGE_INT8
  ):
    updated_component_spec[_QuantizationComponent.COMPONENT_ACTIVATION] = (
        _QuantizationComponentSpec(
            quantization_component=_QuantizationComponent.COMPONENT_ACTIVATION,
            tensor_type=_TensorType.TENSORTYPE_INT_8,
        )
    )
    updated_component_spec[_QuantizationComponent.COMPONENT_WEIGHT] = (
        _QuantizationComponentSpec(
            quantization_component=_QuantizationComponent.COMPONENT_WEIGHT,
            tensor_type=_TensorType.TENSORTYPE_INT_8,
        )
    )
    updated_component_spec[_QuantizationComponent.COMPONENT_BIAS] = (
        _QuantizationComponentSpec(
            quantization_component=_QuantizationComponent.COMPONENT_BIAS,
            tensor_type=_TensorType.TENSORTYPE_INT_32,
        )
    )
  elif (
      quant_method.preset_method
      == _PresetMethod.METHOD_STATIC_RANGE_WEIGHT_ONLY_INT8
  ):
    updated_component_spec[_QuantizationComponent.COMPONENT_WEIGHT] = (
        _QuantizationComponentSpec(
            quantization_component=_QuantizationComponent.COMPONENT_WEIGHT,
            tensor_type=_TensorType.TENSORTYPE_INT_8,
        )
    )

  # Override if quantization_component_spec is specified.
  if quant_method.quantization_component_specs:
    # Check if the component spec is supported configuration in TF-Quant.
    for component_spec in quant_method.quantization_component_specs:
      if component_spec.quantization_component in [
          _QuantizationComponent.COMPONENT_WEIGHT,
          _QuantizationComponent.COMPONENT_ACTIVATION,
      ]:
        if component_spec.tensor_type != _TensorType.TENSORTYPE_INT_8:
          raise ValueError(
              'Only int8 precision is supported for input operands.'
          )
      else:
        if component_spec.tensor_type != _TensorType.TENSORTYPE_INT_32:
          raise ValueError('Only int32 precision is supported for bias.')
      # Update with the custom spec.
      updated_component_spec[component_spec.quantization_component] = (
          component_spec
      )

  # Update the componet spec
  del quant_method.quantization_component_specs[:]
  quant_method.quantization_component_specs.extend(
      updated_component_spec.values()
  )

  if (
      quant_method.preset_method == _PresetMethod.METHOD_STATIC_RANGE_INT8
      or quant_method.preset_method == _PresetMethod.METHOD_DYNAMIC_RANGE_INT8
  ) and (len(quant_method.quantization_component_specs) != 3):
    raise ValueError('Only 3 components are needed for', quant_method)
  elif (
      quant_method.preset_method
      == _PresetMethod.METHOD_STATIC_RANGE_WEIGHT_ONLY_INT8
  ) and len(quant_method.quantization_component_specs) != 1:
    raise ValueError('At least one component spec needs to be specified.')


def _populate_unitwise_quantization_specs(
    quantization_options: _QuantizationOptions,
) -> None:
  """Verifies and pupulates unitwise quantization specs."""
  if not quantization_options.unit_wise_quantization_specs:
    return

  sorted_top_level_component_specs = sorted(
      quantization_options.quantization_method.quantization_component_specs,
      key=lambda x: x.quantization_component,
  )

  for unitwise_spec in quantization_options.unit_wise_quantization_specs:
    if not unitwise_spec.unit:
      raise ValueError(
          'UnitWiseQuantizationSpec must contain at least one unit.'
      )

    for unit in unitwise_spec.unit:
      if not unit.op_type and not unit.node_name:
        raise ValueError('Either `op_type` or `node_name` must be specified.')

    _populate_quantization_component_spec(unitwise_spec.quantization_method)

    component_specs = (
        unitwise_spec.quantization_method.quantization_component_specs
    )
    if component_specs and (
        sorted_top_level_component_specs
        != sorted(component_specs, key=lambda x: x.quantization_component)
    ):
      raise ValueError(
          'Currently unit-wise quantization spec only supports NO_QUANTIZE and'
          ' same quantization method as the top-level `quantization_method`'
      )


def _populate_calibration_options(
    quantization_options: quant_opts_pb2.QuantizationOptions,
):
  """Populates default values for CalibrationOptions.

  Args:
    quantization_options: An instance of QuantizationOptions with a field
      specifying CalibrationOptions
  """

  calib_opts = quantization_options.calibration_options
  if (
      calib_opts.calibration_method
      == _CalibrationMethod.CALIBRATION_METHOD_UNSPECIFIED
  ):
    calib_opts.calibration_method = (
        _CalibrationMethod.CALIBRATION_METHOD_MIN_MAX
    )
  elif (
      calib_opts.calibration_method
      == _CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_PERCENTILE
  ):
    if not calib_opts.calibration_parameters.initial_num_bins:
      calib_opts.calibration_parameters.initial_num_bins = 256
    if not calib_opts.calibration_parameters.min_percentile:
      calib_opts.calibration_parameters.min_percentile = 0.001
    if not calib_opts.calibration_parameters.max_percentile:
      calib_opts.calibration_parameters.max_percentile = 99.999
  # Check the activation_tensor_type of HISTOGRAM_MSE methods.
  elif calib_opts.calibration_method in [
      _CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_MSE_BRUTEFORCE,
      _CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_MSE_MAX_FREQUENCY,
      _CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_MSE_SYMMETRIC,
  ]:
    activation_tensor_type = (
        quantization_options.quantization_method.quantization_component_specs[
            _QuantizationComponent.COMPONENT_ACTIVATION
        ].tensor_type
    )
    # Unlike the HISTOGRAM_PERCENTILE method, the HISTOGRAM_MSE method uses
    # num_bits because it actually quantizes and dequantizes values.
    if activation_tensor_type != _TensorType.TENSORTYPE_INT_8:
      raise ValueError(
          'Only TENSORTYPE_INT_8 is supported for HISTOGRAM_MSE calibration'
          f' methods. calibration_method={calib_opts.calibration_method}'
      )

    if not calib_opts.calibration_parameters.initial_num_bins:
      calib_opts.calibration_parameters.initial_num_bins = 256


def _populate_quantization_options_default_values(
    quantization_options: _QuantizationOptions,
) -> None:
  """Populates default values for QuantizationOptions.

  Populates unspecified or unset fields of QuantizationOptions with the default
  values.

  * If `op_set` is unspecified, it defaults to `OpSet.XLA`.
  * If `freeze_all_variables` is not set, it defaults to `True`.
  * Check if configurations are set correctly:
    - Per-channel quantization is supported for Uniform Quantized opset only.

  Args:
    quantization_options: An instance of QuantizationOptions.
  """
  if quantization_options.op_set == quant_opts_pb2.OpSet.OP_SET_UNSPECIFIED:
    quantization_options.op_set = quant_opts_pb2.OpSet.XLA

  if not quantization_options.tags:
    quantization_options.tags.append(tag_constants.SERVING)

  if not quantization_options.signature_keys:
    quantization_options.signature_keys.append(
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    )

  if not quantization_options.HasField('freeze_all_variables'):
    quantization_options.freeze_all_variables = True

  if quantization_options.enable_legacy_weight_only:
    raise ValueError(
        'Legacy weight-only is deprecated. Use weight-only quantization method.'
    )

  # Converter assumes options are specified. So set SRQ explicitly.
  if (
      quantization_options.quantization_method.preset_method
      == _PresetMethod.METHOD_UNSPECIFIED
  ):
    logging.debug(
        '"preset_method" for QuantizationMethod is not specified.'
        'Static range quantization is used by default.'
    )
    quantization_options.quantization_method.preset_method = (
        _PresetMethod.METHOD_STATIC_RANGE_INT8
    )

  # Check default quantization option values for weight-only quantization.
  # TODO(b/242805842): Find good minimum_elements_for_weights number for server.
  # please also update default value in tflite converter:
  # tensorflow/compiler/mlir/lite/tf_to_tfl_flatbuffer.cc;l=201
  if (
      quantization_options.quantization_method.preset_method
      == _PresetMethod.METHOD_STATIC_RANGE_WEIGHT_ONLY_INT8
  ) or (
      quantization_options.quantization_method.preset_method
      == _PresetMethod.METHOD_DYNAMIC_RANGE_INT8
  ):
    if quantization_options.min_num_elements_for_weights == 0:
      quantization_options.min_num_elements_for_weights = (
          _DYNAMIC_RANGE_DEFAULT_MIN_NUM_ELEMENTS_FOR_WEIGHTS
      )
      logging.warning(
          (
              'QuantizationOptions.min_num_elements_for_weights is not set (0).'
              ' Setting to the default value: %d.'
          ),
          _DYNAMIC_RANGE_DEFAULT_MIN_NUM_ELEMENTS_FOR_WEIGHTS,
      )

  # TODO: b/307900054 - Set the per-channel quantization by default.
  if quantization_options.enable_per_channel_quantization and not (
      (
          quantization_options.op_set == quant_opts_pb2.OpSet.UNIFORM_QUANTIZED
          or quantization_options.quantization_method.preset_method
          == _PresetMethod.METHOD_STATIC_RANGE_WEIGHT_ONLY_INT8
      )
      or (
          quantization_options.op_set == quant_opts_pb2.OpSet.XLA
          and quantization_options.quantization_method.preset_method
          == _PresetMethod.METHOD_STATIC_RANGE_INT8
      )
  ):
    raise ValueError(
        'Currently, per-channel quantization is supported for Uniform Quantized'
        ' opset, weight only quantization, or XLA opset with static range'
        ' quantization.'
    )

  if (
      quantization_options.quantization_method.preset_method
      == _PresetMethod.METHOD_STATIC_RANGE_WEIGHT_ONLY_INT8
      and (
          quantization_options.op_set == quant_opts_pb2.OpSet.UNIFORM_QUANTIZED
          or quantization_options.op_set == quant_opts_pb2.OpSet.TF
      )
  ):
    raise ValueError('TF/Uniform quantized opset does not support weight-only.')

  if (quantization_options.op_set == quant_opts_pb2.OpSet.STABLEHLO) and (
      quantization_options.quantization_method.preset_method
      != _PresetMethod.METHOD_STATIC_RANGE_INT8
  ):
    raise ValueError(
        'StableHLO quantized opset currently only supports static range'
        ' quantization via TF Quantizer.'
    )

  if quantization_options.HasField('debugger_options'):
    # Set `force_graph_mode_calibration` to True to avoid skipping op execution,
    # which are not connected to return ops, during calibration execution.
    # Setting `force_graph_mode_calibration` to True enables execution of the
    # model in graph mode (not eager mode).
    logging.debug(
        'Setting `force_graph_mode_calibration = True` to ensure the debugging '
        'model is executed in graph mode during calibration, rather than eager '
        'mode.'
    )
    quantization_options.force_graph_mode_calibration = True

    if not quantization_options.debugger_options.log_dir_path:
      quantization_options.debugger_options.log_dir_path = '/tmp/dumps'

    if (
        quantization_options.debugger_options.debugger_type
        == quant_opts_pb2.DebuggerOptions.DebuggerType.DEBUGGER_TYPE_UNSPECIFIED
    ):
      raise ValueError(
          'Debugger is enabled but debugger type was not specified.'
      )

    if (
        quantization_options.debugger_options.debugger_type
        == quant_opts_pb2.DebuggerOptions.DebuggerType.DEBUGGER_TYPE_WHOLE_MODEL
        and not quantization_options.debugger_options.unquantized_dump_model_path
    ):
      raise ValueError(
          'Debugger type whole model verify was used but'
          ' unquantized_dump_model_path was not specified.'
      )

  # Check and populate quantization component spec.
  _populate_quantization_component_spec(
      quantization_options.quantization_method
  )
  # Verify and populate unit-wise quantization specs.
  _populate_unitwise_quantization_specs(quantization_options)

  if (
      quantization_options.quantization_method.preset_method
      == _PresetMethod.METHOD_STATIC_RANGE_INT8
  ):
    # Check and populate calibration options.
    _populate_calibration_options(quantization_options)


@tf_export.tf_export('quantization.experimental.quantize_saved_model')
def quantize(
    saved_model_path: str,
    output_directory: Optional[str] = None,
    quantization_options: Optional[_QuantizationOptions] = None,
    representative_dataset: Optional[
        repr_dataset.RepresentativeDatasetOrMapping
    ] = None,
    *,
    overwrite_output_directory: bool = False,
) -> autotrackable.AutoTrackable:
  """Quantizes the SavedModel with the given quantization options.

  Example usage:
  ```python
  # Quantizing a model trained with QAT.
  quantization_options = tf.quantization.experimental.QuantizationOptions(
      signature_keys=['your_signature_key'],
  )
  tf.quantization.experimental.quantize_saved_model(
      '/tmp/input_model',
      '/tmp/output_model',
      quantization_options=quantization_options,
  )

  # When quantizing a model trained without QAT (Post-Training Quantization),
  # a representative dataset is required.
  representative_dataset = [{"input": tf.random.uniform(shape=(3, 3))}
                        for _ in range(256)]
  tf.quantization.experimental.quantize_saved_model(
      '/tmp/input_model',
      '/tmp/output_model',
      quantization_options=quantization_options,
      representative_dataset={'your_signature_key': representative_dataset},
    )

  # In addition to preset quantization methods, fine-grained control of
  # quantization for each component is also supported.
  _QuantizationComponentSpec = (
      tf.quantization.experimental.QuantizationComponentSpec
  )
  quantization_options = tf.quantization.experimental.QuantizationOptions(
      signature_keys=['your_signature_key'],
      quantization_method=tf.quantization.experimental.QuantizationMethod(
          quantization_component_specs=[
              _QuantizationComponentSpec(
                  quantization_component=(
                      _QuantizationComponentSpec.COMPONENT_ACTIVATION
                  ),
                  tensor_type=_QuantizationComponentSpec.TENSORTYPE_INT_8,
              )
          ]
      )
  )
  tf.quantization.experimental.quantize_saved_model(
      '/tmp/input_model',
      '/tmp/output_model',
      quantization_options=quantization_options,
  )
  ```

  Args:
    saved_model_path: Path to the saved model. When representative_dataset is
      not provided, this should be a model trained with QAT.
    output_directory: The path to save the output SavedModel. Set
      `overwrite_output_directory` to `True` to overwrite any existing contents
      in the directory if not empty.
    quantization_options: A set of options for quantization. If None, it uses
      post-training static range quantization with XLA opset by default.
    representative_dataset: an iterator that returns a dictionary of {input_key:
      input_value} or a map from signature key to a dictionary of {input_key:
      input_value} that feeds calibration data for quantizing model. The
      representative should be provided when the model is a PTQ model. It can be
      provided either via this parameter or via the `representative_datasets`
      field in `QuantizationOptions`.
    overwrite_output_directory: If set to true, overwrites the output directory
      iff it isn't empty. The default value is false.

  Returns:
    A SavedModel object with TF quantization applied, or None if no quantization
    is performed.

  Raises:
    ValueError: When 1) representative_dataset is not provided for non QAT model
      for enabling static range quantization, 2) invalid value is provided as
      a quantization method, or 3) provide representative dataset via both
      argument and QuantizationOptions.
    ValueError: When the specified quantization method is not yet supported.
  """
  _verify_output_dir(output_directory, overwrite_output_directory)

  # Set default values for None arguments.
  if output_directory is None:
    output_directory = tempfile.mkdtemp()

  if quantization_options is None:
    quantization_options = _QuantizationOptions()

  _populate_quantization_options_default_values(quantization_options)

  if (
      representative_dataset is not None
      and quantization_options.representative_datasets
  ):
    raise ValueError(
        'Do not specify both the `representative_dataset` argument and'
        ' the `representative_datasets` field in `QuantizationOptions`.'
    )

  if quantization_options.representative_datasets:
    representative_dataset = repr_dataset.RepresentativeDatasetLoader(
        quantization_options.representative_datasets
    ).load()

  method: _QuantizationMethod = quantization_options.quantization_method
  if (
      method.preset_method == _PresetMethod.METHOD_STATIC_RANGE_INT8
      or method.preset_method == _PresetMethod.METHOD_NO_QUANTIZE
  ):
    return _static_range_quantize(
        saved_model_path,
        output_directory,
        quantization_options,
        representative_dataset,
    )
  elif method.preset_method == _PresetMethod.METHOD_DYNAMIC_RANGE_INT8:
    return _dynamic_range_quantize(
        saved_model_path,
        output_directory,
        quantization_options,
    )
  elif (
      method.preset_method == _PresetMethod.METHOD_STATIC_RANGE_WEIGHT_ONLY_INT8
  ):
    return _weight_only_quantize(
        saved_model_path,
        output_directory,
        quantization_options,
    )
  else:
    raise ValueError(
        'Quantization method {method.preset_method} is not supported.'
    )
