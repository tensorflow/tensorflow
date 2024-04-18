# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""StableHLO Quantizer."""
from typing import Mapping

from tensorflow.compiler.mlir.quantization.stablehlo import quantization_config_pb2 as qc
from tensorflow.compiler.mlir.quantization.stablehlo.python import pywrap_quantization
from tensorflow.compiler.mlir.quantization.tensorflow.python import py_function_lib
from tensorflow.compiler.mlir.quantization.tensorflow.python import save_model
from tensorflow.core.protobuf import meta_graph_pb2

# Mapping of signature def key -> SignatureDef.
_SignatureDefMap = Mapping[str, meta_graph_pb2.SignatureDef]


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


def _has_quantization_method(
    quantization_specs: qc.QuantizationSpecs, method: str
) -> bool:
  """Returns whether a given QuantizationSpecs has the given quantization method."""
  for spec in quantization_specs.specs:
    if spec.method.HasField(method):
      return True
  return False


# TODO: b/310594193 - Export API to pip package.
def quantize_saved_model(
    src_saved_model_path: str,
    dst_saved_model_path: str,
    config: qc.QuantizationConfig,
) -> None:
  """Quantizes a saved model.

  Args:
    src_saved_model_path: Path to the directory for the source SavedModel.
    dst_saved_model_path: Path to the directory for the destination SavedModel.
    config: Quantization configuration.

  Raises:
    ValueError: When `config` was not configured for static-range PTQ
    single representative dataset.
  """
  # Updates user-provided `QuantizationConfig`s for the internal quantization
  # pipeline to work with.
  print('=== User-provided QuantizationConfig ===')
  print(config)
  config = qc.QuantizationConfig.FromString(
      pywrap_quantization.populate_default_configs(config.SerializeToString())
  )
  config = qc.QuantizationConfig.FromString(
      pywrap_quantization.expand_preset_configs(config.SerializeToString())
  )
  print('=== Updated QuantizationConfig ===')
  print(config)

  if not (
      _has_quantization_method(config.specs, 'static_range_ptq')
      and len(config.calibration_options.representative_datasets) == 1
  ) and not _has_quantization_method(config.specs, 'weight_only_ptq'):
    raise ValueError(
        '`quantize_saved_model` currently only supports static-range PTQ with a'
        ' single signature or weight-only quantization.'
    )

  signature_def_map = save_model.get_signatures_from_saved_model(
      src_saved_model_path,
      signature_keys=None,
      tags=set(config.tf_saved_model.tags),
  )

  signature_def_map_serialized = _serialize_signature_def_map(signature_def_map)
  # Currently, only StaticRangePtq or WeightOnlyPtq is supported.
  # Consider merging the pipelines to address mixed algorithm models.
  if _has_quantization_method(config.specs, 'static_range_ptq'):
    pywrap_quantization.static_range_ptq(
        src_saved_model_path,
        dst_saved_model_path,
        quantization_config_serialized=config.SerializeToString(),
        signature_keys=list(signature_def_map.keys()),
        signature_def_map_serialized=signature_def_map_serialized,
        py_function_library=py_function_lib.PyFunctionLibrary(),
    )
  elif _has_quantization_method(config.specs, 'weight_only_ptq'):
    pywrap_quantization.weight_only_ptq(
        src_saved_model_path,
        dst_saved_model_path,
        quantization_config_serialized=config.SerializeToString(),
        signature_keys=list(signature_def_map.keys()),
        signature_def_map_serialized=signature_def_map_serialized,
        py_function_library=py_function_lib.PyFunctionLibrary(),
    )
