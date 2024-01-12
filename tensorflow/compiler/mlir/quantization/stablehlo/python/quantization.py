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
from tensorflow.compiler.mlir.quantization.tensorflow import quantization_options_pb2 as quant_opts_pb2
from tensorflow.compiler.mlir.quantization.tensorflow.python import py_function_lib
from tensorflow.compiler.mlir.quantization.tensorflow.python import save_model
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.saved_model import loader_impl

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
  if not (
      config.HasField('static_range_ptq_preset')
      and len(config.static_range_ptq_preset.representative_datasets) == 1
  ):
    raise ValueError(
        '`quantize_saved_model` currently only supports static-range PTQ with a'
        ' single signature.'
    )

  signature_def_map = save_model.get_signatures_from_saved_model(
      src_saved_model_path,
      signature_keys=None,
      tags=set(config.tf_saved_model.tags),
  )

  loader = loader_impl.SavedModelLoader(src_saved_model_path)
  function_aliases = loader.get_meta_graph_def_from_tags(
      config.tf_saved_model.tags
  ).meta_info_def.function_aliases

  # Create a signature key -> `RepresentativeDatasetFile` mapping.
  # `RepresentativeDatsetFile` should be serialized for `static_range_ptq` due
  # to pywrap protobuf compatibility requirements.
  tfrecord_file_path: str = (
      config.static_range_ptq_preset.representative_datasets[0].tf_record.path
  )
  dataset_file_map = {
      'serving_default': quant_opts_pb2.RepresentativeDatasetFile(
          tfrecord_file_path=tfrecord_file_path
      ).SerializeToString()
  }

  signature_def_map_serialized = _serialize_signature_def_map(signature_def_map)
  pywrap_quantization.static_range_ptq(
      src_saved_model_path,
      dst_saved_model_path,
      quantization_config_serialized=config.SerializeToString(),
      signature_keys=list(signature_def_map.keys()),
      signature_def_map_serialized=signature_def_map_serialized,
      function_aliases=dict(function_aliases),
      py_function_library=py_function_lib.PyFunctionLibrary(),
      representative_dataset_file_map_serialized=dataset_file_map,
  )
