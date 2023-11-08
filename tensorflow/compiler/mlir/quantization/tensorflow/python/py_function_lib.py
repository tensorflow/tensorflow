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
"""Defines a wrapper class for overridden python method definitions."""
from typing import Optional
import uuid

from absl import logging

from tensorflow.compiler.mlir.quantization.tensorflow import exported_model_pb2
from tensorflow.compiler.mlir.quantization.tensorflow.python import pywrap_function_lib
from tensorflow.compiler.mlir.quantization.tensorflow.python import save_model
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.lib.io import file_io

# Name of the saved model assets directory.
_ASSETS_DIR = 'assets'
_ASSETS_EXTRA_DIR = 'assets.extra'


def _get_saver_def_or_none(
    exported_model: exported_model_pb2.ExportedModel,
) -> Optional[saver_pb2.SaverDef]:
  """Returns the SaverDef from ExportedModel, None otherwise.

  Args:
    exported_model: ExportedModel to take the SaverDef from.

  Returns:
    SaverDef instance if the field `saver_def` is set. None otherwise.
  """
  if exported_model.HasField('saver_def'):
    return exported_model.saver_def
  return None


def _copy_assets(src_path: str, dst_path: str) -> None:
  """Copies the assets directory of the saved model.

  Clones the contents of the assets/ directory from the source saved model
  directory to the destination saved model directory. Nothing will be copied if
  there are no assets directory in the source directory.

  Args:
    src_path: Source saved model directory.
    dst_path: Destination saved model directory. This directory must exist.
  """
  for assets_dir_name in [_ASSETS_DIR, _ASSETS_EXTRA_DIR]:
    src_assets_path = file_io.join(src_path, assets_dir_name)
    if not file_io.file_exists_v2(src_assets_path):
      # Do nothing if the source assets path does not exist.
      continue

    dst_assets_path = file_io.join(dst_path, assets_dir_name)
    file_io.create_dir_v2(dst_assets_path)

    for curr_dir, _, files in file_io.walk_v2(src_assets_path):
      for asset_file_name in files:
        src_asset_file = file_io.join(curr_dir, asset_file_name)

        # Construct the destination assets file path.
        curr_dst_dir = curr_dir.replace(src_assets_path, dst_assets_path)
        dst_asset_file = file_io.join(curr_dst_dir, asset_file_name)

        file_io.copy_v2(src_asset_file, dst_asset_file)
        logging.info(
            'Copied asset file: %s -> %s', src_asset_file, dst_asset_file
        )


class PyFunctionLibrary(pywrap_function_lib.PyFunctionLibrary):
  """Wrapper class for overridden python method definitions.

  This class contains python methods that overrides C++ virtual functions
  declared in `pywrap_function_lib.PyFunctionLibrary`.
  """

  # LINT.IfChange(assign_ids_to_custom_aggregator_ops)
  def assign_ids_to_custom_aggregator_ops(
      self,
      exported_model_serialized: bytes,
  ) -> bytes:
  # LINT.ThenChange(py_function_lib.h:assign_ids_to_custom_aggregator_ops)
    """Assigns UUIDs to each CustomAggregator op find in the graph def.

    Args:
      exported_model_serialized: Serialized `ExportedModel` instance.

    Returns:
      Serialized `ExportedModel` whose CustomAggregator ops are assigned UUIDs
      to their `id` attributes.
    """
    exported_model = exported_model_pb2.ExportedModel.FromString(
        exported_model_serialized
    )

    graph_def = exported_model.graph_def
    for function_def in graph_def.library.function:
      for node_def in function_def.node_def:
        if node_def.op == 'CustomAggregator':
          node_def.attr['id'].s = uuid.uuid4().hex.encode('ascii')

    return exported_model.SerializeToString()

  # LINT.IfChange(save_exported_model)
  def save_exported_model(
      self,
      dst_saved_model_path: str,
      exported_model_serialized: bytes,
      src_saved_model_path: str,
      tags: set[str],
      serialized_signature_def_map: dict[str, bytes],
  ) -> None:
  # LINT.ThenChange(py_function_lib.h:save_exported_model)
    """Saves `ExportedModel` to `dst_saved_model_path` as a SavedModel.

    Args:
      dst_saved_model_path: Destination path to save the exported model.
      exported_model_serialized: Exported model to export as SavedModel.
      src_saved_model_path: Path to the source SavedModel. This will be used to
        copy the asset files to `dst_saved_model_path`.
      tags: Tags to attach to the saved MetaGraphDef.
      serialized_signature_def_map: Signature key -> serialized SignatureDef.
    """
    exported_model = exported_model_pb2.ExportedModel.FromString(
        exported_model_serialized
    )

    # Deserialize values in signature_def_map.
    signature_def_map = {}
    for key, serialized_signature_def in serialized_signature_def_map.items():
      signature_def_map[key] = meta_graph_pb2.SignatureDef.FromString(
          serialized_signature_def
      )

    save_model.save_model_v1(
        exported_model.graph_def,
        dst_saved_model_path,
        signature_def_map,
        tags,
        init_op_name=exported_model.init_node_name,
        saver_def=_get_saver_def_or_none(exported_model),
        checkpoint_dir=exported_model.checkpoint_dir,
        function_aliases=exported_model.function_aliases,
        asset_file_defs=exported_model.asset_file_defs,
    )

    _copy_assets(src_saved_model_path, dst_saved_model_path)
