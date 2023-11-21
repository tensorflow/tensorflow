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
from typing import Any

class PyFunctionLibrary:

  # LINT.IfChange(assign_ids_to_custom_aggregator_ops)
  def assign_ids_to_custom_aggregator_ops(
      self, exported_model_serialized: bytes
  ) -> bytes: ...
  # LINT.ThenChange()

  # LINT.IfChange(save_exported_model)
  def save_exported_model(
      self,
      dst_saved_model_path: str,
      exported_model_serialized: bytes,
      src_saved_model_path: str,
      tags: set[str],
      serialized_signature_def_map: dict[str, bytes],
  ) -> None: ...
  # LINT.ThenChange()

  # LINT.IfChange(run_calibration)
  def run_calibration(
      self,
      saved_model_path: str,
      exported_model_serialized: bytes,
      quantization_options_serialized: bytes,
      representative_dataset: Any,
  ) -> bytes: ...
  # LINT.ThenChange()

  # LINT.IfChange(enable_dump_tensor)
  def enable_dump_tensor(self, graph_def_serialized: bytes) -> bytes: ...
  # LINT.ThenChange()

  # LINT.IfChange(change_dump_tensor_file_name)
  def change_dump_tensor_file_name(
      self, graph_def_serialized: bytes
  ) -> bytes: ...
  # LINT.ThenChange()
