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
      signature_keys: list[str],
      tags: set[str],
      calibration_options_serialized: bytes,
      force_graph_mode_calibration: bool,
      # Value type: RepresentativeDatasetFile.
      representative_dataset_file_map_serialized: dict[str, bytes],
  ) -> None: ...
  # LINT.ThenChange()

  # LINT.IfChange(get_calibration_min_max_value)
  def get_calibration_min_max_value(
      self,
      calibration_statistics_serialized: bytes,
      calibration_options_serialized: bytes,
  ) -> tuple[float, float]: ...
  # LINT.ThenChange()
