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
from tensorflow.compiler.mlir.quantization.tensorflow.calibrator import calibration_statistics_pb2

# LINT.IfChange(clear_calibrator)
def clear_calibrator() -> None: ...

# LINT.ThenChange()

# LINT.IfChange(clear_data_from_calibrator)
def clear_data_from_calibrator(id: bytes) -> None: ...

# LINT.ThenChange()

# LINT.IfChange(get_statistics_from_calibrator)
def get_statistics_from_calibrator(
    id: bytes,
) -> calibration_statistics_pb2.CalibrationStatistics: ...

# LINT.ThenChange()
