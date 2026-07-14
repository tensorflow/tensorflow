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
"""List of test backend names."""

import enum
import os


# LINT.IfChange(backend_name)
class DTensorTestUtilBackend(enum.Enum):
  """DTensor backend the test is being run on."""
  UNSPECIFIED = 'unspecified'
  CPU = 'cpu'
  GPU = 'gpu'
  GPU_2DEVS_BACKEND = '2gpus'
  TPU = 'tpu'
  TPU_STREAM_EXECUTOR = 'tpu_se'
  TPU_V3_DONUT_BACKEND = 'tpu_v3_2x2'
  TPU_V4_DONUT_BACKEND = 'tpu_v4_2x2'


DTENSOR_TEST_UTIL_BACKEND = DTensorTestUtilBackend(
    os.getenv('DTENSOR_TEST_UTIL_BACKEND', default='unspecified')
)

# LINT.ThenChange()
