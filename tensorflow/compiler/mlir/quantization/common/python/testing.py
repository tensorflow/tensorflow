"""Common testing utilities for quantization libraries."""

# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
import itertools
from typing import Any, Mapping, Sequence


def parameter_combinations(
    test_parameters: Sequence[Mapping[str, Sequence[Any]]]
) -> Sequence[Mapping[str, Any]]:
  """Generate all combinations of test parameters.

  Args:
    test_parameters: List of dictionaries that maps parameter keys and values.

  Returns:
    real_parameters: All possible combinations of the parameters as list of
    dictionaries.
  """
  real_parameters = []
  for parameters in test_parameters:
    keys = parameters.keys()
    for curr in itertools.product(*parameters.values()):
      real_parameters.append(dict(zip(keys, curr)))
  return real_parameters
