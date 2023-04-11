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
"""RefVariable class."""

from tensorflow.python.ops import variables

# Refactor currently in progress to move the following symbols to this file.
# The symbols themselves cannot be moved until all references to them have
#   been updated to this location. This is done to avoid import cycles.
RefVariable = variables.RefVariable
_find_initialized_value_for_variable = variables._find_initialized_value_for_variable  # pylint: disable=protected-access
_safe_initial_value_from_tensor = variables._safe_initial_value_from_tensor  # pylint: disable=protected-access
_safe_initial_value_from_op = variables._safe_initial_value_from_op  # pylint: disable=protected-access
_has_cycle = variables._has_cycle  # pylint: disable=protected-access
_try_guard_against_uninitialized_dependencies = variables._try_guard_against_uninitialized_dependencies  # pylint: disable=protected-access
validate_synchronization_aggregation_trainable = variables.validate_synchronization_aggregation_trainable
