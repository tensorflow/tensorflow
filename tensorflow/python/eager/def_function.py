# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Supports old symbols supplied by this file while the code is refactored."""

# pylint:disable=unused-import,g-bad-import-order

# Config Options
from tensorflow.python.eager.polymorphic_function.eager_function_run import run_functions_eagerly
from tensorflow.python.eager.polymorphic_function.eager_function_run import functions_run_eagerly
from tensorflow.python.eager.polymorphic_function.polymorphic_function import set_dynamic_variable_creation

# tf.function Classes
from tensorflow.python.eager.polymorphic_function.polymorphic_function import Function
from tensorflow.python.eager.polymorphic_function.polymorphic_function import function

# Private attributes
from tensorflow.python.eager.polymorphic_function.polymorphic_function import _tf_function_counter
