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
"""Supports old symbols supplied by this file while the code is refactored."""

# pylint:disable=unused-import,g-bad-import-order

# TODO(b/243822285): Reduce this list as much as possible.
# Constants
from tensorflow.python.eager.polymorphic_function.monomorphic_function import _BACKWARD_PREFIX
from tensorflow.python.eager.polymorphic_function.monomorphic_function import _FORWARD_PREFIX
from tensorflow.python.eager.polymorphic_function.monomorphic_function import _INFERENCE_PREFIX

# Function Classes
from tensorflow.python.eager.polymorphic_function.monomorphic_function import ConcreteFunction
from tensorflow.python.eager.polymorphic_function.tracing_compiler import TracingCompiler as Function
from tensorflow.python.eager.polymorphic_function.atomic_function import EagerDefinedFunction as _EagerDefinedFunction

# Utilities
from tensorflow.python.eager.polymorphic_function.tracing_compiler import TfMethodTarget
from tensorflow.python.eager.polymorphic_function.monomorphic_function import _inference_name

# TODO(b/244360504): Remove in favor of graph transformation API.
# QUARANTINED - Function Callback Modification API
from tensorflow.python.eager.polymorphic_function.quarantine import add_function_callback
from tensorflow.python.eager.polymorphic_function.quarantine import clear_function_callbacks
from tensorflow.python.eager.polymorphic_function.quarantine import remove_function_callback
from tensorflow.python.eager.polymorphic_function.atomic_function import function_callbacks as _function_callbacks

# TODO(b/258247871): Remove in favor of tf.function.
# QUARANTINED - Defun API
from tensorflow.python.eager.polymorphic_function.quarantine import defun_with_attributes
