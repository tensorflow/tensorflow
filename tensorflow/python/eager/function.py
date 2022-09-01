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
# Public Constants
from tensorflow.python.eager.polymorphic_function.monomorphic_function import BACKWARD_FUNCTION_ATTRIBUTE_NAME
from tensorflow.python.eager.polymorphic_function.monomorphic_function import FORWARD_FUNCTION_ATTRIBUTE_NAME
from tensorflow.python.eager.polymorphic_function.monomorphic_function import IMPLEMENTS_ATTRIBUTE_NAME
from tensorflow.python.eager.polymorphic_function.monomorphic_function import SHARED_RENDEZVOUS_ATTRIBUTE_NAME

# Public Classes
from tensorflow.python.eager.polymorphic_function.monomorphic_function import ConcreteFunction
from tensorflow.python.eager.polymorphic_function.monomorphic_function import ConcreteFunctionGarbageCollector
from tensorflow.python.eager.polymorphic_function.polymorphic_function import TfMethodTarget
from tensorflow.python.eager.polymorphic_function.polymorphic_function import Function

# Public Functions
from tensorflow.python.eager.polymorphic_function.monomorphic_function import add_function_callback
from tensorflow.python.eager.polymorphic_function.monomorphic_function import clear_function_callbacks
from tensorflow.python.eager.polymorphic_function.monomorphic_function import common_shape
from tensorflow.python.eager.polymorphic_function.monomorphic_function import remove_function_callback
from tensorflow.python.eager.polymorphic_function.polymorphic_function import class_method_to_instance_method
from tensorflow.python.eager.polymorphic_function.polymorphic_function import defun
from tensorflow.python.eager.polymorphic_function.polymorphic_function import defun_with_attributes
from tensorflow.python.eager.polymorphic_function.polymorphic_function import register

# Private Constants
from tensorflow.python.eager.polymorphic_function.monomorphic_function import _BACKWARD_PREFIX
from tensorflow.python.eager.polymorphic_function.monomorphic_function import _FORWARD_PREFIX
from tensorflow.python.eager.polymorphic_function.monomorphic_function import _INFERENCE_PREFIX

# Private Classes
from tensorflow.python.eager.polymorphic_function.monomorphic_function import _DelayedRewriteGradientFunctions
from tensorflow.python.eager.polymorphic_function.monomorphic_function import _EagerDefinedFunction
from tensorflow.python.eager.polymorphic_function.monomorphic_function import _EagerDefinedFunctionDeleter
from tensorflow.python.eager.polymorphic_function.monomorphic_function import _FirstOrderTapeGradientFunctions
from tensorflow.python.eager.polymorphic_function.monomorphic_function import _ForwardBackwardCall
from tensorflow.python.eager.polymorphic_function.monomorphic_function import _ForwardWrapper
from tensorflow.python.eager.polymorphic_function.monomorphic_function import _HigherOrderTapeGradientFunctions
from tensorflow.python.eager.polymorphic_function.monomorphic_function import _Marker
from tensorflow.python.eager.polymorphic_function.monomorphic_function import _TapeGradientFunctions
from tensorflow.python.eager.polymorphic_function.monomorphic_function import _InterpolateFunctionError

# Private Functions and Attributes
from tensorflow.python.eager.polymorphic_function.monomorphic_function import _backward_name
from tensorflow.python.eager.polymorphic_function.monomorphic_function import _contains_type_spec
from tensorflow.python.eager.polymorphic_function.monomorphic_function import _create_forward_backward_with_graph
from tensorflow.python.eager.polymorphic_function.monomorphic_function import _forward_name
from tensorflow.python.eager.polymorphic_function.monomorphic_function import _function_callbacks
from tensorflow.python.eager.polymorphic_function.monomorphic_function import _inference_name
from tensorflow.python.eager.polymorphic_function.monomorphic_function import _is_type_subset
from tensorflow.python.eager.polymorphic_function.monomorphic_function import _parse_func_attrs
from tensorflow.python.eager.polymorphic_function.monomorphic_function import _structure_summary
from tensorflow.python.eager.polymorphic_function.monomorphic_function import _type_spec_for
from tensorflow.python.eager.polymorphic_function.polymorphic_function import _graph_building_time_counter

