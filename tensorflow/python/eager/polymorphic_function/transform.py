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
"""tf.function transformations implementation."""

# TODO(fmuham): Move this logic to core/function when layered.
# TODO(fmuham): Deprecate and migrate these as AtomicFunction transformations.
FUNC_GRAPH_TRANSFORMS = []
CONCRETE_FUNCTION_CALLBACKS = []


def apply_func_graph_transforms(func_graph):
  """Applies registered transformations to FuncGraph."""
  for transform in FUNC_GRAPH_TRANSFORMS:
    transform(func_graph)


def call_concrete_function_callbacks(concrete_fn):
  """Calls registered callbacks against new ConcreteFunctions."""
  for callback in CONCRETE_FUNCTION_CALLBACKS:
    callback(concrete_fn)
