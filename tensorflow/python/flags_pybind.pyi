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

class Flag:
    def __init__(self, *args, **kwargs) -> None: ...
    def reset(self, arg0: bool) -> None: ...
    def value(self) -> bool: ...

class Flags:
    enable_aggressive_constant_replication: Flag
    enable_nested_function_shape_inference: Flag
    enable_quantized_dtypes_training: Flag
    graph_building_optimization: Flag
    more_stack_traces: Flag
    op_building_optimization: Flag
    publish_function_graphs: Flag
    saved_model_fingerprinting: Flag
    test_only_experiment_1: Flag
    test_only_experiment_2: Flag
    tf_shape_default_int64: Flag
    def __init__(self) -> None: ...
