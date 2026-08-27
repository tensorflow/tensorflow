# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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

"""Definitions for targets in DPB (Delegate Performance Benchmark)."""

def latency_benchmark_extra_deps():
    """Defines extra dependencies for latency benchmark. Currently empty."""
    return []

def accuracy_benchmark_extra_deps():
    """Defines extra dependencies for accuracy benchmark. Currently empty."""
    return []

def latency_benchmark_extra_models():
    """Defines extra models for latency benchmark. Currently empty.

    Returns a list of tuples where each tuple has two fields: 1) the model name and 2) the model target label. Example:
    [
        ("model1.tflite", "@repo//package:model1.tflite"),
        ("model2.tflite", "@repo//package:model2.tflite"),
    ]
    """
    return []

def accuracy_benchmark_extra_models():
    """Defines extra models for accuracy benchmark. Currently empty.

    Returns a list of tuples where each tuple has two fields: 1) the model name and 2) the model target label. Example:
    [
        ("model1_with_validation.tflite", "@repo//package:model1_with_validation.tflite"),
        ("model2_with_validation.tflite", "@repo//package:model2_with_validation.tflite"),
    ]
    """
    return []
