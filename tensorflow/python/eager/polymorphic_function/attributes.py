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

"""This file lists FunctionDef attributes and corresponding allowlists."""

# IMPORTANT: The usage of all the attributes below should be considered tech
# debt and new additions to this list are discouraged.
#
# Historically, attributes have been used as means to pipe extra information
# down to runtime that is not related to the actual function definition itself.
#
# This information is better layered independently and future work is encouraged
# to pursue that direction instead.

API_IMPLEMENTS = "api_implements"
API_PREFERRED_DEVICE = "api_preferred_device"
BACKWARD_FUNCTION = "backward_function_name"
DISABLE_CALL_SHAPE_INFERENCE = "_disable_call_shape_inference"
EAGER_RUNTIME_CONSTRUCTION_CONTEXT = "_construction_context"
FORWARD_FUNCTION = "forward_function_name"
GO_BACKWARDS = "go_backwards"
IMPLEMENTS = "_implements"
INPUT_SHAPES = "_input_shapes"
INTS_ON_DEVICE = "experimental_ints_on_device"
NO_INLINE = "_noinline"
ORIGINAL_FUNCTION_NAME = "_original_func_name"
OUTPUTS_ON_OP_DEVICE = "_OutputsOnOpDevice"
QUANTIZED_COMPOSITE_FUNCTION = "tf_quant.composite_function"
QUANTIZED_OPS = "tf_quant.quantized_ops"
SHARED_RENDEZVOUS = "shared_rendezvous"
TF_DATA_FUNCTION = "_tf_data_function"
TFTRT_ALLOW_BUILD_AT_RUNTIME = "_tftrt_allow_build_at_runtime"
TFTRT_CONVERT_FUNCTION = "_tftrt_convert_function"
TFTRT_IS_DYN_OP = "_tftrt_is_dyn_op"
TFTRT_LOGGER = "_tftrt_trt_logger_name"
TFTRT_MAX_BATCH_SIZE = "_tftrt_max_batch_size"
TFTRT_MAX_CACHED_ENGINES = "_tftrt_max_cached_engines"
TFTRT_MAX_WORKSPACE_SIZE = "_tftrt_max_workspace_size_bytes"
TFTRT_MIN_SEGMENT_SIZE = "_tftrt_minimum_segment_size"
TFTRT_PRECISION_MODE = "_tftrt_precision_mode"
TFTRT_PROFILE_STRATEGY = "_tftrt_profile_strategy"
TFTRT_USE_CALIBRATION = "_tftrt_use_calibration"
TFTRT_USE_IMPLICIT_BATCH = "_tftrt_use_implicit_batch"
TIME_MAJOR = "time_major"
XLA_COMPILE = "_XlaMustCompile"
XLA_COMPILE_OPTIONAL = "_XlaCompile"
XLA_SCOPE = "_XlaScope"
XLA_SEPERATE_COMPILED_GRADIENTS = "_XlaSeparateCompiledGradients"

POLYMORPHIC_FUNCTION_ALLOWLIST = frozenset({
    API_IMPLEMENTS,
    API_PREFERRED_DEVICE,
    GO_BACKWARDS,
    IMPLEMENTS,
    TIME_MAJOR,
})

TRACING_COMPILER_ALLOWLIST = frozenset().union(
    POLYMORPHIC_FUNCTION_ALLOWLIST,
    {
        INTS_ON_DEVICE,
        NO_INLINE,
        OUTPUTS_ON_OP_DEVICE,
        SHARED_RENDEZVOUS,
        TF_DATA_FUNCTION,
        XLA_COMPILE,
    },
)

MONOMORPHIC_FUNCTION_ALLOWLIST = frozenset().union(
    TRACING_COMPILER_ALLOWLIST,
    {
        BACKWARD_FUNCTION,
        DISABLE_CALL_SHAPE_INFERENCE,
        EAGER_RUNTIME_CONSTRUCTION_CONTEXT,
        FORWARD_FUNCTION,
        INPUT_SHAPES,
        ORIGINAL_FUNCTION_NAME,
        QUANTIZED_COMPOSITE_FUNCTION,
        QUANTIZED_OPS,
        TFTRT_ALLOW_BUILD_AT_RUNTIME,
        TFTRT_CONVERT_FUNCTION,
        TFTRT_IS_DYN_OP,
        TFTRT_LOGGER,
        TFTRT_MAX_BATCH_SIZE,
        TFTRT_MAX_CACHED_ENGINES,
        TFTRT_MAX_WORKSPACE_SIZE,
        TFTRT_MIN_SEGMENT_SIZE,
        TFTRT_PRECISION_MODE,
        TFTRT_PROFILE_STRATEGY,
        TFTRT_USE_CALIBRATION,
        TFTRT_USE_IMPLICIT_BATCH,
        XLA_COMPILE_OPTIONAL,
        XLA_SCOPE,
        XLA_SEPERATE_COMPILED_GRADIENTS,
    },
)
