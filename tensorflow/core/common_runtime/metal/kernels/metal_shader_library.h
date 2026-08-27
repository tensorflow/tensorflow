/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_METAL_KERNELS_METAL_SHADER_LIBRARY_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_METAL_KERNELS_METAL_SHADER_LIBRARY_H_

// Objective-C++ only.

#import <Metal/Metal.h>

#include <cstdint>

#include "tensorflow/c/tf_status.h"

namespace tensorflow {
namespace metal {

// Parameters shared by the elementwise shaders. Must stay layout-identical to
// the struct of the same name in the embedded Metal source.
struct ElementwiseParams {
  uint32_t count;
  uint32_t lhs_is_scalar;
  uint32_t rhs_is_scalar;
  uint32_t padding;
};

// Parameters for the fill shaders. Layout must match the Metal struct.
struct FillParams {
  uint32_t count;
  float value;
  uint32_t padding0;
  uint32_t padding1;
};

// Parameters for the counter-based random shaders. Layout must match the Metal
// struct. `counter` is bumped by the host on every call so that repeated runs
// of the same op draw different numbers.
struct RandomParams {
  uint32_t count;
  uint32_t seed_lo;
  uint32_t seed_hi;
  uint32_t counter;
};

// Parameters for the optimizer shaders. Layout must match the Metal structs;
// both take an element count first, so one type covers them.
struct OptimizerParams {
  uint32_t count;
  uint32_t use_nesterov;
  uint32_t padding0;
  uint32_t padding1;
};

// Compute pipeline for one function in the backend's shader library.
//
// The Metal source is compiled from a string the first time any pipeline is
// requested, and both the library and each pipeline state are cached for the
// process. Compiling at runtime rather than building a .metallib keeps the
// Bazel rules to plain objc_library and lets the shaders target whatever
// Metal version the running OS provides, at the cost of a one-off compile on
// the first op.
//
// Returns nil and fails `status` if compilation fails or the function is not
// found; callers must not proceed on nil.
id<MTLComputePipelineState> PipelineFor(id<MTLDevice> device,
                                        const char* function_name,
                                        TF_Status* status);

}  // namespace metal
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_METAL_KERNELS_METAL_SHADER_LIBRARY_H_
