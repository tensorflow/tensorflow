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

// Parameters for the integer random shader. Layout must match the Metal
// struct.
struct RandomIntParams {
  uint32_t count;
  uint32_t seed_lo;
  uint32_t seed_hi;
  uint32_t counter;
  int32_t lo;
  uint32_t span;
  uint32_t padding0;
  uint32_t padding1;
};

// Parameters for the optimizer shaders. Layout must match the Metal structs;
// both take an element count first, so one type covers them.
struct OptimizerParams {
  uint32_t count;
  uint32_t use_nesterov;
  uint32_t padding0;
  uint32_t padding1;
};

// Parameters for the morphological dilation gradients. Layout must match the
// Metal struct.
struct DilationParams {
  uint32_t batch;
  uint32_t in_h;
  uint32_t in_w;
  uint32_t channels;
  uint32_t out_h;
  uint32_t out_w;
  uint32_t kh;
  uint32_t kw;
  uint32_t stride_h;
  uint32_t stride_w;
  uint32_t rate_h;
  uint32_t rate_w;
  int32_t pad_top;
  int32_t pad_left;
  uint32_t count;
  uint32_t padding0;
};

// Parameters for the max-pooling index shaders. Layout must match the Metal
// struct.
struct PoolIndexParams {
  uint32_t batch;
  uint32_t in_h;
  uint32_t in_w;
  uint32_t channels;
  uint32_t out_h;
  uint32_t out_w;
  uint32_t kh;
  uint32_t kw;
  uint32_t stride_h;
  uint32_t stride_w;
  int32_t pad_top;
  int32_t pad_left;
  uint32_t count;
  uint32_t include_batch;
  uint32_t padding0;
  uint32_t padding1;
};

// Parameters for the bin-counting shaders. Layout must match the Metal
// struct. `row_len` is zero for a flat count and the row width for the dense
// two-dimensional form.
struct BincountParams {
  uint32_t count;
  uint32_t size;
  uint32_t row_len;
  uint32_t binary;
  uint32_t has_weights;
  uint32_t padding0;
  uint32_t padding1;
  uint32_t padding2;
};

// Parameters for the crop-and-resize shaders. Layout must match the Metal
// struct. `count` is the number of crop elements, which is what all three
// shaders are dispatched over.
struct CropResizeParams {
  uint32_t batch;
  uint32_t in_h;
  uint32_t in_w;
  uint32_t depth;
  uint32_t num_boxes;
  uint32_t crop_h;
  uint32_t crop_w;
  uint32_t method_nearest;
  float extrapolation;
  uint32_t count;
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
