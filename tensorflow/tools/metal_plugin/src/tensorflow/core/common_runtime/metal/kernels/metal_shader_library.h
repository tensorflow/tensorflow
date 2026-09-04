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

// Parameters for the projective transform shader. Layout must match the Metal
// struct. `fill_mode` is 0 for CONSTANT, 1 for REFLECT, 2 for WRAP and 3 for
// NEAREST, matching TensorFlow's order.
struct TransformParams {
  uint32_t batch;
  uint32_t in_h;
  uint32_t in_w;
  uint32_t depth;
  uint32_t out_h;
  uint32_t out_w;
  uint32_t count;
  uint32_t nearest;
  uint32_t fill_mode;
  uint32_t num_transforms;
  float fill_value;
  uint32_t padding0;
};

// Parameters for the resize-gradient shaders. Layout must match the Metal
// struct. `in_h` and `in_w` are the resized dimensions the gradient arrives
// with; `out_h` and `out_w` are the original image's.
struct ResizeGradParams {
  uint32_t batch;
  uint32_t in_h;
  uint32_t in_w;
  uint32_t channels;
  uint32_t out_h;
  uint32_t out_w;
  float height_scale;
  float width_scale;
  uint32_t half_pixel;
  uint32_t align_corners;
  uint32_t count;
  uint32_t padding0;
};

// Parameters for the volume-patch shader. Layout must match the Metal struct.
struct VolumePatchParams {
  uint32_t batch;
  uint32_t in_d;
  uint32_t in_h;
  uint32_t in_w;
  uint32_t channels;
  uint32_t out_d;
  uint32_t out_h;
  uint32_t out_w;
  uint32_t kd;
  uint32_t kh;
  uint32_t kw;
  uint32_t stride_d;
  uint32_t stride_h;
  uint32_t stride_w;
  int32_t pad_d;
  int32_t pad_h;
  int32_t pad_w;
  uint32_t count;
  uint32_t padding0;
  uint32_t padding1;
};

// Parameters for the parameterised truncated normal. Layout must match the
// Metal struct. `num_params` is 1 when the four range vectors are scalars and
// the batch size otherwise.
struct ParamTruncatedParams {
  uint32_t count;
  uint32_t seed_lo;
  uint32_t seed_hi;
  uint32_t counter;
  uint32_t samples_per_batch;
  uint32_t num_params;
  uint32_t padding0;
  uint32_t padding1;
};

// Parameters for the categorical sampler. Layout must match the Metal struct.
struct MultinomialParams {
  uint32_t count;
  uint32_t seed_lo;
  uint32_t seed_hi;
  uint32_t counter;
  uint32_t batch;
  uint32_t classes;
  uint32_t samples;
  uint32_t padding0;
};

// Parameters for the gamma sampler. Layout must match the Metal struct.
struct GammaParams {
  uint32_t count;
  uint32_t seed_lo;
  uint32_t seed_hi;
  uint32_t counter;
  uint32_t num_alphas;
  uint32_t padding0;
  uint32_t padding1;
  uint32_t padding2;
};

// Parameters for the row gather and scatter shaders. Layout must match the
// Metal struct. `slice` is the row width in 32-bit words, so one pair of
// shaders serves every element type by copying bits.
struct RowMoveParams {
  uint32_t count;
  uint32_t slice;
  uint32_t limit;
  uint32_t padding0;
};

// Parameters for turning LAPACK-style pivots into a permutation. Layout must
// match the Metal struct.
struct PivotParams {
  uint32_t batch;
  uint32_t order;
  uint32_t padding0;
  uint32_t padding1;
};

// Parameters for the dense factorisation shaders. Layout must match the Metal
// struct. `k` is min(rows, columns).
struct FactorParams {
  uint32_t batch;
  uint32_t rows;
  uint32_t cols;
  uint32_t k;
  uint32_t full_matrices;
  uint32_t compute_vectors;
  uint32_t padding0;
  uint32_t padding1;
};

// Parameters for the connectionist temporal classification loss. Layout must
// match the Metal struct. `blank` is the class index the alignment treats as
// the blank, which differs between the two versions of the op.
struct CtcParams {
  uint32_t batch;
  uint32_t max_time;
  uint32_t num_classes;
  uint32_t blank;
  uint32_t max_labels;
  uint32_t padding0;
  uint32_t padding1;
  uint32_t padding2;
};

// Parameters for the sparse segment reductions. Layout must match the Metal
// struct. `mode` is 0 for a plain sum, 1 for a mean and 2 for a square-root
// normalisation.
struct SegmentParams {
  uint32_t num_indices;
  uint32_t inner;
  uint32_t num_segments;
  uint32_t data_rows;
  uint32_t mode;
  uint32_t count;
  uint32_t padding0;
  uint32_t padding1;
};

// Parameters for the Fourier transform shader. Layout must match the Metal
// struct. One call transforms along a single axis, described by how many
// elements precede it (`outer`), its own length (`n`) and the stride between
// its elements (`inner`).
struct FftParams {
  uint32_t outer;
  uint32_t n;
  uint32_t inner;
  uint32_t count;
  uint32_t inverse;
  uint32_t scale;
  uint32_t padding0;
  uint32_t padding1;
};

// Parameters for the crop-and-pad shader that moves a tensor between two
// shapes of the same rank. Layout must match the Metal struct. `mode` is 0 for
// complex to complex, 1 for real to complex and 2 for complex to real.
struct ResizeParams {
  uint32_t rank;
  uint32_t count;
  uint32_t mode;
  uint32_t padding0;
  uint32_t in_shape[8];
  uint32_t out_shape[8];
};

// Parameters for the sparse tensor shaders. Layout must match the Metal
// struct. `shape` carries the dense shape the indices address.
struct SparseParams {
  uint32_t nnz;
  uint32_t rank;
  uint32_t count;
  uint32_t inner;
  uint32_t scalar_values;
  uint32_t adjoint_a;
  uint32_t adjoint_b;
  uint32_t padding0;
  uint32_t shape[8];
};

// Parameters for the incomplete beta. Layout must match the Metal struct. Each
// flag says whether that argument is a single value standing for the whole
// tensor, which is the broadcast the op allows.
struct BetaincParams {
  uint32_t count;
  uint32_t a_is_scalar;
  uint32_t b_is_scalar;
  uint32_t x_is_scalar;
};

// Parameters for the numeric summary shaders. Layout must match the Metal
// struct. `prefix` carries the values the summary reports about the tensor
// rather than about its contents, which the host knows and the device does
// not.
struct DebugParams {
  uint32_t count;
  uint32_t prefix_count;
  uint32_t padding0;
  uint32_t padding1;
  float prefix[10];
};

// Parameters for the check-numerics shaders. Layout must match the Metal
// struct.
struct CheckNumericsParams {
  uint32_t count;
  uint32_t padding0;
  uint32_t padding1;
  uint32_t padding2;
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
