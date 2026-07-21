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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_BROADCAST_LOOP_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_BROADCAST_LOOP_H_

#include <algorithm>
#include <cstddef>

#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"

namespace tflite {
namespace reference_ops {

template <typename Pointer1, typename Pointer2, typename OutputType,
          typename BinaryOp>
void RunBinaryOp(Pointer1 a, Pointer2 b, OutputType* output,
                 const size_t* a_stride, const size_t* b_stride,
                 const size_t* output_stride, const size_t* output_shape,
                 int dim, BinaryOp op) {
  TFLITE_DCHECK_GE(dim, 0);
  size_t output_shape_0 = output_shape[dim];
  size_t output_stride_0 = output_stride[dim];
  size_t a_stride_0 = a_stride[dim];
  size_t b_stride_0 = b_stride[dim];
  if (dim == 0) {
    TFLITE_DCHECK_EQ(output_stride_0, 1);
    if (a_stride_0 == 0) {
      TFLITE_DCHECK_EQ(b_stride_0, 1);
      const auto a_0 = *a;
      for (size_t i = 0; i < output_shape_0; ++i) {
        output[i] = op(a_0, b[i]);
      }
    } else if (b_stride_0 == 0) {
      TFLITE_DCHECK_EQ(a_stride_0, 1);
      const auto b_0 = *b;
      for (size_t i = 0; i < output_shape_0; ++i) {
        output[i] = op(a[i], b_0);
      }
    } else {
      TFLITE_DCHECK_EQ(a_stride_0, 1);
      TFLITE_DCHECK_EQ(b_stride_0, 1);
      for (size_t i = 0; i < output_shape_0; ++i) {
        output[i] = op(a[i], b[i]);
      }
    }
  } else {
    dim -= 1;
    for (size_t i = 0; i < output_shape_0; ++i) {
      RunBinaryOp(a, b, output, a_stride, b_stride, output_stride, output_shape,
                  dim, op);
      a = a + a_stride_0;
      b = b + b_stride_0;
      output = output + output_stride_0;
    }
  }
}

// Returns true if a dimension of a loop nest can be fused with the previous
// dimension in the loop nest.
inline bool CanFuseLoops(size_t output_dim, size_t dim, size_t stride,
                         size_t expected_stride, size_t next_stride) {
  if (output_dim == 1) {
    // We can always fuse an extent 1 dimension.
    return true;
  }
  if (next_stride == 0) {
    // The next loop's stride is 0, the current stride must be 0 too.
    return stride == 0;
  } else {
    // The next loop's stride must match the current loop's stride.
    return stride == expected_stride;
  }
}

template <typename Pointer1, typename Pointer2, typename OutputType,
          typename BinaryOp>
inline void BroadcastBinaryOpSimple(const RuntimeShape& input1_shape,
                                    Pointer1 input1_data,
                                    const RuntimeShape& input2_shape,
                                    Pointer2 input2_data,
                                    const RuntimeShape& output_shape,
                                    OutputType* output_data, BinaryOp op) {
  constexpr int kMaxRank = 8;

  int dims_count = std::max(
      output_shape.DimensionsCount(),
      std::max(input1_shape.DimensionsCount(), input2_shape.DimensionsCount()));

  if (dims_count <= 0) {
    *output_data = op(*input1_data, *input2_data);
    return;
  }

  TFLITE_DCHECK_LE(dims_count, kMaxRank);

  const RuntimeShape extended_output_shape =
      RuntimeShape::ExtendedShape(dims_count, output_shape);
  const RuntimeShape extended_input1_shape =
      RuntimeShape::ExtendedShape(dims_count, input1_shape);
  const RuntimeShape extended_input2_shape =
      RuntimeShape::ExtendedShape(dims_count, input2_shape);

  // This loop works as follows:
  // - Start with the last loop, the stride 1 dimension, populate the innermost
  //   loop (dimension 0).
  // - For all remaining dimensions, if the loop can be fused with the previous
  //   loop do that.
  // - Otherwise, make a new loop.
  size_t a_strides[kMaxRank];
  size_t b_strides[kMaxRank];
  size_t o_strides[kMaxRank];
  size_t o_shape[kMaxRank];

  size_t a_accum_stride = 1;
  size_t b_accum_stride = 1;
  size_t o_accum_stride = 1;
  int next_dim_idx = -1;
  for (int i = dims_count - 1; i >= 0; --i) {
    const int input1_dim = extended_input1_shape.Dims(i);
    const int input2_dim = extended_input2_shape.Dims(i);
    const int output_dim = extended_output_shape.Dims(i);
    if (input1_dim <= 0 || input2_dim <= 0 || output_dim <= 0) {
      // Empty operation.
      return;
    }
    size_t a_stride = (input1_dim == 1 && output_dim != 1) ? 0 : a_accum_stride;
    size_t b_stride = (input2_dim == 1 && output_dim != 1) ? 0 : b_accum_stride;
    size_t o_stride = o_accum_stride;
    if (next_dim_idx >= 0 &&
        CanFuseLoops(output_dim, input1_dim, a_stride, a_accum_stride,
                     a_strides[next_dim_idx]) &&
        CanFuseLoops(output_dim, input2_dim, b_stride, b_accum_stride,
                     b_strides[next_dim_idx]) &&
        CanFuseLoops(output_dim, output_dim, o_stride, o_accum_stride,
                     o_strides[next_dim_idx])) {
      // This dimension can be fused into one loop with the previous
      // dimension.
      o_shape[next_dim_idx] *= output_dim;
    } else {
      ++next_dim_idx;
      a_strides[next_dim_idx] = a_stride;
      b_strides[next_dim_idx] = b_stride;
      o_strides[next_dim_idx] = o_stride;
      o_shape[next_dim_idx] = output_dim;
    }

    a_accum_stride *= input1_dim;
    b_accum_stride *= input2_dim;
    o_accum_stride *= output_dim;
  }

  RunBinaryOp(input1_data, input2_data, output_data, a_strides, b_strides,
              o_strides, o_shape, next_dim_idx, op);
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_BROADCAST_LOOP_H_
