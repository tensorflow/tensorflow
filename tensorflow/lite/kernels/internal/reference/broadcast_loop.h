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

#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"

namespace tflite {
namespace reference_ops {

template <typename Pointer1, typename Pointer2, typename OutputType,
          typename BinaryOp>
void RunBinaryOp(Pointer1 a, Pointer2 b, OutputType* output,
                 const size_t* a_stride, const size_t* b_stride,
                 const size_t* output_stride, const size_t* output_shape,
                 int rank, BinaryOp op) {
  if (rank <= 0) {
    *output = op(*a, *b);
  } else {
    if (rank == 1) {
      TFLITE_DCHECK_EQ(output_stride[0], 1);
      if (a_stride[0] == 0) {
        TFLITE_DCHECK_EQ(b_stride[0], 1);
        const auto a_0 = *a;
        for (size_t i = 0; i < output_shape[0]; ++i) {
          output[i] = op(a_0, b[i]);
        }
      } else if (b_stride[0] == 0) {
        TFLITE_DCHECK_EQ(a_stride[0], 1);
        const auto b_0 = *b;
        for (size_t i = 0; i < output_shape[0]; ++i) {
          output[i] = op(a[i], b_0);
        }
      } else {
        TFLITE_DCHECK_EQ(a_stride[0], 1);
        TFLITE_DCHECK_EQ(b_stride[0], 1);
        for (size_t i = 0; i < output_shape[0]; ++i) {
          output[i] = op(a[i], b[i]);
        }
      }
    } else {
      for (size_t i = 0; i < output_shape[0]; ++i) {
        RunBinaryOp(a + i * a_stride[0], b + i * b_stride[0],
                    output + i * output_stride[0], a_stride + 1, b_stride + 1,
                    output_stride + 1, output_shape + 1, rank - 1, op);
      }
    }
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

  const int dims_count = std::max(
      output_shape.DimensionsCount(),
      std::max(input1_shape.DimensionsCount(), input2_shape.DimensionsCount()));

  TFLITE_DCHECK_LE(dims_count, kMaxRank);

  const RuntimeShape extended_output_shape =
      RuntimeShape::ExtendedShape(dims_count, output_shape);
  const RuntimeShape extended_input1_shape =
      RuntimeShape::ExtendedShape(dims_count, input1_shape);
  const RuntimeShape extended_input2_shape =
      RuntimeShape::ExtendedShape(dims_count, input2_shape);

  size_t a_strides[kMaxRank];
  size_t b_strides[kMaxRank];
  size_t o_strides[kMaxRank];
  size_t o_shape[kMaxRank];

  size_t a_accum_stride = 1;
  size_t b_accum_stride = 1;
  size_t o_accum_stride = 1;
  for (int i = dims_count - 1; i >= 0; --i) {
    a_strides[i] = (extended_input1_shape.Dims(i) == 1 &&
                    extended_output_shape.Dims(i) != 1)
                       ? 0
                       : a_accum_stride;
    b_strides[i] = (extended_input2_shape.Dims(i) == 1 &&
                    extended_output_shape.Dims(i) != 1)
                       ? 0
                       : b_accum_stride;
    o_strides[i] = o_accum_stride;
    o_shape[i] = extended_output_shape.Dims(i);

    a_accum_stride *= extended_input1_shape.Dims(i);
    b_accum_stride *= extended_input2_shape.Dims(i);
    o_accum_stride *= extended_output_shape.Dims(i);
  }

  RunBinaryOp(input1_data, input2_data, output_data, a_strides, b_strides,
              o_strides, o_shape, dims_count, op);
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_BROADCAST_LOOP_H_
