/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_SELECT_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_SELECT_H_

#include <cmath>

#include "ruy/profiler/instrumentation.h"  // from @ruy
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace reference_ops {

template <typename D, typename T>
void Select(const RuntimeShape& input_condition_shape,
            const D* input_condition_data, const RuntimeShape& input_x_shape,
            const T* input_x_data, const RuntimeShape& input_y_shape,
            const T* input_y_data, const RuntimeShape& output_shape,
            T* output_data) {
  ruy::profiler::ScopeLabel label("Select");
  int64_t flatsize;
  // Allow select operator executions on mixed scalar tensors and one element
  // tensors.
  if (input_condition_shape.FlatSize() == 1 && input_x_shape.FlatSize() == 1 &&
      input_y_shape.FlatSize() == 1 && output_shape.FlatSize() == 1) {
    flatsize = 1;
  } else {
    flatsize = MatchingFlatSize(input_condition_shape, input_x_shape,
                                input_y_shape, output_shape);
  }
  for (int64_t i = 0; i < flatsize; ++i) {
    output_data[i] =
        input_condition_data[i] ? input_x_data[i] : input_y_data[i];
  }
}

template <typename D, typename T>
void RankOneSelect(const RuntimeShape& input_condition_shape,
                   const D* input_condition_data,
                   const RuntimeShape& input_x_shape, const T* input_x_data,
                   const RuntimeShape& input_y_shape, const T* input_y_data,
                   const RuntimeShape& output_shape, T* output_data) {
  ruy::profiler::ScopeLabel label("Select/RankOneSelect");
  const int64_t outer_size = input_condition_shape.FlatSize();
  int64_t inner_size;
  if (input_condition_shape.DimensionsCount() == 0) {
    inner_size = MatchingFlatSize(input_x_shape, input_y_shape, output_shape);
  } else {
    TFLITE_DCHECK_EQ(
        MatchingDim(input_x_shape, 0, input_y_shape, 0, output_shape, 0),
        outer_size);
    inner_size =
        MatchingFlatSizeSkipDim(input_x_shape, 0, input_y_shape, output_shape);
  }

  int64_t offset = 0;
  for (int64_t i = 0; i < outer_size; i++) {
    const T* input_data = input_condition_data[i] ? input_x_data : input_y_data;
    memcpy(output_data + offset, input_data + offset, inner_size * sizeof(T));
    offset += inner_size;
  }
}

template <typename D, typename T>
void BroadcastSelect5DSlow(const RuntimeShape& input_condition_shape,
                           const D* input_condition_data,
                           const RuntimeShape& input_x_shape,
                           const T* input_x_data,
                           const RuntimeShape& input_y_shape,
                           const T* input_y_data,
                           const RuntimeShape& output_shape, T* output_data) {
  ruy::profiler::ScopeLabel label("Select/BroadcastSelectSlow");
  TFLITE_DCHECK_LE(input_condition_shape.DimensionsCount(), 5);
  TFLITE_DCHECK_LE(input_x_shape.DimensionsCount(), 5);
  TFLITE_DCHECK_LE(input_y_shape.DimensionsCount(), 5);
  TFLITE_DCHECK_LE(output_shape.DimensionsCount(), 5);

  NdArrayDesc<5> desc_condition;
  NdArrayDesc<5> desc_x;
  NdArrayDesc<5> desc_y;
  NdArrayDesc<5> desc_output;
  const RuntimeShape extended_output_shape =
      RuntimeShape::ExtendedShape(5, output_shape);
  CopyDimsToDesc(extended_output_shape, &desc_output);
  NdArrayDescsForElementwiseBroadcast(input_condition_shape, input_x_shape,
                                      input_y_shape, &desc_condition, &desc_x,
                                      &desc_y);

  // In Tensorflow, the dimensions are canonically named (batch_number, row,
  // col, channel), with extents (batches, height, width, depth), with the
  // trailing dimension changing most rapidly (channels has the smallest
  // stride, typically 1 element).
  //
  // In generated C code, we store arrays with the dimensions reversed. The
  // first dimension has smallest stride.
  //
  // We name our variables by their Tensorflow convention, but generate C code
  // nesting loops such that the innermost loop has the smallest stride for
  // the best cache behavior.
  for (int n = 0; n < desc_output.extents[0]; ++n) {
    int out_idx_n = desc_output.extents[1] * n;
    int cond_idx_n = desc_condition.strides[0] * n;
    int in_idx1_n = desc_x.strides[0] * n;
    int in_idx2_n = desc_y.strides[0] * n;
    for (int b = 0; b < desc_output.extents[1]; ++b) {
      int out_idx_b = (out_idx_n + b) * desc_output.extents[2];
      int cond_idx_b = cond_idx_n + desc_condition.strides[1] * b;
      int in_idx1_b = in_idx1_n + desc_x.strides[1] * b;
      int in_idx2_b = in_idx2_n + desc_y.strides[1] * b;
      for (int y = 0; y < desc_output.extents[2]; ++y) {
        int out_idx_y = (out_idx_b + y) * desc_output.extents[3];
        int cond_idx_y = cond_idx_b + desc_condition.strides[2] * y;
        int in_idx1_y = in_idx1_b + desc_x.strides[2] * y;
        int in_idx2_y = in_idx2_b + desc_y.strides[2] * y;
        for (int x = 0; x < desc_output.extents[3]; ++x) {
          int out_idx = (out_idx_y + x) * desc_output.extents[4];
          int cond_idx = cond_idx_y + desc_condition.strides[3] * x;
          int in_idx1 = in_idx1_y + desc_x.strides[3] * x;
          int in_idx2 = in_idx2_y + desc_y.strides[3] * x;
          for (int c = 0; c < desc_output.extents[4]; ++c) {
            output_data[out_idx] = input_condition_data[cond_idx]
                                       ? input_x_data[in_idx1]
                                       : input_y_data[in_idx2];
            out_idx++;
            cond_idx += desc_condition.strides[4];
            in_idx1 += desc_x.strides[4];
            in_idx2 += desc_y.strides[4];
          }
        }
      }
    }
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_SELECT_H_
