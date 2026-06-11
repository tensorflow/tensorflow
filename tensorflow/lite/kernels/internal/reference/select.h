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

#include <algorithm>
#include <cmath>
#include <cstring>

#include "ruy/profiler/instrumentation.h"  // from @ruy
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/reference/broadcast_loop.h"
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
void RunSelectOp(const D* cond, const T* x, const T* y, T* output,
                 const size_t* cond_stride, const size_t* x_stride,
                 const size_t* y_stride, const size_t* output_stride,
                 const size_t* output_shape, int rank) {
  if (rank <= 0) {
    *output = *cond ? *x : *y;
  } else {
    if (rank == 1) {
      TFLITE_DCHECK_EQ(output_stride[0], 1);
      if (cond_stride[0] == 0) {
        if (*cond) {
          if (x_stride[0] == 0) {
            std::fill_n(output, output_shape[0], *x);
          } else {
            TFLITE_DCHECK_EQ(x_stride[0], 1);
            std::memcpy(output, x, output_shape[0] * sizeof(T));
          }
        } else {
          if (y_stride[0] == 0) {
            std::fill_n(output, output_shape[0], *y);
          } else {
            TFLITE_DCHECK_EQ(y_stride[0], 1);
            std::memcpy(output, y, output_shape[0] * sizeof(T));
          }
        }
      } else {
        TFLITE_DCHECK_EQ(cond_stride[0], 1);
        if (x_stride[0] == 0 && y_stride[0] == 0) {
          for (size_t i = 0; i < output_shape[0]; ++i) {
            output[i] = cond[i] ? *x : *y;
          }
        } else if (x_stride[0] == 0) {
          TFLITE_DCHECK_EQ(y_stride[0], 1);
          for (size_t i = 0; i < output_shape[0]; ++i) {
            output[i] = cond[i] ? *x : y[i];
          }
        } else if (y_stride[0] == 0) {
          TFLITE_DCHECK_EQ(x_stride[0], 1);
          for (size_t i = 0; i < output_shape[0]; ++i) {
            output[i] = cond[i] ? x[i] : *y;
          }
        } else {
          TFLITE_DCHECK_EQ(x_stride[0], 1);
          TFLITE_DCHECK_EQ(y_stride[0], 1);
          for (size_t i = 0; i < output_shape[0]; ++i) {
            output[i] = cond[i] ? x[i] : y[i];
          }
        }
      }
    } else {
      for (size_t i = 0; i < output_shape[0]; ++i) {
        RunSelectOp(cond + i * cond_stride[0], x + i * x_stride[0],
                    y + i * y_stride[0], output + i * output_stride[0],
                    cond_stride + 1, x_stride + 1, y_stride + 1,
                    output_stride + 1, output_shape + 1, rank - 1);
      }
    }
  }
}

template <typename D, typename T>
inline void BroadcastSelectSimple(const RuntimeShape& cond_shape,
                                  const D* cond_data,
                                  const RuntimeShape& x_shape, const T* x_data,
                                  const RuntimeShape& y_shape, const T* y_data,
                                  const RuntimeShape& output_shape,
                                  T* output_data) {
  constexpr int kMaxRank = 8;
  const int dims_count = std::max(
      output_shape.DimensionsCount(),
      std::max(cond_shape.DimensionsCount(),
               std::max(x_shape.DimensionsCount(), y_shape.DimensionsCount())));

  TFLITE_DCHECK_LE(dims_count, kMaxRank);

  const RuntimeShape extended_output_shape =
      RuntimeShape::ExtendedShape(dims_count, output_shape);
  const RuntimeShape extended_cond_shape =
      RuntimeShape::ExtendedShape(dims_count, cond_shape);
  const RuntimeShape extended_x_shape =
      RuntimeShape::ExtendedShape(dims_count, x_shape);
  const RuntimeShape extended_y_shape =
      RuntimeShape::ExtendedShape(dims_count, y_shape);

  size_t cond_strides[kMaxRank];
  size_t x_strides[kMaxRank];
  size_t y_strides[kMaxRank];
  size_t o_strides[kMaxRank];
  size_t o_shape[kMaxRank];

  size_t cond_accum_stride = 1;
  size_t x_accum_stride = 1;
  size_t y_accum_stride = 1;
  size_t o_accum_stride = 1;
  for (int i = dims_count - 1; i >= 0; --i) {
    cond_strides[i] =
        (extended_cond_shape.Dims(i) == 1 && extended_output_shape.Dims(i) != 1)
            ? 0
            : cond_accum_stride;
    x_strides[i] =
        (extended_x_shape.Dims(i) == 1 && extended_output_shape.Dims(i) != 1)
            ? 0
            : x_accum_stride;
    y_strides[i] =
        (extended_y_shape.Dims(i) == 1 && extended_output_shape.Dims(i) != 1)
            ? 0
            : y_accum_stride;
    o_strides[i] = o_accum_stride;
    o_shape[i] = extended_output_shape.Dims(i);

    cond_accum_stride *= extended_cond_shape.Dims(i);
    x_accum_stride *= extended_x_shape.Dims(i);
    y_accum_stride *= extended_y_shape.Dims(i);
    o_accum_stride *= extended_output_shape.Dims(i);
  }

  RunSelectOp(cond_data, x_data, y_data, output_data, cond_strides, x_strides,
              y_strides, o_strides, o_shape, dims_count);
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

  BroadcastSelectSimple(input_condition_shape, input_condition_data,
                        input_x_shape, input_x_data, input_y_shape,
                        input_y_data, output_shape, output_data);
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_SELECT_H_
