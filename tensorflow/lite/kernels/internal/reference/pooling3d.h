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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_POOLING3D_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_POOLING3D_H_

#include <algorithm>
#include <limits>

#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace reference_ops {

template <typename T, typename ActivationT>
inline T RoundAndAverage(ActivationT sum, int count) {
  // Round to the closest integer value.
  return sum > 0 ? (sum + count / 2) / count : (sum - count / 2) / count;
}

template <>
inline float RoundAndAverage(float sum, int count) {
  // No rounding for float type.
  return sum / count;
}

template <typename T, typename ActivationT>
inline void AveragePool3D(const Pool3DParams& params,
                          const RuntimeShape& input_shape, const T* input_data,
                          const RuntimeShape& output_shape, T* output_data) {
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 5);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 5);

  ActivationT activation_min, activation_max;
  GetActivationParams(params, &activation_min, &activation_max);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int channels = MatchingDim(input_shape, 4, output_shape, 4);

  const int in_spatial_dim_1 = input_shape.Dims(1);
  const int in_spatial_dim_2 = input_shape.Dims(2);
  const int in_spatial_dim_3 = input_shape.Dims(3);
  const int out_spatial_dim_1 = output_shape.Dims(1);
  const int out_spatial_dim_2 = output_shape.Dims(2);
  const int out_spatial_dim_3 = output_shape.Dims(3);

  const int stride_spatial_dim_1 = params.stride_depth;
  const int stride_spatial_dim_2 = params.stride_height;
  const int stride_spatial_dim_3 = params.stride_width;
  const int filter_spatial_dim_1 = params.filter_depth;
  const int filter_spatial_dim_2 = params.filter_height;
  const int filter_spatial_dim_3 = params.filter_width;
  const int padding_spatial_dim_1 = params.padding_values.depth;
  const int padding_spatial_dim_2 = params.padding_values.height;
  const int padding_spatial_dim_3 = params.padding_values.width;

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_d1 = 0; out_d1 < out_spatial_dim_1; ++out_d1) {
      const int in_d1_origin =
          (out_d1 * stride_spatial_dim_1) - padding_spatial_dim_1;
      const int filter_d1_start = std::max(0, -in_d1_origin);
      const int filter_d1_end =
          std::min(filter_spatial_dim_1, in_spatial_dim_1 - in_d1_origin);
      for (int out_d2 = 0; out_d2 < out_spatial_dim_2; ++out_d2) {
        const int in_d2_origin =
            (out_d2 * stride_spatial_dim_2) - padding_spatial_dim_2;
        const int filter_d2_start = std::max(0, -in_d2_origin);
        const int filter_d2_end =
            std::min(filter_spatial_dim_2, in_spatial_dim_2 - in_d2_origin);
        for (int out_d3 = 0; out_d3 < out_spatial_dim_3; ++out_d3) {
          const int in_d3_origin =
              (out_d3 * stride_spatial_dim_3) - padding_spatial_dim_3;
          const int filter_d3_start = std::max(0, -in_d3_origin);
          const int filter_d3_end =
              std::min(filter_spatial_dim_3, in_spatial_dim_3 - in_d3_origin);
          for (int channel = 0; channel < channels; ++channel) {
            ActivationT total = 0;
            for (int filter_d1 = filter_d1_start; filter_d1 < filter_d1_end;
                 ++filter_d1) {
              const int in_d1 = in_d1_origin + filter_d1;
              for (int filter_d2 = filter_d2_start; filter_d2 < filter_d2_end;
                   ++filter_d2) {
                const int in_d2 = in_d2_origin + filter_d2;
                for (int filter_d3 = filter_d3_start; filter_d3 < filter_d3_end;
                     ++filter_d3) {
                  const int in_d3 = in_d3_origin + filter_d3;
                  total += input_data[Offset(input_shape, batch, in_d1, in_d2,
                                             in_d3, channel)];
                }
              }
            }
            const int filter_count = (filter_d1_end - filter_d1_start) *
                                     (filter_d2_end - filter_d2_start) *
                                     (filter_d3_end - filter_d3_start);
            T average = RoundAndAverage<T, ActivationT>(total, filter_count);
            average = std::max<T>(average, activation_min);
            average = std::min<T>(average, activation_max);
            output_data[Offset(output_shape, batch, out_d1, out_d2, out_d3,
                               channel)] = average;
          }
        }
      }
    }
  }
}

template <typename T, typename ActivationT>
inline void MaxPool3D(const Pool3DParams& params,
                      const RuntimeShape& input_shape, const T* input_data,
                      const RuntimeShape& output_shape, T* output_data) {
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 5);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 5);

  ActivationT activation_min, activation_max;
  GetActivationParams(params, &activation_min, &activation_max);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int channels = MatchingDim(input_shape, 4, output_shape, 4);

  const int in_spatial_dim_1 = input_shape.Dims(1);
  const int in_spatial_dim_2 = input_shape.Dims(2);
  const int in_spatial_dim_3 = input_shape.Dims(3);
  const int out_spatial_dim_1 = output_shape.Dims(1);
  const int out_spatial_dim_2 = output_shape.Dims(2);
  const int out_spatial_dim_3 = output_shape.Dims(3);

  const int stride_spatial_dim_1 = params.stride_depth;
  const int stride_spatial_dim_2 = params.stride_height;
  const int stride_spatial_dim_3 = params.stride_width;
  const int filter_spatial_dim_1 = params.filter_depth;
  const int filter_spatial_dim_2 = params.filter_height;
  const int filter_spatial_dim_3 = params.filter_width;
  const int padding_spatial_dim_1 = params.padding_values.depth;
  const int padding_spatial_dim_2 = params.padding_values.height;
  const int padding_spatial_dim_3 = params.padding_values.width;

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_d1 = 0; out_d1 < out_spatial_dim_1; ++out_d1) {
      const int in_d1_origin =
          (out_d1 * stride_spatial_dim_1) - padding_spatial_dim_1;
      const int filter_d1_start = std::max(0, -in_d1_origin);
      const int filter_d1_end =
          std::min(filter_spatial_dim_1, in_spatial_dim_1 - in_d1_origin);
      for (int out_d2 = 0; out_d2 < out_spatial_dim_2; ++out_d2) {
        const int in_d2_origin =
            (out_d2 * stride_spatial_dim_2) - padding_spatial_dim_2;
        const int filter_d2_start = std::max(0, -in_d2_origin);
        const int filter_d2_end =
            std::min(filter_spatial_dim_2, in_spatial_dim_2 - in_d2_origin);
        for (int out_d3 = 0; out_d3 < out_spatial_dim_3; ++out_d3) {
          const int in_d3_origin =
              (out_d3 * stride_spatial_dim_3) - padding_spatial_dim_3;
          const int filter_d3_start = std::max(0, -in_d3_origin);
          const int filter_d3_end =
              std::min(filter_spatial_dim_3, in_spatial_dim_3 - in_d3_origin);
          for (int channel = 0; channel < channels; ++channel) {
            T max = std::numeric_limits<T>::lowest();
            for (int filter_d1 = filter_d1_start; filter_d1 < filter_d1_end;
                 ++filter_d1) {
              const int in_d1 = in_d1_origin + filter_d1;
              for (int filter_d2 = filter_d2_start; filter_d2 < filter_d2_end;
                   ++filter_d2) {
                const int in_d2 = in_d2_origin + filter_d2;
                for (int filter_d3 = filter_d3_start; filter_d3 < filter_d3_end;
                     ++filter_d3) {
                  const int in_d3 = in_d3_origin + filter_d3;
                  max =
                      std::max(max, input_data[Offset(input_shape, batch, in_d1,
                                                      in_d2, in_d3, channel)]);
                }
              }
            }
            max = std::max<T>(max, activation_min);
            max = std::min<T>(max, activation_max);
            output_data[Offset(output_shape, batch, out_d1, out_d2, out_d3,
                               channel)] = max;
          }
        }
      }
    }
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_POOLING3D_H_
