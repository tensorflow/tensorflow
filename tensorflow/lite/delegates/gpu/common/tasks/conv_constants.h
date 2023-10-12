/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONV_CONSTANTS_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONV_CONSTANTS_H_

#include <memory>
#include <utility>

#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/buffer_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {

template <DataType S, typename T>
void RearrangeWeightsForConvConstants(
    const tflite::gpu::Tensor<OHWI, S>& weights, absl::Span<T> dst) {
  const int dst_depth = DivideRoundUp(weights.shape.o, 4);
  const int src_depth = DivideRoundUp(weights.shape.i, 4);
  const int kernel_x = weights.shape.w;
  const int kernel_y = weights.shape.h;

  int counter = 0;
  for (int s = 0; s < src_depth; ++s) {
    for (int y = 0; y < kernel_y; ++y) {
      for (int x = 0; x < kernel_x; ++x) {
        for (int d = 0; d < dst_depth; ++d) {
          const int channels_count = std::min(4, weights.shape.i - s * 4);
          T filters[4];
          for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < channels_count; ++j) {
              const int s_ch = s * 4 + j;
              const int d_ch = d * 4 + i;
              if (s_ch < weights.shape.i && d_ch < weights.shape.o) {
                const int f_index =
                    weights.shape.LinearIndex({d_ch, y, x, s_ch});
                filters[j][i] = weights.data[f_index];
              } else {
                filters[j][i] = 0.0f;
              }
            }
          }
          for (int i = 0; i < channels_count; ++i) {
            dst[counter++] = filters[i];
          }
        }
      }
    }
  }
}

template <DataType S, typename T>
void RearrangeWeightsForConvConstantsDot(
    const tflite::gpu::Tensor<OHWI, S>& weights, absl::Span<T> dst) {
  const int dst_depth = DivideRoundUp(weights.shape.o, 4);
  const int src_depth = DivideRoundUp(weights.shape.i, 4);
  const int kernel_x = weights.shape.w;
  const int kernel_y = weights.shape.h;

  int counter = 0;
  for (int s = 0; s < src_depth; ++s) {
    for (int y = 0; y < kernel_y; ++y) {
      for (int x = 0; x < kernel_x; ++x) {
        for (int d = 0; d < dst_depth; ++d) {
          const int channels_count = std::min(4, weights.shape.o - d * 4);
          T filters[4];
          for (int j = 0; j < channels_count; ++j) {
            for (int i = 0; i < 4; ++i) {
              const int s_ch = s * 4 + i;
              const int d_ch = d * 4 + j;
              if (s_ch < weights.shape.i && d_ch < weights.shape.o) {
                const int f_index =
                    weights.shape.LinearIndex({d_ch, y, x, s_ch});
                filters[j][i] = weights.data[f_index];
              } else {
                filters[j][i] = 0.0f;
              }
            }
          }
          for (int i = 0; i < channels_count; ++i) {
            dst[counter++] = filters[i];
          }
        }
      }
    }
  }
}

template <DataType T>
void UploadWeightsForConvConstants(const tflite::gpu::Tensor<OHWI, T>& weights,
                                   const GpuInfo& gpu_info,
                                   CalculationsPrecision precision,
                                   bool use_dot_conv, GPUOperation* op) {
  const int src_depth = DivideRoundUp(weights.shape.i, 4);
  const int dst_depth = DivideRoundUp(weights.shape.o, 4);
  const int kernel_x = weights.shape.w;
  const int kernel_y = weights.shape.h;

  const bool f32_weights = precision == CalculationsPrecision::F32;
  const int float_size = f32_weights ? 4 : 2;
  const int aligned_ch_count = use_dot_conv ? weights.shape.o * src_depth * 4
                                            : weights.shape.i * dst_depth * 4;
  const int float_count = aligned_ch_count * kernel_x * kernel_y;

  BufferDescriptor desc;
  desc.element_type = f32_weights ? DataType::FLOAT32 : DataType::FLOAT16;
  desc.element_size = 4;
  if (gpu_info.IsApiOpenCl() || gpu_info.IsApiMetal()) {
    desc.memory_type = MemoryType::CONSTANT;
  } else {
    desc.memory_type = MemoryType::GLOBAL;
  }
  desc.size = float_size * float_count;
  desc.data.resize(desc.size);

  if (f32_weights) {
    float4* ptr = reinterpret_cast<float4*>(desc.data.data());
    if (use_dot_conv) {
      RearrangeWeightsForConvConstantsDot(weights,
                                          absl::MakeSpan(ptr, float_count / 4));
    } else {
      RearrangeWeightsForConvConstants(weights,
                                       absl::MakeSpan(ptr, float_count / 4));
    }
  } else {
    half4* ptr = reinterpret_cast<half4*>(desc.data.data());
    if (use_dot_conv) {
      RearrangeWeightsForConvConstantsDot(weights,
                                          absl::MakeSpan(ptr, float_count / 4));
    } else {
      RearrangeWeightsForConvConstants(weights,
                                       absl::MakeSpan(ptr, float_count / 4));
    }
  }

  op->args_.AddObject("weights",
                      std::make_unique<BufferDescriptor>(std::move(desc)));
}

bool IsConvConstantsSupported(const GpuInfo& gpu_info,
                              const OperationDef& definition,
                              const Convolution2DAttributes& attr);

GPUOperation CreateConvConstants(const GpuInfo& gpu_info,
                                 const OperationDef& definition,
                                 const Convolution2DAttributes& attr);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONV_CONSTANTS_H_
