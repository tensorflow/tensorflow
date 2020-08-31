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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CONV_CONSTANTS_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CONV_CONSTANTS_H_

#include "tensorflow/lite/delegates/gpu/cl/buffer.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/cl/linear_storage.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor.h"
#include "tensorflow/lite/delegates/gpu/cl/util.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace cl {

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
                filters[i][j] = weights.data[f_index];
              } else {
                filters[i][j] = 0.0f;
              }
            }
          }
          T filters_new[4];
          for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
              filters_new[i][j] = filters[j][i];
            }
          }
          for (int i = 0; i < channels_count; ++i) {
            dst[counter++] = filters_new[i];
          }
        }
      }
    }
  }
}

template <DataType T>
void UploadWeightsForConvConstants(const tflite::gpu::Tensor<OHWI, T>& weights,
                                   CalculationsPrecision precision,
                                   GPUOperation* op) {
  const int dst_depth = DivideRoundUp(weights.shape.o, 4);
  const int kernel_x = weights.shape.w;
  const int kernel_y = weights.shape.h;

  const bool f32_weights = precision == CalculationsPrecision::F32;
  const int float_size = f32_weights ? 4 : 2;
  const int float_count = weights.shape.i * dst_depth * 4 * kernel_x * kernel_y;

  BufferDescriptor desc;
  desc.element_type = f32_weights ? DataType::FLOAT32 : DataType::FLOAT16;
  desc.element_size = 4;
  desc.memory_type = MemoryType::CONSTANT;
  desc.size = float_size * float_count;
  desc.data.resize(desc.size);

  if (f32_weights) {
    float4* ptr = reinterpret_cast<float4*>(desc.data.data());
    RearrangeWeightsForConvConstants(weights,
                                     absl::MakeSpan(ptr, float_count / 4));
  } else {
    half4* ptr = reinterpret_cast<half4*>(desc.data.data());
    RearrangeWeightsForConvConstants(weights,
                                     absl::MakeSpan(ptr, float_count / 4));
  }

  op->args_.AddObject("weigths",
                      absl::make_unique<BufferDescriptor>(std::move(desc)));
}

bool IsConvConstantsSupported(const DeviceInfo& device_info,
                              const OperationDef& definition,
                              const Convolution2DAttributes& attr);

GPUOperation CreateConvConstants(const DeviceInfo& device_info,
                                 const OperationDef& definition,
                                 const Convolution2DAttributes& attr);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CONV_CONSTANTS_H_
