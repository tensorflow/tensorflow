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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONVOLUTION_TRANSPOSED_THIN_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONVOLUTION_TRANSPOSED_THIN_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/buffer_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/texture2d_desc.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {

class ConvolutionTransposedThin : public GPUOperation {
 public:
  ConvolutionTransposedThin() = default;
  int3 GetGridSize() const override;

  // Move only
  ConvolutionTransposedThin(ConvolutionTransposedThin&& operation);
  ConvolutionTransposedThin& operator=(ConvolutionTransposedThin&& operation);
  ConvolutionTransposedThin(const ConvolutionTransposedThin&) = delete;
  ConvolutionTransposedThin& operator=(const ConvolutionTransposedThin&) =
      delete;

 private:
  friend ConvolutionTransposedThin CreateConvolutionTransposedThin(
      const GpuInfo& gpu_info, const OperationDef& definition,
      const ConvolutionTransposedAttributes& attr);
  ConvolutionTransposedThin(const OperationDef& definition,
                            const ConvolutionTransposedAttributes& attr,
                            const GpuInfo& gpu_info);
  template <DataType T>
  void UploadData(const tflite::gpu::Tensor<OHWI, T>& weights,
                  const tflite::gpu::Tensor<Linear, T>& biases);

  template <DataType S, typename T>
  void RearrangeWeightsData(const tflite::gpu::Tensor<OHWI, S>& weights,
                            absl::Span<T> dst);
  std::string GenerateConvolutionTransposedCode(const OperationDef& op_def,
                                                int src_depth, int dst_channels,
                                                const int2& kernel_size);
};

template <DataType T>
void ConvolutionTransposedThin::UploadData(
    const tflite::gpu::Tensor<OHWI, T>& weights,
    const tflite::gpu::Tensor<Linear, T>& biases) {
  const int src_depth = DivideRoundUp(weights.shape.i, 4);
  const int flt4_count =
      weights.shape.w * weights.shape.h * src_depth * weights.shape.o;

  const bool f32_weights = definition_.precision == CalculationsPrecision::F32;
  const int flt4_size = f32_weights ? sizeof(float4) : sizeof(half4);

  BufferDescriptor desc;
  desc.element_type = f32_weights ? DataType::FLOAT32 : DataType::FLOAT16;
  desc.element_size = 4;
  desc.memory_type = MemoryType::CONSTANT;
  desc.size = flt4_size * (flt4_count + 1);
  desc.data.resize(desc.size);

  if (f32_weights) {
    float4* gpu_data = reinterpret_cast<float4*>(desc.data.data());
    RearrangeWeightsData(weights, absl::MakeSpan(gpu_data, flt4_count));
    float4 bias_value(0.0f);
    for (int i = 0; i < weights.shape.o; ++i) {
      bias_value[i] = biases.data[i];
    }
    gpu_data[flt4_count] = bias_value;
  } else {
    half4* gpu_data = reinterpret_cast<half4*>(desc.data.data());
    RearrangeWeightsData(weights, absl::MakeSpan(gpu_data, flt4_count));
    half4 bias_value(0.0f);
    for (int i = 0; i < weights.shape.o; ++i) {
      bias_value[i] = biases.data[i];
    }
    gpu_data[flt4_count] = bias_value;
  }

  args_.AddObject("weights",
                  std::make_unique<BufferDescriptor>(std::move(desc)));
}

template <DataType S, typename T>
void ConvolutionTransposedThin::RearrangeWeightsData(
    const tflite::gpu::Tensor<OHWI, S>& weights, absl::Span<T> dst) {
  const int src_depth = DivideRoundUp(weights.shape.i, 4);
  const int kernel_x = weights.shape.w;
  const int kernel_y = weights.shape.h;

  int counter = 0;
  for (int s = 0; s < src_depth; ++s) {
    for (int y = 0; y < kernel_y; ++y) {
      for (int x = 0; x < kernel_x; ++x) {
        std::vector<T> filters(weights.shape.o);
        for (int j = 0; j < weights.shape.o; ++j) {
          for (int i = 0; i < 4; ++i) {
            const int s_ch = s * 4 + i;
            const int d_ch = j;
            if (s_ch < weights.shape.i && d_ch < weights.shape.o) {
              const int f_index = weights.shape.LinearIndex({d_ch, y, x, s_ch});
              filters[j][i] = weights.data[f_index];
            } else {
              filters[j][i] = 0.0f;
            }
          }
        }
        for (int j = 0; j < weights.shape.o; ++j) {
          dst[counter++] = filters[j];
        }
      }
    }
  }
}

bool IsConvolutionTransposedThinSupported(
    const ConvolutionTransposedAttributes& attr);

ConvolutionTransposedThin CreateConvolutionTransposedThin(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const ConvolutionTransposedAttributes& attr);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONVOLUTION_TRANSPOSED_THIN_H_
