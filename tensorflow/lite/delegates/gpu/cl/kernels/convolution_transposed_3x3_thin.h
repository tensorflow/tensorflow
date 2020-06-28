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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CONVOLUTION_TRANSPOSED_3X3_THIN_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CONVOLUTION_TRANSPOSED_3X3_THIN_H_

#include <vector>

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

class ConvolutionTransposed3x3Thin : public GPUOperation {
 public:
  ConvolutionTransposed3x3Thin() = default;
  absl::Status AddToQueue(CLCommandQueue* queue) override;
  absl::Status Tune(const TuningParameters& params) override;

  absl::Status Compile(const CreationContext& creation_context) override;

  // Move only
  ConvolutionTransposed3x3Thin(ConvolutionTransposed3x3Thin&& operation);
  ConvolutionTransposed3x3Thin& operator=(
      ConvolutionTransposed3x3Thin&& operation);
  ConvolutionTransposed3x3Thin(const ConvolutionTransposed3x3Thin&) = delete;
  ConvolutionTransposed3x3Thin& operator=(const ConvolutionTransposed3x3Thin&) =
      delete;

 private:
  friend absl::Status CreateConvolutionTransposed3x3Thin(
      const CreationContext& creation_context, const OperationDef& definition,
      const ConvolutionTransposedAttributes& attr,
      ConvolutionTransposed3x3Thin* result);
  explicit ConvolutionTransposed3x3Thin(
      const OperationDef& definition,
      const ConvolutionTransposedAttributes& attr);
  template <DataType T>
  absl::Status UploadData(const tflite::gpu::Tensor<OHWI, T>& weights,
                          const tflite::gpu::Tensor<Linear, T>& biases,
                          CLContext* context);

  template <DataType S, typename T>
  void RearrangeWeightsData(const tflite::gpu::Tensor<OHWI, S>& weights,
                            absl::Span<T> dst);

  absl::Status BindArguments();
  int3 GetGridSize() const;

  int src_channels_;
  int dst_channels_;

  CLKernel kernel_;
  int3 work_group_size_ = int3(8, 4, 1);
};

template <DataType T>
absl::Status ConvolutionTransposed3x3Thin::UploadData(
    const tflite::gpu::Tensor<OHWI, T>& weights,
    const tflite::gpu::Tensor<Linear, T>& biases, CLContext* context) {
  const int src_depth = DivideRoundUp(src_channels_, 4);
  const int dst_depth = DivideRoundUp(dst_channels_, 4);
  const int kernel_x = 3;  //  This operation support only 3x3 kernel
  const int kernel_y = 3;
  const int flt4_count = kernel_x * kernel_y * src_depth * dst_depth * 4;

  const bool f32_weights = definition_.precision == CalculationsPrecision::F32;

  BufferDescriptor desc;
  desc.element_type = f32_weights ? DataType::FLOAT32 : DataType::FLOAT16;
  desc.element_size = 4;
  desc.memory_type = MemoryType::CONSTANT;

  Buffer weights_buffer;
  if (f32_weights) {
    std::vector<float4> gpu_data(flt4_count);
    RearrangeWeightsData(weights, absl::MakeSpan(gpu_data));
    for (int i = 0; i < dst_depth; ++i) {
      float4 bias_value(0.0f);
      for (int c = 0; c < 4; ++c) {
        int ch = i * 4 + c;
        bias_value[c] = ch < weights.shape.o ? biases.data[ch] : 0.0f;
      }
      gpu_data.push_back(bias_value);
    }
    RETURN_IF_ERROR(CreateReadOnlyBuffer(sizeof(float4) * gpu_data.size(),
                                         gpu_data.data(), context,
                                         &weights_buffer));
  } else {
    std::vector<half4> gpu_data(flt4_count);
    RearrangeWeightsData(weights, absl::MakeSpan(gpu_data));
    for (int i = 0; i < dst_depth; ++i) {
      half4 bias_value(0.0f);
      for (int c = 0; c < 4; ++c) {
        int ch = i * 4 + c;
        bias_value[c] = ch < weights.shape.o ? biases.data[ch] : 0.0f;
      }
      gpu_data.push_back(bias_value);
    }
    RETURN_IF_ERROR(CreateReadOnlyBuffer(sizeof(half4) * gpu_data.size(),
                                         gpu_data.data(), context,
                                         &weights_buffer));
  }

  args_.AddObject("weights", AccessType::READ,
                  absl::make_unique<Buffer>(std::move(weights_buffer)),
                  absl::make_unique<BufferDescriptor>(desc));

  return absl::OkStatus();
}

template <DataType S, typename T>
void ConvolutionTransposed3x3Thin::RearrangeWeightsData(
    const tflite::gpu::Tensor<OHWI, S>& weights, absl::Span<T> dst) {
  const int src_depth = DivideRoundUp(src_channels_, 4);
  const int dst_depth = DivideRoundUp(dst_channels_, 4);
  const int kernel_x = 3;
  const int kernel_y = 3;

  const int remap[9] = {4, 5, 3, 7, 1, 8, 6, 2, 0};

  int counter = 0;
  for (int s = 0; s < src_depth; ++s) {
    for (int d = 0; d < dst_depth; ++d) {
      for (int y = 0; y < kernel_y; ++y) {
        for (int x = 0; x < kernel_x; ++x) {
          const int kernel_index = remap[y * kernel_x + x];
          const int kernel_index_x = kernel_index % kernel_x;
          const int kernel_index_y = kernel_index / kernel_x;
          T filters[4];
          for (int j = 0; j < 4; ++j) {
            for (int i = 0; i < 4; ++i) {
              const int s_ch = s * 4 + i;
              const int d_ch = d * 4 + j;
              if (s_ch < src_channels_ && d_ch < dst_channels_) {
                const int f_index = weights.shape.LinearIndex(
                    {d_ch, kernel_index_y, kernel_index_x, s_ch});
                filters[i][j] = weights.data[f_index];
              } else {
                filters[i][j] = 0.0f;
              }
            }
          }
          dst[counter++] = filters[0];
          dst[counter++] = filters[1];
          dst[counter++] = filters[2];
          dst[counter++] = filters[3];
        }
      }
    }
  }
}

bool IsConvolutionTransposed3x3ThinSupported(
    const CLDevice& device, const ConvolutionTransposedAttributes& attr);

absl::Status CreateConvolutionTransposed3x3Thin(
    const CreationContext& creation_context, const OperationDef& definition,
    const ConvolutionTransposedAttributes& attr,
    ConvolutionTransposed3x3Thin* result);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CONVOLUTION_TRANSPOSED_3X3_THIN_H_
