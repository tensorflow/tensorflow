/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CONVOLUTION_TRANSPOSED_3X3_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CONVOLUTION_TRANSPOSED_3X3_H_

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

class ConvolutionTransposed3x3 : public GPUOperation {
 public:
  ConvolutionTransposed3x3() = default;
  absl::Status AddToQueue(CLCommandQueue* queue) override;
  absl::Status Compile(const CreationContext& creation_context) override;

  // Move only
  ConvolutionTransposed3x3(ConvolutionTransposed3x3&& operation);
  ConvolutionTransposed3x3& operator=(ConvolutionTransposed3x3&& operation);
  ConvolutionTransposed3x3(const ConvolutionTransposed3x3&) = delete;
  ConvolutionTransposed3x3& operator=(const ConvolutionTransposed3x3&) = delete;

  enum class WeightsUploadType {
    LOCAL_MEM_ASYNC,
    LOCAL_MEM_BY_THREADS,
    GLOBAL_MEM,
    CONSTANT_MEM,
  };

 private:
  ConvolutionTransposed3x3(const OperationDef& definition,
                           const CLDevice& device, int2 padding);
  friend absl::Status CreateConvolutionTransposed3x3(
      const CreationContext& creation_context, const OperationDef& definition,
      const ConvolutionTransposedAttributes& attr,
      ConvolutionTransposed3x3* result);
  template <DataType T>
  absl::Status UploadWeights(const tflite::gpu::Tensor<OHWI, T>& weights,
                             CLContext* context);

  template <DataType S, typename T>
  void RearrangeWeightsData(const tflite::gpu::Tensor<OHWI, S>& weights,
                            absl::Span<T> dst);

  absl::Status BindArguments();
  int3 GetGridSize() const;

  int2 padding_;
  int3 work_group_launch_order_;
  WeightsUploadType weights_upload_type_;

  CLKernel kernel_;
  int3 work_group_size_ = int3(8, 4, 1);
};

template <DataType T>
absl::Status ConvolutionTransposed3x3::UploadWeights(
    const tflite::gpu::Tensor<OHWI, T>& weights, CLContext* context) {
  const int src_depth = DivideRoundUp(weights.shape.i, 4);
  const int dst_depth = DivideRoundUp(weights.shape.o, 4);
  const int kernel_x = 3;  //  This operation support only 3x3 kernel
  const int kernel_y = 3;
  const int flt4_count = kernel_x * kernel_y * src_depth * dst_depth * 4;

  const bool f32_weights = definition_.precision == CalculationsPrecision::F32;
  const int flt4_size = f32_weights ? sizeof(float4) : sizeof(half4);

  Buffer weights_buffer;
  if (f32_weights) {
    std::vector<float4> gpu_data(flt4_count);
    RearrangeWeightsData(weights, absl::MakeSpan(gpu_data));
    RETURN_IF_ERROR(CreateReadOnlyBuffer(
        flt4_size * flt4_count, gpu_data.data(), context, &weights_buffer));
  } else {
    std::vector<half4> gpu_data(flt4_count);
    RearrangeWeightsData(weights, absl::MakeSpan(gpu_data));
    RETURN_IF_ERROR(CreateReadOnlyBuffer(
        flt4_size * flt4_count, gpu_data.data(), context, &weights_buffer));
  }

  BufferDescriptor desc;
  desc.element_type = f32_weights ? DataType::FLOAT32 : DataType::FLOAT16;
  desc.element_size = 4;
  desc.memory_type =
      weights_upload_type_ ==
              ConvolutionTransposed3x3::WeightsUploadType::CONSTANT_MEM
          ? MemoryType::CONSTANT
          : MemoryType::GLOBAL;

  args_.AddObject("weights", AccessType::READ,
                  absl::make_unique<Buffer>(std::move(weights_buffer)),
                  absl::make_unique<BufferDescriptor>(desc));

  return absl::OkStatus();
}

template <DataType S, typename T>
void ConvolutionTransposed3x3::RearrangeWeightsData(
    const tflite::gpu::Tensor<OHWI, S>& weights, absl::Span<T> dst) {
  const int src_depth = DivideRoundUp(weights.shape.i, 4);
  const int dst_depth = DivideRoundUp(weights.shape.o, 4);
  const int kernel_x = 3;
  const int kernel_y = 3;

  const int padding_x_rem = abs(padding_.x) % 2;
  const int padding_y_rem = abs(padding_.y) % 2;

  // we are reorganizing weights to read them sequentially in kernel
  std::vector<int> remap;
  if (padding_x_rem == 1 && padding_y_rem == 1) {
    remap = {4, 5, 3, 7, 1, 8, 6, 2, 0};
  } else if (padding_x_rem == 0 && padding_y_rem == 1) {
    remap = {5, 3, 4, 8, 6, 2, 0, 7, 1};
  } else if (padding_x_rem == 1 && padding_y_rem == 0) {
    remap = {7, 1, 8, 6, 2, 0, 4, 5, 3};
  } else {  // padding_x_rem == 0 && padding_y_rem == 0
    remap = {8, 6, 2, 0, 7, 1, 5, 3, 4};
  }

  int counter = 0;
  for (int d = 0; d < dst_depth; ++d) {
    for (int s = 0; s < src_depth; ++s) {
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
              if (s_ch < weights.shape.i && d_ch < weights.shape.o) {
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

bool IsConvolutionTransposed3x3Supported(
    const CLDevice& device, const OperationDef& definition,
    const ConvolutionTransposedAttributes& attr);

absl::Status CreateConvolutionTransposed3x3(
    const CreationContext& creation_context, const OperationDef& definition,
    const ConvolutionTransposedAttributes& attr,
    ConvolutionTransposed3x3* result);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CONVOLUTION_TRANSPOSED_3X3_H_
