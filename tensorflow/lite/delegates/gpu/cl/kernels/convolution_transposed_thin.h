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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CONVOLUTION_TRANSPOSED_THIN_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CONVOLUTION_TRANSPOSED_THIN_H_

#include <vector>

#include "tensorflow/lite/delegates/gpu/cl/buffer.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/flt_type.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor.h"
#include "tensorflow/lite/delegates/gpu/cl/texture2d.h"
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

class ConvolutionTransposedThin : public GPUOperation {
 public:
  ConvolutionTransposedThin() = default;
  absl::Status AddToQueue(CLCommandQueue* queue) override;
  absl::Status Tune(const TuningParameters& params) override;

  absl::Status Compile(const CreationContext& creation_context) override;

  // Move only
  ConvolutionTransposedThin(ConvolutionTransposedThin&& operation);
  ConvolutionTransposedThin& operator=(ConvolutionTransposedThin&& operation);
  ConvolutionTransposedThin(const ConvolutionTransposedThin&) = delete;
  ConvolutionTransposedThin& operator=(const ConvolutionTransposedThin&) =
      delete;

 private:
  friend absl::Status CreateConvolutionTransposedThin(
      const CreationContext& creation_context, const OperationDef& definition,
      const ConvolutionTransposedAttributes& attr,
      ConvolutionTransposedThin* result);
  ConvolutionTransposedThin(const OperationDef& definition,
                            const ConvolutionTransposedAttributes& attr);
  template <DataType T>
  absl::Status UploadWeights(const tflite::gpu::Tensor<OHWI, T>& weights,
                             CLContext* context);

  template <DataType S, typename T>
  void RearrangeWeightsData(const tflite::gpu::Tensor<OHWI, S>& weights,
                            absl::Span<T> dst);

  absl::Status BindArguments();
  int3 GetGridSize() const;

  Buffer weights_buf_;
  FLT4 bias_value_;

  int2 kernel_size_;
  int src_channels_;
  int dst_channels_;

  CLKernel kernel_;
  int3 work_group_size_ = int3(8, 4, 1);
};

template <DataType T>
absl::Status ConvolutionTransposedThin::UploadWeights(
    const tflite::gpu::Tensor<OHWI, T>& weights, CLContext* context) {
  const int src_depth = DivideRoundUp(src_channels_, 4);
  const int elements_count =
      kernel_size_.x * kernel_size_.y * src_depth * 4 * dst_channels_;

  const int float4_size =
      definition_.precision == CalculationsPrecision::F32 ? 16 : 8;
  if (definition_.GetDataType() == DataType::FLOAT32) {
    std::vector<float4> gpu_data(elements_count);
    RearrangeWeightsData(weights, absl::MakeSpan(gpu_data));
    return CreateReadOnlyBuffer(float4_size * elements_count, gpu_data.data(),
                                context, &weights_buf_);
  } else {
    std::vector<half4> gpu_data(elements_count);
    RearrangeWeightsData(weights, absl::MakeSpan(gpu_data));
    return CreateReadOnlyBuffer(float4_size * elements_count, gpu_data.data(),
                                context, &weights_buf_);
  }
}

template <DataType S, typename T>
void ConvolutionTransposedThin::RearrangeWeightsData(
    const tflite::gpu::Tensor<OHWI, S>& weights, absl::Span<T> dst) {
  const int src_depth = DivideRoundUp(src_channels_, 4);
  const int kernel_x = kernel_size_.x;
  const int kernel_y = kernel_size_.y;

  int counter = 0;
  for (int s = 0; s < src_depth; ++s) {
    for (int y = 0; y < kernel_y; ++y) {
      for (int x = 0; x < kernel_x; ++x) {
        std::vector<T> filters(dst_channels_);
        for (int j = 0; j < dst_channels_; ++j) {
          for (int i = 0; i < 4; ++i) {
            const int s_ch = s * 4 + i;
            const int d_ch = j;
            if (s_ch < src_channels_ && d_ch < dst_channels_) {
              const int f_index = weights.shape.LinearIndex({d_ch, y, x, s_ch});
              filters[j][i] = weights.data[f_index];
            } else {
              filters[j][i] = 0.0f;
            }
          }
        }
        for (int j = 0; j < dst_channels_; ++j) {
          dst[counter++] = filters[j];
        }
      }
    }
  }
}

bool IsConvolutionTransposedThinSupported(
    const CLDevice& device, const ConvolutionTransposedAttributes& attr);

absl::Status CreateConvolutionTransposedThin(
    const CreationContext& creation_context, const OperationDef& definition,
    const ConvolutionTransposedAttributes& attr,
    ConvolutionTransposedThin* result);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CONVOLUTION_TRANSPOSED_THIN_H_
