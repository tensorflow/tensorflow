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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CONVOLUTION_TRANSPOSED_4X4_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CONVOLUTION_TRANSPOSED_4X4_H_

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

class ConvolutionTransposed4x4 : public GPUOperation {
 public:
  ConvolutionTransposed4x4() = default;
  Status AddToQueue(CLCommandQueue* queue) override;
  Status Tune(const TuningParameters& params) override;
  Status Compile(const CreationContext& creation_context) override;

  // Move only
  ConvolutionTransposed4x4(ConvolutionTransposed4x4&& operation);
  ConvolutionTransposed4x4& operator=(ConvolutionTransposed4x4&& operation);
  ConvolutionTransposed4x4(const ConvolutionTransposed4x4&) = delete;
  ConvolutionTransposed4x4& operator=(const ConvolutionTransposed4x4&) = delete;

  enum class WeightsUploadType {
    LOCAL_MEM_ASYNC,
    LOCAL_MEM_BY_THREADS,
    GLOBAL_MEM,
  };

 private:
  ConvolutionTransposed4x4(const OperationDef& definition,
                           const CLDevice& device);
  friend Status CreateConvolutionTransposed4x4(
      const CreationContext& creation_context, const OperationDef& definition,
      const ConvolutionTransposedAttributes& attr,
      ConvolutionTransposed4x4* result);
  template <DataType T>
  Status UploadWeights(const ::tflite::gpu::Tensor<OHWI, T>& weights,
                       CLContext* context);

  template <DataType S, typename T>
  void RearrangeWeightsData(const ::tflite::gpu::Tensor<OHWI, S>& weights,
                            absl::Span<T> dst);

  Status BindArguments();
  int3 GetGridSize() const;

  Buffer weights_;
  WeightsUploadType weights_upload_type_;
  LinearStorage biases_;

  CLKernel kernel_;
  int3 work_group_size_ = int3(8, 4, 1);
};

template <DataType T>
Status ConvolutionTransposed4x4::UploadWeights(
    const ::tflite::gpu::Tensor<OHWI, T>& weights, CLContext* context) {
  const int src_depth = IntegralDivideRoundUp(weights.shape.i, 4);
  const int dst_depth = IntegralDivideRoundUp(weights.shape.o, 4);
  const int kernel_x = 4;  //  This operation support only 4x4 kernel
  const int kernel_y = 4;
  const int flt4_count = kernel_x * kernel_y * src_depth * dst_depth * 4;

  const bool f32_weights = definition_.precision == CalculationsPrecision::F32;
  const int flt4_size = f32_weights ? sizeof(float4) : sizeof(half4);

  if (f32_weights) {
    std::vector<float4> gpu_data(flt4_count);
    RearrangeWeightsData(weights, absl::MakeSpan(gpu_data));
    return CreateReadOnlyBuffer(flt4_size * flt4_count, gpu_data.data(),
                                context, &weights_);
  } else {
    std::vector<half4> gpu_data(flt4_count);
    RearrangeWeightsData(weights, absl::MakeSpan(gpu_data));
    return CreateReadOnlyBuffer(flt4_size * flt4_count, gpu_data.data(),
                                context, &weights_);
  }
}

template <DataType S, typename T>
void ConvolutionTransposed4x4::RearrangeWeightsData(
    const ::tflite::gpu::Tensor<OHWI, S>& weights, absl::Span<T> dst) {
  const int src_depth = IntegralDivideRoundUp(weights.shape.i, 4);
  const int dst_depth = IntegralDivideRoundUp(weights.shape.o, 4);
  const int kernel_x = 4;
  const int kernel_y = 4;

  const int remap[16] = {10, 11, 14, 15, 8, 9, 12, 13, 2, 3, 6, 7, 0, 1, 4, 5};

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

bool IsConvolutionTransposed4x4Supported(
    const CLDevice& device, const OperationDef& definition,
    const ConvolutionTransposedAttributes& attr);

Status CreateConvolutionTransposed4x4(
    const CreationContext& creation_context, const OperationDef& definition,
    const ConvolutionTransposedAttributes& attr,
    ConvolutionTransposed4x4* result);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CONVOLUTION_TRANSPOSED_4X4_H_
