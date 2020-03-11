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

class ConvConstants : public GPUOperation {
 public:
  ConvConstants() = default;
  Status AddToQueue(CLCommandQueue* queue) override;
  Status Tune(const TuningParameters& params) override;

  Status Compile(const CreationContext& creation_context) override;

  // Move only
  ConvConstants(ConvConstants&& kernel);
  ConvConstants& operator=(ConvConstants&& kernel);
  ConvConstants(const ConvConstants&) = delete;
  ConvConstants& operator=(const ConvConstants&) = delete;

 private:
  friend Status CreateConvConstants(const CreationContext& creation_context,
                                    const OperationDef& definition,
                                    const Convolution2DAttributes& attr,
                                    ConvConstants* result);
  explicit ConvConstants(const OperationDef& definition,
                         const Convolution2DAttributes& attr)
      : GPUOperation(definition),
        kernel_size_(attr.weights.shape.w, attr.weights.shape.h),
        stride_(attr.strides.w, attr.strides.h),
        padding_(-attr.padding.prepended.w, -attr.padding.prepended.h),
        dilation_(attr.dilations.w, attr.dilations.h),
        src_channels_(attr.weights.shape.i),
        dst_channels_(attr.weights.shape.o) {}

  template <DataType T>
  Status UploadWeights(const ::tflite::gpu::Tensor<OHWI, T>& weights,
                       CLContext* context);

  template <DataType S, typename T>
  void RearrangeWeightsData(const ::tflite::gpu::Tensor<OHWI, S>& weights,
                            absl::Span<T> dst);

  Status BindArguments();
  int3 GetGridSize() const;

  Buffer weights_;
  LinearStorage biases_;

  int2 kernel_size_;
  int2 stride_;
  int2 padding_;
  int2 dilation_;
  int src_channels_;
  int dst_channels_;

  CLKernel kernel_;
  int3 work_group_size_ = int3(8, 4, 1);
};

template <DataType T>
Status ConvConstants::UploadWeights(
    const ::tflite::gpu::Tensor<OHWI, T>& weights, CLContext* context) {
  const int dst_depth = IntegralDivideRoundUp(weights.shape.o, 4);
  const int kernel_x = weights.shape.w;
  const int kernel_y = weights.shape.h;

  const int float_size =
      definition_.precision == CalculationsPrecision::F32 ? 4 : 2;
  const int float_count = src_channels_ * dst_depth * 4 * kernel_x * kernel_y;

  if (definition_.GetDataType() == DataType::FLOAT32) {
    std::vector<float4> gpu_data(float_count / 4);
    RearrangeWeightsData(weights, absl::MakeSpan(gpu_data));
    return CreateReadOnlyBuffer(float_size * float_count, gpu_data.data(),
                                context, &weights_);
  } else {
    std::vector<half4> gpu_data(float_count / 4);
    RearrangeWeightsData(weights, absl::MakeSpan(gpu_data));
    return CreateReadOnlyBuffer(float_size * float_count, gpu_data.data(),
                                context, &weights_);
  }
}

template <DataType S, typename T>
void ConvConstants::RearrangeWeightsData(
    const ::tflite::gpu::Tensor<OHWI, S>& weights, absl::Span<T> dst) {
  const int dst_depth = IntegralDivideRoundUp(weights.shape.o, 4);
  const int src_depth = IntegralDivideRoundUp(weights.shape.i, 4);
  const int kernel_x = weights.shape.w;
  const int kernel_y = weights.shape.h;

  int counter = 0;
  for (int s = 0; s < src_depth; ++s) {
    for (int y = 0; y < kernel_y; ++y) {
      for (int x = 0; x < kernel_x; ++x) {
        for (int d = 0; d < dst_depth; ++d) {
          const int channels_count = std::min(4, src_channels_ - s * 4);
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

bool IsConvConstantsSupported(const CLDevice& device,
                              const OperationDef& definition,
                              const Convolution2DAttributes& attr);

Status CreateConvConstants(const CreationContext& creation_context,
                           const OperationDef& definition,
                           const Convolution2DAttributes& attr,
                           ConvConstants* result);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CONV_CONSTANTS_H_
