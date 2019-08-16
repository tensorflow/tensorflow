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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CONV_TEXTURE_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CONV_TEXTURE_H_

#include <vector>

#include "tensorflow/lite/delegates/gpu/cl/cl_command_queue.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_context.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/cl/linear_storage.h"
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

// This convolution process 2x2x2(XxYxZ) block of FLT4 values per thread.
class ConvTexture : public GPUOperation {
 public:
  ConvTexture() = default;
  Status AddToQueue(CLCommandQueue* queue) override;
  Status Tune(const TuningParameters& params) override;

  Status Compile(const CreationContext& creation_context) override;

  // Move only
  ConvTexture(ConvTexture&& operation);
  ConvTexture& operator=(ConvTexture&& operation);
  ConvTexture(const ConvTexture&) = delete;
  ConvTexture& operator=(const ConvTexture&) = delete;

 private:
  friend Status CreateConvTexture(const CreationContext& creation_context,
                                  const OperationDef& definition,
                                  const Convolution2DAttributes& attr,
                                  ConvTexture* result);
  ConvTexture(const OperationDef& definition,
              const Convolution2DAttributes& attr);
  template <DataType T>
  Status UploadWeights(const ::tflite::gpu::Tensor<OHWI, T>& weights,
                       CLContext* context);

  template <DataType S, typename T>
  void RearrangeWeightsData(const ::tflite::gpu::Tensor<OHWI, S>& weights,
                            absl::Span<T> dst_0, absl::Span<T> dst_1,
                            absl::Span<T> dst_2, absl::Span<T> dst_3);

  Status BindArguments();
  int3 GetGridSize() const;

  Texture2D weights_0_;
  Texture2D weights_1_;
  Texture2D weights_2_;
  Texture2D weights_3_;
  LinearStorage biases_;

  int2 kernel_size_;
  int2 stride_;
  int2 padding_;
  int2 dilation_;

  CLKernel kernel_;
  int3 work_group_size_;
};

template <DataType T>
Status ConvTexture::UploadWeights(const ::tflite::gpu::Tensor<OHWI, T>& weights,
                                  CLContext* context) {
  const int dst_depth = AlignByN(IntegralDivideRoundUp(weights.shape.o, 4), 2);
  const int src_depth = IntegralDivideRoundUp(weights.shape.i, 4);

  int texture_width = dst_depth;
  int texture_height = src_depth * kernel_size_.x * kernel_size_.y;

  DataType data_type = definition_.GetDataType();

  const int elements_count = texture_width * texture_height;

  if (data_type == DataType::FLOAT32) {
    std::vector<float4> gpu_data_0(elements_count);
    std::vector<float4> gpu_data_1(elements_count);
    std::vector<float4> gpu_data_2(elements_count);
    std::vector<float4> gpu_data_3(elements_count);
    RearrangeWeightsData(weights, absl::MakeSpan(gpu_data_0),
                         absl::MakeSpan(gpu_data_1), absl::MakeSpan(gpu_data_2),
                         absl::MakeSpan(gpu_data_3));
    RETURN_IF_ERROR(CreateTexture2DRGBA(data_type, texture_width,
                                        texture_height, gpu_data_0.data(),
                                        context, &weights_0_));
    RETURN_IF_ERROR(CreateTexture2DRGBA(data_type, texture_width,
                                        texture_height, gpu_data_1.data(),
                                        context, &weights_1_));
    RETURN_IF_ERROR(CreateTexture2DRGBA(data_type, texture_width,
                                        texture_height, gpu_data_2.data(),
                                        context, &weights_2_));
    return CreateTexture2DRGBA(data_type, texture_width, texture_height,
                               gpu_data_3.data(), context, &weights_3_);
  } else {
    std::vector<half4> gpu_data_0(elements_count);
    std::vector<half4> gpu_data_1(elements_count);
    std::vector<half4> gpu_data_2(elements_count);
    std::vector<half4> gpu_data_3(elements_count);
    RearrangeWeightsData(weights, absl::MakeSpan(gpu_data_0),
                         absl::MakeSpan(gpu_data_1), absl::MakeSpan(gpu_data_2),
                         absl::MakeSpan(gpu_data_3));
    RETURN_IF_ERROR(CreateTexture2DRGBA(data_type, texture_width,
                                        texture_height, gpu_data_0.data(),
                                        context, &weights_0_));
    RETURN_IF_ERROR(CreateTexture2DRGBA(data_type, texture_width,
                                        texture_height, gpu_data_1.data(),
                                        context, &weights_1_));
    RETURN_IF_ERROR(CreateTexture2DRGBA(data_type, texture_width,
                                        texture_height, gpu_data_2.data(),
                                        context, &weights_2_));
    return CreateTexture2DRGBA(data_type, texture_width, texture_height,
                               gpu_data_3.data(), context, &weights_3_);
  }
}

template <DataType S, typename T>
void ConvTexture::RearrangeWeightsData(
    const ::tflite::gpu::Tensor<OHWI, S>& weights, absl::Span<T> dst_0,
    absl::Span<T> dst_1, absl::Span<T> dst_2, absl::Span<T> dst_3) {
  const int dst_depth = AlignByN(IntegralDivideRoundUp(weights.shape.o, 4), 2);
  const int src_depth = IntegralDivideRoundUp(weights.shape.i, 4);

  int texture_width = dst_depth;

  for (int d = 0; d < dst_depth / 2; ++d) {
    for (int y = 0; y < kernel_size_.y; ++y) {
      for (int x = 0; x < kernel_size_.x; ++x) {
        for (int s = 0; s < src_depth; ++s) {
          for (int sub_d = 0; sub_d < 2; ++sub_d) {
            T filters[4];
            for (int i = 0; i < 4; ++i) {
              for (int j = 0; j < 4; ++j) {
                const int s_ch = s * 4 + j;
                const int d_ch = (d * 2 + sub_d) * 4 + i;
                if (s_ch < weights.shape.i && d_ch < weights.shape.o) {
                  const int f_index =
                      weights.shape.LinearIndex({d_ch, y, x, s_ch});
                  filters[j][i] = weights.data[f_index];
                } else {
                  filters[j][i] = 0.0f;
                }
              }
            }
            int x_coord = d * 2 + sub_d;
            int y_coord = (y * kernel_size_.x + x) * src_depth + s;
            int offset = y_coord * texture_width + x_coord;
            dst_0[offset] = filters[0];
            dst_1[offset] = filters[1];
            dst_2[offset] = filters[2];
            dst_3[offset] = filters[3];
          }
        }
      }
    }
  }
}

Status CreateConvTexture(const CreationContext& creation_context,
                         const OperationDef& definition,
                         const Convolution2DAttributes& attr,
                         ConvTexture* result);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CONV_TEXTURE_H_
