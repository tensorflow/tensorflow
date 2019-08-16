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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_DEPTH_WISE_CONV_3X3_TEXTURE_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_DEPTH_WISE_CONV_3X3_TEXTURE_H_

#include <memory>
#include <vector>

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

class DepthWiseConv3x3Texture : public GPUOperation {
 public:
  DepthWiseConv3x3Texture() = default;
  Status AddToQueue(CLCommandQueue* queue) override;
  Status Tune(const TuningParameters& params) override;

  Status Compile(const CreationContext& creation_context) override;

  // Move only
  DepthWiseConv3x3Texture(DepthWiseConv3x3Texture&& kernel);
  DepthWiseConv3x3Texture& operator=(DepthWiseConv3x3Texture&& kernel);
  DepthWiseConv3x3Texture(const DepthWiseConv3x3Texture&) = delete;
  DepthWiseConv3x3Texture& operator=(const DepthWiseConv3x3Texture&) = delete;

 private:
  explicit DepthWiseConv3x3Texture(const OperationDef& definition);
  template <DataType T>
  Status UploadWeightsAndBiases(const ::tflite::gpu::Tensor<OHWI, T>& weights,
                                const ::tflite::gpu::Tensor<Linear, T>& biases,
                                CLContext* context);

  friend Status CreateDepthWiseConv3x3Texture(
      const CreationContext& creation_context, const OperationDef& definition,
      const DepthwiseConvolution2DAttributes& attr,
      DepthWiseConv3x3Texture* result);

  template <DataType S, typename T>
  void RearrangeWeightsAndBiasesData(
      const ::tflite::gpu::Tensor<OHWI, S>& weights,
      const ::tflite::gpu::Tensor<Linear, S>& biases, absl::Span<T> dst);

  Status BindArguments();
  int3 GetGridSize() const;

  Texture2D weights_;
  CLKernel kernel_;
  int3 work_group_size_ = int3(8, 4, 1);
};

template <DataType T>
Status DepthWiseConv3x3Texture::UploadWeightsAndBiases(
    const ::tflite::gpu::Tensor<OHWI, T>& weights,
    const ::tflite::gpu::Tensor<Linear, T>& biases, CLContext* context) {
  const int src_depth = IntegralDivideRoundUp(weights.shape.i, 4);
  int texture_width = 10;  // 3x3 kernel + 1 bias
  int texture_height = src_depth;
  const int elements_count = texture_width * texture_height;

  if (definition_.GetDataType() == DataType::FLOAT32) {
    std::vector<float4> gpu_data(elements_count);
    RearrangeWeightsAndBiasesData(weights, biases, absl::MakeSpan(gpu_data));
    return CreateTexture2DRGBA(definition_.GetDataType(), texture_width,
                               texture_height, gpu_data.data(), context,
                               &weights_);
  } else {
    std::vector<half4> gpu_data(elements_count);
    RearrangeWeightsAndBiasesData(weights, biases, absl::MakeSpan(gpu_data));
    return CreateTexture2DRGBA(definition_.GetDataType(), texture_width,
                               texture_height, gpu_data.data(), context,
                               &weights_);
  }
}

template <DataType S, typename T>
void DepthWiseConv3x3Texture::RearrangeWeightsAndBiasesData(
    const ::tflite::gpu::Tensor<OHWI, S>& weights,
    const ::tflite::gpu::Tensor<Linear, S>& biases, absl::Span<T> dst) {
  const int src_depth = IntegralDivideRoundUp(weights.shape.i, 4);

  int counter = 0;
  for (int s = 0; s < src_depth; ++s) {
    for (int y = 0; y < 3; ++y) {
      for (int x = 0; x < 3; ++x) {
        T filter_val;
        for (int i = 0; i < 4; ++i) {
          const int s_ch = s * 4 + i;
          if (s_ch < weights.shape.i) {
            const int f_index = weights.shape.LinearIndex({0, y, x, s_ch});
            filter_val[i] = weights.data[f_index];
          } else {
            filter_val[i] = 0.0f;
          }
        }
        dst[counter++] = filter_val;
      }
    }

    T bias_val;
    for (int i = 0; i < 4; ++i) {
      const int dst_ch = s * 4 + i;
      bias_val[i] = dst_ch >= biases.shape.v ? 0.0f : biases.data[dst_ch];
    }
    dst[counter++] = bias_val;
  }
}

bool IsDepthWiseConv3x3TextureSupported(
    const DepthwiseConvolution2DAttributes& attr);

Status CreateDepthWiseConv3x3Texture(
    const CreationContext& creation_context, const OperationDef& definition,
    const DepthwiseConvolution2DAttributes& attr,
    DepthWiseConv3x3Texture* result);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_DEPTH_WISE_CONV_3X3_TEXTURE_H_
