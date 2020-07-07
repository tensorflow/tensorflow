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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_DEPTHWISE_CONV_3X3_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_DEPTHWISE_CONV_3X3_H_

#include <memory>
#include <vector>

#include "tensorflow/lite/delegates/gpu/cl/buffer.h"
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

class DepthwiseConv3x3 : public GPUOperation {
 public:
  DepthwiseConv3x3() = default;
  absl::Status AddToQueue(CLCommandQueue* queue) override;
  absl::Status Tune(const TuningParameters& params) override;

  absl::Status Compile(const CreationContext& creation_context) override;

  // Move only
  DepthwiseConv3x3(DepthwiseConv3x3&& operation);
  DepthwiseConv3x3& operator=(DepthwiseConv3x3&& operation);
  DepthwiseConv3x3(const DepthwiseConv3x3&) = delete;
  DepthwiseConv3x3& operator=(const DepthwiseConv3x3&) = delete;

 private:
  explicit DepthwiseConv3x3(const OperationDef& definition,
                            bool weights_are_buffer, bool local_mem_uploads);
  template <DataType T>
  absl::Status UploadWeightsAndBiases(
      const tflite::gpu::Tensor<OHWI, T>& weights,
      const tflite::gpu::Tensor<Linear, T>& biases, CLContext* context);

  friend absl::Status CreateDepthwiseConv3x3(
      const CreationContext& creation_context, const OperationDef& definition,
      const DepthwiseConvolution2DAttributes& attr, DepthwiseConv3x3* result);

  template <DataType S, typename T>
  void RearrangeWeightsAndBiasesData(
      const tflite::gpu::Tensor<OHWI, S>& weights,
      const tflite::gpu::Tensor<Linear, S>& biases, absl::Span<T> dst);

  absl::Status BindArguments();
  int3 GetGridSize() const;

  bool weights_are_buffer_;
  bool local_mem_uploads_;

  CLKernel kernel_;
  int3 work_group_size_ = int3(8, 4, 1);
};

template <DataType T>
absl::Status DepthwiseConv3x3::UploadWeightsAndBiases(
    const tflite::gpu::Tensor<OHWI, T>& weights,
    const tflite::gpu::Tensor<Linear, T>& biases, CLContext* context) {
  const int src_depth = DivideRoundUp(weights.shape.i, 4);
  int texture_width = 10;  // 3x3 kernel + 1 bias
  int texture_height = src_depth;
  const int elements_count = texture_width * texture_height;
  const bool fp32_weights = definition_.precision == CalculationsPrecision::F32;
  const int float4_size = fp32_weights ? 16 : 8;

  Texture2D weights_tex2d;
  Buffer weights_buf;
  if (fp32_weights) {
    std::vector<float4> gpu_data(elements_count);
    RearrangeWeightsAndBiasesData(weights, biases, absl::MakeSpan(gpu_data));
    if (weights_are_buffer_) {
      RETURN_IF_ERROR(CreateReadOnlyBuffer(float4_size * elements_count,
                                           gpu_data.data(), context,
                                           &weights_buf));
    } else {
      RETURN_IF_ERROR(CreateTexture2DRGBA(
          definition_.GetDataType(), texture_width, texture_height,
          gpu_data.data(), context, &weights_tex2d));
    }
  } else {
    std::vector<half4> gpu_data(elements_count);
    RearrangeWeightsAndBiasesData(weights, biases, absl::MakeSpan(gpu_data));
    if (weights_are_buffer_) {
      RETURN_IF_ERROR(CreateReadOnlyBuffer(float4_size * elements_count,
                                           gpu_data.data(), context,
                                           &weights_buf));
    } else {
      RETURN_IF_ERROR(CreateTexture2DRGBA(
          definition_.GetDataType(), texture_width, texture_height,
          gpu_data.data(), context, &weights_tex2d));
    }
  }

  if (weights_are_buffer_) {
    BufferDescriptor desc;
    desc.element_type = fp32_weights ? DataType::FLOAT32 : DataType::FLOAT16;
    desc.element_size = 4;
    args_.AddObject("weights", AccessType::READ,
                    absl::make_unique<Buffer>(std::move(weights_buf)),
                    absl::make_unique<BufferDescriptor>(desc));
  } else {
    Texture2DDescriptor desc;
    desc.element_type = fp32_weights ? DataType::FLOAT32 : DataType::FLOAT16;
    args_.AddObject("weights", AccessType::READ,
                    absl::make_unique<Texture2D>(std::move(weights_tex2d)),
                    absl::make_unique<Texture2DDescriptor>(desc));
  }

  return absl::OkStatus();
}

template <DataType S, typename T>
void DepthwiseConv3x3::RearrangeWeightsAndBiasesData(
    const tflite::gpu::Tensor<OHWI, S>& weights,
    const tflite::gpu::Tensor<Linear, S>& biases, absl::Span<T> dst) {
  const int src_depth = DivideRoundUp(weights.shape.i, 4);

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

bool IsDepthwiseConv3x3Supported(const DepthwiseConvolution2DAttributes& attr);

absl::Status CreateDepthwiseConv3x3(
    const CreationContext& creation_context, const OperationDef& definition,
    const DepthwiseConvolution2DAttributes& attr, DepthwiseConv3x3* result);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_DEPTHWISE_CONV_3X3_H_
