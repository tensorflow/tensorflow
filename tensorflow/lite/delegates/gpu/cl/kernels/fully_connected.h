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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_FULLY_CONNECTED_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_FULLY_CONNECTED_H_

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

class FullyConnected : public GPUOperation {
 public:
  FullyConnected() = default;
  Status AddToQueue(CLCommandQueue* queue) override;

  Status Compile(const CreationContext& creation_context) override;

  // Move only
  FullyConnected(FullyConnected&& kernel);
  FullyConnected& operator=(FullyConnected&& kernel);
  FullyConnected(const FullyConnected&) = delete;
  FullyConnected& operator=(const FullyConnected&) = delete;

 private:
  explicit FullyConnected(const OperationDef& definition);
  friend Status CreateFullyConnected(const CreationContext& creation_context,
                                     const OperationDef& definition,
                                     const FullyConnectedAttributes& attr,
                                     FullyConnected* result);

  template <DataType T>
  Status UploadWeights(const ::tflite::gpu::Tensor<OHWI, T>& weights,
                       CLContext* context);

  template <DataType T, typename S>
  void RearrangeWeights(const ::tflite::gpu::Tensor<OHWI, T>& weights,
                        absl::Span<S> dst);

  Buffer weights_;
  LinearStorage biases_;
  CLKernel kernel_;
  int3 work_group_size_ = int3(0, 0, 0);
};

template <DataType T>
Status FullyConnected::UploadWeights(
    const ::tflite::gpu::Tensor<OHWI, T>& weights, CLContext* context) {
  const int src_depth = IntegralDivideRoundUp(weights.shape.i, 4);
  const int dst_depth = IntegralDivideRoundUp(weights.shape.o, 4);

  const int elements_count = src_depth * dst_depth * 4;
  const bool f32_weights = definition_.precision == CalculationsPrecision::F32;

  const int float4_size = f32_weights ? 16 : 8;

  if (definition_.GetDataType() == DataType::FLOAT32) {
    std::vector<float4> gpu_data(dst_depth * src_depth * 4);
    RearrangeWeights(weights, absl::MakeSpan(gpu_data));
    return CreateReadOnlyBuffer(float4_size * elements_count, gpu_data.data(),
                                context, &weights_);
  } else {
    std::vector<half4> gpu_data(dst_depth * src_depth * 4);
    RearrangeWeights(weights, absl::MakeSpan(gpu_data));
    return CreateReadOnlyBuffer(float4_size * elements_count, gpu_data.data(),
                                context, &weights_);
  }
}

template <DataType T, typename S>
void FullyConnected::RearrangeWeights(
    const ::tflite::gpu::Tensor<OHWI, T>& weights, absl::Span<S> dst) {
  const int src_depth = IntegralDivideRoundUp(weights.shape.i, 4);
  const int dst_depth = IntegralDivideRoundUp(weights.shape.o, 4);
  int counter = 0;

  for (int s = 0; s < src_depth; ++s) {
    for (int d = 0; d < dst_depth; ++d) {
      S filters[4];
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          const int dst_ch = d * 4 + i;
          const int src_ch = s * 4 + j;
          if (dst_ch < weights.shape.o && src_ch < weights.shape.i) {
            const int f_index =
                weights.shape.LinearIndex({dst_ch, 0, 0, src_ch});
            filters[i][j] = weights.data[f_index];
          } else {
            filters[i][j] = 0.0;
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

Status CreateFullyConnected(const CreationContext& creation_context,
                            const OperationDef& definition,
                            const FullyConnectedAttributes& attr,
                            FullyConnected* result);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_FULLY_CONNECTED_H_
