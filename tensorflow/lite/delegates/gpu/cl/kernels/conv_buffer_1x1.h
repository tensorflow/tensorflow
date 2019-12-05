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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CONV_BUFFER_1X1_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CONV_BUFFER_1X1_H_

#include "tensorflow/lite/delegates/gpu/cl/buffer.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_kernel.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/linear_storage.h"
#include "tensorflow/lite/delegates/gpu/cl/precision.h"
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

class ConvBuffer1x1 : public GPUOperation {
 public:
  ConvBuffer1x1() = default;

  // Move only
  ConvBuffer1x1(ConvBuffer1x1&& operation);
  ConvBuffer1x1& operator=(ConvBuffer1x1&& operation);
  ConvBuffer1x1(const ConvBuffer1x1&) = delete;
  ConvBuffer1x1& operator=(const ConvBuffer1x1&) = delete;

  Status AddToQueue(CLCommandQueue* queue) override;
  Status Tune(const TuningParameters& params) override;

  Status Compile(const CreationContext& creation_context) override;

 private:
  ConvBuffer1x1(const OperationDef& definition, int flt4_x_count,
                int flt4_y_count, int flt8_x_count, int flt8_y_count);
  friend Status CreateConvBuffer1x1(const CreationContext& creation_context,
                                    const OperationDef& definition,
                                    const Convolution2DAttributes& attr,
                                    ConvBuffer1x1* result);
  friend Status CreateConvBuffer1x1(const CreationContext& creation_context,
                                    const OperationDef& definition,
                                    const FullyConnectedAttributes& attr,
                                    ConvBuffer1x1* result);

  template <DataType T>
  Status UploadData(const ::tflite::gpu::Tensor<OHWI, T>& weights,
                    const ::tflite::gpu::Tensor<Linear, T>& biases,
                    CLContext* context);
  template <DataType T>
  Status UploadWeights(const ::tflite::gpu::Tensor<OHWI, T>& weights,
                       CLContext* context);

  Status BindArguments();
  int3 GetGridSize() const;

  CLKernel* GetKernel(int width);

  Buffer weights_;
  LinearStorage biases_;

  CLKernel kernel_flt4_;
  int flt4_x_count_;
  int flt4_y_count_;

  CLKernel kernel_flt8_;
  int flt8_x_count_;
  int flt8_y_count_;

  int3 work_group_size_;
};

template <DataType T>
Status ConvBuffer1x1::UploadData(const ::tflite::gpu::Tensor<OHWI, T>& weights,
                                 const ::tflite::gpu::Tensor<Linear, T>& biases,
                                 CLContext* context) {
  RETURN_IF_ERROR(UploadWeights(weights, context));
  LinearStorageCreateInfo create_info;
  create_info.storage_type = LinearStorageType::BUFFER;
  create_info.data_type = definition_.GetDataType();
  create_info.aligned_size = weights.shape.o;
  RETURN_IF_ERROR(CreateLinearStorage(create_info, biases, context, &biases_));
  return OkStatus();
}

template <DataType T>
Status ConvBuffer1x1::UploadWeights(
    const ::tflite::gpu::Tensor<OHWI, T>& weights, CLContext* context) {
  const int dst_depth = IntegralDivideRoundUp(weights.shape.o, 4);
  const int src_depth = IntegralDivideRoundUp(weights.shape.i, 4);

  const int float4_size = definition_.precision == CalculationsPrecision::F32
                              ? sizeof(float4)
                              : sizeof(half4);

  const int elements_count =
      weights.shape.h * weights.shape.w * src_depth * dst_depth * 4;

  if (definition_.GetDataType() == DataType::FLOAT32) {
    std::vector<float4> gpu_data(elements_count);
    RearrangeWeightsToOHWIOGroupI4O4(weights, /*out_group_size*/ 1,
                                     absl::MakeSpan(gpu_data));
    return CreateReadOnlyBuffer(float4_size * elements_count, gpu_data.data(),
                                context, &weights_);
  } else {
    std::vector<half4> gpu_data(elements_count);
    RearrangeWeightsToOHWIOGroupI4O4(weights, /*out_group_size*/ 1,
                                     absl::MakeSpan(gpu_data));
    return CreateReadOnlyBuffer(float4_size * elements_count, gpu_data.data(),
                                context, &weights_);
  }
}

bool IsConvBuffer1x1Supported(const OperationDef& definition,
                              const Convolution2DAttributes& attr);

Status CreateConvBuffer1x1(const CreationContext& creation_context,
                           const OperationDef& definition,
                           const Convolution2DAttributes& attr,
                           ConvBuffer1x1* result);

Status CreateConvBuffer1x1(const CreationContext& creation_context,
                           const OperationDef& definition,
                           const FullyConnectedAttributes& attr,
                           ConvBuffer1x1* result);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CONV_BUFFER_1X1_H_
