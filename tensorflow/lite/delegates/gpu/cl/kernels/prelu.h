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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_PRELU_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_PRELU_H_

#include <string>

#include "tensorflow/lite/delegates/gpu/cl/cl_context.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_kernel.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/flt_type.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/cl/linear_storage.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"

namespace tflite {
namespace gpu {
namespace cl {

class PReLU : public ElementwiseOperation {
 public:
  PReLU() = default;
  // Move only
  PReLU(PReLU&& operation);
  PReLU& operator=(PReLU&& operation);
  PReLU(const PReLU&) = delete;
  PReLU& operator=(const PReLU&) = delete;

  void SetLinkIndex(int index) override;
  std::string GetCoreCode(const LinkingContext& context) const override;
  std::string GetArgsDeclaration() const override;
  absl::Status BindArguments(CLKernel* kernel) override;

  friend absl::Status CreatePReLU(const CreationContext& creation_context,
                                  const OperationDef& definition,
                                  const PReLUAttributes& attr, PReLU* result);

 private:
  PReLU(const OperationDef& definition, const PReLUAttributes& attr,
        CalculationsPrecision scalar_precision);

  template <DataType T>
  absl::Status UploadParameters(
      const tflite::gpu::Tensor<Linear, T>& parameters, CLContext* context);

  FLT clip_;
  LinearStorage alpha_;
};

absl::Status CreatePReLU(const CreationContext& creation_context,
                         const OperationDef& definition,
                         const PReLUAttributes& attr, PReLU* result);

template <DataType T>
absl::Status PReLU::UploadParameters(
    const tflite::gpu::Tensor<Linear, T>& parameters, CLContext* context) {
  LinearStorageCreateInfo create_info;
  create_info.storage_type =
      DeduceLinearStorageType(definition_.GetPrimaryStorageType());
  create_info.data_type = definition_.GetPrimaryDataType();
  RETURN_IF_ERROR(
      CreateLinearStorage(create_info, parameters, context, &alpha_));
  return absl::OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_PRELU_H_
