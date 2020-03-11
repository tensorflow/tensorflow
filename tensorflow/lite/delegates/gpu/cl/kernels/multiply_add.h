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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_MULTIPLY_ADD_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_MULTIPLY_ADD_H_

#include <string>

#include "tensorflow/lite/delegates/gpu/cl/cl_context.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/flt_type.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/cl/linear_storage.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace cl {

class MultiplyAdd : public ElementwiseOperation {
 public:
  // Move only
  MultiplyAdd() = default;
  MultiplyAdd(MultiplyAdd&& operation);
  MultiplyAdd& operator=(MultiplyAdd&& operation);
  MultiplyAdd(const MultiplyAdd&) = delete;
  MultiplyAdd& operator=(const MultiplyAdd&) = delete;

  Status UploadMul(const MultiplyAttributes& attr,
                   CalculationsPrecision scalar_precision, CLContext* context);
  Status UploadAdd(const AddAttributes& attr,
                   CalculationsPrecision scalar_precision, CLContext* context);

  template <DataType T>
  Status UploadMul(const ::tflite::gpu::Tensor<Linear, T>& mul,
                   CLContext* context);

  template <DataType T>
  Status UploadAdd(const ::tflite::gpu::Tensor<Linear, T>& add,
                   CLContext* context);

  void SetLinkIndex(int index) override;
  std::string GetCoreCode(const LinkingContext& context) const override;

  std::string GetArgsDeclaration() const override;
  Status BindArguments(CLKernel* kernel) override;

  friend Status CreateMultiplyAdd(const CreationContext& creation_context,
                                  const OperationDef& definition,
                                  const MultiplyAttributes& attr,
                                  MultiplyAdd* result);

  friend Status CreateMultiplyAdd(const CreationContext& creation_context,
                                  const OperationDef& definition,
                                  const AddAttributes& attr,
                                  MultiplyAdd* result);

  friend Status CreateMultiplyAdd(const CreationContext& creation_context,
                                  const OperationDef& definition,
                                  const MultiplyAttributes& mul_attr,
                                  const AddAttributes& add_attr,
                                  MultiplyAdd* result);

 private:
  explicit MultiplyAdd(const OperationDef& definition)
      : ElementwiseOperation(definition),
        use_mul_vec_(false),
        use_add_vec_(false) {}

  LinearStorage mul_vec_;
  LinearStorage add_vec_;
  bool use_mul_vec_;
  bool use_add_vec_;
  FLT scalar_mul_;
  FLT scalar_add_;
};

Status CreateMultiplyAdd(const CreationContext& creation_context,
                         const OperationDef& definition,
                         const MultiplyAttributes& attr, MultiplyAdd* result);

Status CreateMultiplyAdd(const CreationContext& creation_context,
                         const OperationDef& definition,
                         const AddAttributes& attr, MultiplyAdd* result);

Status CreateMultiplyAdd(const CreationContext& creation_context,
                         const OperationDef& definition,
                         const MultiplyAttributes& mul_attr,
                         const AddAttributes& add_attr, MultiplyAdd* result);

template <DataType T>
Status MultiplyAdd::UploadMul(const ::tflite::gpu::Tensor<Linear, T>& mul,
                              CLContext* context) {
  LinearStorageCreateInfo create_info;
  create_info.storage_type =
      DeduceLinearStorageType(definition_.GetPrimaryStorageType());
  create_info.data_type = definition_.GetDataType();
  RETURN_IF_ERROR(CreateLinearStorage(create_info, mul, context, &mul_vec_));
  use_mul_vec_ = true;
  return OkStatus();
}

template <DataType T>
Status MultiplyAdd::UploadAdd(const ::tflite::gpu::Tensor<Linear, T>& add,
                              CLContext* context) {
  LinearStorageCreateInfo create_info;
  create_info.storage_type =
      DeduceLinearStorageType(definition_.GetPrimaryStorageType());
  create_info.data_type = definition_.GetDataType();
  RETURN_IF_ERROR(CreateLinearStorage(create_info, add, context, &add_vec_));
  use_add_vec_ = true;
  return OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_MULTIPLY_ADD_H_
