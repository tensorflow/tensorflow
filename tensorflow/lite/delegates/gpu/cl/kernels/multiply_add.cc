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

#include "tensorflow/lite/delegates/gpu/cl/kernels/multiply_add.h"

#include "absl/types/variant.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/precision.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"

namespace tflite {
namespace gpu {
namespace cl {

MultiplyAdd::MultiplyAdd(MultiplyAdd&& operation)
    : ElementwiseOperation(std::move(operation)),
      mul_vec_(std::move(operation.mul_vec_)),
      add_vec_(std::move(operation.add_vec_)),
      use_mul_vec_(operation.use_mul_vec_),
      use_add_vec_(operation.use_add_vec_),
      scalar_mul_(std::move(operation.scalar_mul_)),
      scalar_add_(std::move(operation.scalar_add_)) {}

MultiplyAdd& MultiplyAdd::operator=(MultiplyAdd&& operation) {
  if (this != &operation) {
    mul_vec_ = std::move(operation.mul_vec_);
    add_vec_ = std::move(operation.add_vec_);
    use_mul_vec_ = operation.use_mul_vec_;
    use_add_vec_ = operation.use_add_vec_;
    scalar_mul_ = std::move(operation.scalar_mul_);
    scalar_add_ = std::move(operation.scalar_add_);
    ElementwiseOperation::operator=(std::move(operation));
  }
  return *this;
}

void MultiplyAdd::SetLinkIndex(int index) {
  scalar_mul_.SetName(absl::StrCat("mad_scalar_mul_", index));
  scalar_add_.SetName(absl::StrCat("mad_scalar_add_", index));
  mul_vec_.SetName(absl::StrCat("mad_mul_", index));
  add_vec_.SetName(absl::StrCat("mad_add_", index));
}

std::string MultiplyAdd::GetCoreCode(const LinkingContext& context) const {
  std::string result = absl::StrCat(context.var_name, " = ", context.var_name);
  if (use_mul_vec_) {
    absl::StrAppend(&result, " * ", mul_vec_.ReadLinearFLT4(context.z_coord));
  }
  if (scalar_mul_.Active()) {
    absl::StrAppend(&result, " * (FLT)(", scalar_mul_.GetName(), ")");
  }
  if (use_add_vec_) {
    absl::StrAppend(&result, " + ", add_vec_.ReadLinearFLT4(context.z_coord));
  }
  if (scalar_add_.Active()) {
    absl::StrAppend(&result, " + (FLT)(", scalar_add_.GetName(), ")");
  }
  return absl::StrCat(result, ";\n");
}

std::string MultiplyAdd::GetArgsDeclaration() const {
  std::string args;
  if (use_mul_vec_) {
    absl::StrAppend(&args, ",\n    ", mul_vec_.GetDeclaration());
  }
  if (use_add_vec_) {
    absl::StrAppend(&args, ",\n    ", add_vec_.GetDeclaration());
  }
  if (scalar_mul_.Active()) {
    absl::StrAppend(&args, ",\n    ", scalar_mul_.GetDeclaration());
  }
  if (scalar_add_.Active()) {
    absl::StrAppend(&args, ",\n    ", scalar_add_.GetDeclaration());
  }
  return args;
}

Status MultiplyAdd::BindArguments(CLKernel* kernel) {
  if (use_mul_vec_) {
    RETURN_IF_ERROR(kernel->SetMemoryAuto(mul_vec_.GetMemoryPtr()));
  }
  if (use_add_vec_) {
    RETURN_IF_ERROR(kernel->SetMemoryAuto(add_vec_.GetMemoryPtr()));
  }
  if (scalar_mul_.Active()) {
    RETURN_IF_ERROR(kernel->SetBytesAuto(scalar_mul_));
  }
  if (scalar_add_.Active()) {
    RETURN_IF_ERROR(kernel->SetBytesAuto(scalar_add_));
  }
  return OkStatus();
}

Status MultiplyAdd::UploadMul(const MultiplyScalarAttributes& attr,
                              CalculationsPrecision scalar_precision,
                              CLContext* context) {
  auto mul = absl::get_if<::tflite::gpu::Tensor<Linear, DataType::FLOAT32>>(
      &attr.param);
  auto mul_scalar = absl::get_if<float>(&attr.param);
  if (mul) {
    RETURN_IF_ERROR(UploadMul(*mul, context));
  } else {
    scalar_mul_ = FLT(scalar_precision, *mul_scalar);
  }
  return OkStatus();
}

Status MultiplyAdd::UploadAdd(const AddAttributes& attr,
                              CalculationsPrecision scalar_precision,
                              CLContext* context) {
  auto add = absl::get_if<::tflite::gpu::Tensor<Linear, DataType::FLOAT32>>(
      &attr.param);
  auto add_scalar = absl::get_if<float>(&attr.param);
  if (add) {
    RETURN_IF_ERROR(UploadAdd(*add, context));
  } else {
    scalar_add_ = FLT(scalar_precision, *add_scalar);
  }
  return OkStatus();
}

Status CreateMultiplyAdd(const CreationContext& creation_context,
                         const OperationDef& definition,
                         const MultiplyScalarAttributes& attr,
                         MultiplyAdd* result) {
  const auto scalar_precision = creation_context.device->IsPowerVR()
                                    ? CalculationsPrecision::F32
                                    : definition.precision;
  *result = MultiplyAdd(definition);
  RETURN_IF_ERROR(
      result->UploadMul(attr, scalar_precision, creation_context.context));
  result->SetLinkIndex(0);
  return OkStatus();
}

Status CreateMultiplyAdd(const CreationContext& creation_context,
                         const OperationDef& definition,
                         const AddAttributes& attr, MultiplyAdd* result) {
  const auto scalar_precision = creation_context.device->IsPowerVR()
                                    ? CalculationsPrecision::F32
                                    : definition.precision;
  *result = MultiplyAdd(definition);
  RETURN_IF_ERROR(
      result->UploadAdd(attr, scalar_precision, creation_context.context));
  result->SetLinkIndex(0);
  return OkStatus();
}

Status CreateMultiplyAdd(const CreationContext& creation_context,
                         const OperationDef& definition,
                         const MultiplyScalarAttributes& mul_attr,
                         const AddAttributes& add_attr, MultiplyAdd* result) {
  const auto scalar_precision = creation_context.device->IsPowerVR()
                                    ? CalculationsPrecision::F32
                                    : definition.precision;
  *result = MultiplyAdd(definition);
  RETURN_IF_ERROR(
      result->UploadMul(mul_attr, scalar_precision, creation_context.context));
  RETURN_IF_ERROR(
      result->UploadAdd(add_attr, scalar_precision, creation_context.context));
  result->SetLinkIndex(0);
  return OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
