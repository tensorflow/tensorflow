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

#include "tensorflow/lite/delegates/gpu/cl/kernels/apply_mask.h"

#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/work_group_picking.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace cl {

ApplyMask::ApplyMask(ApplyMask&& operation)
    : ElementwiseOperation(std::move(operation)),
      mask_type_(operation.mask_type_),
      link_index_(operation.link_index_) {}

ApplyMask& ApplyMask::operator=(ApplyMask&& operation) {
  if (this != &operation) {
    mask_type_ = operation.mask_type_;
    link_index_ = operation.link_index_;
    ElementwiseOperation::operator=(std::move(operation));
  }
  return *this;
}

void ApplyMask::SetLinkIndex(int index) { link_index_ = index; }

std::string ApplyMask::GetCoreCode(const LinkingContext& context) const {
  const std::string size_name = "mask_size_op" + std::to_string(link_index_);
  const std::string tensor_name = absl::StrCat("mask_data_op", link_index_);
  TensorCodeGenerator mask(
      tensor_name, {size_name + ".x", size_name + ".y", size_name + ".z"},
      definition_.src_tensors[1]);
  switch (mask_type_) {
    case MaskType::TENSOR:
      return context.var_name + " *= " +
             mask.Read3D(context.x_coord, context.y_coord, context.z_coord) +
             ";\n";
    case MaskType::CHANNELS:
      return context.var_name +
             " *= " + mask.Read3D("0", "0", context.z_coord) + ";\n";
    case MaskType::LAYER:
      return context.var_name +
             " *= " + mask.Read3D(context.x_coord, context.y_coord, "0") +
             ".x;\n";
  }
}

std::string ApplyMask::GetArgsDeclaration() const {
  std::string args;
  const std::string tensor_name = absl::StrCat("mask_data_op", link_index_);
  absl::StrAppend(&args, ",\n",
                  GetTensorDeclaration(AccessType::READ, tensor_name,
                                       definition_.src_tensors[1]));
  const std::string size_name = "mask_size_op" + std::to_string(link_index_);
  absl::StrAppend(&args, ",\n   int4 ", size_name);
  return args;
}

Status ApplyMask::BindArguments(CLKernel* kernel) {
  RETURN_IF_ERROR(kernel->SetMemoryAuto(src_[1]->GetMemoryPtr()));
  RETURN_IF_ERROR(kernel->SetBytesAuto(src_[1]->GetWBatchedHSB()));
  return OkStatus();
}

ApplyMask CreateApplyMask(const OperationDef& definition, const BHWC& src_shape,
                          const BHWC& mask_shape) {
  ApplyMask::MaskType mask_type;
  if (mask_shape == src_shape) {
    mask_type = ApplyMask::MaskType::TENSOR;
  } else if (mask_shape.c == 1) {
    mask_type = ApplyMask::MaskType::LAYER;
  } else {
    mask_type = ApplyMask::MaskType::CHANNELS;
  }
  ApplyMask operation(definition, mask_type);
  operation.SetLinkIndex(0);
  return operation;
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
