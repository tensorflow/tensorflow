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

#include "tensorflow/lite/delegates/gpu/cl/kernels/gpu_operation.h"

#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/work_group_picking.h"
#include "tensorflow/lite/delegates/gpu/common/access_type.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

std::string GetElementWiseCode(const OperationDef& op_def,
                               bool check_src_slices, Arguments* args) {
  auto src_desc = absl::make_unique<TensorDescriptor>(op_def.src_tensors[0]);
  if (op_def.IsBatchSupported()) {
    src_desc->SetStateVar("BatchedWidth", "true");
  }
  args->AddObjectRef("src_tensor", AccessType::READ, std::move(src_desc));
  auto dst_desc = absl::make_unique<TensorDescriptor>(op_def.dst_tensors[0]);
  if (op_def.IsBatchSupported()) {
    dst_desc->SetStateVar("BatchedWidth", "true");
  }
  args->AddObjectRef("dst_tensor", AccessType::WRITE, std::move(dst_desc));

  std::string c = GetCommonDefines(op_def.precision);

  c += "__kernel void main_function(\n";
  c += "$0) {\n";
  c += "  int X = get_global_id(0);\n";
  c += "  int Y = get_global_id(1);\n";
  c += "  int Z = get_global_id(2);\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "Z >= args.dst_tensor.Slices()) return; \n";
  if (check_src_slices) {
    c += "  FLT4 src = (FLT4)(0.0f);\n";
    c += "  if (Z < args.src_tensor.Slices()) {\n";
    c += "    src = args.src_tensor.Read(X, Y, Z);\n";
    c += "  }\n";
  } else {
    c += "  FLT4 src = args.src_tensor.Read(X, Y, Z);\n";
  }
  c += "  args.dst_tensor.Write(src, X, Y, Z);\n";
  c += "} \n";
  return c;
}

}  // namespace

DataType OperationDef::GetDataType() const {
  return DeduceDataTypeFromPrecision(precision);
}

DataType OperationDef::GetPrimaryDataType() const {
  return src_tensors[0].data_type;
}
TensorStorageType OperationDef::GetPrimaryStorageType() const {
  return src_tensors[0].storage_type;
}

bool OperationDef::HasAllTensorsOfType(TensorStorageType storage_type) const {
  for (const auto& src : src_tensors) {
    if (src.storage_type != storage_type) {
      return false;
    }
  }
  for (const auto& dst : dst_tensors) {
    if (dst.storage_type != storage_type) {
      return false;
    }
  }
  return true;
}

bool OperationDef::IsBatchSupported() const {
  for (const auto& src : src_tensors) {
    if (HasAxis(src.layout, Axis::BATCH)) {
      return true;
    }
  }
  for (const auto& dst : dst_tensors) {
    if (HasAxis(dst.layout, Axis::BATCH)) {
      return true;
    }
  }
  return false;
}

GPUOperation::GPUOperation(const OperationDef& definition)
    : definition_(definition) {}

void GPUOperation::SetSrc(Tensor* ptr, int index) {
  if (index >= src_.size()) {
    src_.resize(index + 1, nullptr);
  }
  src_[index] = ptr;
}

void GPUOperation::SetDst(Tensor* ptr, int index) {
  if (index >= dst_.size()) {
    dst_.resize(index + 1, nullptr);
  }
  dst_[index] = ptr;
}

GPUOperation::GPUOperation(GPUOperation&& operation)
    : definition_(std::move(operation.definition_)),
      src_(std::move(operation.src_)),
      dst_(std::move(operation.dst_)),
      args_(std::move(operation.args_)),
      linked_operations_(std::move(operation.linked_operations_)) {}

GPUOperation& GPUOperation::operator=(GPUOperation&& operation) {
  if (this != &operation) {
    definition_ = std::move(operation.definition_);
    src_ = std::move(operation.src_);
    dst_ = std::move(operation.dst_);
    args_ = std::move(operation.args_);
    linked_operations_ = std::move(operation.linked_operations_);
  }
  return *this;
}

void GPUOperation::AddOperation(ElementwiseOperation* operation) {
  linked_operations_.push_back(operation);
}

ElementwiseOperation::ElementwiseOperation(ElementwiseOperation&& operation)
    : GPUOperation(std::move(operation)),
      check_src_channels_size_(operation.check_src_channels_size_),
      code_(std::move(operation.code_)),
      kernel_(std::move(operation.kernel_)),
      work_group_size_(operation.work_group_size_) {}

ElementwiseOperation& ElementwiseOperation::operator=(
    ElementwiseOperation&& operation) {
  if (this != &operation) {
    check_src_channels_size_ = operation.check_src_channels_size_;
    code_ = std::move(operation.code_);
    kernel_ = std::move(operation.kernel_);
    std::swap(work_group_size_, operation.work_group_size_);
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

absl::Status ElementwiseOperation::BindArguments() {
  RETURN_IF_ERROR(args_.SetObjectRef("src_tensor", src_[0]));
  RETURN_IF_ERROR(args_.SetObjectRef("dst_tensor", dst_[0]));
  RETURN_IF_ERROR(SetArgs("", &args_));
  RETURN_IF_ERROR(SetArguments(linked_operations_, &args_));
  RETURN_IF_ERROR(args_.Bind(kernel_.kernel()));
  return absl::OkStatus();
}

int3 ElementwiseOperation::GetGridSize() const {
  const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
  const int grid_y = dst_[0]->Height();
  const int grid_z = dst_[0]->Slices();
  return int3(grid_x, grid_y, grid_z);
}

absl::Status ElementwiseOperation::Compile(
    const CreationContext& creation_context) {
  std::string code =
      GetElementWiseCode(definition_, check_src_channels_size_, &args_);
  std::string element_wise_code;
  element_wise_code += "{\n" + code_ + "\n}\n";
  RETURN_IF_ERROR(
      MergeOperations(linked_operations_, &args_, &element_wise_code));
  RETURN_IF_ERROR(args_.TransformToCLCode(creation_context.device->GetInfo(),
                                          {{"dst_tensor", element_wise_code}},
                                          &code));
  code = absl::Substitute(code, args_.GetListOfArgs());
  return creation_context.cache->GetOrCreateCLKernel(
      code, "main_function", *creation_context.context,
      *creation_context.device, &kernel_);
}

absl::Status ElementwiseOperation::AddToQueue(CLCommandQueue* queue) {
  RETURN_IF_ERROR(BindArguments());
  return queue->DispatchImplicit(kernel_, GetGridSize(), work_group_size_);
}

absl::Status ElementwiseOperation::Tune(const TuningParameters& params) {
  RETURN_IF_ERROR(BindArguments());
  return GetBestWorkGroup(params, kernel_, GetGridSize(), &work_group_size_);
}

absl::Status MergeOperations(
    const std::vector<ElementwiseOperation*>& linked_ops,
    Arguments* merged_args, std::string* merged_code) {
  for (int i = 0; i < linked_ops.size(); ++i) {
    std::string code = linked_ops[i]->GetCode();
    std::string unique_postfix = absl::StrCat("_link", i + 1);
    auto&& link_args = linked_ops[i]->MoveArgs();
    link_args.RenameArgs(unique_postfix, &code);
    *merged_code += "{\n" + code + "\n}\n";
    RETURN_IF_ERROR(merged_args->Merge(std::move(link_args), unique_postfix));
  }
  return absl::OkStatus();
}

absl::Status SetArguments(const std::vector<ElementwiseOperation*>& linked_ops,
                          Arguments* args) {
  for (int i = 0; i < linked_ops.size(); ++i) {
    std::string unique_postfix = absl::StrCat("_link", i + 1);
    RETURN_IF_ERROR(linked_ops[i]->SetArgs(unique_postfix, args));
  }
  return absl::OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
