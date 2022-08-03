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

#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_replace.h"
#include "tensorflow/lite/delegates/gpu/common/access_type.h"
#include "tensorflow/lite/delegates/gpu/common/task/util.h"
#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"

namespace tflite {
namespace gpu {
namespace {
int3 GetWorkGroupsCountInternal(int grid_dimension, const int3& grid_size,
                                const int3& work_group_size,
                                const int3& work_group_launch_order) {
  int3 work_groups_count;
  if (grid_dimension == 1) {
    work_groups_count.x = DivideRoundUp(grid_size.x, work_group_size.x);
    work_groups_count.y = 1;
    work_groups_count.z = 1;
  } else if (grid_dimension == 2) {
    int3 wgs;
    wgs.x = DivideRoundUp(grid_size.x, work_group_size.x);
    wgs.y = DivideRoundUp(grid_size.y, work_group_size.y);
    work_groups_count.x = wgs[work_group_launch_order[0]];
    work_groups_count.y = wgs[work_group_launch_order[1]];
    work_groups_count.z = 1;
  } else {  // grid_dimension == 3
    int3 wgs;
    wgs.x = DivideRoundUp(grid_size.x, work_group_size.x);
    wgs.y = DivideRoundUp(grid_size.y, work_group_size.y);
    wgs.z = DivideRoundUp(grid_size.z, work_group_size.z);
    work_groups_count.x = wgs[work_group_launch_order[0]];
    work_groups_count.y = wgs[work_group_launch_order[1]];
    work_groups_count.z = wgs[work_group_launch_order[2]];
  }
  return work_groups_count;
}

std::string GetElementWiseCode(const OperationDef& op_def) {
  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id = GLOBAL_ID_0;\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int X = GLOBAL_ID_0;\n";
  }
  c += "  int Y = GLOBAL_ID_1;\n";
  c += "  int Z = GLOBAL_ID_2;\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "Z >= args.dst_tensor.Slices()) return; \n";
  c += "  args.src_tensor::type src = args.src_tensor.Read(X, Y, Z);\n";
  c += "  args.dst_tensor.Write(src, X, Y, Z);\n";
  c += "} \n";
  return c;
}

}  // namespace

DataType OperationDef::GetDataType() const {
  return DeduceDataTypeFromPrecision(precision);
}

DataType OperationDef::GetPrimaryDataType() const {
  return src_tensors[0].GetDataType();
}
TensorStorageType OperationDef::GetPrimaryStorageType() const {
  return src_tensors[0].GetStorageType();
}

bool OperationDef::IsBatchSupported() const {
  for (const auto& src : src_tensors) {
    if (src.HasAxis(Axis::BATCH)) {
      return true;
    }
  }
  for (const auto& dst : dst_tensors) {
    if (dst.HasAxis(Axis::BATCH)) {
      return true;
    }
  }
  return false;
}

GPUOperation::GPUOperation(const OperationDef& definition)
    : definition_(definition) {}

void GPUOperation::SetSrc(GpuSpatialTensor* ptr, int index) {
  if (index >= src_.size()) {
    src_.resize(index + 1, nullptr);
  }
  src_[index] = ptr;
}

void GPUOperation::SetDst(GpuSpatialTensor* ptr, int index) {
  if (index >= dst_.size()) {
    dst_.resize(index + 1, nullptr);
  }
  dst_[index] = ptr;
}

GPUOperation::GPUOperation(GPUOperation&& operation)
    : args_(std::move(operation.args_)),
      code_(std::move(operation.code_)),
      work_group_size_(operation.work_group_size_),
      compiler_options_(std::move(operation.compiler_options_)),
      tensor_to_grid_(operation.tensor_to_grid_),
      flops_(operation.flops_),
      const_args_size_(operation.const_args_size_),
      definition_(std::move(operation.definition_)),
      src_(std::move(operation.src_)),
      dst_(std::move(operation.dst_)),
      grid_dimension_(operation.grid_dimension_),
      work_group_launch_order_(operation.work_group_launch_order_),
      grid_size_(operation.grid_size_),
      src_tensors_names_(std::move(operation.src_tensors_names_)),
      dst_tensors_names_(std::move(operation.dst_tensors_names_)),
      work_groups_count_(operation.work_groups_count_),
      elementwise_(operation.elementwise_),
      linkable_count_(operation.linkable_count_),
      elementwise_code_(std::move(operation.elementwise_code_)) {}

GPUOperation& GPUOperation::operator=(GPUOperation&& operation) {
  if (this != &operation) {
    args_ = std::move(operation.args_);
    code_ = std::move(operation.code_);
    std::swap(work_group_size_, operation.work_group_size_);
    compiler_options_ = std::move(operation.compiler_options_);
    tensor_to_grid_ = operation.tensor_to_grid_;
    flops_ = operation.flops_;
    const_args_size_ = operation.const_args_size_;
    definition_ = std::move(operation.definition_);
    src_ = std::move(operation.src_);
    dst_ = std::move(operation.dst_);
    std::swap(grid_dimension_, operation.grid_dimension_);
    std::swap(work_group_launch_order_, operation.work_group_launch_order_);
    std::swap(grid_size_, operation.grid_size_);
    src_tensors_names_ = std::move(operation.src_tensors_names_);
    dst_tensors_names_ = std::move(operation.dst_tensors_names_);
    std::swap(work_groups_count_, operation.work_groups_count_);
    elementwise_ = operation.elementwise_;
    std::swap(linkable_count_, operation.linkable_count_);
    elementwise_code_ = std::move(operation.elementwise_code_);
  }
  return *this;
}

absl::Status GPUOperation::AddOperation(const GpuInfo& gpu_info,
                                        GPUOperation* operation) {
  const auto prev_type = definition_.dst_tensors[0].GetDataType();
  definition_.dst_tensors[0] = operation->definition_.dst_tensors[0];
  linkable_count_ += (operation->linkable_count_ + 1);
  std::string code = "{\n" + operation->elementwise_code_ + "\n}";
  std::string unique_postfix = absl::StrCat("_link", linkable_count_);
  operation->args_.RenameArgs(unique_postfix, &code);
  if (elementwise_code_.empty()) {
    elementwise_code_ = code;
  } else {
    const std::string new_value_name = "interm_value" + unique_postfix;
    code = absl::StrReplaceAll(code, {{"in_value", new_value_name}});
    elementwise_code_ =
        absl::StrReplaceAll(elementwise_code_, {{"out_value", new_value_name}});
    elementwise_code_ = "{\n" + GetTypeDeclaration(gpu_info, prev_type, 4) +
                        " " + new_value_name + ";\n" + elementwise_code_ +
                        "\n" + code + "\n}\n";
  }
  RETURN_IF_ERROR(args_.Merge(std::move(operation->args_), unique_postfix));
  for (int i = 0; i < operation->src_tensors_names_.size(); ++i) {
    definition_.src_tensors.push_back(
        operation->definition_.src_tensors[i + 1]);
    src_tensors_names_.push_back(operation->src_tensors_names_[i] +
                                 unique_postfix);
  }
  for (int i = 0; i < operation->dst_tensors_names_.size(); ++i) {
    dst_tensors_names_.push_back(operation->dst_tensors_names_[i] +
                                 unique_postfix);
  }
  return absl::OkStatus();
}

void GPUOperation::AddSrcTensor(const std::string& tensor_name,
                                const TensorDescriptor& desc) {
  src_tensors_names_.push_back(tensor_name);
  auto desc_new = std::make_unique<TensorDescriptor>(desc);
  args_.AddObjectRef(tensor_name, AccessType::READ, std::move(desc_new));
}

void GPUOperation::AddSrcBuffer(const std::string& buffer_name,
                                const BufferDescriptor& desc) {
  src_tensors_names_.push_back(buffer_name);
  auto desc_new = std::make_unique<BufferDescriptor>(desc);
  args_.AddObjectRef(buffer_name, AccessType::READ, std::move(desc_new));
}

void GPUOperation::AddSrcTexture2D(const std::string& texture_name,
                                   const Texture2DDescriptor& desc) {
  src_tensors_names_.push_back(texture_name);
  auto desc_new = std::make_unique<Texture2DDescriptor>(desc);
  args_.AddObjectRef(texture_name, AccessType::READ, std::move(desc_new));
}

void GPUOperation::AddDstTensor(const std::string& tensor_name,
                                const TensorDescriptor& desc) {
  dst_tensors_names_.push_back(tensor_name);
  auto desc_new = std::make_unique<TensorDescriptor>(desc);
  args_.AddObjectRef(tensor_name, AccessType::WRITE, std::move(desc_new));
}

absl::Status GPUOperation::AssembleCode(const GpuInfo& gpu_info) {
  if (elementwise_) {
    src_tensors_names_.insert(src_tensors_names_.begin(), "src_tensor");
    args_.AddObjectRef(
        "src_tensor", AccessType::READ,
        std::make_unique<TensorDescriptor>(definition_.src_tensors[0]));

    dst_tensors_names_.insert(dst_tensors_names_.begin(), "dst_tensor");
    args_.AddObjectRef(
        "dst_tensor", AccessType::WRITE,
        std::make_unique<TensorDescriptor>(definition_.dst_tensors[0]));

    code_ = GetElementWiseCode(definition_);
  }
  RETURN_IF_ERROR(args_.Compile(
      gpu_info, {{dst_tensors_names_[0], elementwise_code_}}, &code_));
  CalculateConstArgsSize();
  return absl::OkStatus();
}

void GPUOperation::RecalculateWorkGroupsCount() {
  work_groups_count_ = GetWorkGroupsCountInternal(
      grid_dimension_, grid_size_, work_group_size_, work_group_launch_order_);
}

void GPUOperation::CalculateConstArgsSize() {
  const_args_size_ = 0;
  for (const auto& obj : args_.GetObjects()) {
    const_args_size_ += obj.second->GetSizeInBytes();
  }
}

void GPUOperation::GetPossibleDispatches(
    TuningType tuning_type, const GpuInfo& gpu_info,
    const KernelInfo& kernel_info,
    std::vector<DispatchInfo>* dispatches) const {
  std::vector<int3> work_group_sizes;
  GetPossibleKernelWorkGroups(tuning_type, gpu_info, kernel_info,
                              &work_group_sizes);
  dispatches->resize(work_group_sizes.size());
  for (int i = 0; i < work_group_sizes.size(); ++i) {
    auto& dispatch_info = (*dispatches)[i];
    dispatch_info.work_group_size = work_group_sizes[i];
    dispatch_info.work_groups_count = GetWorkGroupsCountInternal(
        grid_dimension_, grid_size_, work_group_sizes[i],
        work_group_launch_order_);
  }
}

void GPUOperation::GetPossibleKernelWorkGroups(
    TuningType tuning_type, const GpuInfo& gpu_info,
    const KernelInfo& kernel_info, std::vector<int3>* work_groups) const {
  GetPossibleWorkGroups(tuning_type, gpu_info, kernel_info, grid_size_,
                        work_groups);
}

int3 GPUOperation::GetGridSize() const {
  if (tensor_to_grid_ == TensorToGrid::kWBToX_HDToY_SToZ) {
    const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
    const int grid_y = dst_[0]->Height() * dst_[0]->Depth();
    const int grid_z = dst_[0]->Slices();
    return int3(grid_x, grid_y, grid_z);
  }
  if (tensor_to_grid_ == TensorToGrid::kWBToX_HDToY_ZIs1) {
    const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
    const int grid_y = dst_[0]->Height() * dst_[0]->Depth();
    const int grid_z = 1;
    return int3(grid_x, grid_y, grid_z);
  }
  if (tensor_to_grid_ == TensorToGrid::kWBToX_HToY_DToZ) {
    const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
    const int grid_y = dst_[0]->Height();
    const int grid_z = dst_[0]->Depth();
    return int3(grid_x, grid_y, grid_z);
  }
  if (tensor_to_grid_ == TensorToGrid::kBToX_YIs1_ZIs1) {
    const int grid_x = dst_[0]->Batch();
    const int grid_y = 1;
    const int grid_z = 1;
    return int3(grid_x, grid_y, grid_z);
  }
  return grid_size_;
}

GPUOperation CreateGpuOperation(const OperationDef& definition,
                                ElementwiseDescriptor&& descriptor) {
  GPUOperation op(definition);
  op.elementwise_code_ = std::move(descriptor.code);
  op.elementwise_ = true;
  op.args_ = std::move(descriptor.args);
  for (int i = 1; i < definition.src_tensors.size(); ++i) {
    const std::string tensor_name = "src_tensor_" + std::to_string(i);
    auto src_desc = definition.src_tensors[i];
    op.AddSrcTensor(tensor_name, src_desc);
  }
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

}  // namespace gpu
}  // namespace tflite
