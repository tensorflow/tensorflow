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
                               bool check_src_slices) {
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

int3 GetWorkGroupsCount(int grid_dimension, const int3& grid_size,
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
    : args_(std::move(operation.args_)),
      code_(std::move(operation.code_)),
      work_group_size_(operation.work_group_size_),
      compiler_options_(std::move(operation.compiler_options_)),
      tensor_to_grid_(operation.tensor_to_grid_),
      elementwise_(operation.elementwise_),
      linkable_(operation.linkable_),
      check_src_channels_size_(operation.check_src_channels_size_),
      definition_(std::move(operation.definition_)),
      src_(std::move(operation.src_)),
      dst_(std::move(operation.dst_)),
      kernel_(std::move(operation.kernel_)),
      grid_dimension_(operation.grid_dimension_),
      work_group_launch_order_(operation.work_group_launch_order_),
      grid_size_(operation.grid_size_),
      src_tensors_names_(std::move(operation.src_tensors_names_)),
      dst_tensors_names_(std::move(operation.dst_tensors_names_)),
      work_groups_count_(operation.work_groups_count_),
      linkable_count_(operation.linkable_count_),
      elementwise_code_(std::move(operation.elementwise_code_)) {}

GPUOperation& GPUOperation::operator=(GPUOperation&& operation) {
  if (this != &operation) {
    args_ = std::move(operation.args_);
    code_ = std::move(operation.code_);
    std::swap(work_group_size_, operation.work_group_size_);
    compiler_options_ = std::move(operation.compiler_options_);
    tensor_to_grid_ = operation.tensor_to_grid_;
    elementwise_ = operation.elementwise_;
    linkable_ = operation.linkable_;
    check_src_channels_size_ = operation.check_src_channels_size_;
    definition_ = std::move(operation.definition_);
    src_ = std::move(operation.src_);
    dst_ = std::move(operation.dst_);
    kernel_ = std::move(operation.kernel_);
    std::swap(grid_dimension_, operation.grid_dimension_);
    std::swap(work_group_launch_order_, operation.work_group_launch_order_);
    std::swap(grid_size_, operation.grid_size_);
    src_tensors_names_ = std::move(operation.src_tensors_names_);
    dst_tensors_names_ = std::move(operation.dst_tensors_names_);
    std::swap(work_groups_count_, operation.work_groups_count_);
    std::swap(linkable_count_, operation.linkable_count_);
    elementwise_code_ = std::move(operation.elementwise_code_);
  }
  return *this;
}

absl::Status GPUOperation::AddOperation(GPUOperation* operation) {
  linkable_count_ += 1;
  std::string code = operation->code_;
  std::string unique_postfix = absl::StrCat("_link", linkable_count_);
  operation->args_.RenameArgs(unique_postfix, &code);
  elementwise_code_ += "{\n" + code + "\n}\n";
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
  auto desc_new = absl::make_unique<TensorDescriptor>(desc);
  args_.AddObjectRef(tensor_name, AccessType::READ, std::move(desc_new));
}

void GPUOperation::AddSrcBuffer(const std::string& buffer_name,
                                const BufferDescriptor& desc) {
  src_tensors_names_.push_back(buffer_name);
  auto desc_new = absl::make_unique<BufferDescriptor>(desc);
  args_.AddObjectRef(buffer_name, AccessType::READ, std::move(desc_new));
}

void GPUOperation::AddDstTensor(const std::string& tensor_name,
                                const TensorDescriptor& desc) {
  dst_tensors_names_.push_back(tensor_name);
  auto desc_new = absl::make_unique<TensorDescriptor>(desc);
  args_.AddObjectRef(tensor_name, AccessType::WRITE, std::move(desc_new));
}

absl::Status GPUOperation::UpdateParams() {
  for (int i = 0; i < src_tensors_names_.size(); ++i) {
    RETURN_IF_ERROR(args_.SetObjectRef(src_tensors_names_[i], src_[i]));
  }
  for (int i = 0; i < dst_tensors_names_.size(); ++i) {
    RETURN_IF_ERROR(args_.SetObjectRef(dst_tensors_names_[i], dst_[i]));
  }
  RETURN_IF_ERROR(BindArguments(&args_));
  grid_size_ = GetGridSize();
  work_groups_count_ = GetWorkGroupsCount(
      grid_dimension_, grid_size_, work_group_size_, work_group_launch_order_);
  return absl::OkStatus();
}

absl::Status GPUOperation::AssembleCode(const DeviceInfo& device_info,
                                        CLContext* context) {
  if (elementwise_) {
    auto src_desc =
        absl::make_unique<TensorDescriptor>(definition_.src_tensors[0]);
    if (definition_.IsBatchSupported()) {
      src_desc->SetStateVar("BatchedWidth", "true");
    }
    src_tensors_names_.insert(src_tensors_names_.begin(), "src_tensor");
    args_.AddObjectRef("src_tensor", AccessType::READ, std::move(src_desc));

    auto dst_desc =
        absl::make_unique<TensorDescriptor>(definition_.dst_tensors[0]);
    if (definition_.IsBatchSupported()) {
      dst_desc->SetStateVar("BatchedWidth", "true");
    }
    dst_tensors_names_.insert(dst_tensors_names_.begin(), "dst_tensor");
    args_.AddObjectRef("dst_tensor", AccessType::WRITE, std::move(dst_desc));

    elementwise_code_ = "{\n" + code_ + "\n}\n" + elementwise_code_;
    code_ = GetElementWiseCode(definition_, check_src_channels_size_);
    RETURN_IF_ERROR(args_.AllocateObjects(context));
    RETURN_IF_ERROR(args_.TransformToCLCode(
        device_info, {{dst_tensors_names_[0], elementwise_code_}}, &code_));
  } else {
    RETURN_IF_ERROR(args_.AllocateObjects(context));
    RETURN_IF_ERROR(args_.TransformToCLCode(
        device_info, {{dst_tensors_names_[0], elementwise_code_}}, &code_));
  }
  return absl::OkStatus();
}

absl::Status GPUOperation::Compile(const CreationContext& creation_context) {
  RETURN_IF_ERROR(
      AssembleCode(creation_context.GetDeviceInfo(), creation_context.context));
  RETURN_IF_ERROR(creation_context.cache->GetOrCreateCLKernel(
      code_, "main_function", compiler_options_, *creation_context.context,
      *creation_context.device, &kernel_));
  return PostCompileCheck(creation_context.device->info_, kernel_.info_);
}

absl::Status GPUOperation::CompileDeserialized(
    const CreationContext& creation_context) {
  return creation_context.cache->GetOrCreateCLKernel(
      code_, "main_function", compiler_options_, *creation_context.context,
      *creation_context.device, &kernel_);
}

void GPUOperation::GetPossibleKernelWorkGroups(
    TuningType tuning_type, const DeviceInfo& device_info,
    const KernelInfo& kernel_info, std::vector<int3>* work_groups) const {
  GetPossibleWorkGroups(tuning_type, device_info, kernel_info, grid_size_,
                        work_groups);
}

absl::Status GPUOperation::Tune(const TuningParameters& params) {
  std::vector<int3> possible_work_groups;
  GetPossibleKernelWorkGroups(params.tuning_type, *params.info, kernel_.info_,
                              &possible_work_groups);
  if (possible_work_groups.empty()) {
    return absl::NotFoundError(
        "Can not found work_group size to launch kernel");
  }
  if (possible_work_groups.size() == 1) {
    work_group_size_ = possible_work_groups[0];
    work_groups_count_ =
        GetWorkGroupsCount(grid_dimension_, grid_size_, work_group_size_,
                           work_group_launch_order_);
    return absl::OkStatus();
  } else {
    std::vector<int3> work_groups_count(possible_work_groups.size());
    for (int i = 0; i < work_groups_count.size(); ++i) {
      work_groups_count[i] =
          GetWorkGroupsCount(grid_dimension_, grid_size_,
                             possible_work_groups[i], work_group_launch_order_);
    }
    RETURN_IF_ERROR(args_.Bind(kernel_.kernel()));
    int best_work_group_index;
    RETURN_IF_ERROR(params.queue->GetBestWorkGroupIndex(
        kernel_, *params.info, work_groups_count, possible_work_groups,
        &best_work_group_index));
    work_group_size_ = possible_work_groups[best_work_group_index];
    work_groups_count_ =
        GetWorkGroupsCount(grid_dimension_, grid_size_, work_group_size_,
                           work_group_launch_order_);
    return absl::OkStatus();
  }
}

int3 GPUOperation::GetGridSize() const {
  if (elementwise_ || tensor_to_grid_ == TensorToGrid::kWBToX_HDToY_SToZ) {
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

void GPUOperation::AddUniquePostfix(const std::string& unique_postfix) {
  for (int i = 0; i < src_tensors_names_.size(); ++i) {
    src_tensors_names_[i] += unique_postfix;
  }
  for (int i = 0; i < dst_tensors_names_.size(); ++i) {
    dst_tensors_names_[i] += unique_postfix;
  }
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
