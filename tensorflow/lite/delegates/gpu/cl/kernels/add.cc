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

#include "tensorflow/lite/delegates/gpu/cl/kernels/add.h"

#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

bool HasTexture2DStorageType(const OperationDef& def) {
  for (auto& src_tensor : def.src_tensors) {
    if (src_tensor.storage_type == TensorStorageType::TEXTURE_2D) {
      return true;
    }
  }
  return false;
}

}  // namespace

std::string Add::GetElementWiseCode(
    const TensorDescriptor& src_descriptor,
    const TensorDescriptor& dst_descriptor, CalculationsPrecision precision,
    const std::vector<ElementwiseOperation*>& linked_operations) {
  TensorCodeGenerator src_tensor("src_data", "dst_size", src_descriptor);
  TensorCodeGenerator dst_tensor("dst_data", "dst_size", dst_descriptor);

  std::string c = GetCommonDefines(precision);

  c += "__kernel void main_function(\n";
  c += src_tensor.GetDeclaration(AccessType::READ);
  c += GetArgsDeclaration();
  c += ::tflite::gpu::cl::GetArgsDeclaration(linked_operations);
  c += dst_tensor.GetDeclaration(AccessType::WRITE) + ",\n";
  c += "    int4 dst_size\n";
  c += ") {\n";
  c += "  int X = get_global_id(0);\n";
  c += "  int Y = get_global_id(1);\n";
  c += "  int Z = get_global_id(2);\n";
  c += "  if (X >= dst_size.x || Y >= dst_size.y) { \n";
  c += "    return; \n";
  c += "  } \n";
  c += "  FLT4 src = (FLT4)(0.0);\n";
  c += "    " + dst_tensor.GetAddress("address", "X", "Y", "Z") + "\n";
  if (src_depthes_[0] != dst_depth_) {
    c += "  if (Z < " + std::to_string(src_depthes_[0]) + ") {\n";
    if (src_descriptor.storage_type == TensorStorageType::TEXTURE_2D) {
      c += "    float t_y = address.y - Z; \n";
      c += "    int ti_y = (t_y + 0.5) * " + inv_divisor_name_ + "; \n";
      c += "    int2 tmp_add = (int2)(address.x, ti_y  * " +
           std::to_string(src_depthes_[0]) + " + Z);\n";
      c += "    src += " + src_tensor.Read3D("tmp_add") + ";\n";
    } else {
      c += "    src += " + src_tensor.Read3D("address") + ";\n";
    }
    c += "  }\n";
  } else {
    c += "  src += " + src_tensor.Read3D("address") + ";\n";
  }
  c += "  " + GetCoreCode("src", "Z", "address");
  c += PostProcess(linked_operations, "src", "Z", "address");
  c += "  " + dst_tensor.Write3D("src", "address") + "\n";
  c += "} \n";
  return c;
}

Add::Add(const OperationDef& definition, const std::vector<int>& channels,
         int dst_channels)
    : ElementwiseOperation(definition),
      dst_depth_(IntegralDivideRoundUp(dst_channels, 4)) {
  src_depthes_.resize(channels.size());
  for (int i = 0; i < channels.size(); ++i) {
    src_depthes_[i] = IntegralDivideRoundUp(channels[i], 4);
  }
}

Add::Add(Add&& operation)
    : ElementwiseOperation(std::move(operation)),
      link_index_(operation.link_index_),
      inv_divisor_name_(std::move(operation.inv_divisor_name_)),
      src_depthes_(std::move(operation.src_depthes_)),
      dst_depth_(operation.dst_depth_) {}

Add& Add::operator=(Add&& operation) {
  if (this != &operation) {
    link_index_ = operation.link_index_;
    inv_divisor_name_ = std::move(operation.inv_divisor_name_);
    src_depthes_ = std::move(operation.src_depthes_);
    dst_depth_ = operation.dst_depth_;
    ElementwiseOperation::operator=(std::move(operation));
  }
  return *this;
}

void Add::SetLinkIndex(int index) {
  inv_divisor_name_ = absl::StrCat("inv_divisor_", index);
  link_index_ = index;
}

std::string Add::GetCoreCode(const std::string& src, const std::string& z_coord,
                             const std::string& address) const {
  std::string result;
  for (int i = 1; i < src_depthes_.size(); ++i) {
    const std::string tensor_name =
        absl::StrCat("src_data_", link_index_, "_", i);
    TensorCodeGenerator src_tensor(tensor_name, "", definition_.src_tensors[i]);
    if (src_depthes_[i] != dst_depth_) {
      absl::StrAppend(&result, "  if (", z_coord, " < ", src_depthes_[i],
                      ") {\n");
      if (definition_.src_tensors[i].storage_type ==
          TensorStorageType::TEXTURE_2D) {
        absl::StrAppend(&result, "    float t_y = ", address, ".y - ", z_coord,
                        ";\n");
        absl::StrAppend(&result, "    int ti_y = (t_y + 0.5) * ",
                        inv_divisor_name_, ";\n");
        absl::StrAppend(&result, "    int2 tmp_add = (int2)(", address,
                        ".x, ti_y * ", src_depthes_[i], " + ", z_coord, ");\n");
        absl::StrAppend(&result, "    ", src,
                        " += ", src_tensor.Read3D("tmp_add"), ";\n");
      } else {
        absl::StrAppend(&result, "    ", src,
                        " += ", src_tensor.Read3D(address), ";\n");
      }
      absl::StrAppend(&result, "  }\n");
    } else {
      absl::StrAppend(&result, "  ", src,
                      " += ", src_tensor.Read3D(address) + ";\n");
    }
  }
  return result;
}

std::string Add::GetArgsDeclaration() const {
  std::string args;
  for (int i = 1; i < src_depthes_.size(); ++i) {
    const std::string tensor_name =
        absl::StrCat("src_data_", link_index_, "_", i);
    TensorCodeGenerator src_tensor(tensor_name, "", definition_.src_tensors[i]);
    absl::StrAppend(&args, ",\n", src_tensor.GetDeclaration(AccessType::READ));
  }
  if (HasTexture2DStorageType(definition_)) {
    absl::StrAppend(&args, ",\n   float ", inv_divisor_name_);
  }
  return args;
}

Status Add::BindArguments(CLKernel* kernel) {
  for (int i = 1; i < src_depthes_.size(); ++i) {
    RETURN_IF_ERROR(kernel->SetMemoryAuto(src_[i]->GetMemoryPtr()));
  }
  if (HasTexture2DStorageType(definition_)) {
    float inv_divisor = 1.0f / static_cast<float>(dst_depth_);
    RETURN_IF_ERROR(kernel->SetBytesAuto(inv_divisor));
  }
  return OkStatus();
}

Status Add::Compile(const CreationContext& creation_context) {
  const auto code =
      GetElementWiseCode(definition_.src_tensors[0], definition_.dst_tensors[0],
                         definition_.precision, linked_operations_);
  return creation_context.cache->GetOrCreateCLKernel(
      code, "main_function", *creation_context.context,
      *creation_context.device, &kernel_);
}

Add CreateAdd(const OperationDef& definition, const std::vector<int>& channels,
              int dst_channels) {
  Add operation(definition, channels, dst_channels);
  operation.SetLinkIndex(0);
  return operation;
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
