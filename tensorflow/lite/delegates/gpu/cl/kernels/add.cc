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

#include <string>

#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
namespace cl {

std::string Add::GetElementWiseCode(
    const OperationDef& op_def,
    const std::vector<ElementwiseOperation*>& linked_operations) {
  TensorCodeGenerator src_tensor(
      "src_data", WHSPoint{"src_size.x", "src_size.y", "src_size.z"},
      op_def.src_tensors[0]);
  TensorCodeGenerator dst_tensor(
      "dst_data", WHSPoint{"dst_size.x", "dst_size.y", "dst_size.z"},
      op_def.dst_tensors[0]);

  std::string c = GetCommonDefines(op_def.precision);

  c += "__kernel void main_function(\n";
  c += src_tensor.GetDeclaration(AccessType::READ);
  c += GetArgsDeclaration();
  c += ::tflite::gpu::cl::GetArgsDeclaration(linked_operations);
  c += dst_tensor.GetDeclaration(AccessType::WRITE) + ",\n";
  c += "    int4 src_size,\n";
  c += "    int4 dst_size\n";
  c += ") {\n";
  c += "  int X = get_global_id(0);\n";
  c += "  int Y = get_global_id(1);\n";
  c += "  int Z = get_global_id(2);\n";
  c += "  if (X >= dst_size.x || Y >= dst_size.y || Z >= dst_size.z) { \n";
  c += "    return; \n";
  c += "  } \n";
  c += "  FLT4 src = (FLT4)(0.0);\n";
  if (src_depthes_[0] != dst_depth_) {
    c += "  if (Z < " + std::to_string(src_depthes_[0]) + ") {\n";
    c += "    src += " + src_tensor.ReadWHS("X", "Y", "Z") + ";\n";
    c += "  }\n";
  } else {
    c += "  src += " + src_tensor.ReadWHS("X", "Y", "Z") + ";\n";
  }
  const LinkingContext context{"src", "X", "Y", "Z"};
  c += "  " + GetCoreCode(context);
  c += PostProcess(linked_operations, context);
  c += "  " + dst_tensor.WriteWHS("src", "X", "Y", "Z") + "\n";
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
      src_depthes_(std::move(operation.src_depthes_)),
      dst_depth_(operation.dst_depth_) {}

Add& Add::operator=(Add&& operation) {
  if (this != &operation) {
    link_index_ = operation.link_index_;
    src_depthes_ = std::move(operation.src_depthes_);
    dst_depth_ = operation.dst_depth_;
    ElementwiseOperation::operator=(std::move(operation));
  }
  return *this;
}

void Add::SetLinkIndex(int index) {
  link_index_ = index;
}

std::string Add::GetCoreCode(const LinkingContext& context) const {
  std::string result;
  for (int i = 1; i < src_depthes_.size(); ++i) {
    const std::string tensor_name =
        absl::StrCat("src_data_", link_index_, "_", i);
    const std::string size_name =
        "src_size_" + std::to_string(link_index_) + "_" + std::to_string(i);
    TensorCodeGenerator src_tensor(
        tensor_name,
        WHSPoint{size_name + ".x", size_name + ".y", size_name + ".z"},
        definition_.src_tensors[i]);
    if (src_depthes_[i] != dst_depth_) {
      absl::StrAppend(&result, "  if (", context.z_coord, " < ",
                      src_depthes_[i], ") {\n");
      absl::StrAppend(&result, "  ", context.var_name, " += ",
                      src_tensor.ReadWHS(context.x_coord, context.y_coord,
                                         context.z_coord) +
                          ";\n");
      absl::StrAppend(&result, "  }\n");
    } else {
      absl::StrAppend(&result, "  ", context.var_name, " += ",
                      src_tensor.ReadWHS(context.x_coord, context.y_coord,
                                         context.z_coord) +
                          ";\n");
    }
  }
  return result;
}

std::string Add::GetArgsDeclaration() const {
  std::string args;
  for (int i = 1; i < src_depthes_.size(); ++i) {
    const std::string tensor_name =
        absl::StrCat("src_data_", link_index_, "_", i);
    absl::StrAppend(&args, ",\n",
                    GetTensorDeclaration(AccessType::READ, tensor_name,
                                         definition_.src_tensors[i]));
  }
  for (int i = 1; i < src_depthes_.size(); ++i) {
    const std::string size_name =
        "src_size_" + std::to_string(link_index_) + "_" + std::to_string(i);
    absl::StrAppend(&args, ",\n   int4 ", size_name);
  }
  return args;
}

Status Add::BindArguments(CLKernel* kernel) {
  for (int i = 1; i < src_depthes_.size(); ++i) {
    RETURN_IF_ERROR(kernel->SetMemoryAuto(src_[i]->GetMemoryPtr()));
  }
  for (int i = 1; i < src_depthes_.size(); ++i) {
    RETURN_IF_ERROR(kernel->SetBytesAuto(src_[i]->GetWBatchedHSB()));
  }
  return OkStatus();
}

Status Add::Compile(const CreationContext& creation_context) {
  const auto code = GetElementWiseCode(definition_, linked_operations_);
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
