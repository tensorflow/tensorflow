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

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_replace.h"
#include "absl/strings/substitute.h"
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

bool NeedsBroadcast(const TensorDescriptor& desc, const BHWC& shape) {
  bool needs_broadcast = shape.w == 1 || shape.h == 1 || shape.c == 1;
  if (desc.HasAxis(Axis::BATCH)) {
    needs_broadcast = needs_broadcast || shape.b == 1;
  }
  return needs_broadcast;
}

std::string GetStringWithoutComments(const std::string& code) {
  std::string result;
  result.reserve(code.size());
  for (size_t i = 0; i < code.size(); ++i) {
    // skip long comment /*...*/
    if (code[i] == '/' && i + 1 < code.size() && code[i + 1] == '*') {
      i = i + 3;
      for (; i < code.size() && (code[i - 1] != '*' && code[i] != '/'); ++i) {
      }
      continue;
    }
    // skip short comment //...
    if (code[i] == '/' && i + 1 < code.size() && code[i + 1] == '/') {
      i = i + 2;
      for (; i < code.size() && code[i] != '\n'; ++i) {
      }
      if (i != code.size()) {
        // i is '\n'
        result.push_back(code[i]);
      }
      continue;
    }
    result.push_back(code[i]);
  }
  return result;
}

struct LinkableContext {
  std::string code;
  TensorDescriptor* tensor_desc;
};

absl::Status ResolveLinking(
    const GpuInfo& gpu_info,
    const std::map<std::string, LinkableContext>& linkables,
    std::string* code) {
  std::map<std::string, LinkableContext> useful_linkables;
  for (const auto& linkable : linkables) {
    if (!linkable.second.code.empty()) {
      useful_linkables[linkable.first] = linkable.second;
    }
  }
  if (useful_linkables.empty()) {
    return absl::OkStatus();
  }
  constexpr char kArgsPrefix[] = "args.";
  for (size_t position = code->find(kArgsPrefix); position != std::string::npos;
       position = code->find(kArgsPrefix, position)) {
    const size_t args_pos = position;
    position += strlen(kArgsPrefix);
    const std::string object_name = GetNextWord(*code, position);
    position += object_name.size();
    auto linkable = useful_linkables.find(object_name);
    if (linkable == useful_linkables.end()) {
      continue;
    }
    char next = (*code)[position];
    position += 1;
    if (next != '.') {
      continue;
    }
    const std::string selector_name = GetNextWord(*code, position);
    position += selector_name.size();
    if (selector_name != "Write") {
      continue;
    }
    next = (*code)[position];
    std::vector<std::string> template_args;
    if (next == '<') {
      size_t close_bracket_pos;
      RETURN_IF_ERROR(ParseArgsInsideBrackets(
          *code, position, &close_bracket_pos, &template_args));
      position = close_bracket_pos;
      next = (*code)[position];
    }
    if (next != '(') {
      return absl::NotFoundError(absl::StrCat("Expected ( after ", object_name,
                                              ".", selector_name, " call"));
    }
    std::vector<std::string> function_args;
    size_t close_bracket_pos;
    RETURN_IF_ERROR(ParseArgsInsideBrackets(*code, position, &close_bracket_pos,
                                            &function_args));

    std::string value_name, x_coord, y_coord, z_coord, s_coord, b_coord;
    RETURN_IF_ERROR(
        linkable->second.tensor_desc->GetLinkingContextFromWriteSelector(
            function_args, &value_name, &x_coord, &y_coord, &z_coord, &s_coord,
            &b_coord));
    const std::string new_value_name = value_name + "_final";
    const std::string out_var_declaration =
        GetTypeDeclaration(gpu_info,
                           linkable->second.tensor_desc->GetDataType(), 4) +
        " " + new_value_name + ";\n";
    std::string prefix;
    size_t space_pos = args_pos - 1;
    while (space_pos >= 0 &&
           ((*code)[space_pos] == ' ' || (*code)[space_pos] == '\t')) {
      prefix += (*code)[space_pos];
      space_pos -= 1;
    }
    function_args[0] = new_value_name;
    std::string write_code = kArgsPrefix + object_name + ".Write";
    if (!template_args.empty()) {
      write_code += std::string("<") + template_args[0];
      for (int i = 1; i < template_args.size(); ++i) {
        write_code += ", " + template_args[i];
      }
      write_code += ">";
    }
    write_code += std::string("(") + function_args[0];
    for (int i = 1; i < function_args.size(); ++i) {
      write_code += ", " + function_args[i];
    }
    write_code += ")";
    std::string patch =
        "{\n" + absl::Substitute(linkable->second.code, out_var_declaration) +
        "\n" + write_code + ";\n}";
    patch = absl::StrReplaceAll(patch, {{"\n", "\n" + prefix},
                                        {"in_value", value_name},
                                        {"out_value", new_value_name},
                                        {"X_COORD", x_coord},
                                        {"Y_COORD", y_coord},
                                        {"Z_COORD", z_coord},
                                        {"S_COORD", s_coord},
                                        {"B_COORD", b_coord}});
    code->replace(args_pos, close_bracket_pos - args_pos, patch);
    position = args_pos + patch.size();
  }
  return absl::OkStatus();
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
      elementwise_inputs_(operation.elementwise_inputs_),
      second_elementwise_tensor_name_(
          operation.second_elementwise_tensor_name_),
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
    std::swap(elementwise_inputs_, operation.elementwise_inputs_);
    std::swap(second_elementwise_tensor_name_,
              operation.second_elementwise_tensor_name_);
    std::swap(linkable_count_, operation.linkable_count_);
    elementwise_code_ = std::move(operation.elementwise_code_);
  }
  return *this;
}

//    input       input
//      |           |
//    elem0         |
//      |    -->  elem
//    elem1         |
//      |           |
//    output      output
// GPUOperation* operation is elem1
// *this is elem0
absl::Status GPUOperation::FuseSimpleElemWithSimpleElem(
    const GpuInfo& gpu_info, GPUOperation* operation) {
  GPUOperation& elem0 = *this;
  GPUOperation& elem1 = *operation;
  elem0.definition_.dst_tensors[0] = elem1.definition_.dst_tensors[0];
  const auto link_value_type = elem1.definition_.src_tensors[0].GetDataType();
  elem0.linkable_count_ += (elem1.linkable_count_ + 1);
  std::string unique_postfix = absl::StrCat("_link", elem0.linkable_count_);
  elem1.args_.RenameArgs(unique_postfix, &elem1.elementwise_code_);
  const std::string link_value_name = "interm_value" + unique_postfix;
  const std::string value_declaration =
      "\n" + GetTypeDeclaration(gpu_info, link_value_type, 4) + " " +
      link_value_name + ";\n";
  elem1.elementwise_code_ = absl::StrReplaceAll(
      elem1.elementwise_code_, {{"in_value", link_value_name}});
  elem0.elementwise_code_ = absl::StrReplaceAll(
      elem0.elementwise_code_, {{"out_value", link_value_name}});
  elem0.elementwise_code_ =
      absl::Substitute(elem0.elementwise_code_, value_declaration);
  elem0.elementwise_code_ += "\n" + elem1.elementwise_code_;
  return args_.Merge(std::move(elem1.args_), unique_postfix);
}

//      input           input
//     /    \             |
//  elem0    |            |
//     \    /      -->  elem
//     elem1              |
//       |                |
//     output           output
// GPUOperation* operation is elem1
// *this is elem0
absl::Status GPUOperation::Fuse2InputElemWithSimpleElemAsFirstInput(
    const GpuInfo& gpu_info, GPUOperation* operation) {
  GPUOperation& elem0 = *this;
  GPUOperation& elem1 = *operation;
  const auto link_value_type = elem0.definition_.dst_tensors[0].GetDataType();
  elem0.definition_.dst_tensors[0] = elem1.definition_.dst_tensors[0];
  elem0.linkable_count_ += (elem1.linkable_count_ + 1);
  std::string unique_postfix = absl::StrCat("_link", elem0.linkable_count_);
  elem1.args_.RenameArgs(unique_postfix, &elem1.elementwise_code_);
  const std::string link_value_name = "interm_value" + unique_postfix;
  const std::string value_declaration =
      "\n" + GetTypeDeclaration(gpu_info, link_value_type, 4) + " " +
      link_value_name + ";\n";
  elem0.elementwise_code_ = absl::StrReplaceAll(
      elem0.elementwise_code_, {{"out_value", link_value_name}});
  elem0.elementwise_code_ =
      absl::Substitute(elem0.elementwise_code_, value_declaration);
  elem1.elementwise_code_ = absl::StrReplaceAll(elem1.elementwise_code_,
                                                {{"in_value", link_value_name},
                                                 {"READ_SECOND_VALUE", ""},
                                                 {"in2_value", "in_value"}});
  elem0.elementwise_code_ += "\n" + elem1.elementwise_code_;
  return elem0.args_.Merge(std::move(elem1.args_), unique_postfix,
                           {elem1.second_elementwise_tensor_name_});
}

//      input           input
//     /    \             |
//    |    elem0          |
//     \    /      -->  elem
//     elem1              |
//       |                |
//     output           output
// GPUOperation* operation is elem1
// *this is elem0
absl::Status GPUOperation::Fuse2InputElemWithSimpleElemAsSecondInput(
    const GpuInfo& gpu_info, GPUOperation* operation43) {
  GPUOperation& elem0 = *this;
  GPUOperation& elem1 = *operation43;
  const auto link_value_type = elem0.definition_.dst_tensors[0].GetDataType();
  elem0.definition_.dst_tensors[0] = elem1.definition_.dst_tensors[0];
  elem0.linkable_count_ += (elem1.linkable_count_ + 1);
  std::string unique_postfix = absl::StrCat("_link", elem0.linkable_count_);
  elem1.args_.RenameArgs(unique_postfix, &elem1.elementwise_code_);
  const std::string link_value_name = "interm_value" + unique_postfix;
  const std::string value_declaration =
      "\n" + GetTypeDeclaration(gpu_info, link_value_type, 4) + " " +
      link_value_name + ";\n";
  elem0.elementwise_code_ = absl::StrReplaceAll(
      elem0.elementwise_code_, {{"out_value", link_value_name}});
  elem0.elementwise_code_ =
      absl::Substitute(elem0.elementwise_code_, value_declaration);
  elem1.elementwise_code_ = absl::StrReplaceAll(
      elem1.elementwise_code_,
      {{"in2_value", link_value_name}, {"READ_SECOND_VALUE", ""}});
  elem0.elementwise_code_ += "\n" + elem1.elementwise_code_;
  return elem0.args_.Merge(std::move(elem1.args_), unique_postfix,
                           {elem1.second_elementwise_tensor_name_});
}

absl::Status GPUOperation::AddOperation(const GpuInfo& gpu_info,
                                        GPUOperation* operation) {
  const auto prev_type = definition_.dst_tensors[0].GetDataType();
  definition_.dst_tensors[0] = operation->definition_.dst_tensors[0];
  linkable_count_ += (operation->linkable_count_ + 1);
  std::string code = operation->elementwise_code_;
  std::string unique_postfix = absl::StrCat("_link", linkable_count_);
  operation->args_.RenameArgs(unique_postfix, &code);
  operation->second_elementwise_tensor_name_ += unique_postfix;
  if (elementwise_code_.empty()) {
    elementwise_code_ = code;
    elementwise_inputs_ = operation->elementwise_inputs_;
    second_elementwise_tensor_name_ =
        operation->second_elementwise_tensor_name_;
  } else {
    if (operation->elementwise_inputs_ == 2) {
      if (elementwise_inputs_ == 2) {
        // if we have fusion of 2 2-input elementwise ops, we will get 3-input
        // elementwise, but currently we support only max 2-input elementwise.
        // So we will resolve one input here.
        RETURN_IF_ERROR(ResolveSecondElementwiseInput());
      }
      second_elementwise_tensor_name_ =
          operation->second_elementwise_tensor_name_;
      elementwise_inputs_ = 2;
    }
    const std::string new_value_name = "interm_value" + unique_postfix;
    code = absl::StrReplaceAll(code, {{"in_value", new_value_name}});
    elementwise_code_ =
        absl::StrReplaceAll(elementwise_code_, {{"out_value", new_value_name}});
    const std::string out_var_declaration =
        "\n" + GetTypeDeclaration(gpu_info, prev_type, 4) + " " +
        new_value_name + ";\n";
    elementwise_code_ =
        absl::Substitute(elementwise_code_, out_var_declaration);
    elementwise_code_ = elementwise_code_ + "\n" + code;
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

absl::Status GPUOperation::ResolveSecondElementwiseInput() {
  if (elementwise_inputs_ != 2) {
    return absl::FailedPreconditionError(
        "Can not apply ResolveSecondElementwiseInput for non 2 input "
        "elementwise");
  }
  TensorDescriptor* tensor_desc;
  RETURN_IF_ERROR(
      GetTensorDescriptor(second_elementwise_tensor_name_, &tensor_desc));
  std::string coords = "X_COORD, Y_COORD, S_COORD";
  if (tensor_desc->HasAxis(Axis::BATCH)) {
    coords += ", B_COORD";
  }
  const std::string read_code = "args." + second_elementwise_tensor_name_ +
                                "::type second_value = args." +
                                second_elementwise_tensor_name_ + ".Read(" +
                                coords + ");\n";
  elementwise_code_ = absl::StrReplaceAll(
      elementwise_code_,
      {{"in2_value", "second_value"}, {"READ_SECOND_VALUE", read_code}});
  elementwise_inputs_ = 1;
  return absl::OkStatus();
}

absl::Status GPUOperation::GetTensorDescriptor(const std::string& tensor_name,
                                               TensorDescriptor** resutl) {
  for (int i = 0; i < src_tensors_names_.size(); ++i) {
    if (src_tensors_names_[i] == tensor_name) {
      int index = elementwise_ ? i + 1 : i;
      *resutl = &definition_.src_tensors[index];
      return absl::OkStatus();
    }
  }
  for (int i = 0; i < dst_tensors_names_.size(); ++i) {
    if (dst_tensors_names_[i] == tensor_name) {
      int index = elementwise_ ? i + 1 : i;
      *resutl = &definition_.dst_tensors[index];
      return absl::OkStatus();
    }
  }
  return absl::NotFoundError("Can not find tensor with this name");
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

void GPUOperation::AddDstTensor(const std::string& tensor_name,
                                const TensorDescriptor& desc) {
  dst_tensors_names_.push_back(tensor_name);
  auto desc_new = std::make_unique<TensorDescriptor>(desc);
  args_.AddObjectRef(tensor_name, AccessType::WRITE, std::move(desc_new));
}

absl::Status GPUOperation::AssembleCode(const GpuInfo& gpu_info) {
  if (elementwise_inputs_ == 2) {
    RETURN_IF_ERROR(ResolveSecondElementwiseInput());
  }
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
    elementwise_ = false;
  }
  std::map<std::string, LinkableContext> linkables;
  if (!elementwise_code_.empty()) {
    TensorDescriptor* dst_tensor_desc;
    RETURN_IF_ERROR(
        GetTensorDescriptor(dst_tensors_names_[0], &dst_tensor_desc));
    linkables[dst_tensors_names_[0]] = {elementwise_code_, dst_tensor_desc};
  }
  RETURN_IF_ERROR(ResolveLinking(gpu_info, linkables, &code_));
  code_ = GetStringWithoutComments(code_);
  RETURN_IF_ERROR(args_.Compile(gpu_info, &code_));
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
  const BHWC second_shape(2, 2, 2, 2);  // dummy non-broadcasted shape
  return CreateGpuOperation(definition, std::move(descriptor), second_shape);
}

GPUOperation CreateGpuOperation(const OperationDef& definition,
                                ElementwiseDescriptor&& descriptor,
                                const BHWC& second_shape) {
  GPUOperation op(definition);
  op.elementwise_code_ = std::move(descriptor.code);
  op.elementwise_ = true;
  if (definition.src_tensors.size() > 1 &&
      op.elementwise_code_.find("in2_value")) {
    const auto second_tensor_def = definition.src_tensors[1];
    if (NeedsBroadcast(second_tensor_def, second_shape)) {
      const std::string x_coord = second_shape.w == 1 ? "0" : "X_COORD";
      const std::string y_coord = second_shape.h == 1 ? "0" : "Y_COORD";
      const std::string s_coord = second_shape.c == 1 ? "0" : "S_COORD";
      std::string coords = absl::StrCat(x_coord, ", ", y_coord, ", ", s_coord);
      if (second_tensor_def.HasAxis(Axis::BATCH)) {
        const std::string b_coord = second_shape.b == 1 ? "0" : "B_COORD";
        coords += ", " + b_coord;
      }
      std::string read_value_code = absl::StrCat(
          "args.src_tensor_1::type in2_value = args.src_tensor_1.Read(", coords,
          ");\n");
      if (second_shape.c == 1) {
        read_value_code += "  in2_value.y = in2_value.x;\n";
        read_value_code += "  in2_value.z = in2_value.x;\n";
        read_value_code += "  in2_value.w = in2_value.x;\n";
      }
      op.elementwise_code_ =
          "$0{" + read_value_code + op.elementwise_code_ + "}";
      op.elementwise_code_ = absl::StrReplaceAll(
          op.elementwise_code_, {{"in2_value", "second_value"}});
      op.elementwise_inputs_ = 1;
    } else {
      op.elementwise_code_ =
          "$0{READ_SECOND_VALUE" + op.elementwise_code_ + "}";
      op.elementwise_inputs_ = 2;
      op.second_elementwise_tensor_name_ = "src_tensor_1";
    }
  } else {
    op.elementwise_code_ = "$0{" + op.elementwise_code_ + "}";
    op.elementwise_inputs_ = 1;
  }
  op.args_ = std::move(descriptor.args);
  for (int i = 1; i < definition.src_tensors.size(); ++i) {
    const std::string tensor_name = "src_tensor_" + std::to_string(i);
    op.AddSrcTensor(tensor_name, definition.src_tensors[i]);
  }
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

absl::Status Fuse2InputElemWith2SimpleElem(const GpuInfo& gpu_info,
                                           GPUOperation&& elem0,
                                           GPUOperation&& elem1,
                                           GPUOperation&& elem_root,
                                           GPUOperation* result) {
  int linkable_count = std::max(elem0.linkable_count_, elem1.linkable_count_);
  linkable_count = std::max(linkable_count, elem_root.linkable_count_);
  linkable_count += 1;

  std::string unique_postfix = absl::StrCat("_link", linkable_count);
  elem0.args_.RenameArgs(unique_postfix + "l", &elem0.elementwise_code_);
  elem1.args_.RenameArgs(unique_postfix + "r", &elem1.elementwise_code_);
  elem_root.args_.RenameArgs(unique_postfix, &elem_root.elementwise_code_);
  const std::string link_left_value_name = "interm_value_left" + unique_postfix;
  const std::string link_right_value_name =
      "interm_value_right" + unique_postfix;
  const auto link_left_value_type =
      elem0.definition_.dst_tensors[0].GetDataType();
  const std::string left_value_declaration =
      "\n" + GetTypeDeclaration(gpu_info, link_left_value_type, 4) + " " +
      link_left_value_name + ";\n";
  const auto link_right_value_type =
      elem1.definition_.dst_tensors[0].GetDataType();
  const std::string right_value_declaration =
      "\n" + GetTypeDeclaration(gpu_info, link_right_value_type, 4) + " " +
      link_right_value_name + ";\n";
  elem0.elementwise_code_ = absl::StrReplaceAll(
      elem0.elementwise_code_, {{"out_value", link_left_value_name}});
  elem1.elementwise_code_ = absl::StrReplaceAll(
      elem1.elementwise_code_, {{"out_value", link_right_value_name}});
  elem0.elementwise_code_ =
      absl::Substitute(elem0.elementwise_code_, left_value_declaration);
  elem1.elementwise_code_ =
      absl::Substitute(elem1.elementwise_code_, right_value_declaration);
  elem_root.elementwise_code_ = absl::StrReplaceAll(
      elem_root.elementwise_code_, {{"in_value", link_left_value_name},
                                    {"READ_SECOND_VALUE", ""},
                                    {"in2_value", link_right_value_name}});

  OperationDef new_definition = elem0.definition_;
  new_definition.dst_tensors[0] = elem_root.definition_.dst_tensors[0];

  *result = GPUOperation(new_definition);
  result->elementwise_ = true;
  result->elementwise_inputs_ = 1;
  result->tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  result->elementwise_code_ = elem0.elementwise_code_ + "\n" +
                              elem1.elementwise_code_ + "\n" +
                              elem_root.elementwise_code_;
  result->linkable_count_ = linkable_count;
  RETURN_IF_ERROR(
      result->args_.Merge(std::move(elem0.args_), unique_postfix + "l"));
  RETURN_IF_ERROR(
      result->args_.Merge(std::move(elem1.args_), unique_postfix + "r"));
  RETURN_IF_ERROR(
      result->args_.Merge(std::move(elem_root.args_), unique_postfix,
                          {elem_root.second_elementwise_tensor_name_}));
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
