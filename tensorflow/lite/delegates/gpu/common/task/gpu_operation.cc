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

#include "absl/strings/match.h"
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

bool IsWordSymbol(char symbol) {
  return absl::ascii_isalnum(symbol) || symbol == '_';
}

// Word is a token consisting of ascii_isalnum symbols or '_'
// (see above IsWordSymbol(char symbol))
// ReplaceAllWords replace all specified word-tokens(old_word) with new_word
void ReplaceAllWords(const std::string& old_word, const std::string& new_word,
                     std::string* str) {
  for (size_t position = str->find(old_word); position != std::string::npos;
       position = str->find(old_word, position)) {
    const char prev = position == 0 ? '.' : (*str)[position - 1];
    const char next = position + old_word.size() < str->size()
                          ? (*str)[position + old_word.size()]
                          : '.';
    if (IsWordSymbol(prev) || IsWordSymbol(next)) {
      position += 1;
      continue;
    }
    str->replace(position, old_word.size(), new_word);
    position += new_word.size();
  }
}

struct LinkableContext {
  std::string code;
  TensorDescriptor* tensor_desc;
};

absl::Status ResolveLinking(const GpuInfo& gpu_info,
                            const LinkableContext& linkable_context,
                            std::vector<std::string>* function_args,
                            std::string* result) {
  std::string value_name, x_coord, y_coord, z_coord, s_coord, b_coord;
  RETURN_IF_ERROR(
      linkable_context.tensor_desc->GetLinkingContextFromWriteSelector(
          *function_args, &value_name, &x_coord, &y_coord, &z_coord, &s_coord,
          &b_coord));
  const std::string new_value_name = value_name + "_final";
  const std::string out_var_declaration =
      "\n" +
      GetTypeDeclaration(gpu_info, linkable_context.tensor_desc->GetDataType(),
                         4) +
      " " + new_value_name + ";\n";
  *result = out_var_declaration +
            "{  // elementwise code with input:" + value_name +
            " output:" + new_value_name + "\n" +
            absl::Substitute(linkable_context.code, "") + "\n}\n";
  *result = absl::StrReplaceAll(*result, {{"\n", "\n  "},
                                          {"in_value", value_name},
                                          {"out_value", new_value_name},
                                          {"X_COORD", x_coord},
                                          {"Y_COORD", y_coord},
                                          {"Z_COORD", z_coord},
                                          {"S_COORD", s_coord},
                                          {"B_COORD", b_coord}});

  (*function_args)[0] = new_value_name;
  return absl::OkStatus();
}

// resolve constructions of type: args.object_name::const_expr_name
// Example: 'args.dst_tensor::type' can be replaced with 'float4'
absl::Status ResolveConstExprPass(const GpuInfo& gpu_info,
                                  const Arguments& args, std::string* code) {
  std::string result;
  size_t position = 0;
  constexpr char kArgsPrefix[] = "args.";
  size_t next_position = code->find(kArgsPrefix);
  while (next_position != std::string::npos) {
    size_t arg_pos = next_position;
    next_position += strlen(kArgsPrefix);
    std::string object_name = GetNextWord(*code, next_position);
    if (next_position + object_name.size() > code->size() - 2) {
      next_position = code->find(kArgsPrefix, next_position);
      continue;
    }
    char next0 = (*code)[next_position + object_name.size()];
    char next1 = (*code)[next_position + object_name.size() + 1];
    if (next0 == ':' && next1 == ':') {
      next_position += object_name.size() + 2;
      std::string const_expr_name = GetNextWord(*code, next_position);
      next_position += const_expr_name.size();
      std::string patch;
      tflite::gpu::GPUObjectDescriptor* desc_ptr;
      RETURN_IF_ERROR(args.GetDescriptor(object_name, &desc_ptr));
      RETURN_IF_ERROR(
          desc_ptr->PerformConstExpr(gpu_info, const_expr_name, &patch));
      code->replace(arg_pos, next_position - arg_pos, patch);
      position = arg_pos + patch.size();
    } else {
      position = arg_pos + strlen(kArgsPrefix);
    }
    next_position = code->find(kArgsPrefix, position);
  }
  return absl::OkStatus();
}

// resolve constructions of type: args.object_name.method_name(list of args)
// Example: 'args.bias.Read(S)' can be replaced with 'args.bias_buffer[S]'
absl::Status ResolveSelectorsPass(
    const GpuInfo& gpu_info,
    const std::map<std::string, LinkableContext>& linkables,
    const Arguments& args, std::string* code) {
  std::string result;
  size_t position = 0;
  constexpr char kArgsPrefix[] = "args.";
  size_t next_position = code->find(kArgsPrefix);
  while (next_position != std::string::npos) {
    size_t arg_pos = next_position;
    next_position += strlen(kArgsPrefix);
    std::string object_name = GetNextWord(*code, next_position);
    char next = (*code)[next_position + object_name.size()];
    if (next == '.') {
      next_position += object_name.size() + 1;
      std::string selector_name = GetNextWord(*code, next_position);
      next_position += selector_name.size();
      next = (*code)[next_position];
      std::vector<std::string> template_args;
      if (next == '<') {
        size_t close_bracket_pos;
        RETURN_IF_ERROR(ParseArgsInsideBrackets(
            *code, next_position, &close_bracket_pos, &template_args));
        next_position = close_bracket_pos;
        next = (*code)[next_position];
      }
      if (next != '(') {
        return absl::NotFoundError(absl::StrCat(
            "Expected ( after ", object_name, ".", selector_name, " call"));
      }
      std::vector<std::string> function_args;
      size_t close_bracket_pos;
      RETURN_IF_ERROR(ParseArgsInsideBrackets(
          *code, next_position, &close_bracket_pos, &function_args));
      for (auto& arg : function_args) {
        RETURN_IF_ERROR(ResolveSelectorsPass(gpu_info, {}, args, &arg));
      }
      std::string patch;
      std::string linkable_patch;
      GPUObjectDescriptor* desc_ptr;
      RETURN_IF_ERROR(args.GetDescriptor(object_name, &desc_ptr));
      auto names = desc_ptr->GetGPUResources(gpu_info).GetNames();
      if (desc_ptr && !linkables.empty() && selector_name == "Write") {
        auto it = linkables.find(object_name);
        if (it != linkables.end()) {
          RETURN_IF_ERROR(ResolveLinking(gpu_info, it->second, &function_args,
                                         &linkable_patch));
          RETURN_IF_ERROR(
              ResolveConstExprPass(gpu_info, args, &linkable_patch));
          RETURN_IF_ERROR(
              ResolveSelectorsPass(gpu_info, {}, args, &linkable_patch));
        }
      }
      RETURN_IF_ERROR(desc_ptr->PerformSelector(
          gpu_info, selector_name, function_args, template_args, &patch));
      for (const auto& member_name : names) {
        const std::string new_name =
            kArgsPrefix + object_name + "_" + member_name;
        ReplaceAllWords(member_name, new_name, &patch);
      }
      if (!linkable_patch.empty()) {
        patch = "{\n" + linkable_patch + patch + ";\n}";
      }
      code->replace(arg_pos, close_bracket_pos - arg_pos, patch);
      position = arg_pos + patch.size();
    } else {
      position = arg_pos + strlen(kArgsPrefix);
    }
    next_position = code->find(kArgsPrefix, position);
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

absl::Status GPUOperation::AddOperation(const GpuInfo& gpu_info,
                                        GPUOperation* operation) {
  const auto prev_type = definition_.dst_tensors[0].GetDataType();
  definition_.dst_tensors[0] = operation->definition_.dst_tensors[0];
  if (!elementwise_) {
    TensorDescriptor* dst_tensor_desc;
    RETURN_IF_ERROR(
        GetTensorDescriptor(dst_tensors_names_[0], &dst_tensor_desc));
    operation->definition_.dst_tensors[0].CopyWithoutData(dst_tensor_desc);
  }
  linkable_count_ += (operation->linkable_count_ + 1);
  std::string code = operation->elementwise_code_;
  std::string unique_postfix = absl::StrCat("_link", linkable_count_);
  code = absl::StrReplaceAll(
      code, {{"interm_value", "interm_value" + unique_postfix}});
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
  GPUObjectDescriptor* desc_ptr;
  RETURN_IF_ERROR(args_.GetDescriptor(tensor_name, &desc_ptr));
  *resutl = static_cast<TensorDescriptor*>(desc_ptr);
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
  RETURN_IF_ERROR(ResolveConstExprPass(gpu_info, args_, &code_));
  RETURN_IF_ERROR(ResolveSelectorsPass(gpu_info, linkables, args_, &code_));
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
      absl::StrContains(op.elementwise_code_, "in2_value")) {
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

absl::Status FuseElemWithElemInternal(
    const GpuInfo& gpu_info, GPUOperation&& elem0, GPUOperation&& elem1,
    const std::vector<std::pair<std::string, std::string>>& replacements,
    GPUOperation* result) {
  const int linkable_count =
      std::max(elem0.linkable_count_, elem1.linkable_count_) + 1;

  const std::string unique_postfix = absl::StrCat("_link", linkable_count);
  elem1.args_.RenameArgs(unique_postfix, &elem1.elementwise_code_);

  const auto link_value_type = elem0.definition_.dst_tensors[0].GetDataType();
  const std::string link_value_name = "interm_value" + unique_postfix;
  const std::string value_declaration =
      "\n" + GetTypeDeclaration(gpu_info, link_value_type, 4) + " " +
      link_value_name + ";\n";
  elem0.elementwise_code_ = absl::StrReplaceAll(
      elem0.elementwise_code_, {{"out_value", link_value_name}});
  elem0.elementwise_code_ =
      absl::Substitute(elem0.elementwise_code_, value_declaration);

  std::vector<std::pair<const absl::string_view, std::string>> replacements_new;
  for (int i = 0; i < replacements.size(); ++i) {
    if (replacements[i].second == "LINK_VALUE") {
      replacements_new.push_back(
          {absl::string_view(replacements[i].first), link_value_name});
    } else {
      replacements_new.push_back(
          {absl::string_view(replacements[i].first), replacements[i].second});
    }
  }
  elem1.elementwise_code_ =
      absl::StrReplaceAll(elem1.elementwise_code_,
                          {{"interm_value", "interm_value" + unique_postfix}});
  elem1.elementwise_code_ =
      absl::StrReplaceAll(elem1.elementwise_code_, replacements_new);

  OperationDef new_definition = elem0.definition_;
  new_definition.dst_tensors[0] = elem1.definition_.dst_tensors[0];

  *result = GPUOperation(new_definition);
  result->elementwise_ = true;
  result->elementwise_inputs_ = 1;
  result->tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  result->elementwise_code_ =
      elem0.elementwise_code_ + "\n" + elem1.elementwise_code_;
  result->linkable_count_ = linkable_count;
  result->args_ = std::move(elem0.args_);
  RETURN_IF_ERROR(result->args_.Merge(std::move(elem1.args_), unique_postfix,
                                      {elem1.second_elementwise_tensor_name_}));
  return absl::OkStatus();
}

absl::Status FuseSimpleElemWithSimpleElem(const GpuInfo& gpu_info,
                                          GPUOperation&& elem0,
                                          GPUOperation&& elem1,
                                          GPUOperation* result) {
  return FuseElemWithElemInternal(gpu_info, std::move(elem0), std::move(elem1),
                                  {{"in_value", "LINK_VALUE"}}, result);
}

absl::Status Fuse2InputElemWithSimpleElemAsFirstInput(const GpuInfo& gpu_info,
                                                      GPUOperation&& elem0,
                                                      GPUOperation&& elem1,
                                                      GPUOperation* result) {
  return FuseElemWithElemInternal(gpu_info, std::move(elem0), std::move(elem1),
                                  {{"in_value", "LINK_VALUE"},
                                   {"READ_SECOND_VALUE", ""},
                                   {"in2_value", "in_value"}},
                                  result);
}

absl::Status Fuse2InputElemWithSimpleElemAsSecondInput(const GpuInfo& gpu_info,
                                                       GPUOperation&& elem0,
                                                       GPUOperation&& elem1,
                                                       GPUOperation* result) {
  return FuseElemWithElemInternal(
      gpu_info, std::move(elem0), std::move(elem1),
      {{"READ_SECOND_VALUE", ""}, {"in2_value", "LINK_VALUE"}}, result);
}

//      input                input           input
//     /    \               /    \             |
//  elem0  elem1           |    elem1          |
//     \    /      -->      \    /      -->  elem
//   elem_root              elem2              |
//       |                    |                |
//     output               output           output
absl::Status Fuse2InputElemWith2SimpleElem(const GpuInfo& gpu_info,
                                           GPUOperation&& elem0,
                                           GPUOperation&& elem1,
                                           GPUOperation&& elem_root,
                                           GPUOperation* result) {
  elem0.linkable_count_ =
      std::max(elem0.linkable_count_, elem1.linkable_count_);
  elem0.linkable_count_ =
      std::max(elem0.linkable_count_, elem_root.linkable_count_);
  GPUOperation elem2;
  RETURN_IF_ERROR(
      FuseElemWithElemInternal(gpu_info, std::move(elem0), std::move(elem_root),
                               {{"in_value", "LINK_VALUE"}}, &elem2));
  return FuseElemWithElemInternal(
      gpu_info, std::move(elem1), std::move(elem2),
      {{"READ_SECOND_VALUE", ""}, {"in2_value", "LINK_VALUE"}}, result);
}

}  // namespace gpu
}  // namespace tflite
