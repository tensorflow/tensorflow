/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/common/task/arguments.h"

#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/buffer_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_object_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/util.h"

namespace tflite {
namespace gpu {
namespace {
bool IsWordSymbol(char symbol) {
  return absl::ascii_isalnum(symbol) || symbol == '_';
}

void ReplaceAllWords(const std::string& old_word, const std::string& new_word,
                     std::string* str) {
  size_t position = str->find(old_word);
  while (position != std::string::npos) {
    char prev = position == 0 ? '.' : (*str)[position - 1];
    char next = position + old_word.size() < str->size()
                    ? (*str)[position + old_word.size()]
                    : '.';
    if (IsWordSymbol(prev) || IsWordSymbol(next)) {
      position = str->find(old_word, position + 1);
      continue;
    }
    str->replace(position, old_word.size(), new_word);
    position = str->find(old_word, position + new_word.size());
  }
}

std::string GetNextWord(const std::string& code, size_t first_position) {
  size_t pos = first_position;
  char t = code[pos];
  while (IsWordSymbol(t)) {
    pos++;
    t = code[pos];
  }
  return code.substr(first_position, pos - first_position);
}

bool HasWord(const std::string& word, const std::string& text) {
  size_t pos = text.find(word);
  while (pos != std::string::npos) {
    char prev = pos == 0 ? '.' : text[pos - 1];
    char next = pos + word.size() < text.size() ? text[pos + word.size()] : '.';
    if (!IsWordSymbol(prev) && !IsWordSymbol(next)) {
      return true;
    }
    pos = text.find(word, pos + 1);
  }
  return false;
}

std::string RenameArg(const std::vector<std::string>& object_names,
                      const std::string& postfix, const std::string& arg_name) {
  for (const auto& object_name : object_names) {
    if (absl::StartsWith(arg_name, object_name) &&
        arg_name.size() > object_name.size() &&
        arg_name[object_name.size()] == '_') {
      return object_name + postfix +
             arg_name.substr(object_name.size(),
                             arg_name.size() - object_name.size());
    }
  }
  return arg_name + postfix;
}

size_t FindEnclosingBracket(const std::string& text, size_t first_pos,
                            char bracket) {
  const std::map<char, char> brackets = {
      {'(', ')'},
      {'{', '}'},
      {'[', ']'},
      {'<', '>'},
  };
  char b_open = bracket;
  auto it = brackets.find(b_open);
  if (it == brackets.end()) {
    return -1;
  }
  char b_close = it->second;
  size_t pos = first_pos;
  int opened = 1;
  int closed = 0;
  while (opened != closed && pos < text.size()) {
    if (text[pos] == b_open) {
      opened++;
    } else if (text[pos] == b_close) {
      closed++;
    }
    pos++;
  }
  if (opened == closed) {
    return pos;
  } else {
    return -1;
  }
}

absl::Status ParseArgsInsideBrackets(const std::string& text,
                                     size_t open_bracket_pos,
                                     size_t* close_bracket_pos,
                                     std::vector<std::string>* args) {
  *close_bracket_pos =
      FindEnclosingBracket(text, open_bracket_pos + 1, text[open_bracket_pos]);
  if (*close_bracket_pos == -1) {
    return absl::NotFoundError("Not found enclosing bracket");
  }
  std::string str_args = text.substr(open_bracket_pos + 1,
                                     *close_bracket_pos - open_bracket_pos - 2);
  std::vector<absl::string_view> words = absl::StrSplit(str_args, ',');
  args->reserve(words.size());
  for (const auto& word : words) {
    absl::string_view arg = absl::StripAsciiWhitespace(word);
    if (!arg.empty()) {
      args->push_back(std::string(arg));
    }
  }
  return absl::OkStatus();
}

std::string DataTypeToGlType(DataType data_type, int vec_size,
                             bool explicit_f16) {
  if (data_type == DataType::FLOAT32) {
    if (vec_size == 1) {
      return "float";
    } else {
      return "vec" + std::to_string(vec_size);
    }
  } else if (data_type == DataType::FLOAT16) {
    if (vec_size == 1) {
      return explicit_f16 ? "float16_t" : "float";
    } else {
      if (explicit_f16) {
        return "f16vec" + std::to_string(vec_size);
      } else {
        return "vec" + std::to_string(vec_size);
      }
    }
  } else if (data_type == DataType::INT32) {
    if (vec_size == 1) {
      return "int";
    } else {
      return "ivec" + std::to_string(vec_size);
    }
  } else if (data_type == DataType::UINT32) {
    if (vec_size == 1) {
      return "uint";
    } else {
      return "uvec" + std::to_string(vec_size);
    }
  }
  return "unsupported_type";
}

absl::Status BufferToKernelLanguage(const GpuInfo& gpu_info,
                                    const std::string& buffer_name,
                                    const BufferDescriptor* buffer_desc,
                                    std::string* result) {
  if (buffer_desc->element_size != 1) {
    return absl::UnimplementedError("No support of vector types.");
  }
  const int elements_count =
      buffer_desc->size /
      (buffer_desc->element_size * SizeOf(buffer_desc->element_type));
  if (gpu_info.IsGlsl()) {
    const std::string gl_type =
        DataTypeToGlType(buffer_desc->element_type, buffer_desc->element_size,
                         gpu_info.IsGlslSupportsExplicitFp16());
    *result = "const ";
    if (buffer_desc->element_type == DataType::FLOAT16 &&
        !gpu_info.IsGlslSupportsExplicitFp16()) {
      *result += "mediump ";
    }
    *result += gl_type + " " + buffer_name + "_buffer[] = " + gl_type + "[](\n";
  } else if (gpu_info.IsApiMetal()) {
    const std::string metal_type =
        ToMetalDataType(buffer_desc->element_type, buffer_desc->element_size);
    *result = "constant " + metal_type + " " + buffer_name + "_buffer[" +
              std::to_string(elements_count) + "] = {\n";
  } else if (gpu_info.IsApiOpenCl()) {
    const std::string cl_type =
        ToCLDataType(buffer_desc->element_type, buffer_desc->element_size);
    *result = "__constant " + cl_type + " " + buffer_name + "_buffer[" +
              std::to_string(elements_count) + "] = {\n";
  } else {
    return absl::UnimplementedError("Not supported API.");
  }
  if (buffer_desc->element_type == DataType::FLOAT16) {
    std::string postfix = "f";
    if (gpu_info.IsGlsl() && gpu_info.IsGlslSupportsExplicitFp16()) {
      postfix = "hf";
    }
    const half* data_ptr =
        reinterpret_cast<const half*>(buffer_desc->data.data());
    for (int i = 0; i < elements_count; ++i) {
      *result += "  " +
                 absl::StrFormat("%.10f", static_cast<float>(data_ptr[i])) +
                 postfix;
      if (i != elements_count - 1) {
        *result += ",\n";
      }
    }
  } else if (buffer_desc->element_type == DataType::FLOAT32) {
    const float* data_ptr =
        reinterpret_cast<const float*>(buffer_desc->data.data());
    for (int i = 0; i < elements_count; ++i) {
      *result += "  " + absl::StrFormat("%.10f", data_ptr[i]) + "f";
      if (i != elements_count - 1) {
        *result += ",\n";
      }
    }
  } else {
    return absl::UnimplementedError("Not supported type.");
  }
  if (gpu_info.IsGlsl()) {
    *result += ");\n";
  } else {
    *result += "};\n";
  }

  return absl::OkStatus();
}

}  // namespace

// Static
constexpr char Arguments::kArgsPrefix[];

void Arguments::AddFloat(const std::string& name, float value) {
  float_values_[name].value = value;
}
void Arguments::AddHalf(const std::string& name, half value) {
  half_values_[name].value = value;
}
void Arguments::AddInt(const std::string& name, int value) {
  int_values_[name].value = value;
}

absl::Status Arguments::SetInt(const std::string& name, int value) {
  auto it = int_values_.find(name);
  if (it == int_values_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No int argument with name - ", name));
  }
  it->second.value = value;
  return absl::OkStatus();
}
absl::Status Arguments::SetFloat(const std::string& name, float value) {
  auto it = float_values_.find(name);
  if (it == float_values_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No float argument with name - ", name));
  }
  it->second.value = value;
  return absl::OkStatus();
}

absl::Status Arguments::SetHalf(const std::string& name, half value) {
  auto it = half_values_.find(name);
  if (it == half_values_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No half argument with name - ", name));
  }
  it->second.value = value;
  return absl::OkStatus();
}

void Arguments::AddObjectRef(const std::string& name, AccessType access_type,
                             GPUObjectDescriptorPtr&& descriptor_ptr) {
  descriptor_ptr->SetAccess(access_type);
  object_refs_[name] = {std::move(descriptor_ptr)};
}

void Arguments::AddObject(const std::string& name,
                          GPUObjectDescriptorPtr&& descriptor_ptr) {
  descriptor_ptr->SetAccess(AccessType::READ);
  objects_[name] = {std::move(descriptor_ptr)};
}

void Arguments::RenameArgs(const std::string& postfix,
                           std::string* code) const {
  size_t next_position = code->find(kArgsPrefix);
  while (next_position != std::string::npos) {
    size_t arg_pos = next_position + strlen(kArgsPrefix);
    std::string arg_name = GetNextWord(*code, arg_pos);
    code->replace(arg_pos, arg_name.size(), arg_name + postfix);
    next_position = code->find(kArgsPrefix, arg_pos + arg_name.size());
  }
}

absl::Status Arguments::Merge(Arguments&& args, const std::string& postfix,
                              const std::vector<std::string>& exception_names) {
  std::vector<std::string> object_names;
  object_names.reserve(args.object_refs_.size() + args.objects_.size());
  for (auto& v : args.object_refs_) {
    if (std::find(exception_names.begin(), exception_names.end(), v.first) !=
        exception_names.end()) {
      continue;
    }
    object_names.push_back(v.first);
    const std::string name = v.first + postfix;
    if (object_refs_.find(name) != object_refs_.end()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Object reference name collision. Name - ", name));
    }
    object_refs_[name] = {std::move(v.second)};
  }
  for (auto& v : args.objects_) {
    if (std::find(exception_names.begin(), exception_names.end(), v.first) !=
        exception_names.end()) {
      continue;
    }
    object_names.push_back(v.first);
    const std::string name = v.first + postfix;
    if (objects_.find(name) != objects_.end()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Object name collision. Name - ", name));
    }
    objects_[name] = {std::move(v.second)};
  }
  for (const auto& v : args.int_values_) {
    AddInt(RenameArg(object_names, postfix, v.first), v.second.value);
  }
  for (const auto& v : args.float_values_) {
    AddFloat(RenameArg(object_names, postfix, v.first), v.second.value);
  }
  for (const auto& v : args.half_values_) {
    AddHalf(RenameArg(object_names, postfix, v.first), v.second.value);
  }
  return absl::OkStatus();
}

absl::Status Arguments::GetDescriptor(const std::string& name,
                                      GPUObjectDescriptor** descriptor) const {
  auto it_ref = object_refs_.find(name);
  if (it_ref != object_refs_.end()) {
    *descriptor = it_ref->second.get();
    return absl::OkStatus();
  }
  auto it = objects_.find(name);
  if (it != objects_.end()) {
    *descriptor = it->second.get();
    return absl::OkStatus();
  }
  return absl::NotFoundError(absl::StrCat("No GPU object with name - ", name));
}

void Arguments::ReleaseCPURepresentation() {
  for (auto& t : objects_) {
    t.second->Release();
  }
}

void Arguments::GetActiveArguments(const std::string& code) {
  for (auto& float_val : float_values_) {
    float_val.second.active = HasWord(kArgsPrefix + float_val.first, code);
  }
  for (auto& int_val : int_values_) {
    int_val.second.active = HasWord(kArgsPrefix + int_val.first, code);
  }
  for (auto& half_val : half_values_) {
    half_val.second.active = HasWord(kArgsPrefix + half_val.first, code);
  }
}

int Arguments::GetReadTexturesCount(const GpuInfo& gpu_info) const {
  int counter = 0;
  for (auto& t : objects_) {
    counter += t.second->GetGPUResources(gpu_info).GetReadImagesCount();
  }
  for (auto& t : object_refs_) {
    counter += t.second->GetGPUResources(gpu_info).GetReadImagesCount();
  }
  return counter;
}

int Arguments::GetWriteTexturesCount(const GpuInfo& gpu_info) const {
  int counter = 0;
  for (auto& t : objects_) {
    counter += t.second->GetGPUResources(gpu_info).GetWriteImagesCount();
  }
  for (auto& t : object_refs_) {
    counter += t.second->GetGPUResources(gpu_info).GetWriteImagesCount();
  }
  return counter;
}

void Arguments::SetStateValueForAllObjects(const std::string& key,
                                           const std::string& value) {
  for (auto& obj : object_refs_) {
    obj.second->SetStateVar(key, value);
  }
  for (auto& obj : objects_) {
    obj.second->SetStateVar(key, value);
  }
}

absl::Status Arguments::Compile(
    const GpuInfo& gpu_info,
    const std::map<std::string, std::string>& linkables, std::string* code) {
  RETURN_IF_ERROR(AddObjectsScalarArgs(gpu_info));
  RETURN_IF_ERROR(ResolveConstExprPass(gpu_info, code));
  RETURN_IF_ERROR(ResolveSelectorsPass(gpu_info, linkables, code));
  GetActiveArguments(*code);
  RETURN_IF_ERROR(ResolveKernelGlobalSpaceBuffers(gpu_info, code));
  return absl::OkStatus();
}

absl::Status Arguments::ResolveConstExprPass(const GpuInfo& gpu_info,
                                             std::string* code) const {
  std::string result;
  size_t position = 0;
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
      RETURN_IF_ERROR(
          ResolveConstExpr(gpu_info, object_name, const_expr_name, &patch));
      code->replace(arg_pos, next_position - arg_pos, patch);
      position = arg_pos + patch.size();
    } else {
      position = arg_pos + strlen(kArgsPrefix);
    }
    next_position = code->find(kArgsPrefix, position);
  }
  return absl::OkStatus();
}

absl::Status Arguments::ResolveConstExpr(const GpuInfo& gpu_info,
                                         const std::string& object_name,
                                         const std::string& const_expr,
                                         std::string* result) const {
  tflite::gpu::GPUObjectDescriptor* desc_ptr;
  RETURN_IF_ERROR(GetDescriptor(object_name, &desc_ptr));
  RETURN_IF_ERROR(desc_ptr->PerformConstExpr(gpu_info, const_expr, result));
  return absl::OkStatus();
}

absl::Status Arguments::ResolveSelectorsPass(
    const GpuInfo& gpu_info,
    const std::map<std::string, std::string>& linkables,
    std::string* code) const {
  std::string result;
  size_t position = 0;
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
        RETURN_IF_ERROR(ResolveSelectorsPass(gpu_info, {}, &arg));
      }
      std::string patch;
      RETURN_IF_ERROR(ResolveSelector(gpu_info, linkables, object_name,
                                      selector_name, function_args,
                                      template_args, &patch));
      code->replace(arg_pos, close_bracket_pos - arg_pos, patch);
      position = arg_pos + patch.size();
    } else {
      position = arg_pos + strlen(kArgsPrefix);
    }
    next_position = code->find(kArgsPrefix, position);
  }
  return absl::OkStatus();
}

absl::Status Arguments::ResolveSelector(
    const GpuInfo& gpu_info,
    const std::map<std::string, std::string>& linkables,
    const std::string& object_name, const std::string& selector,
    const std::vector<std::string>& function_args,
    const std::vector<std::string>& template_args, std::string* result) const {
  GPUObjectDescriptor* desc_ptr;
  RETURN_IF_ERROR(GetDescriptor(object_name, &desc_ptr));
  auto names = desc_ptr->GetGPUResources(gpu_info).GetNames();
  const auto* tensor_desc = dynamic_cast<const TensorDescriptor*>(desc_ptr);
  std::vector<std::string> function_args_new = function_args;
  if (tensor_desc && !linkables.empty() && selector == "Write") {
    auto it = linkables.find(object_name);
    if (it != linkables.end() && !it->second.empty()) {
      if (desc_ptr->GetAccess() != AccessType::WRITE &&
          desc_ptr->GetAccess() != AccessType::READ_WRITE) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Object with name - ", object_name, " should have Write access."));
      }
      std::string value_name, x_coord, y_coord, z_coord, s_coord, b_coord;
      RETURN_IF_ERROR(tensor_desc->GetLinkingContextFromWriteSelector(
          function_args_new, &value_name, &x_coord, &y_coord, &z_coord,
          &s_coord, &b_coord));
      const std::string new_value_name = value_name + "_final";
      const std::string out_var_declaration =
          "\n" + GetTypeDeclaration(gpu_info, tensor_desc->GetDataType(), 4) +
          " " + new_value_name + ";\n";
      *result = "{  // elementwise code with input:" + value_name +
                absl::Substitute(it->second, out_var_declaration) + "\n";
      *result = absl::StrReplaceAll(*result, {{"\n", "\n  "}});
      ReplaceAllWords("in_value", value_name, result);
      ReplaceAllWords("out_value", new_value_name, result);
      ReplaceAllWords("X_COORD", x_coord, result);
      ReplaceAllWords("Y_COORD", y_coord, result);
      ReplaceAllWords("Z_COORD", z_coord, result);
      ReplaceAllWords("S_COORD", s_coord, result);
      ReplaceAllWords("B_COORD", b_coord, result);
      function_args_new[0] = new_value_name;
      RETURN_IF_ERROR(ResolveConstExprPass(gpu_info, result));
      RETURN_IF_ERROR(ResolveSelectorsPass(gpu_info, {}, result));
    }
  }
  std::string patch;
  RETURN_IF_ERROR(desc_ptr->PerformSelector(
      gpu_info, selector, function_args_new, template_args, &patch));
  ResolveObjectNames(object_name, names, &patch);
  if (result->empty()) {
    *result += patch;
  } else {
    // result has elementwise code
    *result += "// write result to tensor\n  " + patch + ";\n}";
  }
  return absl::OkStatus();
}

void Arguments::ResolveObjectNames(const std::string& object_name,
                                   const std::vector<std::string>& member_names,
                                   std::string* code) const {
  for (const auto& member_name : member_names) {
    const std::string new_name = kArgsPrefix + object_name + "_" + member_name;
    ReplaceAllWords(member_name, new_name, code);
  }
}

absl::Status Arguments::AddObjectsScalarArgs(const GpuInfo& gpu_info) {
  for (auto& t : objects_) {
    const auto resources = t.second->GetGPUResources(gpu_info);
    for (const auto& r : resources.ints) {
      AddInt(absl::StrCat(t.first, "_", r));
    }
    for (const auto& r : resources.floats) {
      AddFloat(absl::StrCat(t.first, "_", r));
    }
  }
  for (auto& t : object_refs_) {
    const auto resources = t.second->GetGPUResources(gpu_info);
    for (const auto& r : resources.ints) {
      AddInt(absl::StrCat(t.first, "_", r));
    }
    for (const auto& r : resources.floats) {
      AddFloat(absl::StrCat(t.first, "_", r));
    }
  }
  return absl::OkStatus();
}

void Arguments::ResolveArgsPass(std::string* code) const {
  size_t position = 0;
  size_t next_position = code->find(kArgsPrefix);
  while (next_position != std::string::npos) {
    size_t arg_pos = next_position;
    next_position += strlen(kArgsPrefix);
    std::string object_name = GetNextWord(*code, next_position);
    std::string new_name = object_name;
    code->replace(arg_pos, object_name.size() + strlen(kArgsPrefix), new_name);
    position = arg_pos + new_name.size();
    next_position = code->find(kArgsPrefix, position);
  }
}

absl::Status Arguments::ResolveKernelGlobalSpaceBuffers(const GpuInfo& gpu_info,
                                                        std::string* code) {
  for (auto it = objects_.begin(); it != objects_.end();) {
    const auto* buffer_desc =
        dynamic_cast<const BufferDescriptor*>(it->second.get());
    if (!buffer_desc || buffer_desc->memory_type != MemoryType::CONSTANT) {
      ++it;
      continue;
    }
    bool is_kernel_global_space = false;
    for (const auto& attribute : buffer_desc->attributes) {
      if (attribute == "kernel_global_space") {
        is_kernel_global_space = true;
        break;
      }
    }
    if (!is_kernel_global_space) {
      ++it;
      continue;
    }
    std::string declaration;
    if (!BufferToKernelLanguage(gpu_info, it->first, buffer_desc, &declaration)
             .ok()) {
      ++it;
      continue;
    }
    *code = declaration + *code;
    objects_.erase(it++);
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
