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
#include <string>
#include <utility>

#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_desc.h"

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
  RETURN_IF_ERROR(ResolveSelectorsPass(gpu_info, linkables, code));
  GetActiveArguments(*code);
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
  if (tensor_desc && (selector == "Write" || selector == "Linking")) {
    auto it = linkables.find(object_name);
    if (it != linkables.end()) {
      if (desc_ptr->GetAccess() != AccessType::WRITE &&
          desc_ptr->GetAccess() != AccessType::READ_WRITE) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Object with name - ", object_name, " should have Write access."));
      }
      std::string value_name, x_coord, y_coord, s_coord;
      RETURN_IF_ERROR(tensor_desc->GetLinkingContextFromWriteSelector(
          function_args, &value_name, &x_coord, &y_coord, &s_coord));
      // x_coord can have batch size property of link_object
      ResolveObjectNames(object_name, names, &x_coord);
      *result = it->second;
      ReplaceAllWords("in_out_value", value_name, result);
      ReplaceAllWords("X_COORD", x_coord, result);
      ReplaceAllWords("Y_COORD", y_coord, result);
      ReplaceAllWords("S_COORD", s_coord, result);
      RETURN_IF_ERROR(ResolveSelectorsPass(gpu_info, {}, result));
      if (selector == "Linking") {
        return absl::OkStatus();
      }
    }
  }
  std::string patch;
  RETURN_IF_ERROR(desc_ptr->PerformSelector(gpu_info, selector, function_args,
                                            template_args, &patch));
  ResolveObjectNames(object_name, names, &patch);
  *result += patch;
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

}  // namespace gpu
}  // namespace tflite
