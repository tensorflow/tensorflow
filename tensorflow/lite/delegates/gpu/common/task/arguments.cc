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

#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace {
bool IsWordSymbol(char symbol) {
  return absl::ascii_isalnum(symbol) || symbol == '_';
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
    if (!IsWordSymbol(prev) & !IsWordSymbol(next)) {
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

}  // namespace

void Arguments::AddFloat(const std::string& name, float value) {
  float_values_[name].value = value;
}
void Arguments::AddHalf(const std::string& name, half value) {
  half_values_[name].value = value;
}
void Arguments::AddInt(const std::string& name, int value) {
  int_values_[name].value = value;
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
  static constexpr char kArgsPrefix[] = "args.";
  size_t next_position = code->find(kArgsPrefix);
  while (next_position != std::string::npos) {
    size_t arg_pos = next_position + strlen(kArgsPrefix);
    std::string arg_name = GetNextWord(*code, arg_pos);
    code->replace(arg_pos, arg_name.size(), arg_name + postfix);
    next_position = code->find(kArgsPrefix, arg_pos + arg_name.size());
  }
}

absl::Status Arguments::Merge(Arguments&& args, const std::string& postfix) {
  std::vector<std::string> object_names;
  object_names.reserve(args.object_refs_.size() + args.objects_.size());
  for (auto& v : args.object_refs_) {
    object_names.push_back(v.first);
    const std::string name = v.first + postfix;
    if (object_refs_.find(name) != object_refs_.end()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Object reference name collision. Name - ", name));
    }
    object_refs_[name] = {std::move(v.second)};
  }
  for (auto& v : args.objects_) {
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

void Arguments::ReleaseCPURepresentation() {
  for (auto& t : objects_) {
    t.second->Release();
  }
}

void Arguments::GetActiveArguments(const std::string& args_prefix,
                                   const std::string& code) {
  for (auto& float_val : float_values_) {
    float_val.second.active = HasWord(args_prefix + float_val.first, code);
  }
  for (auto& int_val : int_values_) {
    int_val.second.active = HasWord(args_prefix + int_val.first, code);
  }
  for (auto& half_val : half_values_) {
    half_val.second.active = HasWord(args_prefix + half_val.first, code);
  }
}

}  // namespace gpu
}  // namespace tflite
