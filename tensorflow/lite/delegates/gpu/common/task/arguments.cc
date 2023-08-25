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
#include "tensorflow/lite/delegates/gpu/common/task/util.h"

namespace tflite {
namespace gpu {
namespace {
bool IsWordSymbol(char symbol) {
  return absl::ascii_isalnum(symbol) || symbol == '_';
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
    const std::string glsl_type = ToGlslShaderDataType(
        buffer_desc->element_type, buffer_desc->element_size,
        /*add_precision*/ false, gpu_info.IsGlslSupportsExplicitFp16());
    const std::string glsl_type_with_precision = ToGlslShaderDataType(
        buffer_desc->element_type, buffer_desc->element_size,
        /*add_precision*/ true, gpu_info.IsGlslSupportsExplicitFp16());
    *result = "const " + glsl_type_with_precision + " " + buffer_name +
              "_buffer[] = " + glsl_type + "[](\n";
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

absl::Status Arguments::Compile(const GpuInfo& gpu_info, std::string* code) {
  RETURN_IF_ERROR(AddObjectsScalarArgs(gpu_info));
  GetActiveArguments(*code);
  RETURN_IF_ERROR(ResolveKernelGlobalSpaceBuffers(gpu_info, code));
  return absl::OkStatus();
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
