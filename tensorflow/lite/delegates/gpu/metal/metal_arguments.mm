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
#include "tensorflow/lite/delegates/gpu/metal/metal_arguments.h"

#include <string>

#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
namespace metal {
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

void AppendArgument(const std::string& arg, std::string* args) {
  if (!args->empty()) {
    absl::StrAppend(args, ",\n");
  }
  absl::StrAppend(args, arg);
}
}  // namespace

// Static
constexpr char MetalArguments::kArgsPrefix[];

absl::Status MetalArguments::Init(int buffer_offset, Arguments* args, std::string* code) {
  args->GetActiveArguments(*code);
  std::string struct_desc = "struct uniforms_buffer {\n";
  int pos = 0;
  for (auto& fvalue : args->float_values_) {
    auto& new_val = float_values_[fvalue.first];
    new_val.value = fvalue.second.value;
    new_val.active = fvalue.second.active;
    if (fvalue.second.active) {
      new_val.bytes_offset = pos * 4;
      pos++;
      struct_desc += "  float " + fvalue.first + ";\n";
      ReplaceAllWords(kArgsPrefix + fvalue.first, "U." + fvalue.first, code);
    }
  }
  for (auto& ivalue : args->int_values_) {
    auto& new_val = int_values_[ivalue.first];
    new_val.value = ivalue.second.value;
    new_val.active = ivalue.second.active;
    if (ivalue.second.active) {
      new_val.bytes_offset = pos * 4;
      pos++;
      struct_desc += "  int " + ivalue.first + ";\n";
      ReplaceAllWords(kArgsPrefix + ivalue.first, "U." + ivalue.first, code);
    }
  }
  if (pos != 0) {
    int aligned_pos = AlignByN(pos, 4);
    for (int i = pos; i < aligned_pos; i++) {
      struct_desc += "  int dummy" + std::to_string(i - pos) + ";\n";
    }
    struct_desc += "};";
    const_data_.resize(aligned_pos * 4);
    for (auto& it : float_values_) {
      if (it.second.active) {
        float* ptr = reinterpret_cast<float*>(&const_data_[it.second.bytes_offset]);
        *ptr = it.second.value;
      }
    }
    for (auto& it : int_values_) {
      if (it.second.active) {
        int32_t* ptr = reinterpret_cast<int32_t*>(&const_data_[it.second.bytes_offset]);
        *ptr = it.second.value;
      }
    }
  } else {
    struct_desc = "";
  }
  *code = absl::Substitute(*code, struct_desc, GetListOfArgs(buffer_offset));
  return absl::OkStatus();
}

absl::Status MetalArguments::SetInt(const std::string& name, int value) {
  auto it = int_values_.find(name);
  if (it == int_values_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No int argument with name - ", name));
  }
  it->second.value = value;
  if (it->second.active) {
    int32_t* ptr = reinterpret_cast<int32_t*>(&const_data_[it->second.bytes_offset]);
    *ptr = value;
  }
  return absl::OkStatus();
}
absl::Status MetalArguments::SetFloat(const std::string& name, float value) {
  auto it = float_values_.find(name);
  if (it == float_values_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No float argument with name - ", name));
  }
  it->second.value = value;
  if (it->second.active) {
    float* ptr = reinterpret_cast<float*>(&const_data_[it->second.bytes_offset]);
    *ptr = value;
  }
  return absl::OkStatus();
}

void MetalArguments::Encode(id<MTLComputeCommandEncoder> encoder, int buffer_offset) const {
  if (!const_data_.empty()) {
    [encoder setBytes:const_data_.data() length:const_data_.size() atIndex:buffer_offset];
  }
}

std::string MetalArguments::GetListOfArgs(int buffer_offset) {
  std::string result;
  for (auto& t : buffers_) {
    std::string attributes;
    for (const auto& attr : t.second.desc.attributes) {
      attributes += absl::StrCat("  __attribute__((", attr, "))");
    }
    AppendArgument(
        absl::StrCat(
            MemoryTypeToMetalType(t.second.desc.memory_type), " ",
            ToMetalDataType(t.second.desc.data_type, t.second.desc.element_size),
            "* ", t.first, "[[buffer(", buffer_offset, ")]]", attributes),
        &result);
    buffer_offset++;
  }
  if (!const_data_.empty()) {
    AppendArgument(
        absl::StrCat("constant uniforms_buffer& U[[buffer(", buffer_offset, ")]]"),
        &result);
    buffer_offset++;
  }
  if (!result.empty()) {
    result += ",\n";
  }
  return result;
}

absl::Status MetalArguments::SetGPUResources(
    const std::string& name, const GPUResourcesWithValue& resources) {
  for (const auto& r : resources.ints) {
    RETURN_IF_ERROR(SetInt(absl::StrCat(name, "_", r.first), r.second));
  }
  for (const auto& r : resources.floats) {
    RETURN_IF_ERROR(SetFloat(absl::StrCat(name, "_", r.first), r.second));
  }
  for (const auto& r : resources.buffers) {
    RETURN_IF_ERROR(SetBuffer(absl::StrCat(name, "_", r.first), r.second));
  }
  return absl::OkStatus();
}

void MetalArguments::AddBuffer(const std::string& name, const GPUBufferDescriptor& desc) {
  buffers_[name].desc = desc;
}

absl::Status MetalArguments::SetBuffer(const std::string& name, id<MTLBuffer> handle) {
  auto it = buffers_.find(name);
  if (it == buffers_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No buffer argument with name - ", name));
  }
  it->second.handle = handle;
  return absl::OkStatus();
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
