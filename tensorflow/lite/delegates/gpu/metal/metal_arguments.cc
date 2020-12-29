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
#include "tensorflow/lite/delegates/gpu/metal/buffer.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/common/task/util.h"

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

std::string GetNextWord(const std::string& code, size_t first_position) {
  size_t pos = first_position;
  char t = code[pos];
  while (IsWordSymbol(t)) {
    pos++;
    t = code[pos];
  }
  return code.substr(first_position, pos - first_position);
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

void AppendArgument(const std::string& arg, std::string* args) {
  if (!args->empty()) {
    absl::StrAppend(args, ",\n");
  }
  absl::StrAppend(args, arg);
}

absl::Status CreateMetalObject(id<MTLDevice> device, GPUObjectDescriptor* desc,
                            GPUObjectPtr* result) {
  const auto* buffer_desc = dynamic_cast<const BufferDescriptor*>(desc);
  if (buffer_desc) {
    Buffer gpu_buffer;
    RETURN_IF_ERROR(
        gpu_buffer.CreateFromBufferDescriptor(*buffer_desc, device));
    *result = absl::make_unique<Buffer>(std::move(gpu_buffer));
    return absl::OkStatus();
  }

  return absl::InvalidArgumentError("Unknown GPU descriptor.");
}
}  // namespace

// Static
constexpr char MetalArguments::kArgsPrefix[];

absl::Status MetalArguments::Init(id<MTLDevice> device, int buffer_offset,
                                  Arguments* args, std::string* code) {
  RETURN_IF_ERROR(AllocateObjects(*args, device));
  RETURN_IF_ERROR(AddObjectArgs(args));
  RETURN_IF_ERROR(ResolveSelectorsPass(*args, {}, code));
  RETURN_IF_ERROR(SetObjectsResources(*args));
  object_refs_ = std::move(args->object_refs_);
  args->GetActiveArguments(kArgsPrefix, *code);
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
        float* ptr =
            reinterpret_cast<float*>(&const_data_[it.second.bytes_offset]);
        *ptr = it.second.value;
      }
    }
    for (auto& it : int_values_) {
      if (it.second.active) {
        int32_t* ptr =
            reinterpret_cast<int32_t*>(&const_data_[it.second.bytes_offset]);
        *ptr = it.second.value;
      }
    }
  } else {
    struct_desc = "";
  }
  ResolveArgsPass(code);
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
    int32_t* ptr =
        reinterpret_cast<int32_t*>(&const_data_[it->second.bytes_offset]);
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
    float* ptr =
        reinterpret_cast<float*>(&const_data_[it->second.bytes_offset]);
    *ptr = value;
  }
  return absl::OkStatus();
}

absl::Status MetalArguments::SetHalf(const std::string& name, half value) {
  return absl::UnimplementedError(
      "No support of half uniforms in Metal backend");
}

absl::Status MetalArguments::SetObjectRef(const std::string& name,
                                          const GPUObject& object) {
  auto it = object_refs_.find(name);
  if (it == object_refs_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No object ref with name - ", name));
  }
  GPUResourcesWithValue resources;
  RETURN_IF_ERROR(object.GetGPUResources(it->second.get(), &resources));
  for (const auto& r : resources.ints) {
    RETURN_IF_ERROR(SetInt(absl::StrCat(name, "_", r.first), r.second));
  }
  for (const auto& r : resources.floats) {
    RETURN_IF_ERROR(SetFloat(absl::StrCat(name, "_", r.first), r.second));
  }
  return absl::OkStatus();
  // return SetGPUResources(name, resources);
}

void MetalArguments::Encode(id<MTLComputeCommandEncoder> encoder,
                            int buffer_offset) const {
  for (auto& b : buffers_) {
    [encoder setBuffer:b.second.handle offset:0 atIndex:buffer_offset];
    buffer_offset++;
  }
  if (!const_data_.empty()) {
    [encoder setBytes:const_data_.data()
               length:const_data_.size()
              atIndex:buffer_offset];
  }
}

absl::Status MetalArguments::AllocateObjects(const Arguments& args,
                                          id<MTLDevice> device) {
  objects_.resize(args.objects_.size());
  int i = 0;
  for (auto& t : args.objects_) {
    RETURN_IF_ERROR(CreateMetalObject(device, t.second.get(), &objects_[i]));
    i++;
  }
  return absl::OkStatus();
}

absl::Status MetalArguments::AddObjectArgs(Arguments* args) {
  for (auto& t : args->objects_) {
    AddGPUResources(t.first, t.second->GetGPUResources(), args);
  }
  for (auto& t : args->object_refs_) {
    auto resources = t.second->GetGPUResources();
    for (const auto& r : resources.ints) {
      args->AddInt(absl::StrCat(t.first, "_", r));
    }
    for (const auto& r : resources.floats) {
      args->AddFloat(absl::StrCat(t.first, "_", r));
    }
    // AddGPUResources(t.first, t.second->GetGPUResources(), args);
  }
  return absl::OkStatus();
}

std::string MetalArguments::GetListOfArgs(int buffer_offset) {
  std::string result;
  for (auto& t : buffers_) {
    std::string attributes;
    for (const auto& attr : t.second.desc.attributes) {
      attributes += absl::StrCat("  __attribute__((", attr, "))");
    }
    AppendArgument(
        absl::StrCat(MemoryTypeToMetalType(t.second.desc.memory_type), " ",
                     ToMetalDataType(t.second.desc.data_type,
                                     t.second.desc.element_size),
                     "* ", t.first, "[[buffer(", buffer_offset, ")]]",
                     attributes),
        &result);
    buffer_offset++;
  }
  if (!const_data_.empty()) {
    AppendArgument(absl::StrCat("constant uniforms_buffer& U[[buffer(",
                                buffer_offset, ")]]"),
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

void MetalArguments::AddBuffer(const std::string& name,
                               const GPUBufferDescriptor& desc) {
  buffers_[name].desc = desc;
}

void MetalArguments::AddGPUResources(const std::string& name,
                                  const GPUResources& resources,
                                  Arguments* args) {
  for (const auto& r : resources.ints) {
    args->AddInt(absl::StrCat(name, "_", r));
  }
  for (const auto& r : resources.floats) {
    args->AddFloat(absl::StrCat(name, "_", r));
  }
  for (const auto& r : resources.buffers) {
    AddBuffer(absl::StrCat(name, "_", r.first), r.second);
  }
}

absl::Status MetalArguments::SetBuffer(const std::string& name,
                                       id<MTLBuffer> handle) {
  auto it = buffers_.find(name);
  if (it == buffers_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No buffer argument with name - ", name));
  }
  it->second.handle = handle;
  return absl::OkStatus();
}

absl::Status MetalArguments::ResolveSelectorsPass(
    const Arguments& args, const std::map<std::string, std::string>& linkables,
    std::string* code) {
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
        RETURN_IF_ERROR(ResolveSelectorsPass(args, {}, &arg));
      }
      std::string patch;
      RETURN_IF_ERROR(ResolveSelector(args, linkables, object_name,
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

absl::Status MetalArguments::ResolveSelector(
    const Arguments& args, const std::map<std::string, std::string>& linkables,
    const std::string& object_name, const std::string& selector,
    const std::vector<std::string>& function_args,
    const std::vector<std::string>& template_args, std::string* result) {
  const GPUObjectDescriptor* desc_ptr;
  auto it_ref = args.object_refs_.find(object_name);
  auto it_obj = args.objects_.find(object_name);
  if (it_ref != args.object_refs_.end()) {
    desc_ptr = it_ref->second.get();
  } else if (it_obj != args.objects_.end()) {
    desc_ptr = it_obj->second.get();
  } else {
    return absl::NotFoundError(
        absl::StrCat("No object with name - ", object_name));
  }
  auto names = desc_ptr->GetGPUResources().GetNames();
  std::string patch;
  RETURN_IF_ERROR(desc_ptr->PerformSelector(selector, function_args,
                                            template_args, &patch));
  ResolveObjectNames(object_name, names, &patch);
  *result += patch;
  return absl::OkStatus();
}

void MetalArguments::ResolveObjectNames(
    const std::string& object_name,
    const std::vector<std::string>& member_names, std::string* code) {
  for (const auto& member_name : member_names) {
    const std::string new_name = kArgsPrefix + object_name + "_" + member_name;
    ReplaceAllWords(member_name, new_name, code);
  }
}

void MetalArguments::ResolveArgsPass(std::string* code) {
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

absl::Status MetalArguments::SetObjectsResources(const Arguments& args) {
  int i = 0;
  for (const auto& t : args.objects_) {
    GPUResourcesWithValue resources;
    RETURN_IF_ERROR(objects_[i]->GetGPUResources(t.second.get(), &resources));
    RETURN_IF_ERROR(SetGPUResources(t.first, resources));
    i++;
  }
  return absl::OkStatus();
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
