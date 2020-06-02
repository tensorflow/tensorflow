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

#include "tensorflow/lite/delegates/gpu/cl/arguments.h"

#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/str_split.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {
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

size_t FindEnclosingBracket(const std::string& text, size_t first_pos,
                            char bracket) {
  const std::map<char, char> brackets = {
      {'(', ')'},
      {'{', '}'},
      {'[', ']'},
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
    absl::StrAppend(args, ",\n  ");
  }
  absl::StrAppend(args, arg);
}

std::string GetImageModifier(AccessType access) {
  switch (access) {
    case AccessType::READ:
      return "__read_only";
    case AccessType::WRITE:
      return "__write_only";
    case AccessType::READ_WRITE:
      return "__read_write";
  }
}

}  // namespace

Arguments::Arguments(Arguments&& args)
    : int_values_(std::move(args.int_values_)),
      shared_int4s_data_(std::move(args.shared_int4s_data_)),
      float_values_(std::move(args.float_values_)),
      shared_float4s_data_(std::move(args.shared_float4s_data_)),
      buffers_(std::move(args.buffers_)),
      images2d_(std::move(args.images2d_)),
      image2d_arrays_(std::move(args.image2d_arrays_)),
      images3d_(std::move(args.images3d_)),
      image_buffers_(std::move(args.image_buffers_)),
      object_refs_(std::move(args.object_refs_)),
      objects_(std::move(args.objects_)) {}
Arguments& Arguments::operator=(Arguments&& args) {
  if (this != &args) {
    int_values_ = std::move(args.int_values_);
    shared_int4s_data_ = std::move(args.shared_int4s_data_);
    float_values_ = std::move(args.float_values_);
    shared_float4s_data_ = std::move(args.shared_float4s_data_);
    buffers_ = std::move(args.buffers_);
    images2d_ = std::move(args.images2d_);
    image2d_arrays_ = std::move(args.image2d_arrays_);
    images3d_ = std::move(args.images3d_);
    image_buffers_ = std::move(args.image_buffers_);
    object_refs_ = std::move(args.object_refs_);
    objects_ = std::move(args.objects_);
  }
  return *this;
}

void Arguments::AddFloat(const std::string& name, float value) {
  float_values_[name].value = value;
}
void Arguments::AddInt(const std::string& name, int value) {
  int_values_[name].value = value;
}
void Arguments::AddBuffer(const std::string& name,
                          const GPUBufferDescriptor& desc) {
  buffers_[name] = desc;
}
void Arguments::AddImage2D(const std::string& name,
                           const GPUImage2DDescriptor& desc) {
  images2d_[name] = desc;
}

void Arguments::AddImage2DArray(const std::string& name,
                                const GPUImage2DArrayDescriptor& desc) {
  image2d_arrays_[name] = desc;
}

void Arguments::AddImage3D(const std::string& name,
                           const GPUImage3DDescriptor& desc) {
  images3d_[name] = desc;
}

void Arguments::AddImageBuffer(const std::string& name,
                               const GPUImageBufferDescriptor& desc) {
  image_buffers_[name] = desc;
}

void Arguments::AddObjectRef(const std::string& name, AccessType access_type,
                             GPUObjectDescriptorPtr&& descriptor_ptr) {
  object_refs_[name] = {access_type, std::move(descriptor_ptr)};
}

void Arguments::AddObject(const std::string& name, AccessType access_type,
                          GPUObjectPtr&& object) {
  objects_[name] = {access_type, std::move(object)};
}

void Arguments::AddGPUResources(const std::string& name,
                                const GPUResources& resources) {
  for (const auto& r : resources.ints) {
    AddInt(absl::StrCat(name, "_", r));
  }
  for (const auto& r : resources.floats) {
    AddFloat(absl::StrCat(name, "_", r));
  }
  for (const auto& r : resources.buffers) {
    AddBuffer(absl::StrCat(name, "_", r.first), r.second);
  }
  for (const auto& r : resources.images2d) {
    AddImage2D(absl::StrCat(name, "_", r.first), r.second);
  }
  for (const auto& r : resources.image2d_arrays) {
    AddImage2DArray(absl::StrCat(name, "_", r.first), r.second);
  }
  for (const auto& r : resources.images3d) {
    AddImage3D(absl::StrCat(name, "_", r.first), r.second);
  }
  for (const auto& r : resources.image_buffers) {
    AddImageBuffer(absl::StrCat(name, "_", r.first), r.second);
  }
}

absl::Status Arguments::SetInt(const std::string& name, int value) {
  auto ii = int_values_.find(name);
  if (ii == int_values_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No int argument with name - ", name));
  }
  ii->second.value = value;
  if (ii->second.active) {
    shared_int4s_data_[ii->second.offset] = value;
  }
  return absl::OkStatus();
}

absl::Status Arguments::SetFloat(const std::string& name, float value) {
  auto fi = float_values_.find(name);
  if (fi == float_values_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No float argument with name - ", name));
  }
  fi->second.value = value;
  if (fi->second.active) {
    shared_float4s_data_[fi->second.offset] = value;
  }
  return absl::OkStatus();
}

absl::Status Arguments::SetImage2D(const std::string& name, cl_mem memory) {
  auto it = images2d_.find(name);
  if (it == images2d_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No image2D argument with name - ", name));
  }
  it->second.memory = memory;
  return absl::OkStatus();
}

absl::Status Arguments::SetBuffer(const std::string& name, cl_mem memory) {
  auto it = buffers_.find(name);
  if (it == buffers_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No buffer argument with name - ", name));
  }
  it->second.memory = memory;
  return absl::OkStatus();
}

absl::Status Arguments::SetImage2DArray(const std::string& name,
                                        cl_mem memory) {
  auto it = image2d_arrays_.find(name);
  if (it == image2d_arrays_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No image2D array argument with name - ", name));
  }
  it->second.memory = memory;
  return absl::OkStatus();
}

absl::Status Arguments::SetImage3D(const std::string& name, cl_mem memory) {
  auto it = images3d_.find(name);
  if (it == images3d_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No image3D argument with name - ", name));
  }
  it->second.memory = memory;
  return absl::OkStatus();
}

absl::Status Arguments::SetImageBuffer(const std::string& name, cl_mem memory) {
  auto it = image_buffers_.find(name);
  if (it == image_buffers_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No image buffer argument with name - ", name));
  }
  it->second.memory = memory;
  return absl::OkStatus();
}

absl::Status Arguments::SetObjectRef(const std::string& name,
                                     const GPUObject* object) {
  auto it = object_refs_.find(name);
  if (it == object_refs_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No object ref with name - ", name));
  }
  return SetGPUResources(name, object->GetGPUResources(it->second.access_type));
}

absl::Status Arguments::SetGPUResources(
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
  for (const auto& r : resources.images2d) {
    RETURN_IF_ERROR(SetImage2D(absl::StrCat(name, "_", r.first), r.second));
  }
  for (const auto& r : resources.image2d_arrays) {
    RETURN_IF_ERROR(
        SetImage2DArray(absl::StrCat(name, "_", r.first), r.second));
  }
  for (const auto& r : resources.images3d) {
    RETURN_IF_ERROR(SetImage3D(absl::StrCat(name, "_", r.first), r.second));
  }
  for (const auto& r : resources.image_buffers) {
    RETURN_IF_ERROR(SetImageBuffer(absl::StrCat(name, "_", r.first), r.second));
  }
  return absl::OkStatus();
}

absl::Status Arguments::TransformToCLCode(std::string* code) {
  RETURN_IF_ERROR(AddObjectArgs());
  RETURN_IF_ERROR(ResolveSelectorsPass(code));
  ResolveArgsPass(code);
  return absl::OkStatus();
}

std::string Arguments::GetListOfArgs() {
  std::string result;
  for (auto& t : buffers_) {
    const std::string type_name =
        t.second.data_type == DataType::FLOAT32 ? "float" : "half";
    AppendArgument(absl::StrCat("__global ", type_name, t.second.element_size,
                                "* ", t.first),
                   &result);
  }
  for (auto& t : image_buffers_) {
    AppendArgument(absl::StrCat(GetImageModifier(t.second.access_type),
                                " image1d_buffer_t ", t.first),
                   &result);
  }
  for (auto& t : images2d_) {
    AppendArgument(absl::StrCat(GetImageModifier(t.second.access_type),
                                " image2d_t ", t.first),
                   &result);
  }
  for (auto& t : image2d_arrays_) {
    AppendArgument(absl::StrCat(GetImageModifier(t.second.access_type),
                                " image2d_array_t ", t.first),
                   &result);
  }
  for (auto& t : images3d_) {
    AppendArgument(absl::StrCat(GetImageModifier(t.second.access_type),
                                " image3d_t ", t.first),
                   &result);
  }
  for (int i = 0; i < shared_int4s_data_.size() / 4; ++i) {
    AppendArgument(absl::StrCat("int4 shared_int4_", i), &result);
  }
  for (int i = 0; i < shared_float4s_data_.size() / 4; ++i) {
    AppendArgument(absl::StrCat("float4 shared_float4_", i), &result);
  }
  return result;
}

absl::Status Arguments::Bind(cl_kernel kernel, int offset) {
  for (auto& t : buffers_) {
    const int error_code =
        clSetKernelArg(kernel, offset, sizeof(cl_mem), &t.second.memory);
    if (error_code != CL_SUCCESS) {
      return absl::UnknownError(absl::StrCat(
          "Failed to set kernel arguments - ", CLErrorCodeToString(error_code),
          "(at index - ", offset, ")"));
    }
    offset++;
  }
  for (auto& t : image_buffers_) {
    const int error_code =
        clSetKernelArg(kernel, offset, sizeof(cl_mem), &t.second.memory);
    if (error_code != CL_SUCCESS) {
      return absl::UnknownError(absl::StrCat(
          "Failed to set kernel arguments - ", CLErrorCodeToString(error_code),
          "(at index - ", offset, ")"));
    }
    offset++;
  }
  for (auto& t : images2d_) {
    const int error_code =
        clSetKernelArg(kernel, offset, sizeof(cl_mem), &t.second.memory);
    if (error_code != CL_SUCCESS) {
      return absl::UnknownError(absl::StrCat(
          "Failed to set kernel arguments - ", CLErrorCodeToString(error_code),
          "(at index - ", offset, ")"));
    }
    offset++;
  }
  for (auto& t : image2d_arrays_) {
    const int error_code =
        clSetKernelArg(kernel, offset, sizeof(cl_mem), &t.second.memory);
    if (error_code != CL_SUCCESS) {
      return absl::UnknownError(absl::StrCat(
          "Failed to set kernel arguments - ", CLErrorCodeToString(error_code),
          "(at index - ", offset, ")"));
    }
    offset++;
  }
  for (auto& t : images3d_) {
    const int error_code =
        clSetKernelArg(kernel, offset, sizeof(cl_mem), &t.second.memory);
    if (error_code != CL_SUCCESS) {
      return absl::UnknownError(absl::StrCat(
          "Failed to set kernel arguments - ", CLErrorCodeToString(error_code),
          "(at index - ", offset, ")"));
    }
    offset++;
  }
  for (int i = 0; i < shared_int4s_data_.size() / 4; ++i) {
    const int error_code = clSetKernelArg(kernel, offset, sizeof(int32_t) * 4,
                                          &shared_int4s_data_[i * 4]);
    if (error_code != CL_SUCCESS) {
      return absl::UnknownError(absl::StrCat(
          "Failed to set kernel arguments - ", CLErrorCodeToString(error_code),
          "(at index - ", offset, ")"));
    }
    offset++;
  }
  for (int i = 0; i < shared_float4s_data_.size() / 4; ++i) {
    const int error_code = clSetKernelArg(kernel, offset, sizeof(int32_t) * 4,
                                          &shared_float4s_data_[i * 4]);
    if (error_code != CL_SUCCESS) {
      return absl::UnknownError(absl::StrCat(
          "Failed to set kernel arguments - ", CLErrorCodeToString(error_code),
          "(at index - ", offset, ")"));
    }
    offset++;
  }
  return absl::OkStatus();
}

std::string Arguments::AddActiveArgument(const std::string& arg_name) {
  if (auto it = int_values_.find(arg_name); it != int_values_.end()) {
    int int_index;
    if (it->second.active) {
      int_index = it->second.offset;
    } else {
      it->second.active = true;
      it->second.offset = shared_int4s_data_.size();
      int_index = it->second.offset;
      shared_int4s_data_.push_back(it->second.value);
    }
    std::string index = std::to_string(int_index / 4);
    std::string postfixes[4] = {"x", "y", "z", "w"};
    return "shared_int4_" + index + "." + postfixes[int_index % 4];
  }
  if (auto it = float_values_.find(arg_name); it != float_values_.end()) {
    int float_index;
    if (it->second.active) {
      float_index = it->second.offset;
    } else {
      it->second.active = true;
      it->second.offset = shared_float4s_data_.size();
      float_index = it->second.offset;
      shared_float4s_data_.push_back(it->second.value);
    }
    std::string index = std::to_string(float_index / 4);
    std::string postfixes[4] = {"x", "y", "z", "w"};
    return "shared_float4_" + index + "." + postfixes[float_index % 4];
  }
  return arg_name;
}

void Arguments::ResolveArgsPass(std::string* code) {
  std::string result;
  size_t position = 0;
  size_t next_position = code->find(kArgsPrefix);
  while (next_position != std::string::npos) {
    size_t arg_pos = next_position;
    next_position += strlen(kArgsPrefix);
    std::string object_name = GetNextWord(*code, next_position);
    std::string new_name = AddActiveArgument(object_name);
    code->replace(arg_pos, object_name.size() + strlen(kArgsPrefix), new_name);
    position = arg_pos + new_name.size();
    next_position = code->find(kArgsPrefix, position);
  }

  int shared_int4s_aligned_size = AlignByN(shared_int4s_data_.size(), 4);
  shared_int4s_data_.resize(shared_int4s_aligned_size);
  int shared_float4s_aligned_size = AlignByN(shared_float4s_data_.size(), 4);
  shared_float4s_data_.resize(shared_float4s_aligned_size);
}

void Arguments::ResolveObjectNames(const std::string& object_name,
                                   const std::vector<std::string>& member_names,
                                   std::string* code) {
  for (const auto& member_name : member_names) {
    const std::string new_name = kArgsPrefix + object_name + "_" + member_name;
    ReplaceAllWords(member_name, new_name, code);
  }
}

absl::Status Arguments::ResolveSelector(const std::string& object_name,
                                        const std::string& selector,
                                        const std::vector<std::string>& args,
                                        std::string* result) {
  const GPUObjectDescriptor* desc_ptr;
  AccessType access_type;
  if (auto it = object_refs_.find(object_name); it != object_refs_.end()) {
    desc_ptr = it->second.descriptor.get();
    access_type = it->second.access_type;
  } else if (auto it = objects_.find(object_name); it != objects_.end()) {
    desc_ptr = it->second.obj_ptr->GetGPUDescriptor();
    access_type = it->second.access_type;
  } else {
    return absl::NotFoundError(
        absl::StrCat("No object with name - ", object_name));
  }
  RETURN_IF_ERROR(desc_ptr->PerformSelector(selector, args, result));
  auto names = desc_ptr->GetGPUResources(access_type).GetNames();
  ResolveObjectNames(object_name, names, result);
  return absl::OkStatus();
}

absl::Status Arguments::ResolveSelectorsPass(std::string* code) {
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
      if (next != '(') {
        return absl::NotFoundError(
            absl::StrCat("Expected ( after function ", selector_name, " call"));
      }
      next_position += 1;
      size_t bracket_pos = FindEnclosingBracket(*code, next_position, '(');
      if (bracket_pos == -1) {
        return absl::NotFoundError(
            absl::StrCat("Not found enclosing bracket for function ",
                         selector_name, " call"));
      }
      std::string str_args =
          code->substr(next_position, bracket_pos - next_position - 1);
      std::vector<absl::string_view> words = absl::StrSplit(str_args, ',');
      std::vector<std::string> args;
      args.reserve(words.size());
      for (const auto& word : words) {
        absl::string_view arg = absl::StripAsciiWhitespace(word);
        if (!arg.empty()) {
          args.push_back(std::string(arg));
        }
      }
      std::string patch;
      RETURN_IF_ERROR(
          ResolveSelector(object_name, selector_name, args, &patch));
      code->replace(arg_pos, bracket_pos - arg_pos, patch);
      position = arg_pos + patch.size();
    } else {
      position = arg_pos + strlen(kArgsPrefix);
    }
    next_position = code->find(kArgsPrefix, position);
  }
  return absl::OkStatus();
}

absl::Status Arguments::AddObjectArgs() {
  for (auto& t : objects_) {
    AddGPUResources(t.first,
                    t.second.obj_ptr->GetGPUDescriptor()->GetGPUResources(
                        t.second.access_type));
    RETURN_IF_ERROR(SetGPUResources(
        t.first, t.second.obj_ptr->GetGPUResources(t.second.access_type)));
  }
  for (auto& t : object_refs_) {
    AddGPUResources(t.first,
                    t.second.descriptor->GetGPUResources(t.second.access_type));
  }
  return absl::OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
