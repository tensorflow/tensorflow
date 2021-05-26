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

#include "tensorflow/lite/delegates/gpu/cl/cl_arguments.h"

#include <string>

#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/cl/buffer.h"
#include "tensorflow/lite/delegates/gpu/cl/gpu_object.h"
#include "tensorflow/lite/delegates/gpu/cl/linear_storage.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor.h"
#include "tensorflow/lite/delegates/gpu/cl/texture2d.h"
#include "tensorflow/lite/delegates/gpu/common/task/util.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
namespace cl {
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

std::string GetDefaultSamplers(const GpuInfo& gpu_info) {
  std::string result;
  result +=
      "__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | "
      "CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;\n";
  if (gpu_info.IsAdreno() && gpu_info.adreno_info.IsAdreno3xx()) {
    // Unfortunately, CLK_ADDRESS_CLAMP is very slow on Adreno3xx and
    // we can observe huge register overhead when compared to other modes.

    // While using CLK_ADDRESS_NONE with out-of-range image coordinates is
    // undefined in the OpenCL specification, we have observed that
    // CLK_ADDRESS_NONE works like CLK_ADDRESS_CLAMP for out-of-range image
    // coordinates for RGBA F16/F32 textures on Adreno3xx devices. Using
    // CLK_ADDRESS_NONE is significantly faster than CLK_ADDRESS_CLAMP on Adreno
    // 3xx.
    result +=
        "__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | "
        "CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;\n";
  } else {
    result +=
        "__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | "
        "CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n";
  }

  return result;
}

absl::Status CreateCLObject(GPUObjectDescriptor* desc, CLContext* context,
                            GPUObjectPtr* result) {
  const auto* buffer_desc = dynamic_cast<const BufferDescriptor*>(desc);
  if (buffer_desc) {
    Buffer gpu_buffer;
    RETURN_IF_ERROR(
        gpu_buffer.CreateFromBufferDescriptor(*buffer_desc, context));
    *result = absl::make_unique<Buffer>(std::move(gpu_buffer));
    return absl::OkStatus();
  }

  const auto* texture_desc = dynamic_cast<const Texture2DDescriptor*>(desc);
  if (texture_desc) {
    Texture2D gpu_texture;
    RETURN_IF_ERROR(
        gpu_texture.CreateFromTexture2DDescriptor(*texture_desc, context));
    *result = absl::make_unique<Texture2D>(std::move(gpu_texture));
    return absl::OkStatus();
  }

  const auto* linear_desc = dynamic_cast<const TensorLinearDescriptor*>(desc);
  if (linear_desc) {
    LinearStorage gpu_storage;
    RETURN_IF_ERROR(
        gpu_storage.CreateFromTensorLinearDescriptor(*linear_desc, context));
    *result = absl::make_unique<LinearStorage>(std::move(gpu_storage));
    return absl::OkStatus();
  }

  const auto* tensor_desc = dynamic_cast<const TensorDescriptor*>(desc);
  if (tensor_desc) {
    Tensor gpu_tensor;
    RETURN_IF_ERROR(gpu_tensor.CreateFromDescriptor(*tensor_desc, context));
    *result = absl::make_unique<Tensor>(std::move(gpu_tensor));
    return absl::OkStatus();
  }

  return absl::InvalidArgumentError("Unknown GPU descriptor.");
}

}  // namespace

// Static
constexpr char CLArguments::kArgsPrefix[];

absl::Status CLArguments::Init(
    const GpuInfo& gpu_info,
    const std::map<std::string, std::string>& linkables, CLContext* context,
    Arguments* args, std::string* code) {
  RETURN_IF_ERROR(AllocateObjects(*args, context));
  RETURN_IF_ERROR(AddObjectArgs(gpu_info, args));
  RETURN_IF_ERROR(ResolveSelectorsPass(gpu_info, *args, linkables, code));
  object_refs_ = std::move(args->object_refs_);
  args->GetActiveArguments(kArgsPrefix, *code);
  const bool use_f32_for_halfs = gpu_info.IsPowerVR();
  CopyArguments(*args, use_f32_for_halfs);
  RETURN_IF_ERROR(SetObjectsResources(*args));
  RenameArgumentsInCode(code);
  ResolveArgsPass(code);
  *code = absl::Substitute(*code, GetListOfArgs());
  if (gpu_info.SupportsImages()) {
    *code = GetDefaultSamplers(gpu_info) + *code;
  }
  return absl::OkStatus();
}

absl::Status CLArguments::Init(const GpuInfo& gpu_info, Arguments* args,
                               CLContext* context) {
  RETURN_IF_ERROR(AllocateObjects(*args, context));
  RETURN_IF_ERROR(AddObjectArgs(gpu_info, args));
  object_refs_ = std::move(args->object_refs_);
  const bool use_f32_for_halfs = gpu_info.IsPowerVR();
  CopyArguments(*args, use_f32_for_halfs);
  RETURN_IF_ERROR(SetObjectsResources(*args));
  return absl::OkStatus();
}

absl::Status CLArguments::AllocateObjects(const Arguments& args,
                                          CLContext* context) {
  objects_.resize(args.objects_.size());
  int i = 0;
  for (auto& t : args.objects_) {
    RETURN_IF_ERROR(CreateCLObject(t.second.get(), context, &objects_[i]));
    i++;
  }
  return absl::OkStatus();
}

absl::Status CLArguments::AddObjectArgs(const GpuInfo& gpu_info,
                                        Arguments* args) {
  for (auto& t : args->objects_) {
    AddGPUResources(t.first, t.second->GetGPUResources(gpu_info), args);
  }
  for (auto& t : args->object_refs_) {
    AddGPUResources(t.first, t.second->GetGPUResources(gpu_info), args);
  }
  return absl::OkStatus();
}

absl::Status CLArguments::SetObjectsResources(const Arguments& args) {
  int i = 0;
  for (const auto& t : args.objects_) {
    GPUResourcesWithValue resources;
    RETURN_IF_ERROR(objects_[i]->GetGPUResources(t.second.get(), &resources));
    RETURN_IF_ERROR(SetGPUResources(t.first, resources));
    i++;
  }
  return absl::OkStatus();
}

absl::Status CLArguments::ResolveSelectorsPass(
    const GpuInfo& gpu_info, const Arguments& args,
    const std::map<std::string, std::string>& linkables, std::string* code) {
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
        RETURN_IF_ERROR(ResolveSelectorsPass(gpu_info, args, {}, &arg));
      }
      std::string patch;
      RETURN_IF_ERROR(ResolveSelector(gpu_info, args, linkables, object_name,
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

void CLArguments::ResolveObjectNames(
    const std::string& object_name,
    const std::vector<std::string>& member_names, std::string* code) {
  for (const auto& member_name : member_names) {
    const std::string new_name = kArgsPrefix + object_name + "_" + member_name;
    ReplaceAllWords(member_name, new_name, code);
  }
}

absl::Status CLArguments::ResolveSelector(
    const GpuInfo& gpu_info, const Arguments& args,
    const std::map<std::string, std::string>& linkables,
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
      RETURN_IF_ERROR(ResolveSelectorsPass(gpu_info, args, {}, result));
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

void CLArguments::ResolveArgsPass(std::string* code) {
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

void CLArguments::CopyScalarValues(Arguments* args) const {
  for (const auto& fvalue : float_values_) {
    args->float_values_[fvalue.first].value = fvalue.second.value;
  }
  for (const auto& ivalue : int_values_) {
    args->int_values_[ivalue.first].value = ivalue.second.value;
  }
  for (const auto& hfvalue : half_values_) {
    args->half_values_[hfvalue.first].value = hfvalue.second.value;
  }
}

void CLArguments::CopyArguments(const Arguments& args, bool use_f32_for_halfs) {
  for (const auto& fvalue : args.float_values_) {
    auto& new_val = float_values_[fvalue.first];
    new_val.value = fvalue.second.value;
    new_val.active = fvalue.second.active;
    if (fvalue.second.active) {
      new_val.offset = shared_float4s_data_.size();
      shared_float4s_data_.push_back(new_val.value);
    }
  }
  for (const auto& ivalue : args.int_values_) {
    auto& new_val = int_values_[ivalue.first];
    new_val.value = ivalue.second.value;
    new_val.active = ivalue.second.active;
    if (ivalue.second.active) {
      new_val.offset = shared_int4s_data_.size();
      shared_int4s_data_.push_back(new_val.value);
    }
  }
  for (const auto& hfvalue : args.half_values_) {
    auto& new_val = half_values_[hfvalue.first];
    new_val.value = hfvalue.second.value;
    new_val.active = hfvalue.second.active;
    if (hfvalue.second.active) {
      if (use_f32_for_halfs) {
        new_val.store_as_f32 = true;
        new_val.offset = shared_float4s_data_.size();
        shared_float4s_data_.push_back(new_val.value);
      } else {
        new_val.store_as_f32 = false;
        new_val.offset = shared_half4s_data_.size();
        shared_half4s_data_.push_back(new_val.value);
      }
    }
  }
  int shared_int4s_aligned_size = AlignByN(shared_int4s_data_.size(), 4);
  shared_int4s_data_.resize(shared_int4s_aligned_size);
  int shared_float4s_aligned_size = AlignByN(shared_float4s_data_.size(), 4);
  shared_float4s_data_.resize(shared_float4s_aligned_size);
  int shared_half4s_aligned_size = AlignByN(shared_half4s_data_.size(), 4);
  shared_half4s_data_.resize(shared_half4s_aligned_size);
}

void CLArguments::RenameArgumentsInCode(std::string* code) {
  const std::string postfixes[4] = {"x", "y", "z", "w"};
  for (const auto& fvalue : float_values_) {
    if (fvalue.second.active) {
      std::string index = std::to_string(fvalue.second.offset / 4);
      std::string new_name =
          "shared_float4_" + index + "." + postfixes[fvalue.second.offset % 4];
      ReplaceAllWords(kArgsPrefix + fvalue.first, new_name, code);
    }
  }
  for (const auto& ivalue : int_values_) {
    if (ivalue.second.active) {
      std::string index = std::to_string(ivalue.second.offset / 4);
      std::string new_name =
          "shared_int4_" + index + "." + postfixes[ivalue.second.offset % 4];
      ReplaceAllWords(kArgsPrefix + ivalue.first, new_name, code);
    }
  }
  for (const auto& hfvalue : half_values_) {
    if (hfvalue.second.active) {
      std::string index = std::to_string(hfvalue.second.offset / 4);
      std::string new_name;
      if (hfvalue.second.store_as_f32) {
        new_name = "(half)(shared_float4_" + index + "." +
                   postfixes[hfvalue.second.offset % 4] + ")";
      } else {
        new_name = "shared_half4_" + index + "." +
                   postfixes[hfvalue.second.offset % 4];
      }
      ReplaceAllWords(kArgsPrefix + hfvalue.first, new_name, code);
    }
  }
}

void CLArguments::AddBuffer(const std::string& name,
                            const GPUBufferDescriptor& desc) {
  buffers_[name].desc = desc;
}
void CLArguments::AddImage2D(const std::string& name,
                             const GPUImage2DDescriptor& desc) {
  images2d_[name].desc = desc;
}

void CLArguments::AddImage2DArray(const std::string& name,
                                  const GPUImage2DArrayDescriptor& desc) {
  image2d_arrays_[name].desc = desc;
}

void CLArguments::AddImage3D(const std::string& name,
                             const GPUImage3DDescriptor& desc) {
  images3d_[name].desc = desc;
}

void CLArguments::AddImageBuffer(const std::string& name,
                                 const GPUImageBufferDescriptor& desc) {
  image_buffers_[name].desc = desc;
}

void CLArguments::AddCustomMemory(const std::string& name,
                                  const GPUCustomMemoryDescriptor& desc) {
  custom_memories_[name].desc = desc;
}

void CLArguments::AddGPUResources(const std::string& name,
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
  for (const auto& r : resources.custom_memories) {
    AddCustomMemory(absl::StrCat(name, "_", r.first), r.second);
  }
}

absl::Status CLArguments::SetInt(const std::string& name, int value) {
  auto it = int_values_.find(name);
  if (it == int_values_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No int argument with name - ", name));
  }
  it->second.value = value;
  if (it->second.active) {
    shared_int4s_data_[it->second.offset] = value;
  }
  return absl::OkStatus();
}
absl::Status CLArguments::SetFloat(const std::string& name, float value) {
  auto it = float_values_.find(name);
  if (it == float_values_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No float argument with name - ", name));
  }
  it->second.value = value;
  if (it->second.active) {
    shared_float4s_data_[it->second.offset] = value;
  }
  return absl::OkStatus();
}

absl::Status CLArguments::SetHalf(const std::string& name, half value) {
  auto it = half_values_.find(name);
  if (it == half_values_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No half argument with name - ", name));
  }
  it->second.value = value;
  if (it->second.active) {
    if (it->second.store_as_f32) {
      shared_float4s_data_[it->second.offset] = value;
    } else {
      shared_half4s_data_[it->second.offset] = value;
    }
  }
  return absl::OkStatus();
}

absl::Status CLArguments::SetImage2D(const std::string& name, cl_mem memory) {
  auto it = images2d_.find(name);
  if (it == images2d_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No image2D argument with name - ", name));
  }
  it->second.memory = memory;
  return absl::OkStatus();
}

absl::Status CLArguments::SetBuffer(const std::string& name, cl_mem memory) {
  auto it = buffers_.find(name);
  if (it == buffers_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No buffer argument with name - ", name));
  }
  it->second.memory = memory;
  return absl::OkStatus();
}

absl::Status CLArguments::SetImage2DArray(const std::string& name,
                                          cl_mem memory) {
  auto it = image2d_arrays_.find(name);
  if (it == image2d_arrays_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No image2D array argument with name - ", name));
  }
  it->second.memory = memory;
  return absl::OkStatus();
}

absl::Status CLArguments::SetImage3D(const std::string& name, cl_mem memory) {
  auto it = images3d_.find(name);
  if (it == images3d_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No image3D argument with name - ", name));
  }
  it->second.memory = memory;
  return absl::OkStatus();
}

absl::Status CLArguments::SetImageBuffer(const std::string& name,
                                         cl_mem memory) {
  auto it = image_buffers_.find(name);
  if (it == image_buffers_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No image buffer argument with name - ", name));
  }
  it->second.memory = memory;
  return absl::OkStatus();
}

absl::Status CLArguments::SetCustomMemory(const std::string& name,
                                          cl_mem memory) {
  auto it = custom_memories_.find(name);
  if (it == custom_memories_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No custom memory argument with name - ", name));
  }
  it->second.memory = memory;
  return absl::OkStatus();
}

absl::Status CLArguments::SetObjectRef(const std::string& name,
                                       const GPUObject* object) {
  auto it = object_refs_.find(name);
  if (it == object_refs_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No object ref with name - ", name));
  }
  GPUResourcesWithValue resources;
  RETURN_IF_ERROR(object->GetGPUResources(it->second.get(), &resources));
  return SetGPUResources(name, resources);
}

absl::Status CLArguments::SetGPUResources(
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
  for (const auto& r : resources.custom_memories) {
    RETURN_IF_ERROR(
        SetCustomMemory(absl::StrCat(name, "_", r.first), r.second));
  }
  return absl::OkStatus();
}

std::string CLArguments::GetListOfArgs() {
  std::string result;
  for (auto& t : buffers_) {
    const std::string type_name =
        t.second.desc.data_type == DataType::FLOAT32 ? "float" : "half";
    std::string attributes;
    for (const auto& attr : t.second.desc.attributes) {
      attributes += absl::StrCat("  __attribute__((", attr, "))");
    }
    AppendArgument(
        absl::StrCat(
            MemoryTypeToCLType(t.second.desc.memory_type), " ",
            ToCLDataType(t.second.desc.data_type, t.second.desc.element_size),
            "* ", t.first, attributes),
        &result);
  }
  for (auto& t : image_buffers_) {
    AppendArgument(absl::StrCat(GetImageModifier(t.second.desc.access_type),
                                " image1d_buffer_t ", t.first),
                   &result);
  }
  for (auto& t : images2d_) {
    AppendArgument(absl::StrCat(GetImageModifier(t.second.desc.access_type),
                                " image2d_t ", t.first),
                   &result);
  }
  for (auto& t : image2d_arrays_) {
    AppendArgument(absl::StrCat(GetImageModifier(t.second.desc.access_type),
                                " image2d_array_t ", t.first),
                   &result);
  }
  for (auto& t : images3d_) {
    AppendArgument(absl::StrCat(GetImageModifier(t.second.desc.access_type),
                                " image3d_t ", t.first),
                   &result);
  }
  for (auto& t : custom_memories_) {
    AppendArgument(absl::StrCat(t.second.desc.type_name, " ", t.first),
                   &result);
  }
  for (int i = 0; i < shared_int4s_data_.size() / 4; ++i) {
    AppendArgument(absl::StrCat("int4 shared_int4_", i), &result);
  }
  for (int i = 0; i < shared_float4s_data_.size() / 4; ++i) {
    AppendArgument(absl::StrCat("float4 shared_float4_", i), &result);
  }
  for (int i = 0; i < shared_half4s_data_.size() / 4; ++i) {
    AppendArgument(absl::StrCat("half4 shared_half4_", i), &result);
  }
  return result;
}

absl::Status CLArguments::Bind(cl_kernel kernel, int offset) {
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
  for (auto& t : custom_memories_) {
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
  for (int i = 0; i < shared_half4s_data_.size() / 4; ++i) {
    const int error_code = clSetKernelArg(kernel, offset, sizeof(int16_t) * 4,
                                          &shared_half4s_data_[i * 4]);
    if (error_code != CL_SUCCESS) {
      return absl::UnknownError(absl::StrCat(
          "Failed to set kernel arguments - ", CLErrorCodeToString(error_code),
          "(at index - ", offset, ")"));
    }
    offset++;
  }
  return absl::OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
