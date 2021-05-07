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
#include "tensorflow/lite/delegates/gpu/common/task/util.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/buffer.h"
#include "tensorflow/lite/delegates/gpu/metal/linear_storage.h"
#include "tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.h"
#include "tensorflow/lite/delegates/gpu/metal/texture2d.h"

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

  const auto* texture_desc = dynamic_cast<const Texture2DDescriptor*>(desc);
  if (texture_desc) {
    Texture2D gpu_texture;
    RETURN_IF_ERROR(
        gpu_texture.CreateFromTexture2DDescriptor(*texture_desc, device));
    *result = absl::make_unique<Texture2D>(std::move(gpu_texture));
    return absl::OkStatus();
  }

  const auto* linear_desc = dynamic_cast<const TensorLinearDescriptor*>(desc);
  if (linear_desc) {
    LinearStorage gpu_storage;
    RETURN_IF_ERROR(
        gpu_storage.CreateFromTensorLinearDescriptor(*linear_desc, device));
    *result = absl::make_unique<LinearStorage>(std::move(gpu_storage));
    return absl::OkStatus();
  }

  const auto* tensor_desc = dynamic_cast<const TensorDescriptor*>(desc);
  if (tensor_desc) {
    MetalSpatialTensor gpu_tensor;
    RETURN_IF_ERROR(gpu_tensor.CreateFromDescriptor(*tensor_desc, device));
    *result = absl::make_unique<MetalSpatialTensor>(std::move(gpu_tensor));
    return absl::OkStatus();
  }

  return absl::InvalidArgumentError("Unknown GPU descriptor.");
}

std::string AccessToMetalTextureAccess(AccessType access_type) {
  if (access_type == AccessType::READ) {
    return "access::read";
  } else if (access_type == AccessType::READ_WRITE) {
    return "access::read_write";
  } else if (access_type == AccessType::WRITE) {
    return "access::write";
  } else {
    return "access::unknown";
  }
}
}  // namespace

// Static
constexpr char MetalArguments::kArgsPrefix[];

absl::Status MetalArguments::Init(
    const std::map<std::string, std::string>& linkables, MetalDevice* device,
    Arguments* args, std::string* code) {
  RETURN_IF_ERROR(AllocateObjects(*args, device->device()));
  RETURN_IF_ERROR(AddObjectArgs(args));
  RETURN_IF_ERROR(
      ResolveSelectorsPass(device->GetInfo(), *args, linkables, code));
  object_refs_ = std::move(args->object_refs_);
  args->GetActiveArguments(kArgsPrefix, *code);
  std::string struct_desc = ScalarArgumentsToStructWithVec4Fields(args, code);
  RETURN_IF_ERROR(SetObjectsResources(*args));
  ResolveArgsPass(code);
  std::string header = R"(
#include <metal_stdlib>
using namespace metal;

)";
  header += struct_desc + "\n";
  *code = header + *code;
  std::string arguments = GetListOfArgs(/*buffer_offset*/ 0);
  const bool use_global_id = code->find("GLOBAL_ID_") != std::string::npos;
  const bool use_local_id = code->find("LOCAL_ID_") != std::string::npos;
  const bool use_group_id = code->find("GROUP_ID_") != std::string::npos;
  const bool use_group_size = code->find("GROUP_SIZE_") != std::string::npos;
  const bool use_simd_id =
      code->find("SUB_GROUP_LOCAL_ID") != std::string::npos;
  if (use_global_id) {
    AppendArgument("uint3 reserved_gid[[thread_position_in_grid]]", &arguments);
  }
  if (use_local_id) {
    AppendArgument("uint3 reserved_lid[[thread_position_in_threadgroup]]",
                   &arguments);
  }
  if (use_group_id) {
    AppendArgument("uint3 reserved_group_id[[threadgroup_position_in_grid]]",
                   &arguments);
  }
  if (use_group_size) {
    AppendArgument("uint3 reserved_group_size[[threads_per_threadgroup]]",
                   &arguments);
  }
  if (use_simd_id) {
    AppendArgument("uint reserved_simd_id[[thread_index_in_simdgroup]]",
                   &arguments);
  }
  if (!use_global_id && !use_local_id && !use_group_id && !use_group_size &&
      !arguments.empty()) {
    arguments += ",\n";
  }
  *code = absl::Substitute(*code, arguments);
  return absl::OkStatus();
}

std::string MetalArguments::ScalarArgumentsToStructWithScalarFields(
    Arguments* args, std::string* code) {
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
  for (const auto& hfvalue : args->half_values_) {
    auto& new_val = float_values_[hfvalue.first];
    new_val.value = hfvalue.second.value;
    new_val.active = hfvalue.second.active;
    if (hfvalue.second.active) {
      new_val.bytes_offset = pos * 4;
      pos++;
      struct_desc += "  float " + hfvalue.first + ";\n";
      ReplaceAllWords(kArgsPrefix + hfvalue.first,
                      "static_cast<half>(U." + hfvalue.first + ")", code);
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
  return struct_desc;
}

std::string MetalArguments::ScalarArgumentsToStructWithVec4Fields(
    Arguments* args, std::string* code) {
  std::string struct_desc = "struct uniforms_buffer {\n";
  int pos = 0;
  std::string channels[4] = {".x", ".y", ".z", ".w"};
  for (auto& fvalue : args->float_values_) {
    auto& new_val = float_values_[fvalue.first];
    new_val.value = fvalue.second.value;
    new_val.active = fvalue.second.active;
    if (fvalue.second.active) {
      new_val.bytes_offset = pos * 4;
      if (pos % 4 == 0) {
        struct_desc += "  float4 cmp_float4_" + std::to_string(pos / 4) + ";\n";
      }
      std::string new_name =
          "U.cmp_float4_" + std::to_string(pos / 4) + channels[pos % 4];
      ReplaceAllWords(kArgsPrefix + fvalue.first, new_name, code);
      pos++;
    }
  }
  for (const auto& hfvalue : args->half_values_) {
    auto& new_val = float_values_[hfvalue.first];
    new_val.value = hfvalue.second.value;
    new_val.active = hfvalue.second.active;
    if (hfvalue.second.active) {
      new_val.bytes_offset = pos * 4;
      if (pos % 4 == 0) {
        struct_desc += "  float4 cmp_float4_" + std::to_string(pos / 4) + ";\n";
      }
      std::string new_name = "static_cast<half>(U.cmp_float4_" +
                             std::to_string(pos / 4) + channels[pos % 4] + ")";
      ReplaceAllWords(kArgsPrefix + hfvalue.first, new_name, code);
      pos++;
    }
  }
  pos = AlignByN(pos, 4);
  for (auto& ivalue : args->int_values_) {
    auto& new_val = int_values_[ivalue.first];
    new_val.value = ivalue.second.value;
    new_val.active = ivalue.second.active;
    if (ivalue.second.active) {
      new_val.bytes_offset = pos * 4;
      if (pos % 4 == 0) {
        struct_desc += "  int4 cmp_int4_" + std::to_string(pos / 4) + ";\n";
      }
      std::string new_name =
          "U.cmp_int4_" + std::to_string(pos / 4) + channels[pos % 4];
      ReplaceAllWords(kArgsPrefix + ivalue.first, new_name, code);
      pos++;
    }
  }
  if (pos != 0) {
    int aligned_pos = AlignByN(pos, 4);
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
  return struct_desc;
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
  auto it = float_values_.find(name);
  if (it == float_values_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No half argument with name - ", name));
  }
  it->second.value = value;
  if (it->second.active) {
    float* ptr =
        reinterpret_cast<float*>(&const_data_[it->second.bytes_offset]);
    *ptr = value;
  }
  return absl::OkStatus();
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
  return SetGPUResources(name, resources);
}

void MetalArguments::Encode(id<MTLComputeCommandEncoder> encoder,
                            int buffer_offset, int texture_offset) const {
  for (auto& b : buffers_) {
    [encoder setBuffer:b.second.handle offset:0 atIndex:buffer_offset];
    buffer_offset++;
  }
  for (auto& image : images2d_) {
    [encoder setTexture:image.second.handle atIndex:texture_offset];
    texture_offset++;
  }
  for (auto& image : image2d_arrays_) {
    [encoder setTexture:image.second.handle atIndex:texture_offset];
    texture_offset++;
  }
  for (auto& image : images3d_) {
    [encoder setTexture:image.second.handle atIndex:texture_offset];
    texture_offset++;
  }
  for (auto& image : image_buffers_) {
    [encoder setTexture:image.second.handle atIndex:texture_offset];
    texture_offset++;
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
    AddGPUResources(t.first, t.second->GetGPUResources(), args);
  }
  return absl::OkStatus();
}

std::string MetalArguments::GetListOfArgs(int buffer_offset,
                                          int textures_offset) {
  std::string result;
  for (auto& t : buffers_) {
    AppendArgument(
        absl::StrCat(MemoryTypeToMetalType(t.second.desc.memory_type), " ",
                     ToMetalDataType(t.second.desc.data_type,
                                     t.second.desc.element_size),
                     "* ", t.first, "[[buffer(", buffer_offset, ")]]"),
        &result);
    buffer_offset++;
  }
  for (auto& t : images2d_) {
    std::string access = AccessToMetalTextureAccess(t.second.desc.access_type);
    std::string data_type = ToMetalDataType(t.second.desc.data_type);
    if (t.second.desc.normalized) {
      data_type = ToMetalDataType(t.second.desc.normalized_type);
    }
    AppendArgument(absl::StrCat("texture2d<", data_type, ", ", access, "> ",
                                t.first, "[[texture(", textures_offset, ")]]"),
                   &result);
    textures_offset++;
  }
  for (auto& t : image2d_arrays_) {
    std::string access = AccessToMetalTextureAccess(t.second.desc.access_type);
    std::string data_type = ToMetalDataType(t.second.desc.data_type);
    AppendArgument(
        absl::StrCat("texture2d_array<", data_type, ", ", access, "> ", t.first,
                     "[[texture(", textures_offset, ")]]"),
        &result);
    textures_offset++;
  }
  for (auto& t : images3d_) {
    std::string access = AccessToMetalTextureAccess(t.second.desc.access_type);
    std::string data_type = ToMetalDataType(t.second.desc.data_type);
    AppendArgument(absl::StrCat("texture3d<", data_type, ", ", access, "> ",
                                t.first, "[[texture(", textures_offset, ")]]"),
                   &result);
    textures_offset++;
  }
  for (auto& t : image_buffers_) {
    std::string access = AccessToMetalTextureAccess(t.second.desc.access_type);
    std::string data_type = ToMetalDataType(t.second.desc.data_type);
    AppendArgument(
        absl::StrCat("texture_buffer<", data_type, ", ", access, "> ", t.first,
                     "[[texture(", textures_offset, ")]]"),
        &result);
    textures_offset++;
  }
  if (!const_data_.empty()) {
    AppendArgument(absl::StrCat("constant uniforms_buffer& U[[buffer(",
                                buffer_offset, ")]]"),
                   &result);
    buffer_offset++;
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

void MetalArguments::AddBuffer(const std::string& name,
                               const GPUBufferDescriptor& desc) {
  buffers_[name].desc = desc;
}

void MetalArguments::AddImage2D(const std::string& name,
                                const GPUImage2DDescriptor& desc) {
  images2d_[name].desc = desc;
}

void MetalArguments::AddImage2DArray(const std::string& name,
                                     const GPUImage2DArrayDescriptor& desc) {
  image2d_arrays_[name].desc = desc;
}

void MetalArguments::AddImage3D(const std::string& name,
                                const GPUImage3DDescriptor& desc) {
  images3d_[name].desc = desc;
}

void MetalArguments::AddImageBuffer(const std::string& name,
                                    const GPUImageBufferDescriptor& desc) {
  image_buffers_[name].desc = desc;
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

absl::Status MetalArguments::SetImage2D(const std::string& name,
                                        id<MTLTexture> handle) {
  auto it = images2d_.find(name);
  if (it == images2d_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No image2d argument with name - ", name));
  }
  it->second.handle = handle;
  return absl::OkStatus();
}

absl::Status MetalArguments::SetImage2DArray(const std::string& name,
                                             id<MTLTexture> handle) {
  auto it = image2d_arrays_.find(name);
  if (it == image2d_arrays_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No image2d array argument with name - ", name));
  }
  it->second.handle = handle;
  return absl::OkStatus();
}

absl::Status MetalArguments::SetImage3D(const std::string& name,
                                        id<MTLTexture> handle) {
  auto it = images3d_.find(name);
  if (it == images3d_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No image3d argument with name - ", name));
  }
  it->second.handle = handle;
  return absl::OkStatus();
}

absl::Status MetalArguments::SetImageBuffer(const std::string& name,
                                            id<MTLTexture> handle) {
  auto it = image_buffers_.find(name);
  if (it == image_buffers_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No image buffer argument with name - ", name));
  }
  it->second.handle = handle;
  return absl::OkStatus();
}

absl::Status MetalArguments::ResolveSelectorsPass(
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

absl::Status MetalArguments::ResolveSelector(
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
  auto names = desc_ptr->GetGPUResources().GetNames();
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
