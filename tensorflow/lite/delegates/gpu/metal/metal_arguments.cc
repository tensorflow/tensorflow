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

#include <cstring>
#include <string>
#include <utility>

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
  if (!str) {
    return;
  }
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
    bool use_arguments_buffer, MetalDevice* device, Arguments* args,
    std::string* code) {
  RETURN_IF_ERROR(AllocateObjects(*args, device->device()));
  RETURN_IF_ERROR(AddObjectArgs(device->GetInfo(), *args));
  args->MoveObjectRefs(&object_refs_);
  std::string call_prefix = use_arguments_buffer ? "args." : "";
  std::string struct_desc =
      CopyScalarArgumentsToStructWithVec4Fields(*args, call_prefix, code);
  RETURN_IF_ERROR(SetObjectsResources(*args));
  if (!use_arguments_buffer) {
    args->ResolveArgsPass(code);
  }
  std::string header = R"(
#include <metal_stdlib>
using namespace metal;

)";
  header += struct_desc + "\n";
  if (use_arguments_buffer) {
    const std::string arg_buf_struct =
        GetArgumentBufferStructDefinition(!struct_desc.empty());
    header += arg_buf_struct + "\n";
  }
  *code = header + *code;
  std::string arguments;
  if (use_arguments_buffer) {
    arguments = "device ArgBuffer& args[[buffer(0)]]";
  } else {
    arguments = GetListOfArgs(/*buffer_offset*/ 0);
  }
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

absl::Status MetalArguments::Init(bool use_arguments_buffer,
                                  MetalDevice* device, Arguments* args) {
  RETURN_IF_ERROR(AllocateObjects(*args, device->device()));
  RETURN_IF_ERROR(AddObjectArgs(device->GetInfo(), *args));
  args->MoveObjectRefs(&object_refs_);
  CopyScalarArgumentsToStructWithVec4Fields(*args);
  RETURN_IF_ERROR(SetObjectsResources(*args));
  return absl::OkStatus();
}

std::string MetalArguments::CopyScalarArgumentsToStructWithScalarFields(
    const Arguments& args, const std::string& call_prefix, std::string* code) {
  std::string struct_desc = "struct uniforms_buffer {\n";
  int pos = 0;
  for (auto& fvalue : args.GetFloatValues()) {
    auto& new_val = float_values_[fvalue.first];
    new_val.value = fvalue.second.value;
    new_val.active = fvalue.second.active;
    if (fvalue.second.active) {
      new_val.bytes_offset = pos * 4;
      pos++;
      struct_desc += "  float " + fvalue.first + ";\n";
      ReplaceAllWords(kArgsPrefix + fvalue.first,
                      call_prefix + "U." + fvalue.first, code);
    }
  }
  for (const auto& hfvalue : args.GetHalfValues()) {
    auto& new_val = float_values_[hfvalue.first];
    new_val.value = hfvalue.second.value;
    new_val.active = hfvalue.second.active;
    if (hfvalue.second.active) {
      new_val.bytes_offset = pos * 4;
      pos++;
      struct_desc += "  float " + hfvalue.first + ";\n";
      ReplaceAllWords(
          kArgsPrefix + hfvalue.first,
          "static_cast<half>(" + call_prefix + "U." + hfvalue.first + ")",
          code);
    }
  }
  for (auto& ivalue : args.GetIntValues()) {
    auto& new_val = int_values_[ivalue.first];
    new_val.value = ivalue.second.value;
    new_val.active = ivalue.second.active;
    if (ivalue.second.active) {
      new_val.bytes_offset = pos * 4;
      pos++;
      struct_desc += "  int " + ivalue.first + ";\n";
      ReplaceAllWords(kArgsPrefix + ivalue.first,
                      call_prefix + "U." + ivalue.first, code);
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

std::string MetalArguments::CopyScalarArgumentsToStructWithVec4Fields(
    const Arguments& args, const std::string& call_prefix, std::string* code) {
  std::string struct_desc = "struct uniforms_buffer {\n";
  int pos = 0;
  std::string channels[4] = {".x", ".y", ".z", ".w"};
  for (auto& fvalue : args.GetFloatValues()) {
    auto& new_val = float_values_[fvalue.first];
    new_val.value = fvalue.second.value;
    new_val.active = fvalue.second.active;
    if (fvalue.second.active) {
      new_val.bytes_offset = pos * 4;
      if (pos % 4 == 0) {
        struct_desc += "  float4 cmp_float4_" + std::to_string(pos / 4) + ";\n";
      }
      std::string new_name = call_prefix + "U.cmp_float4_" +
                             std::to_string(pos / 4) + channels[pos % 4];
      ReplaceAllWords(kArgsPrefix + fvalue.first, new_name, code);
      pos++;
    }
  }
  for (const auto& hfvalue : args.GetHalfValues()) {
    auto& new_val = float_values_[hfvalue.first];
    new_val.value = hfvalue.second.value;
    new_val.active = hfvalue.second.active;
    if (hfvalue.second.active) {
      new_val.bytes_offset = pos * 4;
      if (pos % 4 == 0) {
        struct_desc += "  float4 cmp_float4_" + std::to_string(pos / 4) + ";\n";
      }
      std::string new_name = "static_cast<half>(" + call_prefix +
                             "U.cmp_float4_" + std::to_string(pos / 4) +
                             channels[pos % 4] + ")";
      ReplaceAllWords(kArgsPrefix + hfvalue.first, new_name, code);
      pos++;
    }
  }
  pos = AlignByN(pos, 4);
  for (auto& ivalue : args.GetIntValues()) {
    auto& new_val = int_values_[ivalue.first];
    new_val.value = ivalue.second.value;
    new_val.active = ivalue.second.active;
    if (ivalue.second.active) {
      new_val.bytes_offset = pos * 4;
      if (pos % 4 == 0) {
        struct_desc += "  int4 cmp_int4_" + std::to_string(pos / 4) + ";\n";
      }
      std::string new_name = call_prefix + "U.cmp_int4_" +
                             std::to_string(pos / 4) + channels[pos % 4];
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

std::string MetalArguments::GetArgumentBufferStructDefinition(
    bool add_constants_struct) {
  std::string result;
  result = "struct ArgBuffer {\n";
  int index = 0;
  for (auto& t : buffers_) {
    std::string mem_type = MemoryTypeToMetalType(t.second.desc.memory_type);
    std::string metal_type =
        ToMetalDataType(t.second.desc.data_type, t.second.desc.element_size);
    result += absl::StrCat("  ", mem_type, " ", metal_type, "* ", t.first,
                           "[[id(", index, ")]];\n");
    index++;
  }
  for (auto& t : images2d_) {
    std::string access = AccessToMetalTextureAccess(t.second.desc.access_type);
    std::string data_type =
        ToMetalDataType(ToMetalTextureType(t.second.desc.data_type));
    result += absl::StrCat("  texture2d<", data_type, ", ", access, "> ",
                           t.first, "[[id(", index, ")]];\n");
    index++;
  }
  for (auto& t : image2d_arrays_) {
    std::string access = AccessToMetalTextureAccess(t.second.desc.access_type);
    std::string data_type =
        ToMetalDataType(ToMetalTextureType(t.second.desc.data_type));
    result += absl::StrCat("  texture2d_array<", data_type, ", ", access, "> ",
                           t.first, "[[id(", index, ")]];\n");
    index++;
  }
  for (auto& t : images3d_) {
    std::string access = AccessToMetalTextureAccess(t.second.desc.access_type);
    std::string data_type =
        ToMetalDataType(ToMetalTextureType(t.second.desc.data_type));
    result += absl::StrCat("  texture3d<", data_type, ", ", access, "> ",
                           t.first, "[[id(", index, ")]];\n");
    index++;
  }
  for (auto& t : image_buffers_) {
    std::string access = AccessToMetalTextureAccess(t.second.desc.access_type);
    std::string data_type =
        ToMetalDataType(ToMetalTextureType(t.second.desc.data_type));
    result += absl::StrCat("  texture_buffer<", data_type, ", ", access, "> ",
                           t.first, "[[id(", index, ")]];\n");
    index++;
  }
  if (add_constants_struct) {
    result += "  uniforms_buffer U;\n";
  }
  result += "};";
  return result;
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
    [encoder setBuffer:b.second.handle
                offset:b.second.offset
               atIndex:buffer_offset];
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

API_AVAILABLE(ios(11.0), macos(10.13), tvos(11.0))
void MetalArguments::AddResourcesToEncoder(
    id<MTLComputeCommandEncoder> encoder) const {
  for (auto& b : buffers_) {
    [encoder useResource:b.second.handle
                   usage:MTLResourceUsageRead | MTLResourceUsageWrite];
  }
  for (auto& image : images2d_) {
    [encoder useResource:image.second.handle
                   usage:MTLResourceUsageRead | MTLResourceUsageWrite];
  }
  for (auto& image : image2d_arrays_) {
    [encoder useResource:image.second.handle
                   usage:MTLResourceUsageRead | MTLResourceUsageWrite];
  }
  for (auto& image : images3d_) {
    [encoder useResource:image.second.handle
                   usage:MTLResourceUsageRead | MTLResourceUsageWrite];
  }
  for (auto& image : image_buffers_) {
    [encoder useResource:image.second.handle
                   usage:MTLResourceUsageRead | MTLResourceUsageWrite];
  }
}

API_AVAILABLE(ios(11.0), macos(10.13), tvos(11.0))
void MetalArguments::EncodeArguments(id<MTLArgumentEncoder> arguments_encoder) {
  int index = 0;
  for (auto& b : buffers_) {
    [arguments_encoder setBuffer:b.second.handle
                          offset:b.second.offset
                         atIndex:index];
    index++;
  }
  for (auto& image : images2d_) {
    [arguments_encoder setTexture:image.second.handle atIndex:index];
    index++;
  }
  for (auto& image : image2d_arrays_) {
    [arguments_encoder setTexture:image.second.handle atIndex:index];
    index++;
  }
  for (auto& image : images3d_) {
    [arguments_encoder setTexture:image.second.handle atIndex:index];
    index++;
  }
  for (auto& image : image_buffers_) {
    [arguments_encoder setTexture:image.second.handle atIndex:index];
    index++;
  }
  if (!const_data_.empty()) {
    std::memcpy([arguments_encoder constantDataAtIndex:index],
                const_data_.data(), const_data_.size());
  }
}

absl::Status MetalArguments::AllocateObjects(const Arguments& args,
                                          id<MTLDevice> device) {
  objects_.resize(args.GetObjects().size());
  int i = 0;
  for (auto& t : args.GetObjects()) {
    RETURN_IF_ERROR(CreateMetalObject(device, t.second.get(), &objects_[i]));
    i++;
  }
  return absl::OkStatus();
}

absl::Status MetalArguments::AddObjectArgs(const GpuInfo& gpu_info,
                                           const Arguments& args) {
  for (const auto& t : args.GetObjects()) {
    AddGPUResources(t.first, t.second->GetGPUResources(gpu_info));
  }
  for (const auto& t : args.GetObjectRefs()) {
    AddGPUResources(t.first, t.second->GetGPUResources(gpu_info));
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
    std::string data_type =
        ToMetalDataType(ToMetalTextureType(t.second.desc.data_type));
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
    std::string data_type =
        ToMetalDataType(ToMetalTextureType(t.second.desc.data_type));
    AppendArgument(
        absl::StrCat("texture2d_array<", data_type, ", ", access, "> ", t.first,
                     "[[texture(", textures_offset, ")]]"),
        &result);
    textures_offset++;
  }
  for (auto& t : images3d_) {
    std::string access = AccessToMetalTextureAccess(t.second.desc.access_type);
    std::string data_type =
        ToMetalDataType(ToMetalTextureType(t.second.desc.data_type));
    AppendArgument(absl::StrCat("texture3d<", data_type, ", ", access, "> ",
                                t.first, "[[texture(", textures_offset, ")]]"),
                   &result);
    textures_offset++;
  }
  for (auto& t : image_buffers_) {
    std::string access = AccessToMetalTextureAccess(t.second.desc.access_type);
    std::string data_type =
        ToMetalDataType(ToMetalTextureType(t.second.desc.data_type));
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
    RETURN_IF_ERROR(SetBuffer(absl::StrCat(name, "_", r.first), r.second.handle,
                              r.second.offset));
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
                                     const GPUResources& resources) {
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
                                       id<MTLBuffer> handle, uint64_t offset) {
  auto it = buffers_.find(name);
  if (it == buffers_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No buffer argument with name - ", name));
  }
  it->second.handle = handle;
  it->second.offset = offset;
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

absl::Status MetalArguments::SetObjectsResources(const Arguments& args) {
  int i = 0;
  for (const auto& t : args.GetObjects()) {
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
