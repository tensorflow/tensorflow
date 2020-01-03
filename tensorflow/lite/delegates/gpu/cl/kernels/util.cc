/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"

#include <cmath>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {
std::string GetReadImageFromDataType(DataType data_type) {
  if (data_type == DataType::FLOAT32) {
    return "read_imagef";
  } else if (data_type == DataType::FLOAT16) {
    return "read_imageh";
  } else {
    return "error";
  }
}

std::string GetWriteImageFromDataType(DataType data_type) {
  if (data_type == DataType::FLOAT32) {
    return "write_imagef";
  } else if (data_type == DataType::FLOAT16) {
    return "write_imageh";
  } else {
    return "error";
  }
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

std::string TextureAddressModeToString(TextureAddressMode address_mode) {
  switch (address_mode) {
    case TextureAddressMode::DONT_CARE:
      return "smp_none";
    case TextureAddressMode::ZERO:
      return "smp_zero";
  }
}

}  // namespace

std::string GetCommonDefines(CalculationsPrecision precision) {
  std::string result;

  switch (precision) {
    case CalculationsPrecision::F32:
      result += "#define ACCUM_FLT4 float4\n";
      result += "#define FLT float\n";
      result += "#define FLT2 float2\n";
      result += "#define FLT3 float3\n";
      result += "#define FLT4 float4\n";
      result += "#define TO_FLT4 convert_float4\n";
      result += "#define TO_ACCUM_TYPE convert_float4\n";
      result += "#define TO_ACCUM_FLT convert_float\n";
      result += "#define READ_IMAGE read_imagef\n";
      result += "#define WRITE_IMAGE write_imagef\n";
      break;
    case CalculationsPrecision::F16:
      result += "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
      result += "#define ACCUM_FLT4 half4\n";
      result += "#define FLT half\n";
      result += "#define FLT2 half2\n";
      result += "#define FLT3 half3\n";
      result += "#define FLT4 half4\n";
      result += "#define TO_FLT4 convert_half4\n";
      result += "#define TO_ACCUM_TYPE convert_half4\n";
      result += "#define TO_ACCUM_FLT convert_half\n";
      result += "#define READ_IMAGE read_imageh\n";
      result += "#define WRITE_IMAGE write_imageh\n";
      break;
    case CalculationsPrecision::F32_F16:
      result += "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
      result += "#define ACCUM_FLT4 float4\n";
      result += "#define FLT half\n";
      result += "#define FLT2 half2\n";
      result += "#define FLT3 half3\n";
      result += "#define FLT4 half4\n";
      result += "#define TO_FLT4 convert_half4\n";
      result += "#define TO_ACCUM_TYPE convert_float4\n";
      result += "#define TO_ACCUM_FLT convert_float\n";
      result += "#define READ_IMAGE read_imageh\n";
      result += "#define WRITE_IMAGE write_imageh\n";
      break;
  }

  result +=
      "const sampler_t smp_edge = CLK_NORMALIZED_COORDS_FALSE | "
      "CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;\n";
  result +=
      "const sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | "
      "CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;\n";
  result +=
      "const sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | "
      "CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n";

  return result;
}

TensorCodeGenerator::SizeVariablesNames::SizeVariablesNames(
    const std::string& width_name, const std::string& height_name,
    const std::string& depth_name)
    : width(width_name), height(height_name), depth(depth_name) {}

TensorCodeGenerator::SizeVariablesNames::SizeVariablesNames(
    const std::string& width_name, const std::string& height_name,
    const std::string& depth_name, const std::string& batch_name)
    : width(width_name),
      height(height_name),
      depth(depth_name),
      batch(batch_name) {}

TensorCodeGenerator::TensorCodeGenerator(const std::string& name,
                                         const SizeVariablesNames& sizes,
                                         const TensorDescriptor& descriptor)
    : tensor_name_(name), sizes_(sizes), descriptor_(descriptor) {}

std::string TensorCodeGenerator::GetDeclaration(AccessType access_type) const {
  return GetTensorDeclaration(access_type, tensor_name_, descriptor_);
}

std::string TensorCodeGenerator::Read3D(const std::string& x,
                                        const std::string& y,
                                        const std::string& z,
                                        TextureAddressMode address_mode) const {
  return Read(GetGlobalAddressNoDeclaration(x, y, z), address_mode);
}

std::string TensorCodeGenerator::Read4D(const std::string& x,
                                        const std::string& y,
                                        const std::string& z,
                                        const std::string& b,
                                        TextureAddressMode address_mode) const {
  return Read(GetGlobalAddressNoDeclaration(x, y, z, b), address_mode);
}

std::string TensorCodeGenerator::ReadAsFloat3D(
    const std::string& x, const std::string& y, const std::string& z,
    TextureAddressMode address_mode) const {
  return ReadAsFloat(GetGlobalAddressNoDeclaration(x, y, z), address_mode);
}

std::string TensorCodeGenerator::ReadAsFloat4D(
    const std::string& x, const std::string& y, const std::string& z,
    const std::string& b, TextureAddressMode address_mode) const {
  return ReadAsFloat(GetGlobalAddressNoDeclaration(x, y, z, b), address_mode);
}

std::string TensorCodeGenerator::GetAddress(const std::string& var_name,
                                            const std::string& x,
                                            const std::string& y,
                                            const std::string& z) const {
  return DeclareAddress(var_name, GetGlobalAddressNoDeclaration(x, y, z));
}

std::string TensorCodeGenerator::GetAddress(const std::string& var_name,
                                            const std::string& x,
                                            const std::string& y,
                                            const std::string& z,
                                            const std::string& b) const {
  return DeclareAddress(var_name, GetGlobalAddressNoDeclaration(x, y, z, b));
}

std::string TensorCodeGenerator::GetGlobalAddressNoDeclaration(
    const std::string& x, const std::string& y, const std::string& z) const {
  switch (descriptor_.storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      return absl::Substitute("((($2) * $3 + ($1)) * $4 + ($0))", x, y, z,
                              sizes_.height, sizes_.width);
    case TensorStorageType::TEXTURE_2D:
      return absl::Substitute("(int2)(($0), ($1) * $3 + ($2))", x, y, z,
                              sizes_.depth);
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return absl::StrCat("(int2)(", x, ", ", y, ")");
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_3D:
      return absl::StrCat("(int4)(", x, ", ", y, ", ", z, ", 0)");
    case TensorStorageType::UNKNOWN:
      return "error";
  }
}

std::string TensorCodeGenerator::GetGlobalAddressNoDeclaration(
    const std::string& x, const std::string& y, const std::string& z,
    const std::string& b) const {
  if (b.empty()) {
    return GetGlobalAddressNoDeclaration(x, y, z);
  }
  switch (descriptor_.storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      return absl::Substitute("(((($3) * $4 + $2) * $5 + ($1)) * $6 + ($0))", b,
                              x, y, z, sizes_.height, sizes_.width,
                              sizes_.batch);
    case TensorStorageType::TEXTURE_2D:
      return absl::Substitute("(int2)(($0) * ($4) + ($1), ($2) * $5 + ($3))", x,
                              b, y, z, sizes_.batch, sizes_.depth);
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return absl::Substitute("(int2)(($0) * ($3) + ($1), ($2))", x, b, y,
                              sizes_.batch);
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_3D:
      return absl::Substitute("(int4)(($0) * ($4) + ($1), ($2), ($3), 0)", x, b,
                              y, z, sizes_.batch);
    case TensorStorageType::UNKNOWN:
      return "error";
    default:
      return "error";
  }
}

std::string TensorCodeGenerator::DeclareAddress(
    const std::string& var_name, const std::string& address) const {
  switch (descriptor_.storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      return absl::StrCat("int ", var_name, " = ", address, ";\n");
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return absl::StrCat("int2 ", var_name, " = ", address, ";\n");
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_3D:
      return absl::StrCat("int4 ", var_name, " = ", address, ";\n");
    case TensorStorageType::UNKNOWN:
      return "";
  }
}

std::string TensorCodeGenerator::Write3D(const std::string& var_name,
                                         const std::string& x,
                                         const std::string& y,
                                         const std::string& z) const {
  return Write(var_name, GetGlobalAddressNoDeclaration(x, y, z));
}

std::string TensorCodeGenerator::Write4D(const std::string& var_name,
                                         const std::string& x,
                                         const std::string& y,
                                         const std::string& z,
                                         const std::string& b) const {
  return Write(var_name, GetGlobalAddressNoDeclaration(x, y, z, b));
}

std::string TensorCodeGenerator::Read(const std::string& global_address,
                                      TextureAddressMode address_mode) const {
  switch (descriptor_.storage_type) {
    case TensorStorageType::BUFFER:
      return absl::StrCat(tensor_name_, "[", global_address, "]");
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::TEXTURE_3D:
    case TensorStorageType::SINGLE_TEXTURE_2D:
    case TensorStorageType::TEXTURE_ARRAY:
      return absl::StrCat(
          GetReadImageFromDataType(descriptor_.data_type), "(", tensor_name_,
          ", " + TextureAddressModeToString(address_mode) + ", ",
          global_address, ")");
    case TensorStorageType::IMAGE_BUFFER:
      return absl::StrCat(GetReadImageFromDataType(descriptor_.data_type), "(",
                          tensor_name_, ", ", global_address, ")");
    case TensorStorageType::UNKNOWN:
      return "";
  }
}

std::string TensorCodeGenerator::ReadAsFloat(
    const std::string& global_address, TextureAddressMode address_mode) const {
  switch (descriptor_.storage_type) {
    case TensorStorageType::BUFFER:
      return absl::StrCat("convert_float4(", tensor_name_, "[", global_address,
                          "])");
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::TEXTURE_3D:
    case TensorStorageType::SINGLE_TEXTURE_2D:
    case TensorStorageType::TEXTURE_ARRAY:
      return absl::StrCat(
          "read_imagef(", tensor_name_,
          ", " + TextureAddressModeToString(address_mode) + ", ",
          global_address, ")");
    case TensorStorageType::IMAGE_BUFFER:
      return absl::StrCat("read_imagef(", tensor_name_, ", ", global_address,
                          ")");
    case TensorStorageType::UNKNOWN:
      return "";
  }
}

std::string TensorCodeGenerator::Write(
    const std::string& var_name, const std::string& global_address) const {
  switch (descriptor_.storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      return absl::StrCat(tensor_name_, "[", global_address, "] = ", var_name,
                          ";\n");
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::TEXTURE_3D:
    case TensorStorageType::SINGLE_TEXTURE_2D:
    case TensorStorageType::TEXTURE_ARRAY:
      return absl::StrCat(GetWriteImageFromDataType(descriptor_.data_type), "(",
                          tensor_name_, ", ", global_address, ", ", var_name,
                          ");\n");
    case TensorStorageType::UNKNOWN:
      return "";
  }
}

std::string GetTensorDeclaration(AccessType access,
                                 const std::string& tensor_name,
                                 const TensorDescriptor& descriptor) {
  switch (descriptor.storage_type) {
    case TensorStorageType::BUFFER:
      return absl::StrCat("__global ", ToCLDataType(descriptor.data_type, 4),
                          "* ", tensor_name);
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return GetImageModifier(access) + " image2d_t " + tensor_name;
    case TensorStorageType::TEXTURE_ARRAY:
      return GetImageModifier(access) + " image2d_array_t " + tensor_name;
    case TensorStorageType::TEXTURE_3D:
      return GetImageModifier(access) + " image3d_t " + tensor_name;
    case TensorStorageType::IMAGE_BUFFER:
      if (access == AccessType::WRITE) {
        return absl::StrCat("__global ", ToCLDataType(descriptor.data_type, 4),
                            "* ", tensor_name);
      } else {
        return GetImageModifier(access) + " image1d_buffer_t " + tensor_name;
      }
    case TensorStorageType::UNKNOWN:
      return "error";
  }
}

std::string GetXStrideCorrected(const std::string& src_x,
                                const std::string& batch_size,
                                const std::string& stride_x,
                                const std::string& padding_x) {
  // TODO(sorokin) check perf and optimize with floor() if needed
  // int p0 = src_x / batch_size;\n";
  // int b0 = src_x % batch_size;\n";
  // return p0 * stride_x * batch_size + b0 + padding_x;\n";
  return absl::Substitute("((($0) / $1) * $2 * $1 + (($0) % $1) + $3)", src_x,
                          batch_size, stride_x, padding_x);
}

TextureAddressMode GetFastestZeroMode(const CLDevice& device) {
  return device.IsAdreno3xx() ? TextureAddressMode::DONT_CARE
                              : TextureAddressMode::ZERO;
}

float4 GetMaskForLastPlane(int channels) {
  float4 mask = float4(0.0f);
  const int reminder = channels % 4 == 0 ? 4 : channels % 4;
  for (int i = 0; i < reminder; ++i) {
    mask[i] = 1.0f;
  }
  return mask;
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
