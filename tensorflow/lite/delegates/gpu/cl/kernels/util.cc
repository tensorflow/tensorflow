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

#include <cfloat>
#include <cmath>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/cl/precision.h"
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
      result += "#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable\n";
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
      result += "#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable\n";
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
      result += "#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable\n";
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

TensorCodeGenerator::TensorCodeGenerator(const std::string& name,
                                         const WHSPoint& sizes,
                                         const TensorDescriptor& descriptor)
    : tensor_name_(name),
      width_name_(sizes.w_name),
      height_name_(sizes.h_name),
      slices_name_(sizes.s_name),
      descriptor_(descriptor) {}

TensorCodeGenerator::TensorCodeGenerator(const std::string& name,
                                         const WHSBPoint& sizes,
                                         const TensorDescriptor& descriptor)
    : tensor_name_(name),
      width_name_(sizes.w_name),
      height_name_(sizes.h_name),
      slices_name_(sizes.s_name),
      batch_name_(sizes.b_name),
      descriptor_(descriptor) {}

TensorCodeGenerator::TensorCodeGenerator(const std::string& name,
                                         const WHDSPoint& sizes,
                                         const TensorDescriptor& descriptor)
    : tensor_name_(name),
      width_name_(sizes.w_name),
      height_name_(sizes.h_name),
      depth_name_(sizes.d_name),
      slices_name_(sizes.s_name),
      descriptor_(descriptor) {}

TensorCodeGenerator::TensorCodeGenerator(const std::string& name,
                                         const WHDSBPoint& sizes,
                                         const TensorDescriptor& descriptor)
    : tensor_name_(name),
      width_name_(sizes.w_name),
      height_name_(sizes.h_name),
      depth_name_(sizes.d_name),
      slices_name_(sizes.s_name),
      batch_name_(sizes.b_name),
      descriptor_(descriptor) {}

std::string TensorCodeGenerator::GetDeclaration(AccessType access_type) const {
  return GetTensorDeclaration(access_type, tensor_name_, descriptor_);
}

std::string TensorCodeGenerator::ReadWHS(
    const std::string& x, const std::string& y, const std::string& s,
    TextureAddressMode address_mode) const {
  return Read(GetGlobalAddressNoDeclarationWHS(x, y, s), address_mode);
}

std::string TensorCodeGenerator::ReadWHSB(
    const std::string& x, const std::string& y, const std::string& s,
    const std::string& b, TextureAddressMode address_mode) const {
  return Read(GetGlobalAddressNoDeclarationWHSB(x, y, s, b), address_mode);
}

std::string TensorCodeGenerator::ReadWHDS(
    const std::string& x, const std::string& y, const std::string& z,
    const std::string& s, TextureAddressMode address_mode) const {
  return Read(GetGlobalAddressNoDeclarationWHDS(x, y, z, s), address_mode);
}

std::string TensorCodeGenerator::ReadWHDSB(
    const std::string& x, const std::string& y, const std::string& z,
    const std::string& s, const std::string& b,
    TextureAddressMode address_mode) const {
  return Read(GetGlobalAddressNoDeclarationWHDSB(x, y, z, s, b), address_mode);
}

std::string TensorCodeGenerator::ReadAsFloatWHS(
    const std::string& x, const std::string& y, const std::string& s,
    TextureAddressMode address_mode) const {
  return ReadAsFloat(GetGlobalAddressNoDeclarationWHS(x, y, s), address_mode);
}

std::string TensorCodeGenerator::ReadAsFloatWHSB(
    const std::string& x, const std::string& y, const std::string& s,
    const std::string& b, TextureAddressMode address_mode) const {
  return ReadAsFloat(GetGlobalAddressNoDeclarationWHSB(x, y, s, b),
                     address_mode);
}

std::string TensorCodeGenerator::ReadAsFloatWHDS(
    const std::string& x, const std::string& y, const std::string& z,
    const std::string& s, TextureAddressMode address_mode) const {
  return ReadAsFloat(GetGlobalAddressNoDeclarationWHDS(x, y, z, s),
                     address_mode);
}

std::string TensorCodeGenerator::ReadAsFloatWHDSB(
    const std::string& x, const std::string& y, const std::string& z,
    const std::string& s, const std::string& b,
    TextureAddressMode address_mode) const {
  return ReadAsFloat(GetGlobalAddressNoDeclarationWHDSB(x, y, z, s, b),
                     address_mode);
}

std::string TensorCodeGenerator::ReadAsTypeWHS(
    DataType type, const std::string& x, const std::string& y,
    const std::string& s, TextureAddressMode address_mode) const {
  return ReadAsType(type, GetGlobalAddressNoDeclarationWHS(x, y, s),
                    address_mode);
}

std::string TensorCodeGenerator::ReadAsTypeWHSB(
    DataType type, const std::string& x, const std::string& y,
    const std::string& s, const std::string& b,
    TextureAddressMode address_mode) const {
  return ReadAsType(type, GetGlobalAddressNoDeclarationWHSB(x, y, s, b),
                    address_mode);
}

std::string TensorCodeGenerator::ReadAsTypeWHDS(
    DataType type, const std::string& x, const std::string& y,
    const std::string& z, const std::string& s,
    TextureAddressMode address_mode) const {
  return ReadAsType(type, GetGlobalAddressNoDeclarationWHDS(x, y, z, s),
                    address_mode);
}

std::string TensorCodeGenerator::ReadAsTypeWHDSB(
    DataType type, const std::string& x, const std::string& y,
    const std::string& z, const std::string& s, const std::string& b,
    TextureAddressMode address_mode) const {
  return ReadAsType(type, GetGlobalAddressNoDeclarationWHDSB(x, y, z, s, b),
                    address_mode);
}

std::string TensorCodeGenerator::GetAddressWHS(const std::string& var_name,
                                               const std::string& x,
                                               const std::string& y,
                                               const std::string& s) const {
  return DeclareAddress(var_name, GetGlobalAddressNoDeclarationWHS(x, y, s));
}

std::string TensorCodeGenerator::GetAddressWHSB(const std::string& var_name,
                                                const std::string& x,
                                                const std::string& y,
                                                const std::string& s,
                                                const std::string& b) const {
  return DeclareAddress(var_name,
                        GetGlobalAddressNoDeclarationWHSB(x, y, s, b));
}

std::string TensorCodeGenerator::GetAddressWHDS(const std::string& var_name,
                                                const std::string& x,
                                                const std::string& y,
                                                const std::string& z,
                                                const std::string& s) const {
  return DeclareAddress(var_name,
                        GetGlobalAddressNoDeclarationWHDS(x, y, z, s));
}

std::string TensorCodeGenerator::GetAddressWHDSB(
    const std::string& var_name, const std::string& x, const std::string& y,
    const std::string& z, const std::string& s, const std::string& b) const {
  return DeclareAddress(var_name,
                        GetGlobalAddressNoDeclarationWHDSB(x, y, z, s, b));
}

std::string TensorCodeGenerator::GetGlobalAddressNoDeclarationWHS(
    const std::string& x, const std::string& y, const std::string& s) const {
  switch (descriptor_.storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      return absl::Substitute("((($2) * $3 + ($1)) * $4 + ($0))", x, y, s,
                              height_name_, width_name_);
    case TensorStorageType::TEXTURE_2D:
      return absl::Substitute("(int2)(($0), ($1) * $3 + ($2))", x, y, s,
                              slices_name_);
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return absl::StrCat("(int2)(", x, ", ", y, ")");
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_3D:
      return absl::StrCat("(int4)(", x, ", ", y, ", ", s, ", 0)");
    case TensorStorageType::UNKNOWN:
      return "error";
  }
}

std::string TensorCodeGenerator::GetGlobalAddressNoDeclarationWHSB(
    const std::string& x, const std::string& y, const std::string& s,
    const std::string& b) const {
  if (b.empty()) {
    return GetGlobalAddressNoDeclarationWHS(x, y, s);
  }
  switch (descriptor_.storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      return absl::Substitute("(((($3) * $4 + $2) * $5 + ($1)) * $6 + ($0))", b,
                              x, y, s, height_name_, width_name_, batch_name_);
    case TensorStorageType::TEXTURE_2D:
      return absl::Substitute("(int2)(($0) * $4 + ($1), ($2) * $5 + ($3))", x,
                              b, y, s, batch_name_, slices_name_);
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return absl::Substitute("(int2)(($0) * $3 + ($1), ($2))", x, b, y,
                              batch_name_);
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_3D:
      return absl::Substitute("(int4)(($0) * $4 + ($1), ($2), ($3), 0)", x, b,
                              y, s, batch_name_);
    case TensorStorageType::UNKNOWN:
      return "error";
    default:
      return "error";
  }
}

std::string TensorCodeGenerator::GetGlobalAddressNoDeclarationWHDS(
    const std::string& x, const std::string& y, const std::string& z,
    const std::string& s) const {
  switch (descriptor_.storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      return absl::Substitute("(((($3) * $4 + ($2)) * $5 + ($1)) * $6 + ($0))",
                              x, y, s, z, slices_name_, height_name_,
                              width_name_);
    case TensorStorageType::TEXTURE_2D:
      return absl::Substitute("(int2)(($0) * $4 + ($1), ($2) * $5 + ($3))", x,
                              z, y, s, depth_name_, slices_name_);
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return absl::Substitute("(int2)(($0) * $3 + ($1), ($2))", x, z, y,
                              depth_name_);
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_3D:
      return absl::Substitute("(int4)(($0), ($1), ($2) * $4 + ($3), 0)", x, y,
                              z, s, slices_name_);
    case TensorStorageType::UNKNOWN:
      return "error";
  }
}

std::string TensorCodeGenerator::GetGlobalAddressNoDeclarationWHDSB(
    const std::string& x, const std::string& y, const std::string& z,
    const std::string& s, const std::string& b) const {
  if (b.empty()) {
    return GetGlobalAddressNoDeclarationWHDS(x, y, z, s);
  }
  switch (descriptor_.storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      return absl::Substitute(
          "((((($4) * $5 + ($3)) * $6 + $2) * $7 + ($1)) * $8 + ($0))", b, x, y,
          s, z, slices_name_, height_name_, width_name_, batch_name_);
    case TensorStorageType::TEXTURE_2D:
      return absl::Substitute(
          "(int2)((($0) * $5 + ($1)) * $6 + ($2), ($3) * $7 + ($4))", x, b, z,
          y, s, batch_name_, depth_name_, slices_name_);
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return absl::Substitute("(int2)((($0) * $4 + ($1)) * $5 + ($2), ($3))", x,
                              b, z, y, batch_name_, depth_name_);
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_3D:
      return absl::Substitute(
          "(int4)(($0) * $5 + ($1), ($2), ($3) * $6 + ($4), 0)", x, b, y, z, s,
          batch_name_, slices_name_);
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

std::string TensorCodeGenerator::WriteWHS(const std::string& var_name,
                                          const std::string& x,
                                          const std::string& y,
                                          const std::string& s) const {
  return Write(var_name, GetGlobalAddressNoDeclarationWHS(x, y, s));
}

std::string TensorCodeGenerator::WriteWHSB(const std::string& var_name,
                                           const std::string& x,
                                           const std::string& y,
                                           const std::string& s,
                                           const std::string& b) const {
  return Write(var_name, GetGlobalAddressNoDeclarationWHSB(x, y, s, b));
}

std::string TensorCodeGenerator::WriteWHDS(const std::string& var_name,
                                           const std::string& x,
                                           const std::string& y,
                                           const std::string& z,
                                           const std::string& s) const {
  return Write(var_name, GetGlobalAddressNoDeclarationWHDS(x, y, z, s));
}

std::string TensorCodeGenerator::WriteWHDSB(
    const std::string& var_name, const std::string& x, const std::string& y,
    const std::string& z, const std::string& s, const std::string& b) const {
  return Write(var_name, GetGlobalAddressNoDeclarationWHDSB(x, y, z, s, b));
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

std::string TensorCodeGenerator::ReadAsType(
    DataType type, const std::string& global_address,
    TextureAddressMode address_mode) const {
  const std::string read_as =
      type == DataType::FLOAT16 ? "read_imageh" : "read_imagef";
  switch (descriptor_.storage_type) {
    case TensorStorageType::BUFFER: {
      const std::string reading =
          absl::StrCat(tensor_name_, "[", global_address, "]");
      if (type == descriptor_.data_type) {
        return reading;
      } else {
        const std::string conversion =
            type == DataType::FLOAT16 ? "convert_half4" : "convert_float4";
        return absl::StrCat(conversion, "(", reading, ")");
      }
    }
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::TEXTURE_3D:
    case TensorStorageType::SINGLE_TEXTURE_2D:
    case TensorStorageType::TEXTURE_ARRAY:
      return absl::StrCat(
          read_as, "(", tensor_name_,
          ", " + TextureAddressModeToString(address_mode) + ", ",
          global_address, ")");
    case TensorStorageType::IMAGE_BUFFER:
      return absl::StrCat(read_as, "(", tensor_name_, ", ", global_address,
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

int3 GetFirstSuitableWorkGroup(const std::vector<int3>& wgs, int max_wg_size) {
  for (const auto& wg : wgs) {
    const int wg_size = wg.x * wg.y * wg.z;
    if (wg_size <= max_wg_size) {
      return wg;
    }
  }
  return {1, 1, 1};
}

int GetRecommendedBlockSizeForConv(const CLDevice& device,
                                   CalculationsPrecision precision,
                                   int task_size) {
  const float task_size_per_cu =
      task_size / static_cast<float>(device.GetInfo().compute_units_count);
  int block_size = 1;
  float threshold_1 = FLT_MAX;
  float threshold_2 = FLT_MAX;
  float threshold_4 = FLT_MAX;
  if (!device.IsMali()) {
    return 1;
  }
  MaliInfo mali_info = device.GetInfo().mali_info;
  switch (precision) {
    case CalculationsPrecision::F16:
      if (mali_info.IsBifrostGen1()) {
        threshold_1 = 256.0f;
        threshold_2 = 256.0f * 4.0f;
        threshold_4 = 256.0f * 8.0f;
      } else if (mali_info.IsBifrostGen2()) {
        threshold_1 = 256.0f * 2.0f;
        threshold_2 = 256.0f * 8.0f;
        threshold_4 = 256.0f * 16.0f;
      } else if (mali_info.IsBifrostGen3() || mali_info.IsValhall()) {
        threshold_1 = 256.0f;
        threshold_2 = 256.0f * 6.0f;
        threshold_4 = 256.0f * 16.0f;
      } else if (mali_info.IsMidgard()) {
        threshold_1 = 256.0f * 4.0f;
        threshold_2 = 256.0f * 16.0f;
      }
      break;
    case CalculationsPrecision::F32_F16:
      if (mali_info.IsBifrostGen1()) {
        threshold_1 = 256.0f;
        threshold_2 = 256.0f * 3.0f;
        threshold_4 = 256.0f * 32.0f;
      } else if (mali_info.IsBifrostGen2()) {
        threshold_1 = 256.0f * 2.0f;
        threshold_2 = 256.0f * 8.0f;
      } else if (mali_info.IsBifrostGen3() || mali_info.IsValhall()) {
        threshold_1 = 256.0f;
        threshold_2 = 256.0f * 8.0f;
      } else if (mali_info.IsMidgard()) {
        threshold_1 = 256.0f * 4.0f;
      }
      break;
    case CalculationsPrecision::F32:
      if (mali_info.IsBifrostGen1()) {
        threshold_1 = 256.0f;
        threshold_2 = 256.0f * 4.0f;
      } else if (mali_info.IsBifrostGen2()) {
        threshold_1 = 128.0f;
        threshold_2 = 256.0f * 4.0f;
      } else if (mali_info.IsBifrostGen3() || mali_info.IsValhall()) {
        threshold_1 = 256.0f;
        threshold_2 = 256.0f * 12.0f;
      } else if (mali_info.IsMidgard()) {
        threshold_1 = 256.0f * 16.0f;
      }
      break;
  }
  if (task_size_per_cu <= threshold_1) {
    block_size = 1;
  } else if (task_size_per_cu <= threshold_2) {
    block_size = 2;
  } else if (task_size_per_cu <= threshold_4) {
    block_size = 4;
  } else {
    block_size = 8;
  }
  return block_size;
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
