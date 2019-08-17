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

namespace tflite {
namespace gpu {
namespace cl {

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

std::string GetGlobalAddress(TensorStorageType storage_type,
                             const std::string& size_name,
                             const std::string& var_name, const std::string& x,
                             const std::string& y, const std::string& z) {
  switch (storage_type) {
    case TensorStorageType::BUFFER:
      return absl::StrCat("int ", var_name, " = ((", z, ") * ", size_name,
                          ".y + (", y, ")) * ", size_name, ".x + (", x, ");\n");
    case TensorStorageType::TEXTURE_2D:
      return absl::StrCat("int2 ", var_name, " = (int2)((", x, "), (", y,
                          ") * ", size_name, ".w + (", z, "));\n");
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return absl::StrCat("int2 ", var_name, " = (int2)(", x, ", ", y, ");\n");
    case TensorStorageType::TEXTURE_ARRAY:
      return absl::StrCat("int4 ", var_name, " = (int4)(", x, ", ", y, ", ", z,
                          ", 0);\n");
    case TensorStorageType::UNKNOWN:
      return "";
  }
}

std::string GetReadImageFromDataType(DataType data_type) {
  if (data_type == DataType::FLOAT32) {
    return "read_imagef";
  } else if (data_type == DataType::FLOAT16) {
    return "read_imageh";
  } else {
    return "READ_IMAGE";
  }
}

std::string ReadGlobalFLT4(TensorStorageType storage_type, DataType data_type,
                           const std::string& tensor_name,
                           const std::string& size_name, const std::string& x,
                           const std::string& y, const std::string& z) {
  switch (storage_type) {
    case TensorStorageType::BUFFER:
      return absl::StrCat(tensor_name, "[((", z, ") * ", size_name, ".y + (", y,
                          ")) * ", size_name, ".x + (", x, ")]");
    case TensorStorageType::TEXTURE_2D:
      return absl::StrCat(GetReadImageFromDataType(data_type), "(", tensor_name,
                          ", smp_zero, (int2)((", x, "), (", y, ") * ",
                          size_name, ".w + (", z, ")))");
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return absl::StrCat(GetReadImageFromDataType(data_type), "(", tensor_name,
                          ", smp_zero, (int2)(", x, ", ", y, "))");
    case TensorStorageType::TEXTURE_ARRAY:
      return absl::StrCat(GetReadImageFromDataType(data_type), "(", tensor_name,
                          ", smp_zero, (int4)(", x, ", ", y, ", ", z, ", 0))");
    case TensorStorageType::UNKNOWN:
      return "";
  }
}

std::string ReadGlobalFloat4(TensorStorageType storage_type,
                             const std::string& tensor_name,
                             const std::string& size_name, const std::string& x,
                             const std::string& y, const std::string& z) {
  switch (storage_type) {
    case TensorStorageType::BUFFER:
      return absl::StrCat("convert_float4(", tensor_name, "[((", z, ") * ",
                          size_name, ".y + (", y, ")) * ", size_name, ".x + (",
                          x, ")])");
    case TensorStorageType::TEXTURE_2D:
      return absl::StrCat("read_imagef(", tensor_name, ", smp_zero, (int2)((",
                          x, "), (", y, ") * ", size_name, ".w + (", z, ")))");
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return absl::StrCat("read_imagef(", tensor_name, ", smp_zero, (int2)(", x,
                          ", ", y, "))");
    case TensorStorageType::TEXTURE_ARRAY:
      return absl::StrCat("read_imagef(", tensor_name, ", smp_zero, (int4)(", x,
                          ", ", y, ", ", z, ", 0))");
    case TensorStorageType::UNKNOWN:
      return "";
  }
}

std::string ReadGlobalFLT4(TensorStorageType storage_type, DataType data_type,
                           const std::string& tensor_name,
                           const std::string& global_address) {
  switch (storage_type) {
    case TensorStorageType::BUFFER:
      return absl::StrCat(tensor_name, "[", global_address, "]");
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return absl::StrCat(GetReadImageFromDataType(data_type), "(", tensor_name,
                          ", smp_zero, ", global_address, ")");
    case TensorStorageType::TEXTURE_ARRAY:
      return absl::StrCat(GetReadImageFromDataType(data_type), "(", tensor_name,
                          ", smp_zero, ", global_address, ")");
    case TensorStorageType::UNKNOWN:
      return "";
  }
}

std::string ReadGlobalFloat4(TensorStorageType storage_type,
                             const std::string& tensor_name,
                             const std::string& global_address) {
  switch (storage_type) {
    case TensorStorageType::BUFFER:
      return absl::StrCat("convert_float4(", tensor_name, "[", global_address,
                          "])");
    case TensorStorageType::TEXTURE_2D:
      return absl::StrCat("read_imagef(", tensor_name, ", smp_zero, ",
                          global_address, ")");
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return absl::StrCat("read_imagef(", tensor_name, ", smp_zero, ",
                          global_address, ")");
    case TensorStorageType::TEXTURE_ARRAY:
      return absl::StrCat("read_imagef(", tensor_name, ", smp_zero, ",
                          global_address, ")");
    case TensorStorageType::UNKNOWN:
      return "";
  }
}

std::string GetWriteImageFromDataType(DataType data_type) {
  if (data_type == DataType::FLOAT32) {
    return "write_imagef";
  } else if (data_type == DataType::FLOAT16) {
    return "write_imageh";
  } else {
    return "WRITE_IMAGE";
  }
}

std::string WriteGlobalFLT4(TensorStorageType storage_type, DataType data_type,
                            const std::string& tensor_name,
                            const std::string& size_name,
                            const std::string& var_name, const std::string& x,
                            const std::string& y, const std::string& z) {
  switch (storage_type) {
    case TensorStorageType::BUFFER:
      return absl::StrCat(tensor_name, "[((", z, ") * ", size_name, ".y + (", y,
                          ")) * ", size_name, ".x + (", x, ")] = ", var_name,
                          ";\n");
    case TensorStorageType::TEXTURE_2D:
      return absl::StrCat(GetWriteImageFromDataType(data_type), "(",
                          tensor_name, ", (int2)((", x, "), (", y, ") * ",
                          size_name, ".w + (", z, ")), ", var_name, ");\n");
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return absl::StrCat(GetWriteImageFromDataType(data_type), "(",
                          tensor_name, ", (int2)(", x, ", ", y, "), ", var_name,
                          ");\n");
    case TensorStorageType::TEXTURE_ARRAY:
      return absl::StrCat(GetWriteImageFromDataType(data_type), "(",
                          tensor_name, ", (int4)(", x, ", ", y, ", ", z,
                          ", 0), ", var_name, ");\n");
    case TensorStorageType::UNKNOWN:
      return "";
  }
}

std::string WriteGlobalFLT4(TensorStorageType storage_type, DataType data_type,
                            const std::string& tensor_name,
                            const std::string& var_name,
                            const std::string& global_address) {
  switch (storage_type) {
    case TensorStorageType::BUFFER:
      return absl::StrCat(tensor_name, "[", global_address, "] = ", var_name,
                          ";\n");
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return absl::StrCat(GetWriteImageFromDataType(data_type), "(",
                          tensor_name, ", ", global_address, ", ", var_name,
                          ");\n");
    case TensorStorageType::TEXTURE_ARRAY:
      return absl::StrCat(GetWriteImageFromDataType(data_type), "(",
                          tensor_name, ", ", global_address, ", ", var_name,
                          ");\n");
    case TensorStorageType::UNKNOWN:
      return "";
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

std::string GetDataType(DataType type) {
  switch (type) {
    case DataType::FLOAT16:
      return "half";
    case DataType::FLOAT32:
      return "float";
    default:
      return "FLT";
  }
}

std::string GetDataType4(DataType type) { return GetDataType(type) + "4"; }

std::string GetTensorDeclaration(TensorStorageType storage_type,
                                 AccessType access, DataType data_type) {
  switch (storage_type) {
    case TensorStorageType::BUFFER:
      return absl::StrCat("__global ", GetDataType4(data_type), "*");
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return GetImageModifier(access) + " image2d_t";
    case TensorStorageType::TEXTURE_ARRAY:
      return GetImageModifier(access) + " image2d_array_t";
    case TensorStorageType::UNKNOWN:
      return "";
  }
}

std::string GetTensorDeclaration(TensorStorageType storage_type,
                                 const std::string& tensor_name,
                                 AccessType access, DataType data_type) {
  return absl::StrCat(GetTensorDeclaration(storage_type, access, data_type),
                      " ", tensor_name);
}

std::string GenerateGlobal3DCoords(TensorStorageType storage_type) {
  std::string code;
  switch (storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::SINGLE_TEXTURE_2D:
      code += "  int X = get_global_id(0);\n";
      code += "  int Y = get_global_id(1);\n";
      code += "  int Z = get_global_id(2);\n";
      break;
    case TensorStorageType::UNKNOWN:
      return "";
  }

  return code;
}

TensorCodeGenerator::TensorCodeGenerator(const std::string& name,
                                         const std::string& uniform_size_name,
                                         TensorStorageType storage_type,
                                         AccessType access)
    : name_(name),
      uniform_size_name_(uniform_size_name),
      storage_type_(storage_type),
      access_(access) {}

TensorCodeGenerator::TensorCodeGenerator(const std::string& name,
                                         const std::string& uniform_size_name,
                                         const TensorDescriptor& descriptor)
    : name_(name),
      uniform_size_name_(uniform_size_name),
      storage_type_(descriptor.storage_type),
      data_type_(descriptor.data_type) {}

std::string TensorCodeGenerator::GetDeclaration() const {
  return GetTensorDeclaration(storage_type_, name_, access_, data_type_);
}

std::string TensorCodeGenerator::GetDeclaration(AccessType access_type) const {
  return GetTensorDeclaration(storage_type_, name_, access_type, data_type_);
}

std::string TensorCodeGenerator::Read3D(const std::string& x,
                                        const std::string& y,
                                        const std::string& z) const {
  return ReadGlobalFLT4(storage_type_, data_type_, name_, uniform_size_name_, x,
                        y, z);
}

std::string TensorCodeGenerator::ReadAsFloat3D(const std::string& x,
                                               const std::string& y,
                                               const std::string& z) const {
  return ReadGlobalFloat4(storage_type_, name_, uniform_size_name_, x, y, z);
}

std::string TensorCodeGenerator::Read3D(
    const std::string& global_address) const {
  return ReadGlobalFLT4(storage_type_, data_type_, name_, global_address);
}

std::string TensorCodeGenerator::ReadAsFloat3D(
    const std::string& global_address) const {
  return ReadGlobalFloat4(storage_type_, name_, global_address);
}

std::string TensorCodeGenerator::GetAddress(const std::string& var_name,
                                            const std::string& x,
                                            const std::string& y,
                                            const std::string& z) const {
  return GetGlobalAddress(storage_type_, uniform_size_name_, var_name, x, y, z);
}

std::string TensorCodeGenerator::Write3D(const std::string& var_name,
                                         const std::string& x,
                                         const std::string& y,
                                         const std::string& z) const {
  return WriteGlobalFLT4(storage_type_, data_type_, name_, uniform_size_name_,
                         var_name, x, y, z);
}

std::string TensorCodeGenerator::Write3D(
    const std::string& var_name, const std::string& global_address) const {
  return WriteGlobalFLT4(storage_type_, data_type_, name_, var_name,
                         global_address);
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
