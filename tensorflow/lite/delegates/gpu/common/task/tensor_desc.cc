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

#include "tensorflow/lite/delegates/gpu/common/task/tensor_desc.h"

#include <cstdint>

#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
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

std::string AddressModeToCLSampler(AddressMode address_mode) {
  switch (address_mode) {
    case AddressMode::kDontCare:
      return "smp_none";
    case AddressMode::kZero:
      return "smp_zero";
  }
}

}  // namespace

std::string ToString(TensorStorageType type) {
  switch (type) {
    case TensorStorageType::UNKNOWN:
      return "TensorStorageType::UNKNOWN";
    case TensorStorageType::BUFFER:
      return "TensorStorageType::BUFFER";
    case TensorStorageType::TEXTURE_ARRAY:
      return "TensorStorageType::TEXTURE_ARRAY";
    case TensorStorageType::TEXTURE_2D:
      return "TensorStorageType::TEXTURE_2D";
    case TensorStorageType::TEXTURE_3D:
      return "TensorStorageType::TEXTURE_3D";
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return "TensorStorageType::SINGLE_TEXTURE_2D";
    case TensorStorageType::IMAGE_BUFFER:
      return "TensorStorageType::IMAGE_BUFFER";
  }
}

TensorDescriptor::TensorDescriptor(TensorDescriptor&& desc)
    : GPUObjectDescriptor(std::move(desc)),
      data_type(desc.data_type),
      storage_type(desc.storage_type),
      layout(desc.layout),
      shape(desc.shape),
      data(std::move(desc.data)) {}
TensorDescriptor& TensorDescriptor::operator=(TensorDescriptor&& desc) {
  if (this != &desc) {
    std::swap(data_type, desc.data_type);
    std::swap(storage_type, desc.storage_type);
    std::swap(layout, desc.layout);
    std::swap(shape, desc.shape);
    data = std::move(desc.data);
    GPUObjectDescriptor::operator=(std::move(desc));
  }
  return *this;
}

GPUResources TensorDescriptor::GetGPUResources(const GpuInfo& gpu_info) const {
  GPUResources resources;
  resources.ints.push_back("slice_stride");
  if (HasAxis(Axis::WIDTH)) {
    resources.ints.push_back("width");
  }
  if (HasAxis(Axis::HEIGHT)) {
    resources.ints.push_back("height");
  }
  if (HasAxis(Axis::CHANNELS)) {
    resources.ints.push_back("slices");
    resources.ints.push_back("channels");
  }
  if (HasAxis(Axis::BATCH)) {
    resources.ints.push_back("batch");
  }
  if (HasAxis(Axis::DEPTH)) {
    resources.ints.push_back("depth");
  }
  if (storage_type == TensorStorageType::BUFFER) {
    GPUBufferDescriptor desc;
    desc.data_type = data_type;
    desc.access_type = access_type_;
    desc.element_size = 4;
    auto it1 = state_vars_.find("ElementsX2");
    if (it1 != state_vars_.end() && it1->second == "true") {
      desc.element_size = 8;
    }
    auto it2 = state_vars_.find("ElementsX4");
    if (it2 != state_vars_.end() && it2->second == "true") {
      desc.element_size = 16;
    }
    resources.buffers.push_back({"buffer", desc});
  } else if (storage_type == TensorStorageType::SINGLE_TEXTURE_2D ||
             storage_type == TensorStorageType::TEXTURE_2D) {
    GPUImage2DDescriptor desc;
    desc.data_type = data_type;
    desc.normalized = false;
    desc.access_type = access_type_;
    resources.images2d.push_back({"image2d", desc});
  } else if (storage_type == TensorStorageType::TEXTURE_ARRAY) {
    GPUImage2DArrayDescriptor desc;
    desc.data_type = data_type;
    desc.access_type = access_type_;
    resources.image2d_arrays.push_back({"image2d_array", desc});
  } else if (storage_type == TensorStorageType::TEXTURE_3D) {
    GPUImage3DDescriptor desc;
    desc.data_type = data_type;
    desc.access_type = access_type_;
    resources.images3d.push_back({"image3d", desc});
  } else if (storage_type == TensorStorageType::IMAGE_BUFFER) {
    if (access_type_ == AccessType::READ) {
      GPUImageBufferDescriptor desc;
      desc.data_type = data_type;
      desc.access_type = access_type_;
      resources.image_buffers.push_back({"image_buffer", desc});
    } else {
      GPUBufferDescriptor desc;
      desc.data_type = data_type;
      desc.access_type = access_type_;
      desc.element_size = 4;
      resources.buffers.push_back({"buffer", desc});
    }
  }
  return resources;
}

absl::Status TensorDescriptor::PerformSelector(
    const GpuInfo& gpu_info, const std::string& selector,
    const std::vector<std::string>& args,
    const std::vector<std::string>& template_args, std::string* result) const {
  if (selector == "Width") {
    *result = "width";
    return absl::OkStatus();
  } else if (selector == "Height") {
    *result = "height";
    return absl::OkStatus();
  } else if (selector == "Slices") {
    *result = "slices";
    return absl::OkStatus();
  } else if (selector == "SliceStride") {
    *result = "slice_stride";
    return absl::OkStatus();
  } else if (selector == "Channels") {
    *result = "channels";
    return absl::OkStatus();
  } else if (selector == "Batch") {
    if (HasAxis(Axis::BATCH)) {
      *result = "batch";
    } else {
      *result = "1";
    }
    return absl::OkStatus();
  } else if (selector == "Depth") {
    *result = "depth";
    return absl::OkStatus();
  } else if (selector == "SetBatchRef") {
    if (args.size() != 1) {
      return absl::InvalidArgumentError(
          "Unsupported arguments in SetBatchRef selector");
    }
    state_vars_["batch_id"] = args[0];
    *result = "";
    return absl::OkStatus();
  } else if (selector == "Read") {
    return PerformReadSelector(gpu_info, args, template_args, result);
  } else if (selector == "Write") {
    return PerformWriteSelector(gpu_info, args, result);
  } else if (selector == "WriteLinear") {
    return PerformWriteLinearSelector(gpu_info, args, result);
  } else if (selector == "Write2D") {
    return PerformWrite2DSelector(gpu_info, args, result);
  } else if (selector == "GetAddress") {
    return PerformGetAddressSelector(args, result);
  } else if (selector == "GetPtrWithSliceOffset") {
    return PerformGetPtrWithSliceOffsetSelector(args, result);
  } else if (selector == "GetWHOffset") {
    return PerformGetWHOffsetSelector(args, result);
  } else if (selector == "GetHandle") {
    return PerformGetHandleSelector(args, result);
  } else {
    return absl::NotFoundError(absl::StrCat(
        "TensorDescriptor don't have selector with name - ", selector));
  }
}

absl::Status TensorDescriptor::PerformReadSelector(
    const GpuInfo& gpu_info, const std::vector<std::string>& args,
    const std::vector<std::string>& template_args, std::string* result) const {
  DataType read_as_type = data_type;
  if (!template_args.empty()) {
    if (template_args.size() != 1) {
      return absl::NotFoundError(
          "Unrecognized Read selector template arguments.");
    } else {
      RETURN_IF_ERROR(
          GetDataTypeFromTemplateArgs(template_args[0], &read_as_type));
    }
  }
  if (args.size() == 1) {  // function overload for 1D linear types.
    if (storage_type == TensorStorageType::BUFFER ||
        storage_type == TensorStorageType::IMAGE_BUFFER) {
      *result = Read(gpu_info, read_as_type, {args[0]});
      return absl::OkStatus();
    } else {
      return absl::InvalidArgumentError(
          "Read selector with single argument can be used only with linear "
          "storage types(BUFFER or IMAGE_BUFFER)");
    }
  }
  std::string xc;
  std::string yc;
  std::string zc;
  std::string sc;
  std::string bc;
  bool parsed = ParseCoordsFromArgs(args, 0, &xc, &yc, &zc, &sc, &bc);
  if (args.size() < 2 || !parsed) {
    return absl::NotFoundError("Unrecognized Read selector");
  }

  *result = Read(gpu_info, read_as_type, GetPhysicalCoords(xc, yc, zc, sc, bc));
  return absl::OkStatus();
}

absl::Status TensorDescriptor::GetLinkingContextFromWriteSelector(
    const std::vector<std::string>& args, std::string* value_name,
    std::string* x_coord, std::string* y_coord, std::string* s_coord) const {
  std::string xc;
  std::string yc;
  std::string zc;
  std::string sc;
  std::string bc;
  bool parsed = ParseCoordsFromArgs(args, 1, &xc, &yc, &zc, &sc, &bc);
  if (args.size() < 2 || !parsed) {
    return absl::NotFoundError("Unrecognized Write selector");
  }
  *value_name = args[0];
  if (HasAxis(Axis::BATCH) && !IsBatchedWidth()) {
    *x_coord = absl::StrCat("((", xc, ") * batch + (", bc, "))");
  } else {
    *x_coord = absl::StrCat("(", xc, ")");
  }
  *y_coord = absl::StrCat("(", yc, ")");
  *s_coord = absl::StrCat("(", sc, ")");
  return absl::OkStatus();
}

absl::Status TensorDescriptor::PerformWriteSelector(
    const GpuInfo& gpu_info, const std::vector<std::string>& args,
    std::string* result) const {
  std::string xc;
  std::string yc;
  std::string zc;
  std::string sc;
  std::string bc;
  bool parsed = ParseCoordsFromArgs(args, 1, &xc, &yc, &zc, &sc, &bc);
  if (args.size() < 2 || !parsed) {
    return absl::NotFoundError("Unrecognized Write selector");
  }
  *result = Write(gpu_info, args[0], GetPhysicalCoords(xc, yc, zc, sc, bc));
  return absl::OkStatus();
}

absl::Status TensorDescriptor::PerformWriteLinearSelector(
    const GpuInfo& gpu_info, const std::vector<std::string>& args,
    std::string* result) const {
  if (storage_type != TensorStorageType::BUFFER &&
      storage_type != TensorStorageType::IMAGE_BUFFER) {
    return absl::InvalidArgumentError(
        "WriteLinear selector can be used only with linear "
        "storages(BUFFER/IMAGE_BUFFER)");
  }
  if (args.size() != 2) {
    return absl::NotFoundError("Unrecognized WriteLinear selector");
  }
  *result = Write(gpu_info, args[0], {args[1]});
  return absl::OkStatus();
}

absl::Status TensorDescriptor::PerformWrite2DSelector(
    const GpuInfo& gpu_info, const std::vector<std::string>& args,
    std::string* result) const {
  if (storage_type != TensorStorageType::TEXTURE_2D) {
    return absl::InvalidArgumentError(
        "Write2D selector can be used only with 2d "
        "storages(TEXTURE_2D)");
  }
  if (args.size() != 3) {
    return absl::NotFoundError("Unrecognized Write2D selector");
  }
  *result = Write(gpu_info, args[0], {args[1], args[2]});
  return absl::OkStatus();
}

std::string TensorDescriptor::Read(
    const GpuInfo& gpu_info, DataType read_as_type,
    const std::vector<std::string>& coords) const {
  const std::string read_as =
      read_as_type == DataType::FLOAT16 ? "read_imageh" : "read_imagef";
  const bool need_conversion = read_as_type != data_type;
  const std::string metal_type =
      read_as_type == DataType::FLOAT32 ? "float4" : "half4";
  switch (storage_type) {
    case TensorStorageType::BUFFER:
      if (read_as_type == data_type) {
        return absl::StrCat("buffer[", coords[0], "]");
      } else {
        std::string conversion;
        if (gpu_info.IsApiMetal()) {
          conversion = metal_type;
        } else if (gpu_info.IsApiOpenCl()) {
          if (read_as_type == DataType::FLOAT16) {
            conversion = "convert_half4";
          } else if (read_as_type == DataType::FLOAT32) {
            conversion = "convert_float4";
          }
        }
        return absl::StrCat(conversion, "(buffer[", coords[0], "])");
      }
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::SINGLE_TEXTURE_2D:
      if (gpu_info.IsApiOpenCl()) {
        return absl::Substitute("$0(image2d, $1, (int2)($2, $3))", read_as,
                                AddressModeToCLSampler(AddressModeFromState()),
                                coords[0], coords[1]);
      } else if (gpu_info.IsApiMetal()) {
        std::string result = absl::Substitute("image2d.read(ushort2($0, $1))",
                                              coords[0], coords[1]);
        if (need_conversion) {
          result = metal_type + "(" + result + ")";
        }
        return result;
      } else {
        return "";
      }
    case TensorStorageType::TEXTURE_3D:
      if (gpu_info.IsApiOpenCl()) {
        return absl::Substitute("$0(image3d, $1, (int4)($2, $3, $4, 0))",
                                read_as,
                                AddressModeToCLSampler(AddressModeFromState()),
                                coords[0], coords[1], coords[2]);
      } else if (gpu_info.IsApiMetal()) {
        std::string result =
            absl::Substitute("image3d.read(ushort3($0, $1, $2))", coords[0],
                             coords[1], coords[2]);
        if (need_conversion) {
          result = metal_type + "(" + result + ")";
        }
        return result;
      } else {
        return "";
      }
    case TensorStorageType::TEXTURE_ARRAY:
      if (gpu_info.IsApiOpenCl()) {
        return absl::Substitute("$0(image2d_array, $1, (int4)($2, $3, $4, 0))",
                                read_as,
                                AddressModeToCLSampler(AddressModeFromState()),
                                coords[0], coords[1], coords[2]);
      } else if (gpu_info.IsApiMetal()) {
        std::string result =
            absl::Substitute("image2d_array.read(ushort2($0, $1), $2)",
                             coords[0], coords[1], coords[2]);
        if (need_conversion) {
          result = metal_type + "(" + result + ")";
        }
        return result;
      } else {
        return "";
      }
    case TensorStorageType::IMAGE_BUFFER:
      if (gpu_info.IsApiOpenCl()) {
        return absl::StrCat(read_as, "(image_buffer, ", coords[0], ")");
      } else if (gpu_info.IsApiMetal()) {
        std::string result =
            absl::Substitute("image_buffer.read(uint($0))", coords[0]);
        if (need_conversion) {
          result = metal_type + "(" + result + ")";
        }
        return result;
      } else {
        return "";
      }
    case TensorStorageType::UNKNOWN:
      return "";
  }
}

std::string TensorDescriptor::Write(
    const GpuInfo& gpu_info, const std::string& var_name,
    const std::vector<std::string>& coords) const {
  switch (storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      return absl::StrCat("buffer[", coords[0], "] = ", var_name);
    case TensorStorageType::SINGLE_TEXTURE_2D:
    case TensorStorageType::TEXTURE_2D:
      if (gpu_info.IsApiOpenCl()) {
        return absl::Substitute("$0(image2d, (int2)($1, $2), $3)",
                                GetWriteImageFromDataType(data_type), coords[0],
                                coords[1], var_name);
      } else if (gpu_info.IsApiMetal()) {
        return absl::Substitute("image2d.write($0, ushort2($1, $2))", var_name,
                                coords[0], coords[1]);
      } else {
        return "";
      }
    case TensorStorageType::TEXTURE_3D:
      if (gpu_info.IsApiOpenCl()) {
        return absl::Substitute("$0(image3d, (int4)($1, $2, $3, 0), $4)",
                                GetWriteImageFromDataType(data_type), coords[0],
                                coords[1], coords[2], var_name);
      } else if (gpu_info.IsApiMetal()) {
        return absl::Substitute("image3d.write($0, ushort3($1, $2, $3))",
                                var_name, coords[0], coords[1], coords[2]);
      } else {
        return "";
      }
    case TensorStorageType::TEXTURE_ARRAY:
      if (gpu_info.IsApiOpenCl()) {
        return absl::Substitute("$0(image2d_array, (int4)($1, $2, $3, 0), $4)",
                                GetWriteImageFromDataType(data_type), coords[0],
                                coords[1], coords[2], var_name);
      } else if (gpu_info.IsApiMetal()) {
        return absl::Substitute("image2d_array.write($0, ushort2($1, $2), $3)",
                                var_name, coords[0], coords[1], coords[2]);
      } else {
        return "";
      }
    case TensorStorageType::UNKNOWN:
      return "";
  }
}

absl::Status TensorDescriptor::PerformGetAddressSelector(
    const std::vector<std::string>& args, std::string* result) const {
  std::string xc;
  std::string yc;
  std::string zc;
  std::string sc;
  std::string bc;
  bool parsed = ParseCoordsFromArgs(args, 1, &xc, &yc, &zc, &sc, &bc);
  if (args.size() < 3 || !parsed) {
    return absl::NotFoundError("Unrecognized GetAddress selector");
  }

  *result = DeclareAddress(args[0],
                           GetGlobalAddressNoDeclaration(xc, yc, zc, sc, bc));
  return absl::OkStatus();
}

absl::Status TensorDescriptor::PerformGetPtrWithSliceOffsetSelector(
    const std::vector<std::string>& args, std::string* result) const {
  if (storage_type != TensorStorageType::BUFFER) {
    return absl::InvalidArgumentError(
        "GetPtrWithSliceOffset selector can be used only with BUFFER");
  }
  if (args.size() != 1) {
    return absl::NotFoundError(absl::StrCat(
        "GetPtrWithSliceOffset require one argument(slice coordinate), but ",
        args.size(), " was passed"));
  }
  *result = absl::StrCat("buffer + ", args[0], " * slice_stride");
  return absl::OkStatus();
}

absl::Status TensorDescriptor::PerformGetWHOffsetSelector(
    const std::vector<std::string>& args, std::string* result) const {
  if (storage_type != TensorStorageType::BUFFER &&
      storage_type != TensorStorageType::IMAGE_BUFFER) {
    return absl::InvalidArgumentError(
        "GetWHOffset selector can be used only with BUFFER/IMAGE_BUFFER");
  }
  if (args.size() != 2) {
    return absl::NotFoundError(absl::StrCat(
        "GetWHOffset require two arguments(X and Y coordinates), but ",
        args.size(), " was passed"));
  }
  if (HasAxis(Axis::BATCH) && !IsBatchedWidth()) {
    auto it = state_vars_.find("batch_id");
    std::string batch_id;
    if (it == state_vars_.end()) {
      return absl::NotFoundError(
          "Not found batch_id. Should be setted up by SetBatchRef(). method");
    } else {
      batch_id = it->second;
    }
    *result = absl::StrCat("((", args[1], ") * width + (", args[0],
                           ")) * batch + (", batch_id, ")");
  } else {
    *result = absl::StrCat("(", args[1], ") * width + (", args[0], ")");
  }
  return absl::OkStatus();
}

absl::Status TensorDescriptor::PerformGetHandleSelector(
    const std::vector<std::string>& args, std::string* result) const {
  if (!args.empty()) {
    return absl::NotFoundError(
        absl::StrCat("GetHandle does not require arguments, but ", args.size(),
                     " was passed"));
  }
  switch (storage_type) {
    case TensorStorageType::BUFFER:
      *result = "buffer";
      return absl::OkStatus();
    case TensorStorageType::IMAGE_BUFFER:
      if (access_type_ == AccessType::READ) {
        *result = "image_buffer";
      } else {
        *result = "buffer";
      }
      return absl::OkStatus();
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::SINGLE_TEXTURE_2D:
      *result = "image2d";
      return absl::OkStatus();
    case TensorStorageType::TEXTURE_ARRAY:
      *result = "image2d_array";
      return absl::OkStatus();
    case TensorStorageType::TEXTURE_3D:
      *result = "image3d";
      return absl::OkStatus();
    case TensorStorageType::UNKNOWN:
      return absl::UnavailableError("Unknown type");
  }
}

std::string TensorDescriptor::DeclareAddress(const std::string& var_name,
                                             const std::string& address) const {
  return absl::StrCat(StorageTypeToAddressType(), " ", var_name, " = ", address,
                      ";");
}

std::string TensorDescriptor::StorageTypeToAddressType() const {
  switch (storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      return "int";
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return "int2";
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_3D:
      return "int4";
    case TensorStorageType::UNKNOWN:
      return "";
  }
}

std::vector<std::string> TensorDescriptor::GetPhysicalCoordsWHS(
    const std::string& x, const std::string& y, const std::string& s) const {
  switch (storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      return {
          absl::Substitute("((($2) * height + ($1)) * width + ($0))", x, y, s)};
    case TensorStorageType::TEXTURE_2D:
      return {absl::Substitute("($0)", x),
              absl::Substitute("(($0) * slices + ($1))", y, s)};
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return {absl::Substitute("($0)", x), absl::Substitute("($0)", y)};
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_3D:
      return {absl::Substitute("($0)", x), absl::Substitute("($0)", y),
              absl::Substitute("($0)", s)};
    case TensorStorageType::UNKNOWN:
      return {""};
    default:
      return {""};
  }
}

std::vector<std::string> TensorDescriptor::GetPhysicalCoordsWHSB(
    const std::string& x, const std::string& y, const std::string& s,
    const std::string& b) const {
  switch (storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      return {absl::Substitute(
          "(((($3) * height + $2) * width + ($1)) * batch + ($0))", b, x, y,
          s)};
    case TensorStorageType::TEXTURE_2D:
      return {absl::Substitute("(($0) * batch + ($1))", x, b),
              absl::Substitute("(($0) * slices + ($1))", y, s)};
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return {absl::Substitute("(($0) * batch + ($1))", x, b),
              absl::Substitute("($0)", y)};
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_3D:
      return {absl::Substitute("(($0) * batch + ($1))", x, b),
              absl::Substitute("($0)", y), absl::Substitute("($0)", s)};
    case TensorStorageType::UNKNOWN:
      return {""};
    default:
      return {""};
  }
}

std::vector<std::string> TensorDescriptor::GetPhysicalCoordsWHDS(
    const std::string& x, const std::string& y, const std::string& z,
    const std::string& s) const {
  switch (storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      return {absl::Substitute(
          "(((($3) * slices + ($2)) * height + ($1)) * width + ($0))", x, y, s,
          z)};
    case TensorStorageType::TEXTURE_2D:
      return {absl::Substitute("(($0) * depth + ($1))", x, z),
              absl::Substitute("(($0) * slices + ($1))", y, s)};
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return {absl::Substitute("(($0) * depth + ($1))", x, z),
              absl::Substitute("($0)", y)};
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_3D:
      return {absl::Substitute("($0)", x), absl::Substitute("($0)", y),
              absl::Substitute("(($0) * slices + ($1))", z, s)};
    case TensorStorageType::UNKNOWN:
      return {""};
    default:
      return {""};
  }
}

std::vector<std::string> TensorDescriptor::GetPhysicalCoordsWHDSB(
    const std::string& x, const std::string& y, const std::string& z,
    const std::string& s, const std::string& b) const {
  switch (storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      return {absl::Substitute(
          "((((($4) * slices + ($3)) * height + $2) * width + ($1)) * batch + "
          "($0))",
          b, x, y, s, z)};
    case TensorStorageType::TEXTURE_2D:
      return {absl::Substitute("((($0)*batch + ($1))*depth + ($2))", x, b, z),
              absl::Substitute("(($0) * slices + ($1))", y, s)};
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return {absl::Substitute("((($0)*batch + ($1))*depth + ($2))", x, b, z),
              absl::Substitute("($0)", y)};
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_3D:
      return {absl::Substitute("(($0) * batch + ($1))", x, b),
              absl::Substitute("($0)", y),
              absl::Substitute("(($0) * slices + ($1))", z, s)};
    case TensorStorageType::UNKNOWN:
      return {""};
    default:
      return {""};
  }
}

std::string TensorDescriptor::GetGlobalAddressNoDeclaration(
    const std::string& xc, const std::string& yc, const std::string& zc,
    const std::string& sc, const std::string& bc) const {
  auto coords = GetPhysicalCoords(xc, yc, zc, sc, bc);
  switch (storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER: {
      return coords[0];
    }
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return absl::Substitute("(int2)($0, $1)", coords[0], coords[1]);
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_3D:
      return absl::Substitute("(int4)($0, $1, $2, 0)", coords[0], coords[1],
                              coords[2]);
    case TensorStorageType::UNKNOWN:
      return "error";
  }
}

std::vector<std::string> TensorDescriptor::GetPhysicalCoords(
    const std::string& xc, const std::string& yc, const std::string& zc,
    const std::string& sc, const std::string& bc) const {
  if (layout == Layout::HWC || (IsBatchedWidth() && layout == Layout::BHWC)) {
    return GetPhysicalCoordsWHS(xc, yc, sc);
  } else if (layout == Layout::BHWC) {
    return GetPhysicalCoordsWHSB(xc, yc, sc, bc);
  } else if (layout == Layout::HWDC ||
             (IsBatchedWidth() && layout == Layout::BHWDC)) {
    return GetPhysicalCoordsWHDS(xc, yc, zc, sc);
  } else if (layout == Layout::BHWDC) {
    return GetPhysicalCoordsWHDSB(xc, yc, zc, sc, bc);
  } else {
    return {""};
  }
}

absl::Status TensorDescriptor::GetDataTypeFromTemplateArgs(
    const std::string& template_arg, DataType* result) const {
  std::string read_type = template_arg;
  if (read_type == "FLT" || read_type == "ACCUM_FLT") {
    auto it = state_vars_.find(read_type);
    if (it == state_vars_.end()) {
      return absl::UnavailableError(absl::StrCat(
          "Read selector template argument ", read_type, " uninitialized."));
    } else {
      read_type = it->second;
    }
  }

  if (read_type == "half") {
    *result = DataType::FLOAT16;
  } else if (read_type == "float") {
    *result = DataType::FLOAT32;
  } else {
    return absl::NotFoundError(absl::StrCat(
        "Unrecognized Read selector template argument - ", read_type));
  }
  return absl::OkStatus();
}

bool TensorDescriptor::HasAxis(Axis axis) const {
  if (axis == Axis::WIDTH || axis == Axis::HEIGHT || axis == Axis::CHANNELS) {
    return true;
  }
  if (axis == Axis::BATCH &&
      (layout == Layout::BHWC || layout == Layout::BHWDC)) {
    return true;
  }
  if (axis == Axis::DEPTH &&
      (layout == Layout::HWDC || layout == Layout::BHWDC)) {
    return true;
  }
  return false;
}

int TensorDescriptor::GetWidthSize(BHWDC shape) const {
  int width = shape.w;
  auto it = state_vars_.find("BatchedWidth");
  if (it != state_vars_.end() && it->second == "true") {
    width *= shape.b;
  }
  auto it1 = state_vars_.find("ElementsX2");
  if (it1 != state_vars_.end() && it1->second == "true") {
    width /= 2;
  }
  auto it2 = state_vars_.find("ElementsX4");
  if (it2 != state_vars_.end() && it2->second == "true") {
    width /= 4;
  }
  return width;
}

int TensorDescriptor::GetSliceStrideSize(BHWDC shape) const {
  if (IsBatchedWidth()) {
    return GetWidthSize(shape) * shape.h;
  } else {
    if (HasAxis(Axis::BATCH)) {
      return GetWidthSize(shape) * shape.h * shape.b;
    } else {
      return GetWidthSize(shape) * shape.h;
    }
  }
}

void TensorDescriptor::SetAddressMode(AddressMode mode) {
  if (mode == AddressMode::kZero) {
    state_vars_["TextureMode"] = "ZERO";
  } else {
    state_vars_["TextureMode"] = "DONT_CARE";
  }
}

bool TensorDescriptor::ParseCoordsFromArgs(const std::vector<std::string>& args,
                                           int offset, std::string* xc,
                                           std::string* yc, std::string* zc,
                                           std::string* sc,
                                           std::string* bc) const {
  if (HasAxis(Axis::WIDTH)) {
    if (offset >= args.size()) return false;
    *xc = args[offset++];
  }
  if (HasAxis(Axis::HEIGHT)) {
    if (offset >= args.size()) return false;
    *yc = args[offset++];
  }
  if (HasAxis(Axis::DEPTH)) {
    if (offset >= args.size()) return false;
    *zc = args[offset++];
  }
  if (HasAxis(Axis::CHANNELS)) {
    if (offset >= args.size()) {
      auto it = state_vars_.find("slice_id");
      if (it == state_vars_.end()) {
        return false;
      } else {
        *sc = it->second;
      }
    } else {
      *sc = args[offset++];
    }
  }
  if (HasAxis(Axis::BATCH) && !IsBatchedWidth()) {
    if (offset >= args.size()) {
      auto it = state_vars_.find("batch_id");
      if (it == state_vars_.end()) {
        return false;
      } else {
        *bc = it->second;
      }
    } else {
      *bc = args[offset++];
    }
  }
  return true;
}

bool TensorDescriptor::IsBatchedWidth() const {
  auto it = state_vars_.find("BatchedWidth");
  return it != state_vars_.end() && it->second == "true";
}

AddressMode TensorDescriptor::AddressModeFromState() const {
  auto it = state_vars_.find("TextureMode");
  if (it != state_vars_.end()) {
    if (it->second == "ZERO") {
      return AddressMode::kZero;
    } else {
      return AddressMode::kDontCare;
    }
  } else {
    return AddressMode::kDontCare;
  }
}

void TensorDescriptor::UploadData(
    const tflite::gpu::Tensor<BHWC, DataType::FLOAT32>& src) {
  shape = BHWDC(src.shape.b, src.shape.h, src.shape.w, 1, src.shape.c);
  UploadData(src.data.data());
}

void TensorDescriptor::UploadData(
    const tflite::gpu::Tensor<HWC, DataType::FLOAT32>& src) {
  shape = BHWDC(1, src.shape.h, src.shape.w, 1, src.shape.c);
  UploadData(src.data.data());
}

void TensorDescriptor::UploadData(
    const tflite::gpu::Tensor<Linear, DataType::FLOAT32>& src) {
  shape = BHWDC(1, 1, 1, 1, src.shape.v);
  UploadData(src.data.data());
}

void TensorDescriptor::UploadData(const float* src) {
  int aligned_channels = storage_type == TensorStorageType::SINGLE_TEXTURE_2D
                             ? shape.c
                             : AlignByN(shape.c, 4);
  int elements_count = shape.b * shape.w * shape.h * shape.d * aligned_channels;
  data.resize(elements_count * SizeOf(data_type));
  if (data_type == DataType::FLOAT32) {
    float* gpu_data = reinterpret_cast<float*>(data.data());
    DataFromBHWDC(src, shape, *this, gpu_data);
  } else {
    half* gpu_data = reinterpret_cast<half*>(data.data());
    DataFromBHWDC(src, shape, *this, gpu_data);
  }
}

bool TensorDescriptor::SupportsZeroClamp(const Axis& axis) const {
  switch (storage_type) {
    case TensorStorageType::UNKNOWN:
      return false;
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      return false;
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return axis == Axis::WIDTH || axis == Axis::HEIGHT;
    case TensorStorageType::TEXTURE_3D:
      return axis == Axis::WIDTH || axis == Axis::HEIGHT || axis == Axis::DEPTH;
  }
}

bool TensorDescriptor::CanReadOutOfBorder(const Axis& axis) const {
  switch (storage_type) {
    case TensorStorageType::UNKNOWN:
      return false;
    case TensorStorageType::BUFFER:
      return false;
    case TensorStorageType::IMAGE_BUFFER:
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::TEXTURE_3D:
    case TensorStorageType::SINGLE_TEXTURE_2D:
    case TensorStorageType::TEXTURE_ARRAY:
      return true;
  }
}

bool TensorDescriptor::IsLinear() const {
  return storage_type == TensorStorageType::BUFFER ||
         storage_type == TensorStorageType::IMAGE_BUFFER;
}

bool TensorDescriptor::ReturnsZeroForNegOneRead() const {
  return storage_type == TensorStorageType::IMAGE_BUFFER;
}

namespace {
int GetLinearIndex(const TensorDescriptor& desc, const BHWDC& shape, int b,
                   int x, int y, int d, int s, int sub_c) {
  const int slices = DivideRoundUp(shape.c, 4);
  switch (desc.storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_3D:
      return ((((d * slices + s) * shape.h + y) * shape.w + x) * shape.b + b) *
                 4 +
             sub_c;  // DSHWBC4
    case TensorStorageType::TEXTURE_2D:
      return ((((y * slices + s) * shape.w + x) * shape.b + b) * shape.d + d) *
                 4 +
             sub_c;  // HSWBDC4
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return (((y * shape.w + x) * shape.b + b) * shape.d + d) * shape.c +
             sub_c;  // HWBDC
    case TensorStorageType::UNKNOWN:
      return -1;
  }
}

int GetChannelsAlignment(const TensorDescriptor& desc, const BHWDC& shape) {
  return desc.storage_type == TensorStorageType::SINGLE_TEXTURE_2D ? shape.c
                                                                   : 4;
}
}  // namespace

template <typename FromType, typename ToType>
void DataFromBHWDC(const FromType* src, const BHWDC& shape,
                   const TensorDescriptor& desc, ToType* dst) {
  const int channels_alignment = GetChannelsAlignment(desc, shape);
  const int slices = DivideRoundUp(shape.c, 4);
  for (int b = 0; b < shape.b; ++b) {
    for (int s = 0; s < slices; ++s) {
      for (int y = 0; y < shape.h; ++y) {
        for (int x = 0; x < shape.w; ++x) {
          for (int d = 0; d < shape.d; ++d) {
            for (int c = 0; c < channels_alignment; ++c) {
              FromType value;
              if (s * 4 + c < shape.c) {
                const int cpu_index =
                    shape.LinearIndex({b, y, x, d, s * 4 + c});
                value = src[cpu_index];
              } else {
                value = 0;
              }
              int gpu_index = GetLinearIndex(desc, shape, b, x, y, d, s, c);
              dst[gpu_index] = value;
            }
          }
        }
      }
    }
  }
}

template void DataFromBHWDC<float, float>(const float* src, const BHWDC& shape,
                                          const TensorDescriptor& desc,
                                          float* dst);
template void DataFromBHWDC<float, half>(const float* src, const BHWDC& shape,
                                         const TensorDescriptor& desc,
                                         half* dst);
template void DataFromBHWDC<int32_t, int32_t>(const int32_t* src,
                                              const BHWDC& shape,
                                              const TensorDescriptor& desc,
                                              int32_t* dst);
template void DataFromBHWDC<int16_t, int16_t>(const int16_t* src,
                                              const BHWDC& shape,
                                              const TensorDescriptor& desc,
                                              int16_t* dst);
template void DataFromBHWDC<int8_t, int8_t>(const int8_t* src,
                                            const BHWDC& shape,
                                            const TensorDescriptor& desc,
                                            int8_t* dst);
template void DataFromBHWDC<uint32_t, uint32_t>(const uint32_t* src,
                                                const BHWDC& shape,
                                                const TensorDescriptor& desc,
                                                uint32_t* dst);
template void DataFromBHWDC<uint16_t, uint16_t>(const uint16_t* src,
                                                const BHWDC& shape,
                                                const TensorDescriptor& desc,
                                                uint16_t* dst);
template void DataFromBHWDC<uint8_t, uint8_t>(const uint8_t* src,
                                              const BHWDC& shape,
                                              const TensorDescriptor& desc,
                                              uint8_t* dst);

template <typename FromType, typename ToType>
void DataToBHWDC(const FromType* src, const BHWDC& shape,
                 const TensorDescriptor& desc, ToType* dst) {
  const int channels_alignment = GetChannelsAlignment(desc, shape);
  const int slices = DivideRoundUp(shape.c, 4);
  for (int b = 0; b < shape.b; ++b) {
    for (int s = 0; s < slices; ++s) {
      for (int y = 0; y < shape.h; ++y) {
        for (int x = 0; x < shape.w; ++x) {
          for (int d = 0; d < shape.d; ++d) {
            for (int c = 0; c < channels_alignment; ++c) {
              if (s * 4 + c >= shape.c) {
                continue;
              }
              int cpu_index = shape.LinearIndex({b, y, x, d, s * 4 + c});
              int gpu_index = GetLinearIndex(desc, shape, b, x, y, d, s, c);
              dst[cpu_index] = src[gpu_index];
            }
          }
        }
      }
    }
  }
}

template void DataToBHWDC<float, float>(const float* src, const BHWDC& shape,
                                        const TensorDescriptor& desc,
                                        float* dst);
template void DataToBHWDC<half, float>(const half* src, const BHWDC& shape,
                                       const TensorDescriptor& desc,
                                       float* dst);
template void DataToBHWDC<int32_t, int32_t>(const int32_t* src,
                                            const BHWDC& shape,
                                            const TensorDescriptor& desc,
                                            int32_t* dst);
template void DataToBHWDC<int16_t, int16_t>(const int16_t* src,
                                            const BHWDC& shape,
                                            const TensorDescriptor& desc,
                                            int16_t* dst);
template void DataToBHWDC<int8_t, int8_t>(const int8_t* src, const BHWDC& shape,
                                          const TensorDescriptor& desc,
                                          int8_t* dst);
template void DataToBHWDC<uint32_t, uint32_t>(const uint32_t* src,
                                              const BHWDC& shape,
                                              const TensorDescriptor& desc,
                                              uint32_t* dst);
template void DataToBHWDC<uint16_t, uint16_t>(const uint16_t* src,
                                              const BHWDC& shape,
                                              const TensorDescriptor& desc,
                                              uint16_t* dst);
template void DataToBHWDC<uint8_t, uint8_t>(const uint8_t* src,
                                            const BHWDC& shape,
                                            const TensorDescriptor& desc,
                                            uint8_t* dst);

}  // namespace gpu
}  // namespace tflite
