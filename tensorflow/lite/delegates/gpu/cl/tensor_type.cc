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

#include "tensorflow/lite/delegates/gpu/cl/tensor_type.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"

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

}  // namespace

std::string TextureAddressModeToString(TextureAddressMode address_mode) {
  switch (address_mode) {
    case TextureAddressMode::DONT_CARE:
      return "smp_none";
    case TextureAddressMode::ZERO:
      return "smp_zero";
  }
}

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

GPUResources TensorDescriptor::GetGPUResources() const {
  GPUResources resources;
  if (HasAxis(Axis::WIDTH)) {
    resources.ints.push_back("width");
    resources.ints.push_back("width_div2");
    resources.ints.push_back("width_div4");
    resources.ints.push_back("width_batched");
    resources.ints.push_back("width_batched_div2");
    resources.ints.push_back("width_batched_div4");
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
    const std::string& selector, const std::vector<std::string>& args,
    const std::vector<std::string>& template_args, std::string* result) const {
  if (selector == "Width") {
    *result = GetWidth();
    return absl::OkStatus();
  } else if (selector == "Height") {
    *result = "height";
    return absl::OkStatus();
  } else if (selector == "Slices") {
    *result = "slices";
    return absl::OkStatus();
  } else if (selector == "SliceStride") {
    *result = GetSliceStride();
    return absl::OkStatus();
  } else if (selector == "Channels") {
    *result = "channels";
    return absl::OkStatus();
  } else if (selector == "Batch") {
    *result = "batch";
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
    return PerformReadSelector(args, template_args, result);
  } else if (selector == "Write") {
    return PerformWriteSelector(args, result);
  } else if (selector == "WriteLinear") {
    return PerformWriteLinearSelector(args, result);
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
    const std::vector<std::string>& args,
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
      *result = Read(read_as_type, args[0]);
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

  *result =
      Read(read_as_type, GetGlobalAddressNoDeclaration(xc, yc, zc, sc, bc));
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
    const std::vector<std::string>& args, std::string* result) const {
  std::string xc;
  std::string yc;
  std::string zc;
  std::string sc;
  std::string bc;
  bool parsed = ParseCoordsFromArgs(args, 1, &xc, &yc, &zc, &sc, &bc);
  if (args.size() < 2 || !parsed) {
    return absl::NotFoundError("Unrecognized Write selector");
  }
  *result = Write(args[0], GetGlobalAddressNoDeclaration(xc, yc, zc, sc, bc));
  return absl::OkStatus();
}

absl::Status TensorDescriptor::PerformWriteLinearSelector(
    const std::vector<std::string>& args, std::string* result) const {
  if (storage_type != TensorStorageType::BUFFER &&
      storage_type != TensorStorageType::IMAGE_BUFFER) {
    return absl::InvalidArgumentError(
        "WriteLinear selector can be used only with linear "
        "storages(BUFFER/IMAGE_BUFFER)");
  }
  if (args.size() != 2) {
    return absl::NotFoundError("Unrecognized WriteLinear selector");
  }
  *result = Write(args[0], "(" + args[1] + ")");
  return absl::OkStatus();
}

std::string TensorDescriptor::Read(DataType read_as_type,
                                   const std::string& global_address) const {
  const std::string read_as =
      read_as_type == DataType::FLOAT16 ? "read_imageh" : "read_imagef";
  std::string image_type;
  if (storage_type == TensorStorageType::TEXTURE_2D ||
      storage_type == TensorStorageType::SINGLE_TEXTURE_2D) {
    image_type = "image2d";
  } else if (storage_type == TensorStorageType::TEXTURE_3D) {
    image_type = "image3d";
  } else if (storage_type == TensorStorageType::TEXTURE_ARRAY) {
    image_type = "image2d_array";
  }
  switch (storage_type) {
    case TensorStorageType::BUFFER:
      if (read_as_type == data_type) {
        return absl::StrCat("buffer[", global_address, "]");
      } else {
        const std::string conversion = read_as_type == DataType::FLOAT16
                                           ? "convert_half4"
                                           : "convert_float4";
        return absl::StrCat(conversion, "(buffer[", global_address, "])");
      }
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::TEXTURE_3D:
    case TensorStorageType::SINGLE_TEXTURE_2D:
    case TensorStorageType::TEXTURE_ARRAY:
      return absl::StrCat(
          read_as, "(", image_type,
          ", " + TextureAddressModeToString(ModeFromState()) + ", ",
          global_address, ")");
    case TensorStorageType::IMAGE_BUFFER:
      return absl::StrCat(read_as, "(image_buffer, ", global_address, ")");
    case TensorStorageType::UNKNOWN:
      return "";
  }
}

std::string TensorDescriptor::Write(const std::string& var_name,
                                    const std::string& global_address) const {
  std::string image_type;
  if (storage_type == TensorStorageType::TEXTURE_2D ||
      storage_type == TensorStorageType::SINGLE_TEXTURE_2D) {
    image_type = "image2d";
  } else if (storage_type == TensorStorageType::TEXTURE_3D) {
    image_type = "image3d";
  } else if (storage_type == TensorStorageType::TEXTURE_ARRAY) {
    image_type = "image2d_array";
  }
  switch (storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      return absl::StrCat("buffer[", global_address, "] = ", var_name, ";\n");
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::TEXTURE_3D:
    case TensorStorageType::SINGLE_TEXTURE_2D:
    case TensorStorageType::TEXTURE_ARRAY:
      return absl::StrCat(GetWriteImageFromDataType(data_type), "(", image_type,
                          ", ", global_address, ", ", var_name, ");\n");
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
  *result = absl::StrCat("buffer + ", args[0], " * ", GetSliceStride());
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
  *result = absl::StrCat(args[1], " * ", GetWidth(), " + ", args[0]);
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

std::string TensorDescriptor::GetGlobalAddressNoDeclarationWHS(
    const std::string& x, const std::string& y, const std::string& s) const {
  switch (storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER: {
      return absl::Substitute("((($2) * height + ($1)) * $3 + ($0))", x, y, s,
                              GetWidth());
    }
    case TensorStorageType::TEXTURE_2D:
      return absl::Substitute("(int2)(($0), ($1) * slices + ($2))", x, y, s);
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return absl::StrCat("(int2)(", x, ", ", y, ")");
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_3D:
      return absl::StrCat("(int4)(", x, ", ", y, ", ", s, ", 0)");
    case TensorStorageType::UNKNOWN:
      return "error";
  }
}

std::string TensorDescriptor::GetGlobalAddressNoDeclarationWHSB(
    const std::string& x, const std::string& y, const std::string& s,
    const std::string& b) const {
  switch (storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      return absl::Substitute(
          "(((($3) * height + $2) * width + ($1)) * batch + ($0))", b, x, y, s);
    case TensorStorageType::TEXTURE_2D:
      return absl::Substitute(
          "(int2)(($0) * batch + ($1), ($2) * slices + ($3))", x, b, y, s);
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return absl::Substitute("(int2)(($0) * batch + ($1), ($2))", x, b, y);
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_3D:
      return absl::Substitute("(int4)(($0) * batch + ($1), ($2), ($3), 0)", x,
                              b, y, s);
    case TensorStorageType::UNKNOWN:
      return "error";
    default:
      return "error";
  }
}

std::string TensorDescriptor::GetGlobalAddressNoDeclarationWHDS(
    const std::string& x, const std::string& y, const std::string& z,
    const std::string& s) const {
  switch (storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER: {
      return absl::Substitute(
          "(((($3) * slices + ($2)) * height + ($1)) * $4 + ($0))", x, y, s, z,
          GetWidth());
    }
    case TensorStorageType::TEXTURE_2D:
      return absl::Substitute(
          "(int2)(($0) * depth + ($1), ($2) * slices + ($3))", x, z, y, s);
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return absl::Substitute("(int2)(($0) * depth + ($1), ($2))", x, z, y);
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_3D:
      return absl::Substitute("(int4)(($0), ($1), ($2) * slices + ($3), 0)", x,
                              y, z, s);
    case TensorStorageType::UNKNOWN:
      return "error";
  }
}

std::string TensorDescriptor::GetGlobalAddressNoDeclarationWHDSB(
    const std::string& x, const std::string& y, const std::string& z,
    const std::string& s, const std::string& b) const {
  switch (storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      return absl::Substitute(
          "((((($4) * slices + ($3)) * height + $2) * width + ($1)) * batch + "
          "($0))",
          b, x, y, s, z);
    case TensorStorageType::TEXTURE_2D:
      return absl::Substitute(
          "(int2)((($0) * batch + ($1)) * depth + ($2), ($3) * slices + ($4))",
          x, b, z, y, s);
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return absl::Substitute(
          "(int2)((($0) * batch + ($1)) * depth + ($2), ($3))", x, b, z, y);
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_3D:
      return absl::Substitute(
          "(int4)(($0) * batch + ($1), ($2), ($3) * slices + ($4), 0)", x, b, y,
          z, s);
    case TensorStorageType::UNKNOWN:
      return "error";
    default:
      return "error";
  }
}

std::string TensorDescriptor::GetGlobalAddressNoDeclaration(
    const std::string& xc, const std::string& yc, const std::string& zc,
    const std::string& sc, const std::string& bc) const {
  if (layout == Layout::HWC || (IsBatchedWidth() && layout == Layout::BHWC)) {
    return GetGlobalAddressNoDeclarationWHS(xc, yc, sc);
  } else if (layout == Layout::BHWC) {
    return GetGlobalAddressNoDeclarationWHSB(xc, yc, sc, bc);
  } else if (layout == Layout::HWDC ||
             (IsBatchedWidth() && layout == Layout::BHWDC)) {
    return GetGlobalAddressNoDeclarationWHDS(xc, yc, zc, sc);
  } else if (layout == Layout::BHWDC) {
    return GetGlobalAddressNoDeclarationWHDSB(xc, yc, zc, sc, bc);
  } else {
    return "Unsupported layout";
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

void TensorDescriptor::SetTextureAddressMode(TextureAddressMode mode) {
  if (mode == TextureAddressMode::ZERO) {
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

std::string TensorDescriptor::GetWidth() const {
  std::string div;
  auto it1 = state_vars_.find("ElementsX2");
  if (it1 != state_vars_.end() && it1->second == "true") {
    div = "_div2";
  }
  auto it2 = state_vars_.find("ElementsX4");
  if (it2 != state_vars_.end() && it2->second == "true") {
    div = "_div4";
  }
  auto it = state_vars_.find("BatchedWidth");
  if (it != state_vars_.end() && it->second == "true") {
    return "width_batched" + div;
  } else {
    return "width" + div;
  }
}

std::string TensorDescriptor::GetSliceStride() const {
  if (IsBatchedWidth()) {
    return GetWidth() + " * height";
  } else {
    if (HasAxis(Axis::BATCH)) {
      return GetWidth() + " * height * batch";
    } else {
      return GetWidth() + " * height";
    }
  }
}

TextureAddressMode TensorDescriptor::ModeFromState() const {
  auto it = state_vars_.find("TextureMode");
  if (it != state_vars_.end()) {
    if (it->second == "ZERO") {
      return TextureAddressMode::ZERO;
    } else {
      return TextureAddressMode::DONT_CARE;
    }
  } else {
    return TextureAddressMode::DONT_CARE;
  }
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
