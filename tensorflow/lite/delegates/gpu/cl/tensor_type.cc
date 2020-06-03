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
std::string GetGlobalAddressNoDeclarationWHS(const std::string& x,
                                             const std::string& y,
                                             const std::string& s,
                                             TensorStorageType storage_type) {
  switch (storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      return absl::Substitute("((($2) * height + ($1)) * width + ($0))", x, y,
                              s);
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

std::string GetGlobalAddressNoDeclarationWHSB(const std::string& x,
                                              const std::string& y,
                                              const std::string& s,
                                              const std::string& b,
                                              TensorStorageType storage_type) {
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

std::string GetGlobalAddressNoDeclarationWHDS(const std::string& x,
                                              const std::string& y,
                                              const std::string& z,
                                              const std::string& s,
                                              TensorStorageType storage_type) {
  switch (storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      return absl::Substitute(
          "(((($3) * slices + ($2)) * height + ($1)) * width + ($0))", x, y, s,
          z);
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

std::string GetGlobalAddressNoDeclarationWHDSB(const std::string& x,
                                               const std::string& y,
                                               const std::string& z,
                                               const std::string& s,
                                               const std::string& b,
                                               TensorStorageType storage_type) {
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

GPUResources TensorDescriptor::GetGPUResources(AccessType access_type) const {
  GPUResources resources;
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
    desc.access_type = access_type;
    desc.element_size = 4;
    resources.buffers.push_back({"buffer", desc});
  } else if (storage_type == TensorStorageType::SINGLE_TEXTURE_2D ||
             storage_type == TensorStorageType::TEXTURE_2D) {
    GPUImage2DDescriptor desc;
    desc.data_type = data_type;
    desc.access_type = access_type;
    resources.images2d.push_back({"image2d", desc});
  } else if (storage_type == TensorStorageType::TEXTURE_ARRAY) {
    GPUImage2DArrayDescriptor desc;
    desc.data_type = data_type;
    desc.access_type = access_type;
    resources.image2d_arrays.push_back({"image2d_array", desc});
  } else if (storage_type == TensorStorageType::TEXTURE_3D) {
    GPUImage3DDescriptor desc;
    desc.data_type = data_type;
    desc.access_type = access_type;
    resources.images3d.push_back({"image3d", desc});
  } else if (storage_type == TensorStorageType::IMAGE_BUFFER) {
    if (access_type == AccessType::READ) {
      GPUImageBufferDescriptor desc;
      desc.data_type = data_type;
      desc.access_type = access_type;
      resources.image_buffers.push_back({"image_buffer", desc});
    } else {
      GPUBufferDescriptor desc;
      desc.data_type = data_type;
      desc.access_type = access_type;
      desc.element_size = 4;
      resources.buffers.push_back({"buffer", desc});
    }
  }
  return resources;
}

absl::Status TensorDescriptor::PerformSelector(
    const std::string& selector, const std::vector<std::string>& args,
    std::string* result) const {
  if (selector == "Width") {
    *result = "width";
    return absl::OkStatus();
  } else if (selector == "Height") {
    *result = "height";
    return absl::OkStatus();
  } else if (selector == "Slices") {
    *result = "slices";
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
    return PerformReadSelector(args, result);
  } else if (selector == "Write") {
    return PerformWriteSelector(args, result);
  } else {
    return absl::NotFoundError(absl::StrCat(
        "TensorDescriptor don't have selector with name - ", selector));
  }
}

absl::Status TensorDescriptor::PerformReadSelector(
    const std::vector<std::string>& args, std::string* result) const {
  std::string xc;
  std::string yc;
  std::string zc;
  std::string sc;
  std::string bc;
  bool parsed = ParseCoordsFromArgs(args, 0, &xc, &yc, &zc, &sc, &bc);
  if (args.size() < 2 || !parsed) {
    return absl::NotFoundError("Unrecognized Read selector");
  }

  if (layout == Layout::HWC) {
    *result = Read(GetGlobalAddressNoDeclarationWHS(xc, yc, sc, storage_type));
    return absl::OkStatus();
  } else if (layout == Layout::BHWC) {
    *result =
        Read(GetGlobalAddressNoDeclarationWHSB(xc, yc, sc, bc, storage_type));
    return absl::OkStatus();
  } else if (layout == Layout::HWDC) {
    *result =
        Read(GetGlobalAddressNoDeclarationWHDS(xc, yc, zc, sc, storage_type));
    return absl::OkStatus();
  } else if (layout == Layout::BHWDC) {
    *result = Read(
        GetGlobalAddressNoDeclarationWHDSB(xc, yc, zc, sc, bc, storage_type));
    return absl::OkStatus();
  } else {
    return absl::NotFoundError("Unsupported layout");
  }
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

  if (layout == Layout::HWC) {
    *result = Write(args[0],
                    GetGlobalAddressNoDeclarationWHS(xc, yc, sc, storage_type));
    return absl::OkStatus();
  } else if (layout == Layout::BHWC) {
    *result = Write(args[0], GetGlobalAddressNoDeclarationWHSB(xc, yc, sc, bc,
                                                               storage_type));
    return absl::OkStatus();
  } else if (layout == Layout::HWDC) {
    *result = Write(args[0], GetGlobalAddressNoDeclarationWHDS(xc, yc, zc, sc,
                                                               storage_type));
    return absl::OkStatus();
  } else if (layout == Layout::BHWDC) {
    *result = Write(args[0], GetGlobalAddressNoDeclarationWHDSB(
                                 xc, yc, zc, sc, bc, storage_type));
    return absl::OkStatus();
  } else {
    return absl::NotFoundError("Unsupported layout");
  }
}

std::string TensorDescriptor::Read(const std::string& global_address) const {
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
      return absl::StrCat("buffer[", global_address, "]");
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::TEXTURE_3D:
    case TensorStorageType::SINGLE_TEXTURE_2D:
    case TensorStorageType::TEXTURE_ARRAY:
      return absl::StrCat(GetReadImageFromDataType(data_type), "(", image_type,
                          ", smp_none, ", global_address, ")");
    case TensorStorageType::IMAGE_BUFFER:
      return absl::StrCat(GetReadImageFromDataType(data_type),
                          "(image_buffer, ", global_address, ")");
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
  if (HasAxis(Axis::BATCH)) {
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

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
