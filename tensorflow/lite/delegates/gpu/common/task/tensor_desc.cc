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
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/task/util.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
namespace {
std::string GetReadImageFromDataType(DataType data_type) {
  if (data_type == DataType::FLOAT32) {
    return "read_imagef";
  } else if (data_type == DataType::FLOAT16) {
    return "read_imageh";
  } else if (data_type == DataType::INT8 || data_type == DataType::INT16 ||
             data_type == DataType::INT32) {
    return "read_imagei";
  } else if (data_type == DataType::UINT8 || data_type == DataType::UINT16 ||
             data_type == DataType::UINT32 || data_type == DataType::BOOL) {
    return "read_imageui";
  } else {
    return "error";
  }
}

DataType ToClTextureType(DataType data_type) {
  switch (data_type) {
    case DataType::FLOAT32:
    case DataType::FLOAT16:
    case DataType::INT32:
    case DataType::UINT32:
      return data_type;
    case DataType::INT16:
    case DataType::INT8:
      return DataType::INT32;
    case DataType::BOOL:
    case DataType::UINT16:
    case DataType::UINT8:
      return DataType::UINT32;
    default:
      return DataType::UNKNOWN;
  }
}

std::string GetWriteImageFromDataType(DataType data_type) {
  if (data_type == DataType::FLOAT32) {
    return "write_imagef";
  } else if (data_type == DataType::FLOAT16) {
    return "write_imageh";
  } else if (data_type == DataType::INT8 || data_type == DataType::INT16 ||
             data_type == DataType::INT32) {
    return "write_imagei";
  } else if (data_type == DataType::UINT8 || data_type == DataType::UINT16 ||
             data_type == DataType::UINT32 || data_type == DataType::BOOL) {
    return "write_imageui";
  } else {
    return "error";
  }
}

std::string GetConversionForImage(const GpuInfo& gpu_info, DataType src_type,
                                  DataType dst_type) {
  DataType interm_type = src_type;
  if (gpu_info.IsApiOpenCl()) {
    if (src_type == DataType::FLOAT16 && dst_type == DataType::FLOAT32) {
      return "";
    }
    interm_type = ToClTextureType(src_type);
  } else if (gpu_info.IsApiMetal()) {
    interm_type = ToMetalTextureType(src_type);
  }
  return GetTypeConversion(gpu_info, interm_type, dst_type, 4);
}

std::string GetConversion(const GpuInfo& gpu_info,
                          TensorStorageType storage_type, DataType src_type,
                          DataType dst_type) {
  if (storage_type == TensorStorageType::BUFFER) {
    return GetTypeConversion(gpu_info, src_type, dst_type, 4);
  } else {
    return GetConversionForImage(gpu_info, src_type, dst_type);
  }
}

void MayBeAddConversion(const std::string& conversion, std::string* result) {
  if (!conversion.empty()) {
    *result = conversion + "(" + *result + ")";
  }
}

absl::optional<std::string> GetLinearIndexFromTemplateArgs(
    const std::vector<std::string>& template_args) {
  for (const auto& template_arg : template_args) {
    const std::string kTokenLinearIndex = "LinearIndex::";
    size_t pos = template_arg.find(kTokenLinearIndex);
    if (pos != std::string::npos) {
      pos += kTokenLinearIndex.size();
      return template_arg.substr(pos, template_arg.size() - pos);
    }
  }
  return absl::nullopt;
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
      data_type_(desc.data_type_),
      storage_type_(desc.storage_type_),
      layout_(desc.layout_),
      use_buffer_for_write_only_2d_texture_(
          desc.use_buffer_for_write_only_2d_texture_),
      use_buffer_for_write_only_image_buffer_(
          desc.use_buffer_for_write_only_image_buffer_),
      shape_(desc.shape_),
      data_(std::move(desc.data_)) {}
TensorDescriptor& TensorDescriptor::operator=(TensorDescriptor&& desc) {
  if (this != &desc) {
    std::swap(data_type_, desc.data_type_);
    std::swap(storage_type_, desc.storage_type_);
    std::swap(layout_, desc.layout_);
    std::swap(use_buffer_for_write_only_2d_texture_,
              desc.use_buffer_for_write_only_2d_texture_);
    std::swap(use_buffer_for_write_only_image_buffer_,
              desc.use_buffer_for_write_only_image_buffer_);
    std::swap(shape_, desc.shape_);
    data_ = std::move(desc.data_);
    GPUObjectDescriptor::operator=(std::move(desc));
  }
  return *this;
}

void TensorDescriptor::CopyWithoutData(TensorDescriptor* desc) const {
  desc->data_type_ = data_type_;
  desc->storage_type_ = storage_type_;
  desc->layout_ = layout_;
  desc->use_buffer_for_write_only_2d_texture_ =
      use_buffer_for_write_only_2d_texture_;
  desc->use_buffer_for_write_only_image_buffer_ =
      use_buffer_for_write_only_image_buffer_;
  desc->shape_ = shape_;
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
  if (storage_type_ == TensorStorageType::BUFFER) {
    GPUBufferDescriptor desc;
    desc.data_type = data_type_;
    desc.access_type = access_type_;
    desc.element_size = 4;
    resources.buffers.push_back({"buffer", desc});
  } else if (storage_type_ == TensorStorageType::SINGLE_TEXTURE_2D ||
             storage_type_ == TensorStorageType::TEXTURE_2D) {
    if (access_type_ == AccessType::WRITE &&
        use_buffer_for_write_only_2d_texture_) {
      resources.ints.push_back("aligned_texture_width");
      GPUBufferDescriptor desc;
      desc.data_type = data_type_;
      desc.access_type = access_type_;
      desc.element_size = 4;
      resources.buffers.push_back({"buffer", desc});
    } else {
      GPUImage2DDescriptor desc;
      desc.data_type = data_type_;
      desc.normalized = false;
      desc.access_type = access_type_;
      resources.images2d.push_back({"image2d", desc});
    }
  } else if (storage_type_ == TensorStorageType::TEXTURE_ARRAY) {
    GPUImage2DArrayDescriptor desc;
    desc.data_type = data_type_;
    desc.access_type = access_type_;
    resources.image2d_arrays.push_back({"image2d_array", desc});
  } else if (storage_type_ == TensorStorageType::TEXTURE_3D) {
    GPUImage3DDescriptor desc;
    desc.data_type = data_type_;
    desc.access_type = access_type_;
    resources.images3d.push_back({"image3d", desc});
  } else if (storage_type_ == TensorStorageType::IMAGE_BUFFER) {
    if (access_type_ == AccessType::WRITE &&
        use_buffer_for_write_only_image_buffer_) {
      GPUBufferDescriptor desc;
      desc.data_type = data_type_;
      desc.access_type = access_type_;
      desc.element_size = 4;
      resources.buffers.push_back({"buffer", desc});
    } else {
      GPUImageBufferDescriptor desc;
      desc.data_type = data_type_;
      desc.access_type = access_type_;
      resources.image_buffers.push_back({"image_buffer", desc});
    }
  }
  return resources;
}

void TensorDescriptor::GetGpuResources(
    const BHWDC& tensor_shape, GenericGPUResourcesWithValue* resources) const {
  resources->AddInt("slice_stride", GetSliceStrideSize(tensor_shape));
  if (HasAxis(Axis::WIDTH)) {
    resources->AddInt("width", GetWidthSize(tensor_shape));
  }
  if (HasAxis(Axis::HEIGHT)) {
    resources->AddInt("height", tensor_shape.h);
  }
  if (HasAxis(Axis::CHANNELS)) {
    resources->AddInt("slices", DivideRoundUp(tensor_shape.c, 4));
    resources->AddInt("channels", tensor_shape.c);
  }
  if (HasAxis(Axis::BATCH)) {
    resources->AddInt("batch", tensor_shape.b);
  }
  if (HasAxis(Axis::DEPTH)) {
    resources->AddInt("depth", tensor_shape.d);
  }
}

absl::Status TensorDescriptor::PerformConstExpr(const GpuInfo& gpu_info,
                                                const std::string& const_expr,
                                                std::string* result) const {
  if (const_expr == "type" || const_expr == "scalar_type") {
    const int vec_size = const_expr == "scalar_type" ? 1 : 4;
    *result = GetTypeDeclaration(gpu_info, data_type_, vec_size);
    return absl::OkStatus();
  } else if (const_expr == "zero_value" || const_expr == "scalar_zero_value") {
    const int vec_size = const_expr == "scalar_zero_value" ? 1 : 4;
    *result = GetZeroValue(gpu_info, data_type_, vec_size);
    return absl::OkStatus();
  } else {
    return absl::UnimplementedError(
        absl::StrCat("Can not resolve constant expression - ", const_expr));
  }
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
  } else if (selector == "ReadNearest") {
    return PerformReadNearestSelector(gpu_info, args, result);
  } else if (selector == "ReadBilinear") {
    return PerformReadBilinearSelector(gpu_info, args, result);
  } else if (selector == "ReadPerChannel") {
    return PerformReadPerChannelSelector(gpu_info, args, template_args, result);
  } else if (selector == "Write") {
    return PerformWriteSelector(gpu_info, args, template_args, result);
  } else if (selector == "WriteLinear") {
    return PerformWriteLinearSelector(gpu_info, args, template_args, result);
  } else if (selector == "Write2D") {
    return PerformWrite2DSelector(gpu_info, args, template_args, result);
  } else if (selector == "GetAddress") {
    return PerformGetAddressSelector(args, result);
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
  DataType read_as_type = data_type_;
  RETURN_IF_ERROR(
      MaybeGetDataTypeFromTemplateArgs(template_args, &read_as_type));
  if (args.size() == 1) {  // function overload for 1D linear types.
    if (storage_type_ == TensorStorageType::BUFFER ||
        storage_type_ == TensorStorageType::IMAGE_BUFFER) {
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

absl::Status TensorDescriptor::PerformReadNearestSelector(
    const GpuInfo& gpu_info, const std::vector<std::string>& args,
    std::string* result) const {
  if (IsBatchedWidth()) {
    return absl::NotFoundError(
        "ReadNearest can not be used with BatchedWidth.");
  }
  // ReadNearest(result, fc_x, fc_y, {fc_z}, slice);
  if (!((args.size() == 5 && HasAxis(Axis::DEPTH)) || args.size() == 4)) {
    return absl::NotFoundError("Unrecognized ReadNearest selector");
  }
  std::vector<std::string> coord_args =
      std::vector<std::string>(args.begin() + 1, args.end());
  std::string c;
  c += "  {\n";
  c += "  int coord_x_TMP = INIT_INT(" + coord_args[0] + ");\n";
  c += "  coord_x_TMP = max(coord_x_TMP, 0);\n";
  c += "  coord_x_TMP = min(coord_x_TMP, width - 1);\n";
  coord_args[0] = "coord_x_TMP";
  c += "  int coord_y_TMP = INIT_INT(" + coord_args[1] + ");\n";
  c += "  coord_y_TMP = max(coord_y_TMP, 0);\n";
  c += "  coord_y_TMP = min(coord_y_TMP, height - 1);\n";
  coord_args[1] = "coord_y_TMP";
  if (HasAxis(Axis::DEPTH)) {
    c += "  int coord_z_TMP = INIT_INT(" + coord_args[2] + ");\n";
    c += "  coord_z_TMP = max(coord_z_TMP, 0);\n";
    c += "  coord_z_TMP = min(coord_z_TMP, depth - 1);\n";
    coord_args[2] = "coord_z_TMP";
  }
  std::string src_value;
  RETURN_IF_ERROR(PerformReadSelector(gpu_info, coord_args, {}, &src_value));
  c += "  " + args[0] + " = " + src_value + ";\n";
  c += "  }";
  *result = c;
  return absl::OkStatus();
}

absl::Status TensorDescriptor::PerformReadBilinearSelector(
    const GpuInfo& gpu_info, const std::vector<std::string>& args,
    std::string* result) const {
  if (IsBatchedWidth()) {
    return absl::NotFoundError(
        "ReadBilinear can not be used with BatchedWidth.");
  }
  // ReadBilinear(result, fc_x, fc_y, {fc_z}, slice);
  if (!((args.size() == 5 && HasAxis(Axis::DEPTH)) || args.size() == 4)) {
    return absl::NotFoundError("Unrecognized ReadBilinear selector");
  }
  std::vector<std::string> coord_args =
      std::vector<std::string>(args.begin() + 1, args.end());
  std::string c;
  c += "  {\n";
  c += "  float f_x_TMP = floor(" + coord_args[0] + ");\n";
  c += "  float x_scale_TMP = (" + coord_args[0] + ") - f_x_TMP;\n";
  c += "  int i_x_TMP = INIT_INT(f_x_TMP);\n";
  c += "  int start_x_TMP = max(i_x_TMP, 0);\n";
  c += "  int end_x_TMP = min(i_x_TMP + 1, width - 1);\n";
  c += "  float f_y_TMP = floor(" + coord_args[1] + ");\n";
  c += "  float y_scale_TMP = (" + coord_args[1] + ") - f_y_TMP;\n";
  c += "  int i_y_TMP = INIT_INT(f_y_TMP);\n";
  c += "  int start_y_TMP = max(i_y_TMP, 0);\n";
  c += "  int end_y_TMP = min(i_y_TMP + 1, height - 1);\n";
  if (HasAxis(Axis::DEPTH)) {
    // 3d bilinear read, x, y, z
    c += "  float f_z_TMP = floor(" + coord_args[2] + ");\n";
    c += "  float z_scale_TMP = (" + coord_args[2] + ") - f_z_TMP;\n";
    c += "  int i_z_TMP = INIT_INT(f_z_TMP);\n";
    c += "  int start_z_TMP = max(i_z_TMP, 0);\n";
    c += "  int end_z_TMP = min(i_z_TMP + 1, depth - 1);\n";
    int index = 0;
    for (const auto& src_z : {"start_z_TMP", "end_z_TMP"}) {
      for (const auto& src_y : {"start_y_TMP", "end_y_TMP"}) {
        for (const auto& src_x : {"start_x_TMP", "end_x_TMP"}) {
          coord_args[0] = src_x;
          coord_args[1] = src_y;
          coord_args[2] = src_z;
          std::string src_value;
          RETURN_IF_ERROR(
              PerformReadSelector(gpu_info, coord_args, {"float"}, &src_value));
          c += "  float4 src" + std::to_string(index) + "_TMP = " + src_value +
               ";\n";
          index++;
        }
      }
    }
    c += "  float4 t0_TMP = mix(mix(src0_TMP, src1_TMP, x_scale_TMP), "
         "mix(src2_TMP, src3_TMP, x_scale_TMP), y_scale_TMP);\n";
    c += "  float4 t1_TMP = mix(mix(src4_TMP, src5_TMP, x_scale_TMP), "
         "mix(src6_TMP, src7_TMP, x_scale_TMP), y_scale_TMP);\n";
    c += "  " + args[0] + " = TO_FLT4(mix(t0_TMP, t1_TMP, z_scale_TMP));\n";
  } else {
    // 2d bilinear read, x, y
    int index = 0;
    for (const auto& src_y : {"start_y_TMP", "end_y_TMP"}) {
      for (const auto& src_x : {"start_x_TMP", "end_x_TMP"}) {
        coord_args[0] = src_x;
        coord_args[1] = src_y;
        std::string src_value;
        RETURN_IF_ERROR(
            PerformReadSelector(gpu_info, coord_args, {"float"}, &src_value));
        c += "  float4 src" + std::to_string(index) + "_TMP = " + src_value +
             ";\n";
        index++;
      }
    }
    c += "  " + args[0] +
         " = TO_FLT4(mix(mix(src0_TMP, src1_TMP, x_scale_TMP), mix(src2_TMP, "
         "src3_TMP, x_scale_TMP), y_scale_TMP));\n";
  }
  c += "  }";
  *result = c;
  return absl::OkStatus();
}

absl::Status TensorDescriptor::PerformReadPerChannelSelector(
    const GpuInfo& gpu_info, const std::vector<std::string>& args,
    const std::vector<std::string>& template_args, std::string* result) const {
  std::vector<std::string> coord_args =
      std::vector<std::string>(args.begin() + 1, args.end());
  int channels_index = 0;
  if (HasAxis(Axis::WIDTH)) {
    channels_index++;
  }
  if (HasAxis(Axis::HEIGHT)) {
    channels_index++;
  }
  if (HasAxis(Axis::DEPTH)) {
    channels_index++;
  }
  if (channels_index >= coord_args.size()) {
    return absl::NotFoundError(
        "Wrong number of coordinates in ReadPerChannel.");
  }
  std::string c = "  {\n";
  c += "  int slice_coord_TMP = (" + coord_args[channels_index] + ") / 4;\n";
  c += "  int sub_ch_coord_TMP = (" + coord_args[channels_index] + ") % 4;\n";
  coord_args[channels_index] = "slice_coord_TMP";
  std::string src_value;
  RETURN_IF_ERROR(
      PerformReadSelector(gpu_info, coord_args, template_args, &src_value));
  if (gpu_info.IsApiOpenCl()) {
    DataType dst_type = data_type_;
    RETURN_IF_ERROR(MaybeGetDataTypeFromTemplateArgs(template_args, &dst_type));
    c += "  " + GetTypeDeclaration(gpu_info, dst_type, 4) +
         " src_TMP = " + src_value + ";\n";
    c +=
        "  " + args[0] + " = (" + ToCLDataType(dst_type, 1) +
        "[4]){src_TMP.x, src_TMP.y, src_TMP.z, src_TMP.w}[sub_ch_coord_TMP];\n";
  } else {
    c += "  " + args[0] + " = " + src_value + "[sub_ch_coord_TMP];\n";
  }

  c += "  }";
  *result = c;
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
    const std::vector<std::string>& template_args, std::string* result) const {
  if (IsLinear()) {
    const auto linear_index = GetLinearIndexFromTemplateArgs(template_args);
    if (linear_index.has_value()) {
      std::vector<std::string> new_args = {args[0], linear_index.value()};
      return PerformWriteLinearSelector(gpu_info, new_args, template_args,
                                        result);
    }
  }
  std::string xc;
  std::string yc;
  std::string zc;
  std::string sc;
  std::string bc;
  bool parsed = ParseCoordsFromArgs(args, 1, &xc, &yc, &zc, &sc, &bc);
  if (args.size() < 2 || !parsed) {
    return absl::NotFoundError("Unrecognized Write selector");
  }
  DataType write_type = data_type_;
  RETURN_IF_ERROR(MaybeGetDataTypeFromTemplateArgs(template_args, &write_type));
  *result = Write(gpu_info, write_type, args[0],
                  GetPhysicalCoords(xc, yc, zc, sc, bc));
  return absl::OkStatus();
}

absl::Status TensorDescriptor::PerformWriteLinearSelector(
    const GpuInfo& gpu_info, const std::vector<std::string>& args,
    const std::vector<std::string>& template_args, std::string* result) const {
  if (storage_type_ != TensorStorageType::BUFFER &&
      storage_type_ != TensorStorageType::IMAGE_BUFFER) {
    return absl::InvalidArgumentError(
        "WriteLinear selector can be used only with linear "
        "storages(BUFFER/IMAGE_BUFFER)");
  }
  if (args.size() != 2) {
    return absl::NotFoundError("Unrecognized WriteLinear selector");
  }
  DataType write_type = data_type_;
  RETURN_IF_ERROR(MaybeGetDataTypeFromTemplateArgs(template_args, &write_type));
  *result = Write(gpu_info, write_type, args[0], {args[1]});
  return absl::OkStatus();
}

absl::Status TensorDescriptor::PerformWrite2DSelector(
    const GpuInfo& gpu_info, const std::vector<std::string>& args,
    const std::vector<std::string>& template_args, std::string* result) const {
  if (storage_type_ != TensorStorageType::TEXTURE_2D) {
    return absl::InvalidArgumentError(
        "Write2D selector can be used only with 2d "
        "storages(TEXTURE_2D)");
  }
  if (args.size() != 3) {
    return absl::NotFoundError("Unrecognized Write2D selector");
  }
  DataType write_type = data_type_;
  RETURN_IF_ERROR(MaybeGetDataTypeFromTemplateArgs(template_args, &write_type));
  *result = Write(gpu_info, write_type, args[0], {args[1], args[2]});
  return absl::OkStatus();
}

std::string TensorDescriptor::Read(
    const GpuInfo& gpu_info, DataType read_as_type,
    const std::vector<std::string>& coords) const {
  const std::string conversion =
      GetConversion(gpu_info, storage_type_, data_type_, read_as_type);
  if (gpu_info.IsApiOpenCl() &&
      !(data_type_ == DataType::FLOAT16 && read_as_type == DataType::FLOAT32)) {
    read_as_type = data_type_;
  }
  switch (storage_type_) {
    case TensorStorageType::BUFFER: {
      std::string result;
      if (gpu_info.IsGlsl() && data_type_ == DataType::FLOAT16 &&
          !gpu_info.IsGlslSupportsExplicitFp16()) {
        result =
            absl::StrCat("vec4(unpackHalf2x16(buffer[", coords[0],
                         "].x), unpackHalf2x16(buffer[", coords[0], "].y))");
      } else {
        result = absl::StrCat("buffer[", coords[0], "]");
      }
      MayBeAddConversion(conversion, &result);
      return result;
    }
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::SINGLE_TEXTURE_2D: {
      std::string result;
      if (gpu_info.IsApiOpenCl()) {
        result = absl::Substitute("$0(image2d, smp_zero, (int2)($1, $2))",
                                  GetReadImageFromDataType(read_as_type),
                                  coords[0], coords[1]);
      } else if (gpu_info.IsApiMetal()) {
        result = absl::Substitute("image2d.read(ushort2($0, $1))", coords[0],
                                  coords[1]);
      } else if (gpu_info.IsGlsl()) {
        result = "texelFetch(image2d, ivec2(" + coords[0] + ", " + coords[1] +
                 "), 0)";
        if (data_type_ == DataType::FLOAT16 &&
            gpu_info.IsGlslSupportsExplicitFp16()) {
          result = "f16vec4(" + result + ")";
        }
      }
      MayBeAddConversion(conversion, &result);
      return result;
    }
    case TensorStorageType::TEXTURE_3D: {
      std::string result;
      if (gpu_info.IsApiOpenCl()) {
        result =
            absl::Substitute("$0(image3d, smp_zero, (int4)($1, $2, $3, 0))",
                             GetReadImageFromDataType(read_as_type), coords[0],
                             coords[1], coords[2]);
      } else if (gpu_info.IsApiMetal()) {
        result = absl::Substitute("image3d.read(ushort3($0, $1, $2))",
                                  coords[0], coords[1], coords[2]);
      } else if (gpu_info.IsGlsl()) {
        result = "texelFetch(image3d, ivec3(" + coords[0] + ", " + coords[1] +
                 ", " + coords[2] + "), 0)";
        if (data_type_ == DataType::FLOAT16 &&
            gpu_info.IsGlslSupportsExplicitFp16()) {
          result = "f16vec4(" + result + ")";
        }
      }
      MayBeAddConversion(conversion, &result);
      return result;
    }
    case TensorStorageType::TEXTURE_ARRAY: {
      std::string result;
      if (gpu_info.IsApiOpenCl()) {
        result = absl::Substitute(
            "$0(image2d_array, smp_zero, (int4)($1, $2, $3, 0))",
            GetReadImageFromDataType(read_as_type), coords[0], coords[1],
            coords[2]);
      } else if (gpu_info.IsApiMetal()) {
        result = absl::Substitute("image2d_array.read(ushort2($0, $1), $2)",
                                  coords[0], coords[1], coords[2]);
      } else if (gpu_info.IsGlsl()) {
        result = "texelFetch(image2d_array, ivec3(" + coords[0] + ", " +
                 coords[1] + ", " + coords[2] + "), 0)";
        if (data_type_ == DataType::FLOAT16 &&
            gpu_info.IsGlslSupportsExplicitFp16()) {
          result = "f16vec4(" + result + ")";
        }
      }
      MayBeAddConversion(conversion, &result);
      return result;
    }
    case TensorStorageType::IMAGE_BUFFER: {
      std::string result;
      if (gpu_info.IsApiOpenCl()) {
        result = absl::StrCat(GetReadImageFromDataType(read_as_type),
                              "(image_buffer, ", coords[0], ")");
      } else if (gpu_info.IsApiMetal()) {
        result = absl::Substitute("image_buffer.read(uint($0))", coords[0]);
      } else if (gpu_info.IsGlsl()) {
        result = "texelFetch(image_buffer, " + coords[0] + ")";
        if (data_type_ == DataType::FLOAT16 &&
            gpu_info.IsGlslSupportsExplicitFp16()) {
          result = "f16vec4(" + result + ")";
        }
      }
      MayBeAddConversion(conversion, &result);
      return result;
    }
    case TensorStorageType::UNKNOWN:
      return "";
  }
}

std::string TensorDescriptor::Write(
    const GpuInfo& gpu_info, DataType write_type, const std::string& var_name,
    const std::vector<std::string>& coords) const {
  bool is_texture_write = storage_type_ == TensorStorageType::IMAGE_BUFFER ||
                          storage_type_ == TensorStorageType::TEXTURE_2D ||
                          storage_type_ == TensorStorageType::TEXTURE_ARRAY ||
                          storage_type_ == TensorStorageType::TEXTURE_3D;
  if (storage_type_ == TensorStorageType::IMAGE_BUFFER &&
      use_buffer_for_write_only_image_buffer_) {
    is_texture_write = false;
  }
  if (storage_type_ == TensorStorageType::TEXTURE_2D &&
      use_buffer_for_write_only_2d_texture_) {
    is_texture_write = false;
  }
  DataType write_required_type = data_type_;
  if (is_texture_write) {
    if (gpu_info.IsApiOpenCl()) {
      write_required_type = ToClTextureType(data_type_);
    } else if (gpu_info.IsApiMetal()) {
      write_required_type = ToMetalTextureType(data_type_);
    }
  }
  std::string write_expr = var_name;
  if (write_type != write_required_type) {
    const std::string conversion =
        GetTypeConversion(gpu_info, write_type, write_required_type, 4);
    if (!conversion.empty()) {
      write_expr = conversion + "(" + write_expr + ")";
    }
  }
  switch (storage_type_) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      if (gpu_info.IsApiOpenCl()) {
        if (use_buffer_for_write_only_image_buffer_) {
          return absl::StrCat("buffer[", coords[0], "] = ", write_expr);
        } else {
          return absl::Substitute("$0(image_buffer, $1, $2)",
                                  GetWriteImageFromDataType(data_type_),
                                  coords[0], write_expr);
        }
      } else if (gpu_info.IsApiMetal()) {
        if (use_buffer_for_write_only_image_buffer_) {
          return absl::StrCat("buffer[", coords[0], "] = ", write_expr);
        } else {
          return absl::Substitute("image_buffer.write($0, uint($1))",
                                  write_expr, coords[0]);
        }
      } else if (gpu_info.IsGlsl()) {
        if (data_type_ == DataType::FLOAT16 &&
            !gpu_info.IsGlslSupportsExplicitFp16()) {
          return absl::StrCat("buffer[", coords[0], "] = uvec2(packHalf2x16(",
                              write_expr, ".xy), packHalf2x16(", write_expr,
                              ".zw))");
        } else {
          return absl::StrCat("buffer[", coords[0], "] = ", write_expr);
        }
      } else {
        return absl::StrCat("buffer[", coords[0], "] = ", write_expr);
      }
    case TensorStorageType::SINGLE_TEXTURE_2D:
    case TensorStorageType::TEXTURE_2D:
      if (gpu_info.IsApiOpenCl()) {
        if (use_buffer_for_write_only_2d_texture_) {
          return absl::Substitute(
              "buffer[($2) * aligned_texture_width + ($1)] = $0", write_expr,
              coords[0], coords[1]);
        } else {
          return absl::Substitute("$0(image2d, (int2)($1, $2), $3)",
                                  GetWriteImageFromDataType(data_type_),
                                  coords[0], coords[1], write_expr);
        }
      } else if (gpu_info.IsApiMetal()) {
        if (use_buffer_for_write_only_2d_texture_) {
          return absl::Substitute(
              "buffer[($2) * aligned_texture_width + ($1)] = $0", write_expr,
              coords[0], coords[1]);
        } else {
          return absl::Substitute("image2d.write($0, ushort2($1, $2))",
                                  write_expr, coords[0], coords[1]);
        }
      } else if (gpu_info.IsGlsl()) {
        return absl::Substitute("imageStore(image2d, ivec2($0, $1), $2)",
                                coords[0], coords[1], write_expr);
      } else {
        return "";
      }
    case TensorStorageType::TEXTURE_3D:
      if (gpu_info.IsApiOpenCl()) {
        return absl::Substitute("$0(image3d, (int4)($1, $2, $3, 0), $4)",
                                GetWriteImageFromDataType(data_type_),
                                coords[0], coords[1], coords[2], write_expr);
      } else if (gpu_info.IsApiMetal()) {
        return absl::Substitute("image3d.write($0, ushort3($1, $2, $3))",
                                write_expr, coords[0], coords[1], coords[2]);
      } else if (gpu_info.IsGlsl()) {
        return absl::Substitute("imageStore(image3d, ivec3($0, $1, $2), $3)",
                                coords[0], coords[1], coords[2], write_expr);
      } else {
        return "";
      }
    case TensorStorageType::TEXTURE_ARRAY:
      if (gpu_info.IsApiOpenCl()) {
        return absl::Substitute("$0(image2d_array, (int4)($1, $2, $3, 0), $4)",
                                GetWriteImageFromDataType(data_type_),
                                coords[0], coords[1], coords[2], write_expr);
      } else if (gpu_info.IsApiMetal()) {
        return absl::Substitute("image2d_array.write($0, ushort2($1, $2), $3)",
                                write_expr, coords[0], coords[1], coords[2]);
      } else if (gpu_info.IsGlsl()) {
        return absl::Substitute(
            "imageStore(image2d_array, ivec3($0, $1, $2), $3)", coords[0],
            coords[1], coords[2], write_expr);
      } else {
        return "";
      }
    case TensorStorageType::UNKNOWN:
      return "";
  }
}

absl::Status TensorDescriptor::PerformGetAddressSelector(
    const std::vector<std::string>& args, std::string* result) const {
  std::string xc, yc, zc, sc, bc;
  bool parsed = ParseCoordsFromArgs(args, 0, &xc, &yc, &zc, &sc, &bc);
  if (!parsed) {
    return absl::NotFoundError("Unrecognized GetAddress selector");
  }

  *result = GetGlobalAddressNoDeclaration(xc, yc, zc, sc, bc);
  return absl::OkStatus();
}

absl::Status TensorDescriptor::PerformGetHandleSelector(
    const std::vector<std::string>& args, std::string* result) const {
  if (!args.empty()) {
    return absl::NotFoundError(
        absl::StrCat("GetHandle does not require arguments, but ", args.size(),
                     " was passed"));
  }
  switch (storage_type_) {
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

std::string TensorDescriptor::StorageTypeToAddressType() const {
  switch (storage_type_) {
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
  switch (storage_type_) {
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
  switch (storage_type_) {
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
  switch (storage_type_) {
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
  switch (storage_type_) {
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
  switch (storage_type_) {
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
  if (layout_ == Layout::HWC || (IsBatchedWidth() && layout_ == Layout::BHWC)) {
    return GetPhysicalCoordsWHS(xc, yc, sc);
  } else if (layout_ == Layout::BHWC) {
    return GetPhysicalCoordsWHSB(xc, yc, sc, bc);
  } else if (layout_ == Layout::HWDC ||
             (IsBatchedWidth() && layout_ == Layout::BHWDC)) {
    return GetPhysicalCoordsWHDS(xc, yc, zc, sc);
  } else if (layout_ == Layout::BHWDC) {
    return GetPhysicalCoordsWHDSB(xc, yc, zc, sc, bc);
  } else {
    return {""};
  }
}

absl::Status TensorDescriptor::MaybeGetDataTypeFromTemplateArgs(
    const std::vector<std::string>& template_args, DataType* result) const {
  for (const auto& template_arg : template_args) {
    std::string read_type = template_arg;
    if (read_type == "FLT" || read_type == "ACCUM_FLT") {
      auto it = state_vars_.find(read_type);
      if (it == state_vars_.end()) {
        return absl::UnavailableError(
            absl::StrCat("Template argument ", read_type, " uninitialized."));
      } else {
        read_type = it->second;
      }
    }

    if (read_type == "half") {
      *result = DataType::FLOAT16;
      return absl::OkStatus();
    } else if (read_type == "float") {
      *result = DataType::FLOAT32;
      return absl::OkStatus();
    } else if (read_type == "int") {
      *result = DataType::INT32;
      return absl::OkStatus();
    } else if (read_type == "short") {
      *result = DataType::INT16;
      return absl::OkStatus();
    } else if (read_type == "char") {
      *result = DataType::INT8;
      return absl::OkStatus();
    } else if (read_type == "uint") {
      *result = DataType::UINT32;
      return absl::OkStatus();
    } else if (read_type == "ushort") {
      *result = DataType::UINT16;
      return absl::OkStatus();
    } else if (read_type == "uchar") {
      *result = DataType::UINT8;
      return absl::OkStatus();
    } else if (read_type == "bool") {
      *result = DataType::BOOL;
      return absl::OkStatus();
    }
  }
  return absl::OkStatus();
}

bool TensorDescriptor::HasAxis(Axis axis) const {
  if (axis == Axis::WIDTH || axis == Axis::HEIGHT || axis == Axis::CHANNELS) {
    return true;
  }
  if (axis == Axis::BATCH &&
      (layout_ == Layout::BHWC || layout_ == Layout::BHWDC)) {
    return true;
  }
  if (axis == Axis::DEPTH &&
      (layout_ == Layout::HWDC || layout_ == Layout::BHWDC)) {
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
    if (offset >= args.size()) return false;
    *sc = args[offset++];
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

size_t TensorDescriptor::GetSizeInBytesForShape(const BHWDC& shape5d) const {
  int aligned_channels = storage_type_ == TensorStorageType::SINGLE_TEXTURE_2D
                             ? shape5d.c
                             : AlignByN(shape5d.c, 4);
  int elements_count =
      shape5d.b * shape5d.w * shape5d.h * shape5d.d * aligned_channels;
  return elements_count * SizeOf(data_type_);
}

int TensorDescriptor::GetLinearIndex(const BHWDC& shape5d, int b, int x, int y,
                                     int d, int s, int sub_c) const {
  const int slices = DivideRoundUp(shape5d.c, 4);
  switch (storage_type_) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_3D:
      return ((((d * slices + s) * shape5d.h + y) * shape5d.w + x) * shape5d.b +
              b) *
                 4 +
             sub_c;  // DSHWBC4
    case TensorStorageType::TEXTURE_2D:
      return ((((y * slices + s) * shape5d.w + x) * shape5d.b + b) * shape5d.d +
              d) *
                 4 +
             sub_c;  // HSWBDC4
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return (((y * shape5d.w + x) * shape5d.b + b) * shape5d.d + d) *
                 shape5d.c +
             sub_c;  // HWBDC
    case TensorStorageType::UNKNOWN:
      return -1;
  }
}

void TensorDescriptor::UploadData(
    const tflite::gpu::Tensor<HWC, DataType::FLOAT32>& src) {
  shape_ = BHWDC(1, src.shape.h, src.shape.w, 1, src.shape.c);
  UploadData(src.data.data());
}

void TensorDescriptor::UploadData(
    const tflite::gpu::Tensor<Linear, DataType::FLOAT32>& src) {
  shape_ = BHWDC(1, 1, 1, 1, src.shape.v);
  UploadData(src.data.data());
}

bool TensorDescriptor::SupportsZeroClamp(const Axis& axis,
                                         const GpuInfo& gpu_info) const {
  switch (storage_type_) {
    case TensorStorageType::UNKNOWN:
      return false;
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      return false;
    case TensorStorageType::TEXTURE_ARRAY:
      return (axis == Axis::WIDTH || axis == Axis::HEIGHT) &&
             gpu_info.SupportsZeroClampForImages();
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return (axis == Axis::WIDTH || axis == Axis::HEIGHT) &&
             gpu_info.SupportsZeroClampForImages();
    case TensorStorageType::TEXTURE_3D:
      return (axis == Axis::WIDTH || axis == Axis::HEIGHT ||
              axis == Axis::DEPTH) &&
             gpu_info.SupportsZeroClampForImages();
  }
}

bool TensorDescriptor::CanReadOutOfBorder(const Axis& axis) const {
  switch (storage_type_) {
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
  return storage_type_ == TensorStorageType::BUFFER ||
         storage_type_ == TensorStorageType::IMAGE_BUFFER;
}

bool TensorDescriptor::ReturnsZeroForNegOneRead(const GpuInfo& gpu_info) const {
  return storage_type_ == TensorStorageType::IMAGE_BUFFER &&
         gpu_info.SupportsZeroClampForImageBuffer();
}

absl::Status TensorDescriptor::CanCreateTensorWithShape(
    const GpuInfo& gpu_info, const BHWDC& shape) const {
  const int slices = DivideRoundUp(shape.c, 4);
  const uint64_t allocation_size = GetSizeInBytesForShape(shape);
  const std::string common_desc = "Shape - " + ToString(shape) +
                                  ", data type - " + ToString(data_type_) + ".";
  if (allocation_size > gpu_info.GetMaxMemoryAllocationSize()) {
    return absl::ResourceExhaustedError(absl::StrCat(
        "Requested allocation size - ", allocation_size,
        " bytes. Max allocation size for this GPU - ",
        gpu_info.GetMaxMemoryAllocationSize(), " bytes. ", common_desc));
  }
  switch (storage_type_) {
    case TensorStorageType::BUFFER: {
      if (allocation_size > gpu_info.GetMaxBufferSize()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Buffer with size - ", allocation_size,
            " bytes can not be created. Max buffer size for this GPU - ",
            gpu_info.GetMaxBufferSize(), " bytes. ", common_desc));
      } else {
        return absl::OkStatus();
      }
    }
    case TensorStorageType::IMAGE_BUFFER: {
      const uint64_t element_size = 4 * SizeOf(data_type_);
      const uint64_t image_width = allocation_size / element_size;
      if (image_width > gpu_info.GetMaxImageBufferWidth()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image buffer with width - ", image_width,
            " can not be created. Max image buffer width for this GPU - ",
            gpu_info.GetMaxImageBufferWidth(), ". ", common_desc));
      } else if (allocation_size > gpu_info.GetMaxBufferSize()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Buffer with size - ", allocation_size,
            " bytes can not be created. Max buffer size for this GPU - ",
            gpu_info.GetMaxBufferSize(), " bytes. ", common_desc));
      } else {
        return absl::OkStatus();
      }
    }
    case TensorStorageType::TEXTURE_3D: {
      if (gpu_info.IsApiOpenCl() &&
          gpu_info.opencl_info.cl_version < OpenClVersion::kCl1_2 &&
          slices == 1) {
        return absl::InternalError(
            "clCreateImage3D (that used in CL 1.0/1.1) can not create image "
            "with depth = 1 by specification.");
      }
      const int image_width = shape.w * shape.b;
      const int image_height = shape.h;
      const int image_depth = slices * shape.d;
      if (image_width > gpu_info.GetMaxImage3DWidth()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image3D with width - ", image_width,
            " can not be created. Max Image3D width for this GPU - ",
            gpu_info.GetMaxImage3DWidth(), ". ", common_desc));
      } else if (image_height > gpu_info.GetMaxImage3DHeight()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image3D with height - ", image_height,
            " can not be created. Max Image3D height for this GPU - ",
            gpu_info.GetMaxImage3DHeight(), ". ", common_desc));
      } else if (image_depth > gpu_info.GetMaxImage3DDepth()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image3D with depth - ", image_depth,
            " can not be created. Max Image3D depth for this GPU - ",
            gpu_info.GetMaxImage3DDepth(), ". ", common_desc));
      } else {
        return absl::OkStatus();
      }
    }
    case TensorStorageType::TEXTURE_ARRAY: {
      // Bug on some Adreno. b/131099086
      if (gpu_info.IsApiOpenCl() && slices == 1 && gpu_info.IsAdreno() &&
          !gpu_info.adreno_info.support_one_layer_texture_array) {
        return absl::InternalError(
            "Image2DArray with layer = 1 works incorrect on some Adreno in "
            "OpenCL. Can not be created.");
      }
      const int image_width = shape.w * shape.b;
      const int image_height = shape.h;
      const int image_layers = slices * shape.d;
      if (image_width > gpu_info.GetMaxImage2DWidth()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image2DArray with width - ", image_width,
            " can not be created. Max Image2DArray width for this GPU - ",
            gpu_info.GetMaxImage2DWidth(), ". ", common_desc));
      } else if (image_height > gpu_info.GetMaxImage2DHeight()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image2DArray with height - ", image_height,
            " can not be created. Max Image2DArray height for this GPU - ",
            gpu_info.GetMaxImage2DHeight(), ". ", common_desc));
      } else if (image_layers > gpu_info.GetMaxImage2DArrayLayers()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image2DArray with layers - ", image_layers,
            " can not be created. Max Image2DArray layers for this GPU - ",
            gpu_info.GetMaxImage2DArrayLayers(), ". ", common_desc));
      } else {
        return absl::OkStatus();
      }
    }
    case TensorStorageType::TEXTURE_2D: {
      const int image_width = shape.w * shape.b * shape.d;
      const int image_height = shape.h * slices;
      if (image_width > gpu_info.GetMaxImage2DWidth()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image2D with width - ", image_width,
            " can not be created. Max Image2D width for this GPU - ",
            gpu_info.GetMaxImage2DWidth(), ". ", common_desc));
      } else if (image_height > gpu_info.GetMaxImage2DHeight()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image2D with height - ", image_height,
            " can not be created. Max Image2D height for this GPU - ",
            gpu_info.GetMaxImage2DHeight(), ". ", common_desc));
      } else {
        return absl::OkStatus();
      }
    }
    case TensorStorageType::SINGLE_TEXTURE_2D: {
      const int image_width = shape.w * shape.b * shape.d;
      const int image_height = shape.h;
      if (shape.c > 4) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image2D with channels - ", shape.c, " can not be created."));
      } else if (!gpu_info.SupportsFloatImage2D(data_type_, shape.c)) {
        return absl::ResourceExhaustedError(
            "Image2D doesn't support this pixel layout.");
      } else if (image_width > gpu_info.GetMaxImage2DWidth()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image2D with width - ", image_width,
            " can not be created. Max Image2D width for this GPU - ",
            gpu_info.GetMaxImage2DWidth(), ". ", common_desc));
      } else if (image_height > gpu_info.GetMaxImage2DHeight()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image2D with height - ", image_height,
            " can not be created. Max Image2D height for this GPU - ",
            gpu_info.GetMaxImage2DHeight(), ". ", common_desc));
      } else {
        return absl::OkStatus();
      }
    }
    default:
      return absl::UnimplementedError(
          "Can not create resources for unknown storage type.");
  }
}

absl::Status TensorDescriptor::CanCreateTensorWithShape(
    const GpuInfo& gpu_info, const BHWC& shape) const {
  const BHWDC shape5D(shape.b, shape.h, shape.w, 1, shape.c);
  return CanCreateTensorWithShape(gpu_info, shape5D);
}

absl::Status TensorDescriptor::UpdateToSupportedStorageType(
    const GpuInfo& gpu_info, const BHWC& shape) {
  if (CanCreateTensorWithShape(gpu_info, shape).ok()) {
    return absl::OkStatus();
  }
  if (gpu_info.IsApiMetal()) {
    storage_type_ = TensorStorageType::BUFFER;
    return CanCreateTensorWithShape(gpu_info, shape);
  }

  storage_type_ = TensorStorageType::IMAGE_BUFFER;
  if (gpu_info.SupportsImageBuffer() &&
      CanCreateTensorWithShape(gpu_info, shape).ok()) {
    return absl::OkStatus();
  }
  storage_type_ = TensorStorageType::BUFFER;
  return CanCreateTensorWithShape(gpu_info, shape);
}

TensorDescriptor CreateBhwcTensorDescriptor(DataType data_type,
                                            TensorStorageType storage_type,
                                            const BHWC& shape) {
  TensorDescriptor tensor_desc =
      TensorDescriptor(data_type, storage_type, Layout::BHWC);
  tensor_desc.SetBHWCShape(shape);
  return tensor_desc;
}

TensorDescriptor CreateHwcTensorDescriptor(DataType data_type,
                                           TensorStorageType storage_type,
                                           const HWC& shape) {
  TensorDescriptor tensor_desc =
      TensorDescriptor(data_type, storage_type, Layout::HWC);
  tensor_desc.SetBHWCShape(BHWC(1, shape.h, shape.w, shape.c));
  return tensor_desc;
}
}  // namespace gpu
}  // namespace tflite
