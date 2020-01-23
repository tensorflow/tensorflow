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

#include "tensorflow/lite/delegates/gpu/cl/kernels/depth_wise_conv_3d.h"

#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/delegates/gpu/cl/cl_device.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/work_group_picking.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

bool IsSpecializedCase(int channel_multiplier) {
  return channel_multiplier == 1 || channel_multiplier == 2 ||
         channel_multiplier == 4;
}

std::string GetSrcValue(const TensorCodeGenerator& src_tensor,
                        int channel_multiplier,
                        TextureAddressMode address_mode) {
  std::string c;
  if (channel_multiplier == 1) {
    c += "        FLT4 src_final =" +
         src_tensor.ReadWHDS("x_c", "y_c", "z_c", "S", address_mode) + ";\n";
  } else if (channel_multiplier == 2) {
    c += "        int z_layer = S / 2;\n";
    c += "        FLT4 src =" +
         src_tensor.ReadWHDS("x_c", "y_c", "z_c", "z_layer", address_mode) +
         ";\n";
    c += "        FLT2 t0 = S % 2 == 0 ? src.xy : src.zw;\n";
    c += "        FLT4 src_final = (FLT4)(t0.x, t0.x, t0.y, t0.y);\n";
  } else if (channel_multiplier == 4) {
    c += "        int z_layer = S / 4;\n";
    c += "        FLT4 src =" +
         src_tensor.ReadWHDS("x_c", "y_c", "z_c", "z_layer", address_mode) +
         ";\n";
    c += "        FLT t0 = src.x;\n";
    c += "        int reminder = S % 4;\n";
    c += "        if (reminder == 1) t0 = src.y;\n";
    c += "        if (reminder == 2) t0 = src.z;\n";
    c += "        if (reminder == 3) t0 = src.w;\n";
    c += "        FLT4 src_final = (FLT4)(t0, t0, t0, t0);\n";
  } else {
    c += "        int z_layer = S / channel_multiplier;\n";
    c += "        FLT4 src =" +
         src_tensor.ReadWHDS("x_c", "y_c", "z_c", "z_layer", address_mode) +
         ";\n";
    c += "        int z_offset = (S % channel_multiplier) * 4;\n";
    c += "        FLT4 src_final;\n";
    c += "        FLT temp_arr[4] = {src.x, src.y, src.z, src.w};\n";
    c += "        src_final.x = temp_arr[(z_offset + 0) / "
         "channel_multiplier];\n";
    c += "        src_final.y = temp_arr[(z_offset + 1) / "
         "channel_multiplier];\n";
    c += "        src_final.z = temp_arr[(z_offset + 2) / "
         "channel_multiplier];\n";
    c += "        src_final.w = temp_arr[(z_offset + 3) / "
         "channel_multiplier];\n";
  }

  return c;
}

std::string GenerateDepthWiseConvolution3DCode(
    const OperationDef& op_def, bool stride_correction,
    const LinearStorage& biases, int channel_multiplier,
    bool weights_are_buffer,
    const std::vector<ElementwiseOperation*>& linked_operations,
    const CLDevice& device) {
  TensorCodeGenerator src_tensor(
      "src_data",
      WHDSPoint{"src_size.x", "src_size.y", "src_size.z", "src_size.w"},
      op_def.src_tensors[0]);
  TensorCodeGenerator dst_tensor(
      "dst_data",
      WHDSPoint{"dst_size.x", "dst_size.y", "dst_size.z", "dst_size.w"},
      op_def.dst_tensors[0]);
  const auto src_tensor_type = op_def.src_tensors[0].storage_type;

  std::string c = GetCommonDefines(op_def.precision);

  const bool manual_clamp = src_tensor_type == TensorStorageType::BUFFER ||
                            src_tensor_type == TensorStorageType::IMAGE_BUFFER;

  c += "__kernel void main_function(\n";
  c += src_tensor.GetDeclaration(AccessType::READ) + ",\n";
  if (weights_are_buffer) {
    c += "    __global FLT4* filters,  \n";
  } else {
    c += "    __read_only image2d_t filters,  \n";
  }
  c += biases.GetDeclaration();
  c += GetArgsDeclaration(linked_operations);
  c += dst_tensor.GetDeclaration(AccessType::WRITE) + ",\n";
  c += "    int4 kernel_size,                \n";
  c += "    int4 stride,                     \n";
  c += "    int4 padding,                    \n";
  c += "    int4 dilation,                   \n";
  if (!IsSpecializedCase(channel_multiplier)) {
    c += "    int channel_multiplier,            \n";
  }
  if (op_def.IsBatchSupported()) {
    c += "    int batch_size,          \n";
  }
  c += "    int4 src_size,                   \n";
  c += "    int4 dst_size                    \n";
  c += ") {\n";
  c += "  int X = get_global_id(0);\n";
  c += "  int Y = get_global_id(1);\n";
  c += "  int linear_id_z = get_global_id(2);\n";
  c += "  int S = linear_id_z % dst_size.w;\n";
  c += "  int Z = linear_id_z / dst_size.w;\n";
  c += "  if (X >= dst_size.x || Y >= dst_size.y || Z >= dst_size.z) return;\n";
  c += "  ACCUM_FLT4 r = (ACCUM_FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n";
  if (stride_correction) {
    c += "  int x_offseted = " +
         GetXStrideCorrected("X", "batch_size", "stride.x", "padding.x") +
         ";\n";
  } else {
    c += "  int x_offseted = X * stride.x + padding.x;\n";
  }
  c += "  int y_offseted = Y * stride.y + padding.y;\n";
  c += "  int z_offseted = Z * stride.z + padding.z;\n";
  if (weights_are_buffer) {
    c += "  int fx_c = S * kernel_size.x * kernel_size.y * kernel_size.z;\n";
  } else {
    c += "  int fx_c = 0;\n";
  }

  if (manual_clamp) {
    c += "  for (int kz = 0; kz < kernel_size.z; ++kz) {\n";
    c += "    int z_c = z_offseted + kz * dilation.z;\n";
    c += "    bool outside_z = z_c < 0 || z_c >= src_size.z;\n";
    c += "    for (int ky = 0; ky < kernel_size.y; ++ky) {\n";
    c += "      int y_c = y_offseted + ky * dilation.y;\n";
    c += "      bool outside_y = y_c < 0 || y_c >= src_size.y;\n";
    c += "      for (int kx = 0; kx < kernel_size.x; ++kx) {\n";
    c += "        int x_c = x_offseted + kx * dilation.x;\n";
    c += "        bool outside_x = x_c < 0 || x_c >= src_size.x;\n";
    c += "        if (!outside_x && !outside_y && !outside_z) {\n";
    if (weights_are_buffer) {
      c += "          FLT4 f = filters[fx_c];\n";
    } else {
      c += "          FLT4 f = READ_IMAGE(filters, smp_none, (int2)(fx_c, "
           "S));\n";
    }
    c += GetSrcValue(src_tensor, channel_multiplier,
                     TextureAddressMode::DONT_CARE);
    c += "          r += TO_ACCUM_TYPE(src_final * f);\n";
    c += "        };\n";
    c += "        fx_c++;\n";
    c += "      }\n";
    c += "    }\n";
    c += "  }\n";
  } else {  // Texture types with ZERO clamping
    c += "  for (int kz = 0; kz < kernel_size.z; ++kz) {\n";
    c += "    int z_c = z_offseted + kz * dilation.z;\n";
    if (src_tensor_type !=
        TensorStorageType::TEXTURE_3D) {  // Only TEXTURE_3D supports clamping
                                          // in DEPTH dimension
      c += "    if (z_c < 0 || z_c >= src_size.z) {\n";
      c += "      fx_c += kernel_size.y * kernel_size.x;\n";
      c += "      continue;\n";
      c += "    }\n";
    }
    c += "    for (int ky = 0; ky < kernel_size.y; ++ky) {\n";
    c += "      int y_c = y_offseted + ky * dilation.y;\n";
    c += "      for (int kx = 0; kx < kernel_size.x; ++kx) {\n";
    c += "        int x_c = x_offseted + kx * dilation.x;\n";
    const auto access_mode = GetFastestZeroMode(device);
    c += GetSrcValue(src_tensor, channel_multiplier, access_mode);
    if (weights_are_buffer) {
      c += "        FLT4 f = filters[fx_c];\n";
    } else {
      c += "        FLT4 f = READ_IMAGE(filters, smp_none, (int2)(fx_c, S));\n";
    }
    c += "        fx_c++;\n";
    c += "        r += TO_ACCUM_TYPE(src_final * f);\n";
    c += "      }\n";
    c += "    }\n";
    c += "  }\n";
  }
  c += "  FLT4 bias_val = " + biases.ReadLinearFLT4("S") + ";\n";
  c += "  FLT4 res0 = TO_FLT4(r) + bias_val;\n";
  const LinkingContext context{"res0", "X", "Y", "S"};
  c += PostProcess(linked_operations, context);
  c += "  " + dst_tensor.WriteWHDS("res0", "X", "Y", "Z", "S") + "\n";
  c += "}\n";
  return c;
}
}  // namespace

DepthWiseConvolution3D::DepthWiseConvolution3D(
    const OperationDef& definition,
    const DepthwiseConvolution3DAttributes& attr, const CLDevice& device)
    : GPUOperation(definition),
      weights_are_buffer_(device.IsMali()),
      kernel_size_(attr.weights.shape.w, attr.weights.shape.h,
                   attr.weights.shape.d),
      stride_(attr.strides.w, attr.strides.h, attr.strides.d),
      padding_(-attr.padding.prepended.w, -attr.padding.prepended.h,
               -attr.padding.prepended.d),
      dilation_(attr.dilations.w, attr.dilations.h, attr.dilations.d),
      channel_multiplier_(attr.weights.shape.o),
      work_group_size_(8, 8, 1) {}

DepthWiseConvolution3D::DepthWiseConvolution3D(
    DepthWiseConvolution3D&& operation)
    : GPUOperation(std::move(operation)),
      weights_tex2d_(std::move(operation.weights_tex2d_)),
      weights_buf_(std::move(operation.weights_buf_)),
      weights_are_buffer_(operation.weights_are_buffer_),
      biases_(std::move(operation.biases_)),
      kernel_size_(operation.kernel_size_),
      stride_(operation.stride_),
      padding_(operation.padding_),
      dilation_(operation.dilation_),
      channel_multiplier_(operation.channel_multiplier_),
      kernel_(std::move(operation.kernel_)),
      work_group_size_(operation.work_group_size_) {}

DepthWiseConvolution3D& DepthWiseConvolution3D::operator=(
    DepthWiseConvolution3D&& operation) {
  if (this != &operation) {
    weights_tex2d_ = std::move(operation.weights_tex2d_);
    weights_buf_ = std::move(operation.weights_buf_);
    std::swap(weights_are_buffer_, operation.weights_are_buffer_);
    biases_ = std::move(operation.biases_);
    std::swap(kernel_size_, operation.kernel_size_);
    std::swap(stride_, operation.stride_);
    std::swap(padding_, operation.padding_);
    std::swap(dilation_, operation.dilation_);
    std::swap(channel_multiplier_, operation.channel_multiplier_);
    kernel_ = std::move(operation.kernel_);
    std::swap(work_group_size_, operation.work_group_size_);
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

Status DepthWiseConvolution3D::Compile(
    const CreationContext& creation_context) {
  const bool stride_correction =
      definition_.IsBatchSupported() && stride_.x != 1;
  const auto code = GenerateDepthWiseConvolution3DCode(
      definition_, stride_correction, biases_, channel_multiplier_,
      weights_are_buffer_, linked_operations_, *creation_context.device);
  return creation_context.cache->GetOrCreateCLKernel(
      code, "main_function", *creation_context.context,
      *creation_context.device, &kernel_);
}

Status DepthWiseConvolution3D::BindArguments() {
  kernel_.ResetBindingCounter();
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(src_[0]->GetMemoryPtr()));
  if (weights_are_buffer_) {
    RETURN_IF_ERROR(kernel_.SetMemoryAuto(weights_buf_.GetMemoryPtr()));
  } else {
    RETURN_IF_ERROR(kernel_.SetMemoryAuto(weights_tex2d_.GetMemoryPtr()));
  }
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(biases_.GetMemoryPtr()));
  RETURN_IF_ERROR(BindArgs(&kernel_, linked_operations_));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(dst_[0]->GetMemoryPtrForWriting()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(
      int4(kernel_size_.x, kernel_size_.y, kernel_size_.z, 1)));
  RETURN_IF_ERROR(
      kernel_.SetBytesAuto(int4(stride_.x, stride_.y, stride_.z, 1)));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(
      int4(padding_.x * src_[0]->Batch(), padding_.y, padding_.z, 1)));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(
      int4(dilation_.x * src_[0]->Batch(), dilation_.y, dilation_.z, 1)));
  if (!IsSpecializedCase(channel_multiplier_)) {
    RETURN_IF_ERROR(kernel_.SetBytesAuto(int32_t(channel_multiplier_)));
  }
  if (definition_.IsBatchSupported()) {
    RETURN_IF_ERROR(kernel_.SetBytesAuto(src_[0]->Batch()));
  }
  RETURN_IF_ERROR(kernel_.SetBytesAuto(src_[0]->GetWBatchedHDS()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(dst_[0]->GetWBatchedHDS()));
  return OkStatus();
}

int3 DepthWiseConvolution3D::GetGridSize() const {
  const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
  const int grid_y = dst_[0]->Height();
  const int grid_z = dst_[0]->Slices() * dst_[0]->Depth();
  return int3(grid_x, grid_y, grid_z);
}

Status DepthWiseConvolution3D::Tune(const TuningParameters& params) {
  RETURN_IF_ERROR(BindArguments());
  return GetBestWorkGroup(params, kernel_, GetGridSize(), &work_group_size_);
}

Status DepthWiseConvolution3D::AddToQueue(CLCommandQueue* queue) {
  RETURN_IF_ERROR(BindArguments());
  return queue->DispatchImplicit(kernel_, GetGridSize(), work_group_size_);
}

Status CreateDepthWiseConvolution3D(
    const CreationContext& creation_context, const OperationDef& definition,
    const DepthwiseConvolution3DAttributes& attr,
    DepthWiseConvolution3D* result) {
  *result = DepthWiseConvolution3D(definition, attr, *creation_context.device);
  RETURN_IF_ERROR(
      result->UploadWeights(attr.weights, creation_context.context));
  LinearStorageCreateInfo create_info;
  create_info.storage_type =
      DeduceLinearStorageType(definition.GetPrimaryStorageType());
  create_info.data_type = definition.GetDataType();
  create_info.name = "biases";
  create_info.aligned_size = attr.weights.shape.o * attr.weights.shape.i;
  RETURN_IF_ERROR(CreateLinearStorage(
      create_info, attr.bias, creation_context.context, &result->biases_));
  return OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
