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

#include "tensorflow/lite/delegates/gpu/cl/kernels/depthwise_conv.h"

#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/delegates/gpu/cl/cl_device.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/work_group_picking.h"
#include "tensorflow/lite/delegates/gpu/cl/linear_storage.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

bool IsSpecializedCase(int channel_multiplier) {
  return channel_multiplier == 1 || channel_multiplier == 2 ||
         channel_multiplier == 4;
}

std::string GetSrcValue(int channel_multiplier, const std::string coords) {
  std::string c;
  if (channel_multiplier == 1) {
    c += "      FLT4 src_final = args.src_tensor.Read(" + coords + ", S);\n";
  } else if (channel_multiplier == 2) {
    c += "      int s_layer = S / 2;\n";
    c += "      FLT4 src = args.src_tensor.Read(" + coords + ", s_layer);\n";
    c += "      FLT2 t0 = S % 2 == 0 ? src.xy : src.zw;\n";
    c += "      FLT4 src_final = (FLT4)(t0.x, t0.x, t0.y, t0.y);\n";
  } else if (channel_multiplier == 4) {
    c += "      int s_layer = S / 4;\n";
    c += "      FLT4 src = args.src_tensor.Read(" + coords + ", s_layer);\n";
    c += "      FLT t0 = src.x;\n";
    c += "      int reminder = S % 4;\n";
    c += "      if (reminder == 1) t0 = src.y;\n";
    c += "      if (reminder == 2) t0 = src.z;\n";
    c += "      if (reminder == 3) t0 = src.w;\n";
    c += "      FLT4 src_final = (FLT4)(t0, t0, t0, t0);\n";
  } else {
    c += "      int s_layer = S / args.ch_multiplier;\n";
    c += "      FLT4 src = args.src_tensor.Read(" + coords + ", s_layer);\n";
    c += "      int s_offset = (S % args.ch_multiplier) * 4;\n";
    c += "      FLT4 src_final;\n";
    c += "      FLT temp_arr[4] = {src.x, src.y, src.z, src.w};\n";
    c += "      src_final.x = temp_arr[(s_offset + 0) / args.ch_multiplier];\n";
    c += "      src_final.y = temp_arr[(s_offset + 1) / args.ch_multiplier];\n";
    c += "      src_final.z = temp_arr[(s_offset + 2) / args.ch_multiplier];\n";
    c += "      src_final.w = temp_arr[(s_offset + 3) / args.ch_multiplier];\n";
  }

  return c;
}

std::string GenerateDepthwiseConvolutionCode(
    const OperationDef& op_def, bool stride_correction, int channel_multiplier,
    bool weights_are_buffer, const CLDevice& device, Arguments* args) {
  auto src_desc = absl::make_unique<TensorDescriptor>(op_def.src_tensors[0]);
  src_desc->SetTextureAddressMode(GetFastestZeroMode(device));
  if (op_def.IsBatchSupported()) {
    src_desc->SetStateVar("BatchedWidth", "true");
  }
  args->AddObjectRef("src_tensor", AccessType::READ, std::move(src_desc));
  auto dst_desc = absl::make_unique<TensorDescriptor>(op_def.dst_tensors[0]);
  if (op_def.IsBatchSupported()) {
    dst_desc->SetStateVar("BatchedWidth", "true");
  }
  args->AddObjectRef("dst_tensor", AccessType::WRITE, std::move(dst_desc));
  args->AddInt("kernel_size_x");
  args->AddInt("stride_x");
  args->AddInt("padding_x");
  args->AddInt("dilation_x");
  args->AddInt("kernel_size_y");
  args->AddInt("stride_y");
  args->AddInt("padding_y");
  args->AddInt("dilation_y");
  if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    args->AddInt("kernel_size_z");
    args->AddInt("stride_z");
    args->AddInt("padding_z");
    args->AddInt("dilation_z");
  }
  if (!IsSpecializedCase(channel_multiplier)) {
    args->AddInt("ch_multiplier");
  }

  const auto src_tensor_type = op_def.src_tensors[0].storage_type;

  std::string c = GetCommonDefines(op_def.precision);

  const bool manual_clamp = src_tensor_type == TensorStorageType::BUFFER ||
                            src_tensor_type == TensorStorageType::IMAGE_BUFFER;

  c += "__kernel void main_function(\n";
  c += "$0) {\n";
  c += "  int X = get_global_id(0);\n";
  c += "  int Y = get_global_id(1);\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "  int linear_id_2 = get_global_id(2);\n";
    c += "  int S = linear_id_2 / args.dst_tensor.Depth();\n";
    c += "  int Z = linear_id_2 % args.dst_tensor.Depth();\n";
  } else {
    c += "  int S = get_global_id(2);\n";
  }
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "S >= args.dst_tensor.Slices()) { \n";
  c += "    return; \n";
  c += "  } \n";
  c += "  ACCUM_FLT4 r = (ACCUM_FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n";
  if (stride_correction) {
    c += "  int x_offseted = " +
         GetXStrideCorrected("X", "args.src_tensor.Batch()", "args.stride_x",
                             "args.padding_x") +
         ";\n";
  } else {
    c += "  int x_offseted = X * args.stride_x + args.padding_x;\n";
  }
  c += "  int y_offseted = Y * args.stride_y + args.padding_y;\n";
  std::string weights_offset = "args.kernel_size_x * args.kernel_size_y";
  if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "  int z_offseted = Z * args.stride_z + args.padding_z;\n";
    weights_offset += " * args.kernel_size_z";
  }
  if (weights_are_buffer) {
    c += "  int fx_c = S * " + weights_offset + ";\n";
  } else {
    c += "  int fx_c = 0;\n";
  }

  std::string flat_coords = "x_c, y_c";
  if (manual_clamp) {
    std::string check = "!outside_x && !outside_y";
    if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
      check += " && !outside_z";
      flat_coords += ", z_c";
      c += "  for (int kz = 0; kz < args.kernel_size_z; ++kz) {\n";
      c += "    int z_c = z_offseted + kz * args.dilation_z;\n";
      c += "    bool outside_z = z_c < 0 || z_c >= args.src_tensor.Depth();\n";
    }
    c += "  for (int ky = 0; ky < args.kernel_size_y; ++ky) {\n";
    c += "    int y_c = y_offseted + ky * args.dilation_y;\n";
    c += "    bool outside_y = y_c < 0 || y_c >= args.src_tensor.Height();\n";
    c += "    for (int kx = 0; kx < args.kernel_size_x; ++kx) {\n";
    c += "      int x_c = x_offseted + kx * args.dilation_x;\n";
    c += "      bool outside_x = x_c < 0 || x_c >= args.src_tensor.Width();\n";
    c += "      if (" + check + ") {\n";
    if (weights_are_buffer) {
      c += "        FLT4 f = args.weights.Read(fx_c);\n";
    } else {
      c += "        FLT4 f = args.weights.Read(fx_c, S);\n";
    }
    c += GetSrcValue(channel_multiplier, flat_coords);
    c += "        r += TO_ACCUM_TYPE(src_final * f);\n";
    c += "      };\n";
    c += "      fx_c++;\n";
    c += "    }\n";
    c += "  }\n";
    if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
      c += "  }\n";
    }
  } else {  // Texture types with ZERO clamping
    if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
      flat_coords += ", z_c";
      c += "  for (int kz = 0; kz < args.kernel_size_z; ++kz) {\n";
      c += "    int z_c = z_offseted + kz * args.dilation_z;\n";
      if (src_tensor_type !=
          TensorStorageType::TEXTURE_3D) {  // Only TEXTURE_3D supports clamping
                                            // in DEPTH dimension
        c += "    if (z_c < 0 || z_c >= args.src_tensor.Depth()) {\n";
        c += "      fx_c += args.kernel_size_y * args.kernel_size_x;\n";
        c += "      continue;\n";
        c += "    }\n";
      }
    }
    c += "  for (int ky = 0; ky < args.kernel_size_y; ++ky) {\n";
    c += "    int y_c = y_offseted + ky * args.dilation_y;\n";
    c += "    for (int kx = 0; kx < args.kernel_size_x; ++kx) {\n";
    c += "      int x_c = x_offseted + kx * args.dilation_x;\n";
    c += GetSrcValue(channel_multiplier, flat_coords);
    if (weights_are_buffer) {
      c += "      FLT4 f = args.weights.Read(fx_c);\n";
    } else {
      c += "      FLT4 f = args.weights.Read(fx_c, S);\n";
    }
    c += "      fx_c++;\n";
    c += "      r += TO_ACCUM_TYPE(src_final * f);\n";
    c += "    }\n";
    c += "  }\n";
    if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
      c += "  }\n";
    }
  }
  c += "  FLT4 res0 = TO_FLT4(r) + args.biases.Read(S);\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "  args.dst_tensor.Write(res0, X, Y, Z, S);\n";
  } else {
    c += "  args.dst_tensor.Write(res0, X, Y, S);\n";
  }
  c += "}\n";

  return c;
}
}  // namespace

DepthwiseConvolution::DepthwiseConvolution(
    const OperationDef& definition,
    const DepthwiseConvolution2DAttributes& attr, bool weights_are_buffer)
    : GPUOperation(definition),
      weights_are_buffer_(weights_are_buffer),
      kernel_size_(attr.weights.shape.w, attr.weights.shape.h, 0, 0),
      stride_(attr.strides.w, attr.strides.h, 0, 0),
      padding_(-attr.padding.prepended.w, -attr.padding.prepended.h, 0, 0),
      dilation_(attr.dilations.w, attr.dilations.h, 0, 0),
      channel_multiplier_(attr.weights.shape.o),
      work_group_size_(8, 8, 1) {}

DepthwiseConvolution::DepthwiseConvolution(
    const OperationDef& definition,
    const DepthwiseConvolution3DAttributes& attr, bool weights_are_buffer)
    : GPUOperation(definition),
      weights_are_buffer_(weights_are_buffer),
      kernel_size_(attr.weights.shape.w, attr.weights.shape.h,
                   attr.weights.shape.d, 0),
      stride_(attr.strides.w, attr.strides.h, attr.strides.d, 0),
      padding_(-attr.padding.prepended.w, -attr.padding.prepended.h,
               -attr.padding.prepended.d, 0),
      dilation_(attr.dilations.w, attr.dilations.h, attr.dilations.d, 0),
      channel_multiplier_(attr.weights.shape.o),
      work_group_size_(8, 8, 1) {}

DepthwiseConvolution::DepthwiseConvolution(DepthwiseConvolution&& operation)
    : GPUOperation(std::move(operation)),
      weights_are_buffer_(operation.weights_are_buffer_),
      kernel_size_(operation.kernel_size_),
      stride_(operation.stride_),
      padding_(operation.padding_),
      dilation_(operation.dilation_),
      channel_multiplier_(operation.channel_multiplier_),
      kernel_(std::move(operation.kernel_)),
      work_group_size_(operation.work_group_size_) {}

DepthwiseConvolution& DepthwiseConvolution::operator=(
    DepthwiseConvolution&& operation) {
  if (this != &operation) {
    std::swap(weights_are_buffer_, operation.weights_are_buffer_);
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

absl::Status DepthwiseConvolution::Compile(
    const CreationContext& creation_context) {
  const bool stride_correction =
      definition_.IsBatchSupported() && stride_.x != 1;
  std::string code = GenerateDepthwiseConvolutionCode(
      definition_, stride_correction, channel_multiplier_, weights_are_buffer_,
      *creation_context.device, &args_);
  std::string element_wise_code;
  RETURN_IF_ERROR(
      MergeOperations(linked_operations_, &args_, &element_wise_code));
  RETURN_IF_ERROR(args_.TransformToCLCode(creation_context.device->GetInfo(),
                                          {{"dst_tensor", element_wise_code}},
                                          &code));
  return creation_context.cache->GetOrCreateCLKernel(
      code, "main_function", *creation_context.context,
      *creation_context.device, &kernel_);
}

absl::Status DepthwiseConvolution::BindArguments() {
  RETURN_IF_ERROR(args_.SetObjectRef("src_tensor", src_[0]));
  RETURN_IF_ERROR(args_.SetObjectRef("dst_tensor", dst_[0]));
  RETURN_IF_ERROR(args_.SetInt("kernel_size_x", kernel_size_.x));
  RETURN_IF_ERROR(args_.SetInt("stride_x", stride_.x));
  RETURN_IF_ERROR(args_.SetInt("padding_x", padding_.x * src_[0]->Batch()));
  RETURN_IF_ERROR(args_.SetInt("dilation_x", dilation_.x * src_[0]->Batch()));
  RETURN_IF_ERROR(args_.SetInt("kernel_size_y", kernel_size_.y));
  RETURN_IF_ERROR(args_.SetInt("stride_y", stride_.y));
  RETURN_IF_ERROR(args_.SetInt("padding_y", padding_.y));
  RETURN_IF_ERROR(args_.SetInt("dilation_y", dilation_.y));
  if (definition_.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    RETURN_IF_ERROR(args_.SetInt("kernel_size_z", kernel_size_.z));
    RETURN_IF_ERROR(args_.SetInt("stride_z", stride_.z));
    RETURN_IF_ERROR(args_.SetInt("padding_z", padding_.z));
    RETURN_IF_ERROR(args_.SetInt("dilation_z", dilation_.z));
  }
  if (!IsSpecializedCase(channel_multiplier_)) {
    RETURN_IF_ERROR(args_.SetInt("ch_multiplier", channel_multiplier_));
  }
  RETURN_IF_ERROR(SetArguments(linked_operations_, &args_));
  return args_.Bind(kernel_.kernel());
}

int3 DepthwiseConvolution::GetGridSize() const {
  const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
  const int grid_y = dst_[0]->Height();
  const int grid_z = dst_[0]->Slices() * dst_[0]->Depth();
  return int3(grid_x, grid_y, grid_z);
}

absl::Status DepthwiseConvolution::Tune(const TuningParameters& params) {
  RETURN_IF_ERROR(BindArguments());
  return GetBestWorkGroup(params, kernel_, GetGridSize(), &work_group_size_);
}

absl::Status DepthwiseConvolution::AddToQueue(CLCommandQueue* queue) {
  RETURN_IF_ERROR(BindArguments());
  return queue->DispatchImplicit(kernel_, GetGridSize(), work_group_size_);
}

absl::Status CreateDepthwiseConvolution(
    const CreationContext& creation_context, const OperationDef& definition,
    const DepthwiseConvolution2DAttributes& attr,
    DepthwiseConvolution* result) {
  bool weights_are_buffer = creation_context.device->IsMali();
  *result = DepthwiseConvolution(definition, attr, weights_are_buffer);
  RETURN_IF_ERROR(
      result->UploadWeights(attr.weights, creation_context.context));

  TensorLinearDescriptor desc;
  desc.storage_type = weights_are_buffer ? LinearStorageType::BUFFER
                                         : LinearStorageType::TEXTURE_2D;
  desc.element_type = definition.GetDataType();

  LinearStorage lt;
  RETURN_IF_ERROR(
      CreateLinearStorage(desc, attr.bias, creation_context.context, &lt));
  result->args_.AddObject("biases", AccessType::READ,
                          absl::make_unique<LinearStorage>(std::move(lt)),
                          absl::make_unique<TensorLinearDescriptor>(desc));
  return absl::OkStatus();
}

absl::Status CreateDepthwiseConvolution(
    const CreationContext& creation_context, const OperationDef& definition,
    const DepthwiseConvolution3DAttributes& attr,
    DepthwiseConvolution* result) {
  bool weights_are_buffer = creation_context.device->IsMali();
  *result = DepthwiseConvolution(definition, attr, weights_are_buffer);
  RETURN_IF_ERROR(
      result->UploadWeights(attr.weights, creation_context.context));

  TensorLinearDescriptor desc;
  desc.storage_type = weights_are_buffer ? LinearStorageType::BUFFER
                                         : LinearStorageType::TEXTURE_2D;
  desc.element_type = definition.GetDataType();

  LinearStorage lt;
  RETURN_IF_ERROR(
      CreateLinearStorage(desc, attr.bias, creation_context.context, &lt));
  result->args_.AddObject("biases", AccessType::READ,
                          absl::make_unique<LinearStorage>(std::move(lt)),
                          absl::make_unique<TensorLinearDescriptor>(desc));
  return absl::OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
