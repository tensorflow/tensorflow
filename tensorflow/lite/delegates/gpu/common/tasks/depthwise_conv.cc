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

#include "tensorflow/lite/delegates/gpu/common/tasks/depthwise_conv.h"

#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/task/tensor_linear_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/util.h"
#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"

namespace tflite {
namespace gpu {

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
    c += "      FLT4 src_final = INIT_FLT4v4(t0.x, t0.x, t0.y, t0.y);\n";
  } else if (channel_multiplier == 4) {
    c += "      int s_layer = S / 4;\n";
    c += "      FLT4 src = args.src_tensor.Read(" + coords + ", s_layer);\n";
    c += "      FLT t0 = src.x;\n";
    c += "      int reminder = S % 4;\n";
    c += "      if (reminder == 1) t0 = src.y;\n";
    c += "      if (reminder == 2) t0 = src.z;\n";
    c += "      if (reminder == 3) t0 = src.w;\n";
    c += "      FLT4 src_final = INIT_FLT4v4(t0, t0, t0, t0);\n";
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
    const GpuInfo& gpu_info, const OperationDef& op_def, bool stride_correction,
    int channel_multiplier, bool weights_are_buffer, bool dynamic_weights,
    GPUOperation* op) {
  auto src_desc = op_def.src_tensors[0];
  if (op_def.IsBatchSupported()) {
    src_desc.SetStateVar("BatchedWidth", "true");
  }
  op->AddSrcTensor("src_tensor", src_desc);
  if (dynamic_weights) {
    op->AddSrcTensor("weights", op_def.src_tensors[1]);
  }

  auto dst_desc = op_def.dst_tensors[0];
  if (op_def.IsBatchSupported()) {
    dst_desc.SetStateVar("BatchedWidth", "true");
  }
  op->AddDstTensor("dst_tensor", dst_desc);

  std::string c;

  c += "MAIN_FUNCTION(\n";
  c += "$0) {\n";
  c += "  int X = GLOBAL_ID_0;\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "  int linear_id_1 = GLOBAL_ID_1;\n";
    c += "  int Y = linear_id_1 / args.dst_tensor.Depth();\n";
    c += "  int Z = linear_id_1 % args.dst_tensor.Depth();\n";
  } else {
    c += "  int Y = GLOBAL_ID_1;\n";
  }
  c += "  int S = GLOBAL_ID_2;\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "S >= args.dst_tensor.Slices()) { \n";
  c += "    return; \n";
  c += "  } \n";
  c += "  ACCUM_FLT4 r = INIT_ACCUM_FLT4(0.0f);\n";
  if (stride_correction) {
    c += "  int x_offseted = " +
         GetXStrideCorrectedV2("X", "args.src_tensor.Batch()", "args.stride_x",
                               "args.padding_x") +
         ";\n";
  } else {
    if (op_def.IsBatchSupported()) {
      c += "  int x_offseted = X * args.stride_x + args.padding_x * "
           "args.src_tensor.Batch();\n";
    } else {
      c += "  int x_offseted = X * args.stride_x + args.padding_x;\n";
    }
  }
  c += "  int y_offseted = Y * args.stride_y + args.padding_y;\n";
  if (!dynamic_weights) {
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
  }
  std::string kernel_size_x =
      dynamic_weights ? "args.weights.Width()" : "args.kernel_size_x";
  std::string kernel_size_y =
      dynamic_weights ? "args.weights.Height()" : "args.kernel_size_y";
  std::string kernel_size_z =
      dynamic_weights ? "args.weights.Depth()" : "args.kernel_size_z";

  auto generate_check = [&]() {
    std::string check;
    const std::vector<Axis> axes{Axis::WIDTH, Axis::HEIGHT, Axis::DEPTH};
    const std::vector<std::string> names{"outside_x", "outside_y", "outside_z"};
    for (int i = 0; i < axes.size(); ++i) {
      const auto& axis = axes[i];
      if (src_desc.HasAxis(axis) &&
          !src_desc.SupportsZeroClamp(axis, gpu_info)) {
        if (!check.empty()) {
          check += " && ";
        }
        check += "!" + names[i];
      }
    }
    return check;
  };
  auto generate_coords = [&]() {
    std::string check;
    const std::vector<Axis> axes{Axis::WIDTH, Axis::HEIGHT, Axis::DEPTH};
    const std::vector<std::string> names{"x_c", "y_c", "z_c"};
    for (int i = 0; i < axes.size(); ++i) {
      const auto& axis = axes[i];
      if (src_desc.HasAxis(axis)) {
        if (!check.empty()) {
          check += ", ";
        }
        check += names[i];
      }
    }
    return check;
  };
  const std::string check = generate_check();
  const std::string coords = generate_coords();

  if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "  for (int kz = 0; kz < " + kernel_size_z + "; ++kz) {\n";
    c += "    int z_c = z_offseted + kz * args.dilation_z;\n";
    if (!src_desc.SupportsZeroClamp(Axis::DEPTH, gpu_info)) {
      c += "    bool outside_z = z_c < 0 || z_c >= args.src_tensor.Depth();\n";
    }
  }
  if (op_def.dst_tensors[0].HasAxis(Axis::HEIGHT)) {
    c += "  for (int ky = 0; ky < " + kernel_size_y + "; ++ky) {\n";
    c += "    int y_c = y_offseted + ky * args.dilation_y;\n";
    if (!src_desc.SupportsZeroClamp(Axis::HEIGHT, gpu_info)) {
      c += "    bool outside_y = y_c < 0 || y_c >= args.src_tensor.Height();\n";
    }
  }
  if (op_def.dst_tensors[0].HasAxis(Axis::WIDTH)) {
    c += "  for (int kx = 0; kx < " + kernel_size_x + "; ++kx) {\n";
    const std::string dilation_x =
        op_def.IsBatchSupported() ? "args.dilation_x * args.src_tensor.Batch()"
                                  : "args.dilation_x";
    c += "    int x_c = x_offseted + kx * " + dilation_x + ";\n";
    if (!src_desc.SupportsZeroClamp(Axis::WIDTH, gpu_info)) {
      c += "    bool outside_x = x_c < 0 || x_c >= args.src_tensor.Width();\n";
    }
  }
  if (!check.empty()) {
    c += "    if (" + check + ") {\n";
  }
  if (dynamic_weights) {
    c += "      FLT4 f = args.weights.Read(kx, ky, S);\n";
  } else {
    if (weights_are_buffer) {
      c += "      FLT4 f = args.weights.Read(fx_c);\n";
    } else {
      c += "      FLT4 f = args.weights.Read(fx_c, S);\n";
    }
  }
  c += GetSrcValue(channel_multiplier, coords);
  c += "      r += TO_ACCUM_TYPE(src_final * f);\n";
  if (!check.empty()) {
    c += "    }\n";
  }
  if (!dynamic_weights) {
    c += "    fx_c++;\n";
  }
  if (op_def.dst_tensors[0].HasAxis(Axis::WIDTH)) {
    c += "  }\n";
  }
  if (op_def.dst_tensors[0].HasAxis(Axis::HEIGHT)) {
    c += "  }\n";
  }
  if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "  }\n";
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

bool UseBuffersForWeights(const GpuInfo& gpu_info) {
  if (gpu_info.IsApple()) {
    if (gpu_info.apple_info.IsA7GenerationGpu() ||
        gpu_info.apple_info.IsA8GenerationGpu()) {
      return false;
    }
  }
  return !gpu_info.SupportsImages() || gpu_info.IsMali() || gpu_info.IsApple();
}
}  // namespace

GPUOperation CreateDepthwiseConvolution2D(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const DepthwiseConvolution2DAttributes& attr) {
  const bool weights_are_buffer = UseBuffersForWeights(gpu_info);
  GPUOperation op(definition);
  op.args_.AddInt("kernel_size_x", attr.weights.shape.w);
  op.args_.AddInt("stride_x", attr.strides.w);
  op.args_.AddInt("padding_x", -attr.padding.prepended.w);
  op.args_.AddInt("dilation_x", attr.dilations.w);
  op.args_.AddInt("kernel_size_y", attr.weights.shape.h);
  op.args_.AddInt("stride_y", attr.strides.h);
  op.args_.AddInt("padding_y", -attr.padding.prepended.h);
  op.args_.AddInt("dilation_y", attr.dilations.h);
  if (!IsSpecializedCase(attr.weights.shape.o)) {
    op.args_.AddInt("ch_multiplier", attr.weights.shape.o);
  }
  const bool stride_correction =
      definition.IsBatchSupported() && attr.strides.w != 1;
  op.code_ = GenerateDepthwiseConvolutionCode(
      gpu_info, definition, stride_correction, attr.weights.shape.o,
      weights_are_buffer, false, &op);
  UploadWeightsForDWConv2D(attr.weights, weights_are_buffer,
                           definition.precision, &op);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;

  TensorLinearDescriptor desc;
  desc.storage_type = weights_are_buffer ? LinearStorageType::BUFFER
                                         : LinearStorageType::TEXTURE_2D;
  desc.element_type = definition.GetDataType();
  desc.UploadLinearData(attr.bias);
  op.args_.AddObject(
      "biases", absl::make_unique<TensorLinearDescriptor>(std::move(desc)));
  return op;
}

GPUOperation CreateDepthwiseConvolution2DDynamicWeights(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const DepthwiseConvolution2DAttributes& attr) {
  GPUOperation op(definition);
  op.args_.AddInt("stride_x", attr.strides.w);
  op.args_.AddInt("padding_x", -attr.padding.prepended.w);
  op.args_.AddInt("dilation_x", attr.dilations.w);
  op.args_.AddInt("stride_y", attr.strides.h);
  op.args_.AddInt("padding_y", -attr.padding.prepended.h);
  op.args_.AddInt("dilation_y", attr.dilations.h);
  const bool stride_correction =
      definition.IsBatchSupported() && attr.strides.w != 1;
  op.code_ = GenerateDepthwiseConvolutionCode(
      gpu_info, definition, stride_correction, 1, false, true, &op);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;

  TensorLinearDescriptor desc;
  desc.storage_type =
      !gpu_info.SupportsImages() || gpu_info.IsMali() || gpu_info.IsApple()
          ? LinearStorageType::BUFFER
          : LinearStorageType::TEXTURE_2D;
  desc.element_type = definition.GetDataType();
  desc.UploadLinearData(attr.bias);
  op.args_.AddObject(
      "biases", absl::make_unique<TensorLinearDescriptor>(std::move(desc)));
  return op;
}

GPUOperation CreateDepthwiseConvolution3D(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const DepthwiseConvolution3DAttributes& attr) {
  const bool weights_are_buffer = UseBuffersForWeights(gpu_info);
  GPUOperation op(definition);
  op.args_.AddInt("kernel_size_x", attr.weights.shape.w);
  op.args_.AddInt("stride_x", attr.strides.w);
  op.args_.AddInt("padding_x", -attr.padding.prepended.w);
  op.args_.AddInt("dilation_x", attr.dilations.w);
  op.args_.AddInt("kernel_size_y", attr.weights.shape.h);
  op.args_.AddInt("stride_y", attr.strides.h);
  op.args_.AddInt("padding_y", -attr.padding.prepended.h);
  op.args_.AddInt("dilation_y", attr.dilations.h);
  op.args_.AddInt("kernel_size_z", attr.weights.shape.d);
  op.args_.AddInt("stride_z", attr.strides.d);
  op.args_.AddInt("padding_z", -attr.padding.prepended.d);
  op.args_.AddInt("dilation_z", attr.dilations.d);
  if (!IsSpecializedCase(attr.weights.shape.o)) {
    op.args_.AddInt("ch_multiplier", attr.weights.shape.o);
  }
  const bool stride_correction =
      definition.IsBatchSupported() && attr.strides.w != 1;
  op.code_ = GenerateDepthwiseConvolutionCode(
      gpu_info, definition, stride_correction, attr.weights.shape.o,
      weights_are_buffer, false, &op);
  UploadWeightsForDWConv3D(attr.weights, weights_are_buffer,
                           definition.precision, &op);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;

  TensorLinearDescriptor desc;
  desc.storage_type = weights_are_buffer ? LinearStorageType::BUFFER
                                         : LinearStorageType::TEXTURE_2D;
  desc.element_type = definition.GetDataType();
  desc.UploadLinearData(attr.bias);
  op.args_.AddObject(
      "biases", absl::make_unique<TensorLinearDescriptor>(std::move(desc)));
  return op;
}

}  // namespace gpu
}  // namespace tflite
