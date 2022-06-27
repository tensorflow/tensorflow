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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/task/tensor_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_linear_desc.h"
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

bool UseBuffersForWeights(const GpuInfo& gpu_info) {
  if (gpu_info.IsApple()) {
    if (gpu_info.apple_info.IsA7GenerationGpu() ||
        gpu_info.apple_info.IsA8GenerationGpu()) {
      return false;
    }
  }
  return !gpu_info.SupportsImages() || gpu_info.IsMali() ||
         gpu_info.IsApple() || gpu_info.IsAMD();
}

void AppendToBack(const std::string& value, const std::string& delimeter,
                  std::string* result) {
  if (!result->empty()) {
    *result += delimeter;
  }
  *result += value;
}

void AppendToFront(const std::string& value, const std::string& delimeter,
                   std::string* result) {
  if (!result->empty()) {
    *result = delimeter + *result;
  }
  *result = value + *result;
}
}  // namespace

DepthwiseConv::DepthwiseConv(const OperationDef& definition,
                             const DepthwiseConvParams& params)
    : GPUOperation(definition), params_(params) {
  if (params.UseLocalMem()) {
    work_group_size_ = params.work_group_size;
  }
}

int3 DepthwiseConv::GetGridSize() const {
  const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
  const int grid_y = dst_[0]->Height() * dst_[0]->Depth();
  const int grid_z = dst_[0]->Slices();
  return int3(grid_x, grid_y, grid_z);
}

void DepthwiseConv::GetPossibleKernelWorkGroups(
    TuningType tuning_type, const GpuInfo& gpu_info,
    const KernelInfo& kernel_info, std::vector<int3>* work_groups) const {
  if (params_.UseLocalMem()) {
    work_groups->push_back(work_group_size_);
    return;
  }
  GetPossibleWorkGroups(tuning_type, gpu_info, kernel_info, grid_size_,
                        work_groups);
}

std::string DepthwiseConv::GenerateWeightsUpload(const GpuInfo& gpu_info) {
  const bool weights_are_buffer = UseBuffersForWeights(gpu_info);
  auto read_weight = [](bool weights_are_buffer, const std::string& lid,
                        int work_group_total_size) {
    if (weights_are_buffer) {
      return "args.weights.Read(S * " + std::to_string(work_group_total_size) +
             " + " + lid + ")";
    } else {
      return "args.weights.Read(" + lid + ", S)";
    }
  };
  std::string c;
  const int work_group_total_size = params_.GetWorkGroupTotalSize();
  c += "  __local FLT4 weights_cache[" +
       std::to_string(params_.GetKernelsTotalSize()) + "];\n";
  c += "  int linear_local_id = (LOCAL_ID_2 * GROUP_SIZE_1 + LOCAL_ID_1) * "
       "GROUP_SIZE_0 + LOCAL_ID_0;\n";
  const int groups = params_.GetKernelsTotalSize() / work_group_total_size;
  const int reminder = params_.GetKernelsTotalSize() % work_group_total_size;
  for (int i = 0; i < groups; ++i) {
    const std::string lid =
        "linear_local_id + " + std::to_string(work_group_total_size * i);
    c += "  weights_cache[" + lid +
         "] = " + read_weight(weights_are_buffer, lid, work_group_total_size) +
         ";\n";
  }
  if (reminder != 0) {
    const std::string lid =
        "linear_local_id + " + std::to_string(work_group_total_size * groups);
    c += "  if (linear_local_id < " + std::to_string(reminder) + ") {\n";
    c += "    weights_cache[" + lid +
         "] = " + read_weight(weights_are_buffer, lid, work_group_total_size) +
         ";\n";
    c += "  }\n";
  }
  return c;
}

std::string DepthwiseConv::GenerateCode(const GpuInfo& gpu_info) {
  const bool weights_are_buffer = UseBuffersForWeights(gpu_info);
  const bool dynamic_weights = definition_.src_tensors.size() == 2;
  AddSrcTensor("src_tensor", definition_.src_tensors[0]);
  if (dynamic_weights) {
    AddSrcTensor("weights", definition_.src_tensors[1]);
  }
  AddDstTensor("dst_tensor", definition_.dst_tensors[0]);

  std::string c;

  c += "MAIN_FUNCTION($0) {\n";
  if (definition_.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id = GLOBAL_ID_0;\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int X = GLOBAL_ID_0;\n";
  }
  if (definition_.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "  int linear_id_1 = GLOBAL_ID_1;\n";
    c += "  int Y = linear_id_1 / args.dst_tensor.Depth();\n";
    c += "  int Z = linear_id_1 % args.dst_tensor.Depth();\n";
  } else {
    c += "  int Y = GLOBAL_ID_1;\n";
  }
  c += "  int S = GLOBAL_ID_2;\n";
  if (params_.use_weights_caching) {
    c += GenerateWeightsUpload(gpu_info);
  }
  if (params_.UseLocalMem()) {
    c += "  LOCAL_MEM_BARRIER;\n";
  }
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "S >= args.dst_tensor.Slices()) { \n";
  c += "    return; \n";
  c += "  } \n";
  c += "  ACCUM_FLT4 r = INIT_ACCUM_FLT4(0.0f);\n";
  c += "  int x_offseted = X * args.stride_x + args.padding_x;\n";
  c += "  int y_offseted = Y * args.stride_y + args.padding_y;\n";
  if (definition_.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "  int z_offseted = Z * args.stride_z + args.padding_z;\n";
  }
  if (!dynamic_weights && !params_.use_weights_caching) {
    if (weights_are_buffer) {
      c += "  int fx_c = S * args.kernels_total_size;\n";
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

  std::string check, coords;
  if (definition_.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "  for (int kz = 0; kz < " + kernel_size_z + "; ++kz) {\n";
    c += "    int z_c = z_offseted + kz * args.dilation_z;\n";
    AppendToFront("z_c", ", ", &coords);
    if (!definition_.src_tensors[0].SupportsZeroClamp(Axis::DEPTH, gpu_info)) {
      c += "    bool inside_z = z_c >= 0 && z_c < args.src_tensor.Depth();\n";
      c += "    z_c = clamp(z_c, 0, args.src_tensor.Depth() - 1);\n";
      AppendToBack("inside_z", " && ", &check);
    }
  }
  if (definition_.dst_tensors[0].HasAxis(Axis::HEIGHT)) {
    c += "  for (int ky = 0; ky < " + kernel_size_y + "; ++ky) {\n";
    c += "    int y_c = y_offseted + ky * args.dilation_y;\n";
    AppendToFront("y_c", ", ", &coords);
    if (!definition_.src_tensors[0].SupportsZeroClamp(Axis::HEIGHT, gpu_info)) {
      c += "    bool inside_y = y_c >= 0 && y_c < args.src_tensor.Height();\n";
      c += "    y_c = clamp(y_c, 0, args.src_tensor.Height() - 1);\n";
      AppendToBack("inside_y", " && ", &check);
    }
  }
  if (definition_.dst_tensors[0].HasAxis(Axis::WIDTH)) {
    c += "  for (int kx = 0; kx < " + kernel_size_x + "; ++kx) {\n";
    c += "    int x_c = x_offseted + kx * args.dilation_x;\n";
    AppendToFront("x_c", ", ", &coords);
    if (!definition_.src_tensors[0].SupportsZeroClamp(Axis::WIDTH, gpu_info)) {
      c += "    bool inside_x = x_c >= 0 && x_c < args.src_tensor.Width();\n";
      c += "    x_c = clamp(x_c, 0, args.src_tensor.Width() - 1);\n";
      AppendToBack("inside_x", " && ", &check);
    }
  }
  std::string weight_value;
  if (params_.use_weights_caching) {
    std::string weight_index = "ky";
    if (definition_.dst_tensors[0].HasAxis(Axis::DEPTH)) {
      weight_index =
          "(kz * " + std::to_string(params_.y_kernel_size) + " + ky)";
    }
    weight_value = "weights_cache[" + weight_index + " * " +
                   std::to_string(params_.x_kernel_size) + " + kx]";
  } else {
    weight_value = "f";
    if (dynamic_weights) {
      c += "      FLT4 f = args.weights.Read(kx, ky, S);\n";
    } else {
      if (weights_are_buffer) {
        c += "      FLT4 f = args.weights.Read(fx_c);\n";
      } else {
        c += "      FLT4 f = args.weights.Read(fx_c, S);\n";
      }
    }
  }
  c += GetSrcValue(params_.channel_multiplier, coords);
  if (!check.empty()) {
    c += "      src_final = src_final * INIT_FLT(" + check + ");\n";
  }
  c += "      r += TO_ACCUM_TYPE(src_final * " + weight_value + ");\n";
  if (!dynamic_weights && !params_.use_weights_caching) {
    c += "    fx_c++;\n";
  }
  if (definition_.dst_tensors[0].HasAxis(Axis::WIDTH)) {
    c += "  }\n";
  }
  if (definition_.dst_tensors[0].HasAxis(Axis::HEIGHT)) {
    c += "  }\n";
  }
  if (definition_.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "  }\n";
  }
  c += "  FLT4 res0 = TO_FLT4(r) + args.biases.Read(S);\n";
  if (definition_.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "  args.dst_tensor.Write(res0, X, Y, Z, S);\n";
  } else {
    c += "  args.dst_tensor.Write(res0, X, Y, S);\n";
  }
  c += "}\n";
  return c;
}

DepthwiseConv CreateDepthwiseConvolution2D(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const DepthwiseConvolution2DAttributes& attr) {
  const bool weights_are_buffer = UseBuffersForWeights(gpu_info);
  DepthwiseConv::DepthwiseConvParams params;
  params.channel_multiplier = attr.weights.shape.o;
  DepthwiseConv op(definition, params);
  op.args_.AddInt("kernel_size_x", attr.weights.shape.w);
  op.args_.AddInt("stride_x", attr.strides.w);
  op.args_.AddInt("padding_x", -attr.padding.prepended.w);
  op.args_.AddInt("dilation_x", attr.dilations.w);
  op.args_.AddInt("kernel_size_y", attr.weights.shape.h);
  op.args_.AddInt("stride_y", attr.strides.h);
  op.args_.AddInt("padding_y", -attr.padding.prepended.h);
  op.args_.AddInt("dilation_y", attr.dilations.h);
  op.args_.AddInt("kernels_total_size",
                  attr.weights.shape.w * attr.weights.shape.h);
  if (!IsSpecializedCase(attr.weights.shape.o)) {
    op.args_.AddInt("ch_multiplier", attr.weights.shape.o);
  }
  op.code_ = op.GenerateCode(gpu_info);
  op.UploadWeightsForDWConv2D(attr.weights, weights_are_buffer);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;

  TensorLinearDescriptor desc;
  desc.storage_type = weights_are_buffer ? LinearStorageType::BUFFER
                                         : LinearStorageType::TEXTURE_2D;
  desc.element_type = definition.GetDataType();
  desc.UploadLinearData(attr.bias);
  op.args_.AddObject("biases",
                     std::make_unique<TensorLinearDescriptor>(std::move(desc)));
  return op;
}

DepthwiseConv CreateDepthwiseConvolution2DDynamicWeights(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const DepthwiseConvolution2DAttributes& attr) {
  DepthwiseConv::DepthwiseConvParams params;
  params.channel_multiplier = 1;
  DepthwiseConv op(definition, params);
  op.args_.AddInt("stride_x", attr.strides.w);
  op.args_.AddInt("padding_x", -attr.padding.prepended.w);
  op.args_.AddInt("dilation_x", attr.dilations.w);
  op.args_.AddInt("stride_y", attr.strides.h);
  op.args_.AddInt("padding_y", -attr.padding.prepended.h);
  op.args_.AddInt("dilation_y", attr.dilations.h);
  op.code_ = op.GenerateCode(gpu_info);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;

  TensorLinearDescriptor desc;
  desc.storage_type =
      !gpu_info.SupportsImages() || gpu_info.IsMali() || gpu_info.IsApple()
          ? LinearStorageType::BUFFER
          : LinearStorageType::TEXTURE_2D;
  desc.element_type = definition.GetDataType();
  desc.UploadLinearData(attr.bias);
  op.args_.AddObject("biases",
                     std::make_unique<TensorLinearDescriptor>(std::move(desc)));
  return op;
}

DepthwiseConv CreateDepthwiseConvolution3D(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const DepthwiseConvolution3DAttributes& attr) {
  const bool weights_are_buffer = UseBuffersForWeights(gpu_info);
  DepthwiseConv::DepthwiseConvParams params;
  params.channel_multiplier = attr.weights.shape.o;
  DepthwiseConv op(definition, params);
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
  op.args_.AddInt(
      "kernels_total_size",
      attr.weights.shape.w * attr.weights.shape.h * attr.weights.shape.d);
  if (!IsSpecializedCase(attr.weights.shape.o)) {
    op.args_.AddInt("ch_multiplier", attr.weights.shape.o);
  }
  op.code_ = op.GenerateCode(gpu_info);
  op.UploadWeightsForDWConv3D(attr.weights, weights_are_buffer);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;

  TensorLinearDescriptor desc;
  desc.storage_type = weights_are_buffer ? LinearStorageType::BUFFER
                                         : LinearStorageType::TEXTURE_2D;
  desc.element_type = definition.GetDataType();
  desc.UploadLinearData(attr.bias);
  op.args_.AddObject("biases",
                     std::make_unique<TensorLinearDescriptor>(std::move(desc)));
  return op;
}

}  // namespace gpu
}  // namespace tflite
