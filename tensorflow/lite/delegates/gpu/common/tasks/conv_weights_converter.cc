/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/common/tasks/conv_weights_converter.h"

#include <cstring>
#include <string>
#include <utility>

#include "tensorflow/lite/delegates/gpu/common/task/util.h"
#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"

namespace tflite {
namespace gpu {

ConverterToConvWeights::ConverterToConvWeights(
    const OperationDef& definition, const WeightsDescription& weights_desc)
    : GPUOperation(definition), weights_desc_(weights_desc) {
  code_ = GetConverterToConvWeightsCode(definition_, weights_desc_);
}

ConverterToConvWeights::ConverterToConvWeights(
    ConverterToConvWeights&& operation)
    : GPUOperation(std::move(operation)),
      weights_desc_(std::move(operation.weights_desc_)) {}

ConverterToConvWeights& ConverterToConvWeights::operator=(
    ConverterToConvWeights&& operation) {
  if (this != &operation) {
    weights_desc_ = std::move(operation.weights_desc_);
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

std::string ConverterToConvWeights::GetConverterToConvWeightsCode(
    const OperationDef& op_def, const WeightsDescription& conv_weights_desc) {
  AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  args_.AddFloat("mask_x");
  args_.AddFloat("mask_y");
  args_.AddFloat("mask_z");
  args_.AddFloat("mask_w");
  args_.AddInt("grid_x_size");

  if (conv_weights_desc.layout == WeightsLayout::kOICustomSpatialI4O4 ||
      conv_weights_desc.layout == WeightsLayout::kOICustomSpatialO4I4) {
    std::vector<int32_t> remap(conv_weights_desc.spatial_remap.size());
    for (int i = 0; i < remap.size(); ++i) {
      remap[i] = conv_weights_desc.spatial_remap[i];
    }
    BufferDescriptor desc;
    desc.element_type = DataType::INT32;
    desc.element_size = 1;
    desc.memory_type = MemoryType::GLOBAL;
    desc.size = remap.size() * sizeof(int32_t);
    desc.data.resize(desc.size);
    std::memcpy(desc.data.data(), remap.data(), desc.size);
    args_.AddObject("spatial_remap",
                    absl::make_unique<BufferDescriptor>(std::move(desc)));
  }

  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  c += "  int O = GLOBAL_ID_0;\n";
  c += "  int I = GLOBAL_ID_1;\n";
  c += "  int Z = GLOBAL_ID_2;\n";
  c += "  int W = Z % args.src_tensor.Width();\n";
  c += "  int H = Z / args.src_tensor.Width();\n";
  c += "  if (O >= args.grid_x_size || I >= args.src_tensor.Slices() || "
       "H >= args.src_tensor.Height()) return;\n";
  c += "  O *= 4;\n";
  std::string x_kern = "W";
  std::string y_kern = "H";
  if (conv_weights_desc.layout == WeightsLayout::kOICustomSpatialI4O4 ||
      conv_weights_desc.layout == WeightsLayout::kOICustomSpatialO4I4) {
    c += "  int spatial_linear = H * args.src_tensor.Width() + W;\n";
    c += "  int linear_remap = args.spatial_remap.Read(spatial_linear);\n";
    c += "  int w_remap = linear_remap % args.src_tensor.Width();\n";
    c += "  int h_remap = linear_remap / args.src_tensor.Width();\n";
    x_kern = "w_remap";
    y_kern = "h_remap";
  }
  const std::string coords = x_kern + ", " + y_kern;
  c += "  FLT4 v0 = INIT_FLT4(0.0f);\n";
  c += "  FLT4 v1 = INIT_FLT4(0.0f);\n";
  c += "  FLT4 v2 = INIT_FLT4(0.0f);\n";
  c += "  FLT4 v3 = INIT_FLT4(0.0f);\n";
  c += "  if (O < args.src_tensor.Batch()) {\n";
  c += "    v0 = args.src_tensor.Read(" + coords + ", I, O);\n";
  c += "  }\n";
  c += "  if (O + 1 < args.src_tensor.Batch()) {\n";
  c += "    v1 = args.src_tensor.Read(" + coords + ", I, O + 1);\n";
  c += "  }\n";
  c += "  if (O + 2 < args.src_tensor.Batch()) {\n";
  c += "    v2 = args.src_tensor.Read(" + coords + ", I, O + 2);\n";
  c += "  }\n";
  c += "  if (O + 3 < args.src_tensor.Batch()) {\n";
  c += "    v3 = args.src_tensor.Read(" + coords + ", I, O + 3);\n";
  c += "  }\n";
  c += "  if (I == args.src_tensor.Slices() - 1) {\n";
  c += "    FLT4 mask = INIT_FLT4v4(args.mask_x, args.mask_y, args.mask_z, "
       "args.mask_w);\n";
  c += "    v0 *= mask;\n";
  c += "    v1 *= mask;\n";
  c += "    v2 *= mask;\n";
  c += "    v3 *= mask;\n";
  c += "  }\n";
  if (conv_weights_desc.IsI4O4()) {
    c += "  FLT4 r0 = INIT_FLT4v4(v0.x, v1.x, v2.x, v3.x);\n";
    c += "  FLT4 r1 = INIT_FLT4v4(v0.y, v1.y, v2.y, v3.y);\n";
    c += "  FLT4 r2 = INIT_FLT4v4(v0.z, v1.z, v2.z, v3.z);\n";
    c += "  FLT4 r3 = INIT_FLT4v4(v0.w, v1.w, v2.w, v3.w);\n";
  } else if (conv_weights_desc.IsO4I4()) {
    c += "  FLT4 r0 = v0;\n";
    c += "  FLT4 r1 = v1;\n";
    c += "  FLT4 r2 = v2;\n";
    c += "  FLT4 r3 = v3;\n";
  }
  if (conv_weights_desc.layout ==
          WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4 ||
      conv_weights_desc.layout ==
          WeightsLayout::k2DX4O4YIsSpatialIAndXIsOOGroupI4) {
    // Writing to 4X Textures 2D
    AddDstTensor("dst_tensor0", op_def.dst_tensors[0]);
    AddDstTensor("dst_tensor1", op_def.dst_tensors[1]);
    AddDstTensor("dst_tensor2", op_def.dst_tensors[2]);
    AddDstTensor("dst_tensor3", op_def.dst_tensors[3]);
    c += "  int yc = (H * args.src_tensor.Width() + W) * "
         "args.src_tensor.Slices() + I;\n";
    c += "  args.dst_tensor0.Write2D(r0, O / 4, yc);\n";
    c += "  args.dst_tensor1.Write2D(r1, O / 4, yc);\n";
    c += "  args.dst_tensor2.Write2D(r2, O / 4, yc);\n";
    c += "  args.dst_tensor3.Write2D(r3, O / 4, yc);\n";
    c += "}\n";
  } else {
    // Writing to linear buffer
    AddDstTensor("dst_tensor", op_def.dst_tensors[0]);
    c += "  int GROUP_SIZE = " +
         std::to_string(conv_weights_desc.GetOutputGroupSize()) + ";\n";
    c += "  int d_index = O / (GROUP_SIZE * 4);\n";
    c += "  int k_index = (O % (GROUP_SIZE * 4)) / 4;\n";
    std::string index;
    if (conv_weights_desc.layout == WeightsLayout::kOICustomSpatialI4O4 ||
        conv_weights_desc.layout == WeightsLayout::kOICustomSpatialO4I4) {
      index =
          "((d_index * args.src_tensor.Slices() + I) * "
          "args.src_tensor.Height() "
          "+ H) * args.src_tensor.Width() + W";
    } else if (conv_weights_desc.layout ==
                   WeightsLayout::kOSpatialIOGroupI4O4 ||
               conv_weights_desc.layout ==
                   WeightsLayout::kOSpatialIOGroupO4I4) {
      index =
          "((d_index * args.src_tensor.Height() + H) * args.src_tensor.Width() "
          "+ "
          "W) * args.src_tensor.Slices() + I";
    }
    c += "  int dst_offset = (" + index + ") * GROUP_SIZE + k_index;\n";
    c += "  args.dst_tensor.WriteLinear(r0, dst_offset * 4 + 0);\n";
    c += "  args.dst_tensor.WriteLinear(r1, dst_offset * 4 + 1);\n";
    c += "  args.dst_tensor.WriteLinear(r2, dst_offset * 4 + 2);\n";
    c += "  args.dst_tensor.WriteLinear(r3, dst_offset * 4 + 3);\n";
    c += "}\n";
  }
  return c;
}

absl::Status ConverterToConvWeights::BindArguments(ArgumentsBinder* args) {
  const int out_group_size = weights_desc_.GetOutputGroupSize();
  const int grid_x =
      DivideRoundUp(AlignByN(src_[0]->Batch(), 4 * out_group_size), 4);
  RETURN_IF_ERROR(args->SetInt("grid_x_size", grid_x));
  float4 mask = GetMaskForLastPlane(src_[0]->Channels());
  RETURN_IF_ERROR(args->SetFloat("mask_x", mask.x));
  RETURN_IF_ERROR(args->SetFloat("mask_y", mask.y));
  RETURN_IF_ERROR(args->SetFloat("mask_z", mask.z));
  return args->SetFloat("mask_w", mask.w);
}

int3 ConverterToConvWeights::GetGridSize() const {
  const int out_group_size = weights_desc_.GetOutputGroupSize();
  const int grid_x =
      DivideRoundUp(AlignByN(src_[0]->Batch(), 4 * out_group_size), 4);
  const int grid_y = src_[0]->Slices();
  const int grid_z = src_[0]->Width() * src_[0]->Height();
  return int3(grid_x, grid_y, grid_z);
}

ConverterToConvWeights CreateConverterToConvWeights(
    const OperationDef& definition, const WeightsDescription& weights_desc) {
  return ConverterToConvWeights(definition, weights_desc);
}

}  // namespace gpu
}  // namespace tflite
