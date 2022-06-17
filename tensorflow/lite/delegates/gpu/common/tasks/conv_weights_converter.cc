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
#include <memory>
#include <string>
#include <utility>

#include "tensorflow/lite/delegates/gpu/common/task/util.h"

namespace tflite {
namespace gpu {

ConverterToConvWeights::ConverterToConvWeights(
    const OperationDef& definition, const WeightsDescription& weights_desc)
    : GPUOperation(definition), weights_desc_(weights_desc) {
  code_ = GetConverterToConvWeightsCode();
}

std::string ConverterToConvWeights::GetConverterToConvWeightsCode() {
  AddSrcTensor("src_tensor", definition_.src_tensors[0]);
  args_.AddFloat("mask_x");
  args_.AddFloat("mask_y");
  args_.AddFloat("mask_z");
  args_.AddFloat("mask_w");
  args_.AddInt("out_ch");
  args_.AddInt("out_ch_x4_groups");
  args_.AddInt("in_ch");
  args_.AddInt("in_ch_x4_groups");
  args_.AddInt("kernel_width");
  args_.AddInt("kernel_height");
  args_.AddInt("kernel_spatial_size");

  if (weights_desc_.layout == WeightsLayout::kOICustomSpatialI4O4 ||
      weights_desc_.layout == WeightsLayout::kOICustomSpatialO4I4) {
    std::vector<int32_t> remap(weights_desc_.spatial_remap.size());
    for (int i = 0; i < remap.size(); ++i) {
      remap[i] = weights_desc_.spatial_remap[i];
    }
    BufferDescriptor desc;
    desc.element_type = DataType::INT32;
    desc.element_size = 1;
    desc.memory_type = MemoryType::GLOBAL;
    desc.size = remap.size() * sizeof(int32_t);
    desc.data.resize(desc.size);
    std::memcpy(desc.data.data(), remap.data(), desc.size);
    args_.AddObject("spatial_remap",
                    std::make_unique<BufferDescriptor>(std::move(desc)));
  }

  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  c += "  int O = GLOBAL_ID_0;\n";
  c += "  int I = GLOBAL_ID_1;\n";
  c += "  int spatial_linear = GLOBAL_ID_2;\n";
  c += "  if (O >= args.out_ch_x4_groups) return;\n";
  c += "  if (I >= args.in_ch_x4_groups) return;\n";
  c += "  if (spatial_linear >= args.kernel_spatial_size) return;\n";
  if (weights_desc_.layout == WeightsLayout::kOICustomSpatialI4O4 ||
      weights_desc_.layout == WeightsLayout::kOICustomSpatialO4I4) {
    c += "  int linear_remap = args.spatial_remap.Read(spatial_linear);\n";
    c += "  int W = linear_remap % args.kernel_width;\n";
    c += "  int H = linear_remap / args.kernel_width;\n";
  } else {
    c += "  int W = spatial_linear % args.kernel_width;\n";
    c += "  int H = spatial_linear / args.kernel_width;\n";
  }
  // W and H is src coordinates, spatial_linear is dst coordinate
  c += "  FLT4 v0 = INIT_FLT4(0.0f);\n";
  c += "  FLT4 v1 = INIT_FLT4(0.0f);\n";
  c += "  FLT4 v2 = INIT_FLT4(0.0f);\n";
  c += "  FLT4 v3 = INIT_FLT4(0.0f);\n";
  // OHWI as BHWC: Read(WHSB) - Read(WHIO)
  c += "  if (O * 4 < args.out_ch) {\n";
  c += "    v0 = args.src_tensor.Read(W, H, I, O * 4);\n";
  c += "  }\n";
  c += "  if (O * 4 + 1 < args.out_ch) {\n";
  c += "    v1 = args.src_tensor.Read(W, H, I, O * 4 + 1);\n";
  c += "  }\n";
  c += "  if (O * 4 + 2 < args.out_ch) {\n";
  c += "    v2 = args.src_tensor.Read(W, H, I, O * 4 + 2);\n";
  c += "  }\n";
  c += "  if (O * 4 + 3 < args.out_ch) {\n";
  c += "    v3 = args.src_tensor.Read(W, H, I, O * 4 + 3);\n";
  c += "  }\n";
  c += "  if (I == args.src_tensor.Slices() - 1) {\n";
  c += "    FLT4 mask = INIT_FLT4v4(args.mask_x, args.mask_y, args.mask_z, "
       "args.mask_w);\n";
  c += "    v0 *= mask;\n";
  c += "    v1 *= mask;\n";
  c += "    v2 *= mask;\n";
  c += "    v3 *= mask;\n";
  c += "  }\n";
  const bool need_transpose = weights_desc_.IsI4O4();
  if (need_transpose) {
    c += "  FLT4 r0 = INIT_FLT4v4(v0.x, v1.x, v2.x, v3.x);\n";
    c += "  FLT4 r1 = INIT_FLT4v4(v0.y, v1.y, v2.y, v3.y);\n";
    c += "  FLT4 r2 = INIT_FLT4v4(v0.z, v1.z, v2.z, v3.z);\n";
    c += "  FLT4 r3 = INIT_FLT4v4(v0.w, v1.w, v2.w, v3.w);\n";
  } else {
    c += "  FLT4 r0 = v0;\n";
    c += "  FLT4 r1 = v1;\n";
    c += "  FLT4 r2 = v2;\n";
    c += "  FLT4 r3 = v3;\n";
  }
  if (weights_desc_.layout ==
          WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4 ||
      weights_desc_.layout ==
          WeightsLayout::k2DX4O4YIsSpatialIAndXIsOOGroupI4) {
    // Writing to 4X Textures 2D
    AddDstTensor("dst_tensor0", definition_.dst_tensors[0]);
    AddDstTensor("dst_tensor1", definition_.dst_tensors[1]);
    AddDstTensor("dst_tensor2", definition_.dst_tensors[2]);
    AddDstTensor("dst_tensor3", definition_.dst_tensors[3]);
    c += "  int yc = spatial_linear *  args.in_ch_x4_groups + I;\n";
    c += "  args.dst_tensor0.Write2D(r0, O, yc);\n";
    c += "  args.dst_tensor1.Write2D(r1, O, yc);\n";
    c += "  args.dst_tensor2.Write2D(r2, O, yc);\n";
    c += "  args.dst_tensor3.Write2D(r3, O, yc);\n";
    c += "}\n";
  } else {
    // Writing to linear buffer
    AddDstTensor("dst_tensor", definition_.dst_tensors[0]);
    c += "  int OUTPUT_GROUP_SIZE = " +
         std::to_string(weights_desc_.GetOutputGroupSize()) + ";\n";
    c += "  int d_index = (O * 4) / (OUTPUT_GROUP_SIZE * 4);\n";
    c += "  int k_index = ((O * 4) % (OUTPUT_GROUP_SIZE * 4)) / 4;\n";
    std::string index;
    if (weights_desc_.layout == WeightsLayout::kOICustomSpatialI4O4 ||
        weights_desc_.layout == WeightsLayout::kOICustomSpatialO4I4) {
      index =
          "(d_index * args.in_ch_x4_groups + I) * args.kernel_spatial_size + "
          "spatial_linear";
    } else if (weights_desc_.layout == WeightsLayout::kOSpatialIOGroupI4O4 ||
               weights_desc_.layout == WeightsLayout::kOSpatialIOGroupO4I4) {
      index =
          "(d_index * args.kernel_spatial_size + spatial_linear) * "
          "args.in_ch_x4_groups + I";
    }
    c += "  int dst_offset = (" + index + ") * OUTPUT_GROUP_SIZE + k_index;\n";
    c += "  args.dst_tensor.WriteLinear(r0, dst_offset * 4 + 0);\n";
    c += "  args.dst_tensor.WriteLinear(r1, dst_offset * 4 + 1);\n";
    c += "  args.dst_tensor.WriteLinear(r2, dst_offset * 4 + 2);\n";
    c += "  args.dst_tensor.WriteLinear(r3, dst_offset * 4 + 3);\n";
    c += "}\n";
  }
  return c;
}

OHWI ConverterToConvWeights::GetWeightsSize() const {
  const int output_channels = src_[0]->Batch();
  const int input_channels = src_[0]->Channels();
  const int kernel_width = src_[0]->Width();
  const int kernel_height = src_[0]->Height();
  return OHWI(output_channels, kernel_height, kernel_width, input_channels);
}

absl::Status ConverterToConvWeights::BindArguments(ArgumentsBinder* args) {
  const auto& weights_shape = GetWeightsSize();
  const int output_channels_x4_groups = DivideRoundUp(
      AlignByN(weights_shape.o, 4 * weights_desc_.GetOutputGroupSize()), 4);
  RETURN_IF_ERROR(args->SetInt("out_ch", weights_shape.o));
  RETURN_IF_ERROR(args->SetInt("out_ch_x4_groups", output_channels_x4_groups));
  RETURN_IF_ERROR(args->SetInt("in_ch", weights_shape.i));
  RETURN_IF_ERROR(
      args->SetInt("in_ch_x4_groups", DivideRoundUp(weights_shape.i, 4)));
  RETURN_IF_ERROR(args->SetInt("kernel_width", weights_shape.w));
  RETURN_IF_ERROR(args->SetInt("kernel_height", weights_shape.h));
  RETURN_IF_ERROR(
      args->SetInt("kernel_spatial_size", weights_shape.w * weights_shape.h));
  float4 mask = GetMaskForLastPlane(src_[0]->Channels());
  RETURN_IF_ERROR(args->SetFloat("mask_x", mask.x));
  RETURN_IF_ERROR(args->SetFloat("mask_y", mask.y));
  RETURN_IF_ERROR(args->SetFloat("mask_z", mask.z));
  return args->SetFloat("mask_w", mask.w);
}

int3 ConverterToConvWeights::GetGridSize() const {
  const auto& weights_shape = GetWeightsSize();
  const int out_group_size = weights_desc_.GetOutputGroupSize();
  const int grid_x =
      DivideRoundUp(AlignByN(weights_shape.o, 4 * out_group_size), 4);
  const int grid_y = DivideRoundUp(weights_shape.i, 4);
  const int grid_z = weights_shape.w * weights_shape.h;
  return int3(grid_x, grid_y, grid_z);
}

ConverterToConvWeights CreateConverterToConvWeights(
    const OperationDef& definition, const WeightsDescription& weights_desc) {
  return ConverterToConvWeights(definition, weights_desc);
}

}  // namespace gpu
}  // namespace tflite
