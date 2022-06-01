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

#include "tensorflow/lite/delegates/gpu/common/tasks/convolution_transposed_3x3_thin.h"

#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"

namespace tflite {
namespace gpu {

ConvolutionTransposed3x3Thin::ConvolutionTransposed3x3Thin(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const ConvolutionTransposedAttributes& attr)
    : GPUOperation(definition) {
  if (gpu_info.IsApple()) {
    weights_layout_ = WeightsLayout::kOICustomSpatialO4I4;
  } else {
    weights_layout_ = WeightsLayout::kOICustomSpatialI4O4;
  }
  code_ = GenerateConvolutionTransposedCode(
      definition_, DivideRoundUp(attr.weights.shape.i, 4),
      DivideRoundUp(attr.weights.shape.o, 4));
}

std::string ConvolutionTransposed3x3Thin::GenerateConvolutionTransposedCode(
    const OperationDef& op_def, int src_depth, int dst_depth) {
  auto src_desc = op_def.src_tensors[0];
  src_desc.SetAddressMode(AddressMode::kZero);
  AddSrcTensor("src_tensor", src_desc);
  AddDstTensor("dst_tensor", op_def.dst_tensors[0]);

  if (op_def.src_tensors.size() == 2) {
    // dynamic weights
    BufferDescriptor desc;
    desc.element_type = op_def.src_tensors[1].data_type;
    desc.element_size = 4;
    desc.memory_type = MemoryType::CONSTANT;
    AddSrcBuffer("weights", desc);
  }

  const auto src_tensor_type = op_def.src_tensors[0].storage_type;

  std::string c;

  if (GetWeightsDescription().IsI4O4()) {
    switch (op_def.precision) {
      case CalculationsPrecision::F32:
      case CalculationsPrecision::F16:
        c += "#define CONV(R, SRC, F, i) \\\n";
        c += "  R += SRC.x * F[i + 0]; \\\n";
        c += "  R += SRC.y * F[i + 1]; \\\n";
        c += "  R += SRC.z * F[i + 2]; \\\n";
        c += "  R += SRC.w * F[i + 3];   \n";
        break;
      case CalculationsPrecision::F32_F16:
        c += "#define CONV(R, SRC, F, i) \\\n";
        c += "  R += TO_ACCUM_TYPE(SRC.x * F[i + 0] + SRC.y * F[i + 1]";
        c += "+ SRC.z * F[i + 2] + SRC.w * F[i + 3]);\n";
        break;
    }
  } else {
    // O4I4
    c += "#define CONV(R, SRC, F, i) \\\n";
    c += "  R.x += dot(SRC, F[i + 0]); \\\n";
    c += "  R.y += dot(SRC, F[i + 1]); \\\n";
    c += "  R.z += dot(SRC, F[i + 2]); \\\n";
    c += "  R.w += dot(SRC, F[i + 3]);   \n";
  }

  c += "MAIN_FUNCTION($0) {\n";
  if (op_def.IsBatchSupported()) {
    c += "  int linear_id = GLOBAL_ID_0;\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int X = GLOBAL_ID_0;\n";
  }
  c += "  int Y = GLOBAL_ID_1;\n";
  c += "  if (X >= args.src_tensor.Width() || Y >= args.src_tensor.Height()) "
       "return;\n";
  for (int d = 0; d < dst_depth; ++d) {
    const std::string layer = std::to_string(d);
    c += "  ACCUM_FLT4 r" + layer + "[2][2];\n";
    c += "  r" + layer + "[0][0] = INIT_ACCUM_FLT4(0.0f);\n";
    c += "  r" + layer + "[0][1] = INIT_ACCUM_FLT4(0.0f);\n";
    c += "  r" + layer + "[1][0] = INIT_ACCUM_FLT4(0.0f);\n";
    c += "  r" + layer + "[1][1] = INIT_ACCUM_FLT4(0.0f);\n";
  }
  int filters_index = 0;
  for (int s = 0; s < src_depth; ++s) {
    const std::string z = std::to_string(s);
    c += "  {\n";
    if (src_tensor_type == TensorStorageType::BUFFER) {
      c += "  bool x_in = X + 1 < args.src_tensor.Width();\n";
      c += "  bool y_in = Y + 1 < args.src_tensor.Height();\n";
      c += "  FLT4 src0 = args.src_tensor.Read(X, Y, " + z + ");\n";
      c += "  FLT4 src1 = INIT_FLT4(0.0);\n";
      c += "  FLT4 src2 = INIT_FLT4(0.0);\n";
      c += "  FLT4 src3 = INIT_FLT4(0.0);\n";
      c += "  if (x_in) {\n";
      c += "    src1 = args.src_tensor.Read(X + 1, Y, " + z + ");\n";
      c += "  }\n";
      c += "  if (y_in) {\n";
      c += "    src2 = args.src_tensor.Read(X, Y + 1, " + z + ");\n";
      c += "  }\n";
      c += "  if (x_in && y_in) {\n";
      c += "    src3 = args.src_tensor.Read(X + 1, Y + 1, " + z + ");\n";
      c += "  }\n";
    } else if (src_tensor_type == TensorStorageType::IMAGE_BUFFER) {
      c += "  args.src_tensor.GetAddress(c0, X, Y, " + z + ");\n";
      c += "  args.src_tensor.GetAddress(c1, X + 1, Y, " + z + ");\n";
      c += "  args.src_tensor.GetAddress(c2, X, Y + 1, " + z + ");\n";
      c += "  args.src_tensor.GetAddress(c3, X + 1, Y + 1, " + z + ");\n";
      c += "  bool x_in = X + 1 < args.src_tensor.Width();\n";
      c += "  bool y_in = Y + 1 < args.src_tensor.Height();\n";
      c += "  c1 = select(-1, c1, x_in);\n";
      c += "  c2 = select(-1, c2, y_in);\n";
      c += "  c3 = select(-1, c3, x_in && y_in);\n";
      c += "  FLT4 src0 = args.src_tensor.Read(c0);\n";
      c += "  FLT4 src1 = args.src_tensor.Read(c1);\n";
      c += "  FLT4 src2 = args.src_tensor.Read(c2);\n";
      c += "  FLT4 src3 = args.src_tensor.Read(c3);\n";
    } else {
      c += "  FLT4 src0 = args.src_tensor.Read(X, Y, " + z + ");\n";
      c += "  FLT4 src1 = args.src_tensor.Read(X + 1, Y, " + z + ");\n";
      c += "  FLT4 src2 = args.src_tensor.Read(X, Y + 1, " + z + ");\n";
      c += "  FLT4 src3 = args.src_tensor.Read(X + 1, Y + 1, " + z + ");\n";
    }
    for (int d = 0; d < dst_depth; ++d) {
      const std::string layer = std::to_string(d);
      const std::string f_offset = std::to_string(filters_index);
      filters_index++;
      c += "  {\n";
      c += "  __constant FLT4* L0 = args.weights.GetPtr() + 36 * " + f_offset +
           ";\n";
      c += "  CONV(r" + layer + "[0][0], src0, L0, 0);\n";
      c += "  CONV(r" + layer + "[0][1], src0, L0, 4);\n";
      c += "  CONV(r" + layer + "[0][1], src1, L0, 8);\n";
      c += "  CONV(r" + layer + "[1][0], src0, L0, 12);\n";
      c += "  CONV(r" + layer + "[1][0], src2, L0, 16);\n";
      c += "  CONV(r" + layer + "[1][1], src0, L0, 20);\n";
      c += "  CONV(r" + layer + "[1][1], src1, L0, 24);\n";
      c += "  CONV(r" + layer + "[1][1], src2, L0, 28);\n";
      c += "  CONV(r" + layer + "[1][1], src3, L0, 32);\n";
      c += "  }\n";
    }
    c += "  }\n";
  }
  c += "  X *= 2;\n";
  c += "  Y *= 2;\n";
  for (int d = 0; d < dst_depth; ++d) {
    const std::string layer = std::to_string(d);
    c += "  {\n";
    c += "  FLT4 bias_val = args.biases.Read(" + layer + ");\n";
    for (int y = 0; y < 2; ++y) {
      for (int x = 0; x < 2; ++x) {
        const std::string x_coord = "X + " + std::to_string(x);
        const std::string y_coord = "Y + " + std::to_string(y);
        c += "  {\n";
        c += "    FLT4 result = TO_FLT4(r" + layer + "[" + std::to_string(y) +
             "][" + std::to_string(x) + "]) + bias_val;\n";
        c += "    args.dst_tensor.Write(result, " + x_coord + ", " + y_coord +
             ", " + layer + ");\n";
        c += "  }\n";
      }
    }
    c += "  }\n";
  }
  c += "}\n";

  return c;
}

int3 ConvolutionTransposed3x3Thin::GetGridSize() const {
  const int grid_x = src_[0]->Width() * dst_[0]->Batch();
  const int grid_y = src_[0]->Height();
  const int grid_z = 1;
  return int3(grid_x, grid_y, grid_z);
}

std::vector<int> ConvolutionTransposed3x3Thin::GetSpatialWeightsRemap() const {
  return std::vector<int>{4, 5, 3, 7, 1, 8, 6, 2, 0};
}

void ConvolutionTransposed3x3Thin::UploadWeights(
    const tflite::gpu::Tensor<OHWI, DataType::FLOAT32>& weights) {
  const auto weights_desc = GetWeightsDescription();
  const int flt_count =
      GetTotalElementsCountForLayout(weights_desc, weights.shape);

  BufferDescriptor desc;
  desc.element_type = weights_desc.type;
  desc.element_size = 4;
  desc.memory_type = MemoryType::CONSTANT;
  desc.size = flt_count * SizeOf(desc.element_type);
  desc.data.resize(desc.size);

  RearrangeWeights(weights, weights_desc, absl::MakeSpan(desc.data));

  args_.AddObject("weights",
                  absl::make_unique<BufferDescriptor>(std::move(desc)));
}

bool IsConvolutionTransposed3x3ThinSupported(
    const ConvolutionTransposedAttributes& attr) {
  return attr.weights.shape.o <= 8 && attr.weights.shape.w == 3 &&
         attr.weights.shape.h == 3 && attr.stride.w == 2 &&
         attr.stride.h == 2 && attr.padding.prepended.w == 1 &&
         attr.padding.prepended.h == 1 && attr.padding.appended.w == 1 &&
         attr.padding.appended.h == 1;
}

ConvolutionTransposed3x3Thin CreateConvolutionTransposed3x3Thin(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const ConvolutionTransposedAttributes& attr) {
  ConvolutionTransposed3x3Thin result(gpu_info, definition, attr);
  result.UploadWeights(attr.weights);

  TensorLinearDescriptor desc;
  desc.storage_type = LinearStorageType::TEXTURE_2D;
  desc.element_type = definition.GetDataType();
  desc.UploadLinearData(attr.bias);
  result.args_.AddObject(
      "biases", absl::make_unique<TensorLinearDescriptor>(std::move(desc)));
  return result;
}

ConvolutionTransposed3x3Thin CreateConvolutionTransposed3x3ThinDynamicWeights(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const ConvolutionTransposedAttributes& attr) {
  OperationDef new_def = definition;
  new_def.src_tensors = {
      definition.src_tensors[0]};  // leaving only src_tensor def, weights defs
                                   // will be added later
  const DataType weights_type = definition.GetDataType();
  // add 1 src_tensor(buffer) for weights
  new_def.src_tensors.push_back(
      {weights_type, TensorStorageType::BUFFER, Layout::HWC});
  ConvolutionTransposed3x3Thin result(gpu_info, new_def, attr);

  TensorLinearDescriptor desc;
  desc.storage_type = LinearStorageType::TEXTURE_2D;
  desc.element_type = new_def.GetDataType();
  desc.UploadLinearData(attr.bias);
  result.args_.AddObject(
      "biases", absl::make_unique<TensorLinearDescriptor>(std::move(desc)));
  return result;
}

}  // namespace gpu
}  // namespace tflite
