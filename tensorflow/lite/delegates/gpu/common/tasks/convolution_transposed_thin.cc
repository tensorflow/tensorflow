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

#include "tensorflow/lite/delegates/gpu/common/tasks/convolution_transposed_thin.h"

#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"

namespace tflite {
namespace gpu {

ConvolutionTransposedThin::ConvolutionTransposedThin(
    const OperationDef& definition, const ConvolutionTransposedAttributes& attr,
    const GpuInfo& gpu_info)
    : GPUOperation(definition) {
  code_ = GenerateConvolutionTransposedCode(
      definition_, DivideRoundUp(attr.weights.shape.i, 4), attr.weights.shape.o,
      int2(attr.weights.shape.w, attr.weights.shape.h));
  if (definition_.precision == CalculationsPrecision::F16 &&
      gpu_info.IsAdreno() && gpu_info.adreno_info.IsAdreno3xx()) {
    compiler_options_.push_back(CompilerOptions::kAdrenoFullSimd);
  }
}

ConvolutionTransposedThin::ConvolutionTransposedThin(
    ConvolutionTransposedThin&& operation)
    : GPUOperation(std::move(operation)) {}

ConvolutionTransposedThin& ConvolutionTransposedThin::operator=(
    ConvolutionTransposedThin&& operation) {
  if (this != &operation) {
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

std::string ConvolutionTransposedThin::GenerateConvolutionTransposedCode(
    const OperationDef& op_def, int src_depth, int dst_channels,
    const int2& kernel_size) {
  AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  AddDstTensor("dst_tensor", op_def.dst_tensors[0]);

  const std::string channel_x = dst_channels == 1 ? "" : ".x";
  const std::vector<std::string> postfix = {channel_x, ".y", ".z", ".w"};
  const std::vector<std::string> channel = {".x", ".y", ".z", ".w"};

  const std::string type_postfix =
      dst_channels == 1 ? "" : std::to_string(dst_channels);

  std::string accum_type;

  switch (op_def.precision) {
    case CalculationsPrecision::F32:
    case CalculationsPrecision::F32_F16:
      accum_type = "float" + type_postfix;
      break;
    case CalculationsPrecision::F16:
      accum_type = "half" + type_postfix;
      break;
  }

  std::string c;
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
  c += "  " + accum_type + " r[" + std::to_string(kernel_size.y) + "][" +
       std::to_string(kernel_size.x) + "];\n";
  c += "  {\n";
  c += "  FLT4 src = args.src_tensor.Read(X, Y, 0);\n";
  int index = 0;
  for (int y = 0; y < kernel_size.y; ++y) {
    for (int x = 0; x < kernel_size.x; ++x) {
      std::string r_s =
          "  r[" + std::to_string(y) + "][" + std::to_string(x) + "]";
      for (int d = 0; d < dst_channels; ++d) {
        c += r_s + postfix[d] + " = dot(src, args.weights.Read(" +
             std::to_string(index) + "));\n";
        index++;
      }
    }
  }
  c += "  }\n";
  for (int i = 1; i < src_depth; ++i) {
    c += "  if (X > " + std::to_string(-i) +
         ") {  // always true, to reduce registers usage\n";
    c +=
        "  FLT4 src = args.src_tensor.Read(X, Y, " + std::to_string(i) + ");\n";
    for (int y = 0; y < kernel_size.y; ++y) {
      for (int x = 0; x < kernel_size.x; ++x) {
        std::string r_s =
            "  r[" + std::to_string(y) + "][" + std::to_string(x) + "]";
        for (int d = 0; d < dst_channels; ++d) {
          c += r_s + postfix[d] + " += dot(src, args.weights.Read(" +
               std::to_string(index) + "));\n";
          index++;
        }
      }
    }
    c += "  }\n";
  }
  c += "  X *= " + std::to_string(kernel_size.x) + ";\n";
  c += "  Y *= " + std::to_string(kernel_size.y) + ";\n";
  for (int y = 0; y < kernel_size.y; ++y) {
    for (int x = 0; x < kernel_size.x; ++x) {
      const std::string x_coord = "X + " + std::to_string(x);
      const std::string y_coord = "Y + " + std::to_string(y);
      c += "  if (" + x_coord + " < args.dst_tensor.Width() && " + y_coord +
           " < args.dst_tensor.Height()) {\n";
      c += "    FLT4 result = args.weights.Read(" + std::to_string(index) +
           ");\n";
      for (int d = 0; d < dst_channels; ++d) {
        c += "    result" + channel[d] + " += r[" + std::to_string(y) + "][" +
             std::to_string(x) + "]" + postfix[d] + ";\n";
      }
      c += "    args.dst_tensor.Write(result, " + x_coord + ", " + y_coord +
           ", 0);\n";
      c += "  }\n";
    }
  }
  c += "}\n";

  return c;
}

int3 ConvolutionTransposedThin::GetGridSize() const {
  const int grid_x = src_[0]->Width() * dst_[0]->Batch();
  const int grid_y = src_[0]->Height();
  const int grid_z = 1;
  return int3(grid_x, grid_y, grid_z);
}

bool IsConvolutionTransposedThinSupported(
    const ConvolutionTransposedAttributes& attr) {
  return attr.weights.shape.o <= 4 && attr.weights.shape.w == attr.stride.w &&
         attr.weights.shape.h == attr.stride.h &&
         attr.padding.prepended.w == 0 && attr.padding.prepended.h == 0 &&
         attr.padding.appended.w == 0 && attr.padding.appended.h == 0;
}

ConvolutionTransposedThin CreateConvolutionTransposedThin(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const ConvolutionTransposedAttributes& attr) {
  ConvolutionTransposedThin result(definition, attr, gpu_info);
  result.UploadData(attr.weights, attr.bias);
  return result;
}

}  // namespace gpu
}  // namespace tflite
