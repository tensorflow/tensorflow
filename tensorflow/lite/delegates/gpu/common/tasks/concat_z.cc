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

#include "tensorflow/lite/delegates/gpu/common/tasks/concat_z.h"

#include <algorithm>
#include <string>

#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace {

bool IsAllChannelsX4(const std::vector<int>& channels) {
  for (int channel : channels) {
    if (channel % 4 != 0) {
      return false;
    }
  }
  return true;
}

std::string GetConcatKernelCode(const OperationDef& op_def,
                                const std::vector<int>& channels) {
  std::vector<std::string> tensor_names(op_def.src_tensors.size());
  for (int i = 0; i < op_def.src_tensors.size(); ++i) {
    tensor_names[i] = "src_tensor_" + std::to_string(i);
  }

  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id = GLOBAL_ID_0;\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
    for (int i = 0; i < op_def.src_tensors.size(); ++i) {
      c += "  args." + tensor_names[i] + ".SetBatchRef(B);\n";
    }
  } else {
    c += "  int X = GLOBAL_ID_0;\n";
  }
  c += "  int Y = GLOBAL_ID_1;\n";
  std::string coords = "X, Y";
  if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "  int Z = GLOBAL_ID_2;\n";
    c += "  if (Z >= args.dst_tensor.Depth()) return;\n";
    coords = "X, Y, Z";
  }
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height()) "
       "return; \n";

  if (IsAllChannelsX4(channels)) {
    // When all channels % 4 == 0 we can read/assign/write VEC4 elements easily.
    // Also it is easy to write a loop in this case, to prevent long kernel
    // generation.
    c += "  int S = 0;\n";
    for (int i = 0; i < channels.size(); ++i) {
      std::string t_name = "args." + tensor_names[i];
      const int src_depth = DivideRoundUp(channels[i], 4);
      if (src_depth % 2 == 0) {
        // We can read more at once inside of loop in case src_depth % 2 == 0
        // it should be better for reading latency hiding
        c += "  for (int i = 0; i < " + t_name + ".Slices(); i += 2) {\n";
        c += "    " + t_name + "::type result0 = " + t_name + ".Read(" +
             coords + ", i);\n";
        c += "    " + t_name + "::type result1 = " + t_name + ".Read(" +
             coords + ", i + 1);\n";
        c += "    args.dst_tensor.Write(result0, " + coords + ", S);\n";
        c += "    args.dst_tensor.Write(result1, " + coords + ", S + 1);\n";
        c += "    S += 2;\n";
        c += "  }\n";
      } else {
        c += "  for (int i = 0; i < " + t_name + ".Slices(); ++i) {\n";
        c += "    " + t_name + "::type result = " + t_name + ".Read(" + coords +
             ", i);\n";
        c += "    args.dst_tensor.Write(result, " + coords + ", S);\n";
        c += "    S++;\n";
        c += "  }\n";
      }
    }
  } else {
    c += "  args.src_tensor_0::type result = args.src_tensor_0::zero_value;\n";
    int out_channel = 0;
    int read_index = 0;
    int z = 0;
    const std::string postfix[] = {".x", ".y", ".z", ".w"};
    for (int i = 0; i < channels.size(); ++i) {
      std::string tensor_name = "args." + tensor_names[i];
      const int depth = DivideRoundUp(channels[i], 4);
      for (int d = 0; d < depth; ++d) {
        const int channels_in_group = std::min(4, channels[i] - d * 4);
        const std::string temp_name = "t" + std::to_string(read_index);
        c += "  " + tensor_name + "::type " + temp_name + " = " + tensor_name +
             ".Read(" + coords + ", " + std::to_string(d) + ");\n";
        for (int ch = 0; ch < channels_in_group; ++ch) {
          c += "  result" + postfix[out_channel] + " = ";
          c += temp_name + postfix[ch] + ";\n";
          out_channel++;
          if (out_channel == 4) {
            out_channel = 0;
            c += "  args.dst_tensor.Write(result, " + coords + ", " +
                 std::to_string(z) + ");\n";
            z++;
          }
        }
        read_index++;
      }
    }
    if (out_channel != 0) {
      c += "  args.dst_tensor.Write(result, " + coords + ", " +
           std::to_string(z) + ");\n";
    }
  }
  c += "}\n";
  return c;
}

}  // namespace

GPUOperation CreateConcatZ(const OperationDef& definition,
                           const std::vector<int>& channels,
                           const GpuInfo& gpu_info) {
  GPUOperation op(definition);
  for (int i = 0; i < definition.src_tensors.size(); ++i) {
    const std::string name = "src_tensor_" + std::to_string(i);
    op.AddSrcTensor(name, definition.src_tensors[i]);
  }
  op.AddDstTensor("dst_tensor", definition.dst_tensors[0]);
  op.code_ = GetConcatKernelCode(definition, channels);
  if (gpu_info.IsPowerVR() &&
      definition.precision == CalculationsPrecision::F32 &&
      !IsAllChannelsX4(channels)) {
    // BUG, some PowerVRs (GE8320) produce incorrect result without it
    op.compiler_options_.push_back(CompilerOptions::kClDisableOptimizations);
  }
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HToY_DToZ;
  return op;
}

}  // namespace gpu
}  // namespace tflite
