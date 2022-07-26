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

#include "tensorflow/lite/delegates/gpu/common/tasks/add.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"

namespace tflite {
namespace gpu {
namespace {
GPUOperation CreateUnequalAdd(const OperationDef& op_def) {
  GPUOperation op(op_def);
  op.AddDstTensor("dst_tensor", op_def.dst_tensors[0]);
  for (int i = 0; i < op_def.src_tensors.size(); ++i) {
    const std::string tensor_name = absl::StrCat("src_tensor_", i);
    op.AddSrcTensor(tensor_name, op_def.src_tensors[i]);
  }
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id = GLOBAL_ID_0;\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
    for (int i = 0; i < op_def.src_tensors.size(); ++i) {
      const std::string tensor_name = absl::StrCat("src_tensor_", i);
      c += "  args." + tensor_name + ".SetBatchRef(B);\n";
    }
  } else {
    c += "  int X = GLOBAL_ID_0;\n";
  }
  c += "  int Y = GLOBAL_ID_1;\n";
  c += "  int S = GLOBAL_ID_2;\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "S >= args.dst_tensor.Slices()) return; \n";
  c += "  args.src_tensor_0::type src = args.src_tensor_0::zero_value;\n";
  for (int i = 0; i < op_def.src_tensors.size(); ++i) {
    const std::string tensor_name = absl::StrCat("src_tensor_", i);
    c += "  if (S < args." + tensor_name + ".Slices()) {\n";
    c += "    src += args." + tensor_name + ".Read(X, Y, S);\n";
    c += "  }\n";
  }
  c += "  args.dst_tensor.Write(src, X, Y, S);\n";
  c += "} \n";
  op.code_ = std::move(c);
  return op;
}
}  // namespace

GPUOperation CreateAdd(const OperationDef& definition,
                       const std::vector<int>& channels, int dst_channels) {
  if (dst_channels != channels[0]) {
    return CreateUnequalAdd(definition);
  }
  ElementwiseDescriptor op_desc;
  op_desc.code = "  out_value = in_value;\n";
  for (int i = 1; i < definition.src_tensors.size(); ++i) {
    const std::string tensor_name = absl::StrCat("src_tensor_", i);
    op_desc.code += "if (S_COORD < args." + tensor_name + ".Slices()) {\n";
    op_desc.code += "  out_value += args." + tensor_name +
                    ".Read(X_COORD, Y_COORD, S_COORD);\n";
    op_desc.code += "}\n";
  }
  return CreateGpuOperation(definition, std::move(op_desc));
}

}  // namespace gpu
}  // namespace tflite
