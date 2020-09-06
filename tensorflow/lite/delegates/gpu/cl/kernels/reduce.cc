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

#include "tensorflow/lite/delegates/gpu/cl/kernels/reduce.h"

#include <string>

#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/precision.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {
std::string GetReduceChannelsKernelCode(const OperationDef& op_def,
                                        const OperationType& op_type) {
  std::string c = GetCommonDefines(op_def.precision);
  if (op_type == OperationType::REDUCE_SUM) {
    c += "#define OP(a, b) ((a) + (b))\n";
  } else if (op_type == OperationType::REDUCE_PRODUCT) {
    c += "#define OP(a, b) ((a) * (b))\n";
  } else if (op_type == OperationType::REDUCE_MAXIMUM) {
    c += "#define OP(a, b) max(a, b)\n";
  } else if (op_type == OperationType::REDUCE_MINIMUM) {
    c += "#define OP(a, b) min(a, b)\n";
  }
  c += "__kernel void main_function($0) {\n";
  c += "  int X = get_global_id(0);\n";
  c += "  int Y = get_global_id(1);\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height()) "
       "return;\n";
  if (op_type == OperationType::REDUCE_SUM) {
    c += "  FLT4 reduced = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n";
  } else if (op_type == OperationType::REDUCE_PRODUCT) {
    c += "  FLT4 reduced = (FLT4)(1.0f, 1.0f, 1.0f, 1.0f);\n";
  } else {
    c += "  FLT4 V0 = args.src_tensor.Read(X, Y, 0);\n";
    c += "  FLT4 reduced = (FLT4)(V0.x, V0.x, V0.x, V0.x);\n";
  }
  c += "  int s = 0;\n";
  c += "  for (; s < args.src_tensor.Slices() - 1; ++s) {\n";
  c += "    FLT4 V = args.src_tensor.Read(X, Y, s);\n";
  c += "    reduced = OP(reduced, V);\n";
  c += "  }\n";
  c += "  FLT reduced_final = OP(OP(reduced.x, reduced.y), OP(reduced.z, "
       "reduced.w));\n";
  c += "  FLT last_reduce;\n";
  c += "  FLT4 last_val = args.src_tensor.Read(X, Y, s);\n";
  c += "  int ch_rem = args.src_tensor.Channels() % 4;\n";
  c += "  if (ch_rem == 0) {\n";
  c += "    last_reduce = OP(OP(last_val.x, last_val.y), OP(last_val.z, "
       "last_val.w));\n";
  c += "  } else if (ch_rem == 1) {\n";
  c += "    last_reduce = OP(OP(last_val.x, last_val.y), last_val.z);\n";
  c += "  } else if (ch_rem == 2) {\n";
  c += "    last_reduce = OP(last_val.x, last_val.y);\n";
  c += "  } else {\n";
  c += "    last_reduce = last_val.x;\n";
  c += "  }\n";
  c += "  reduced_final = OP(reduced_final, last_reduce);\n";
  c += "  FLT4 result = (FLT4)(reduced_final, 0.0f, 0.0f, 0.0f);\n";
  c += "  args.dst_tensor.Write(result, X, Y, 0);\n";
  c += "}\n";
  return c;
}
}  // namespace

GPUOperation CreateReduce(const OperationDef& definition,
                          const ReduceAttributes& attr,
                          const OperationType& op_type) {
  GPUOperation op(definition);
  auto src_desc = definition.src_tensors[0];
  if (definition.IsBatchSupported()) {
    src_desc.SetStateVar("BatchedWidth", "true");
  }
  op.AddSrcTensor("src_tensor", src_desc);
  auto dst_desc = definition.dst_tensors[0];
  if (definition.IsBatchSupported()) {
    dst_desc.SetStateVar("BatchedWidth", "true");
  }
  op.AddDstTensor("dst_tensor", dst_desc);
  op.code_ = GetReduceChannelsKernelCode(definition, op_type);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_ZIs1;
  return op;
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
