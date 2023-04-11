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

#include "tensorflow/lite/delegates/gpu/common/tasks/space_to_depth.h"

#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"

namespace tflite {
namespace gpu {

namespace {
std::string GetSpaceToDepthCode(const OperationDef& op_def) {
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
  c += "  int S = GLOBAL_ID_2;\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "S >= args.dst_tensor.Slices()) { \n";
  c += "    return; \n";
  c += "  } \n";
  c += "  args.src_tensor::scalar_type tmp[4];\n";
  c += "  tmp[0] = args.src_tensor::scalar_zero_value;\n";
  c += "  tmp[1] = args.src_tensor::scalar_zero_value;\n";
  c += "  tmp[2] = args.src_tensor::scalar_zero_value;\n";
  c += "  tmp[3] = args.src_tensor::scalar_zero_value;\n";
  c += "  for (int i = 0; i < 4; ++i) {\n";
  c += "    int dst_c = 4 * S + i;\n";
  c += "    int block_id = dst_c / args.src_tensor.Channels();\n";
  c += "    int src_x = X * args.block_size + block_id % args.block_size;\n";
  c += "    int src_y = Y * args.block_size + block_id / args.block_size;\n";
  c += "    int src_c = dst_c % args.src_tensor.Channels();\n";
  c += "    args.src_tensor.ReadPerChannel(tmp[i], src_x, src_y, src_c);\n";
  c += "  }\n";
  c += "  args.src_tensor::type result;\n";
  c += "  result.x = tmp[0];\n";
  c += "  result.y = tmp[1];\n";
  c += "  result.z = tmp[2];\n";
  c += "  result.w = tmp[3];\n";
  c += "  args.dst_tensor.Write(result, X, Y, S);\n";
  c += "}\n";
  return c;
}

std::string GetDepthToSpaceCode(const OperationDef& op_def) {
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
  c += "  int S = GLOBAL_ID_2;\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "S >= args.dst_tensor.Slices()) { \n";
  c += "    return; \n";
  c += "  } \n";
  c += "  FLT tmp[4];\n";
  c += "  tmp[0] = INIT_FLT(0.0f);\n";
  c += "  tmp[1] = INIT_FLT(0.0f);\n";
  c += "  tmp[2] = INIT_FLT(0.0f);\n";
  c += "  tmp[3] = INIT_FLT(0.0f);\n";
  c += "  for (int i = 0; i < 4; ++i) {\n";
  c += "    int dst_c = 4 * S + i;\n";
  c += "    int block_x = X % args.block_size;\n";
  c += "    int src_x = X / args.block_size;\n";
  c += "    int block_y = Y % args.block_size;\n";
  c += "    int src_y = Y / args.block_size;\n";
  c += "    int block_id = block_y * args.block_size + block_x;\n";
  c += "    int src_c = block_id * args.dst_tensor.Channels() + dst_c;\n";
  c += "    src_c = min(src_c, args.src_tensor.Channels() - 1);\n";
  c += "    args.src_tensor.ReadPerChannel(tmp[i], src_x, src_y, src_c);\n";
  c += "  }\n";
  c += "  FLT4 result;\n";
  c += "  result.x = tmp[0];\n";
  c += "  result.y = tmp[1];\n";
  c += "  result.z = tmp[2];\n";
  c += "  result.w = tmp[3];\n";
  c += "  args.dst_tensor.Write(result, X, Y, S);\n";
  c += "}\n";
  return c;
}
}  // namespace

GPUOperation CreateSpaceToDepth(const OperationDef& op_def,
                                const SpaceToDepthAttributes& attr) {
  GPUOperation op(op_def);
  op.AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  op.AddDstTensor("dst_tensor", op_def.dst_tensors[0]);
  op.args_.AddInt("block_size", attr.block_size);
  op.code_ = GetSpaceToDepthCode(op_def);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

GPUOperation CreateDepthToSpace(const OperationDef& op_def,
                                const SpaceToDepthAttributes& attr) {
  GPUOperation op(op_def);
  op.AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  op.AddDstTensor("dst_tensor", op_def.dst_tensors[0]);
  op.args_.AddInt("block_size", attr.block_size);
  op.code_ = GetDepthToSpaceCode(op_def);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

}  // namespace gpu
}  // namespace tflite
