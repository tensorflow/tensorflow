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

#include "tensorflow/lite/delegates/gpu/common/tasks/max_unpooling.h"

#include <string>

namespace tflite {
namespace gpu {
namespace {
void AppendConditionally(const std::string& value, const std::string& delimeter,
                         std::string* result) {
  if (!result->empty()) {
    *result += delimeter;
  }
  *result += value;
}

std::string GetMaxUnpoolingKernelCode(const GpuInfo& gpu_info,
                                      const OperationDef& op_def,
                                      GPUOperation* op) {
  op->AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  op->AddSrcTensor("src_indices", op_def.src_tensors[1]);
  op->AddDstTensor("dst_tensor", op_def.dst_tensors[0]);

  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id = GLOBAL_ID_0;\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
    c += "  args.src_indices.SetBatchRef(B);\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int X = GLOBAL_ID_0;\n";
  }
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
  c += "  int src_x = (X + args.padding_x) / args.stride_x;\n";
  c += "  int t_x = X - (src_x * args.stride_x - args.padding_x);\n";
  c += "  int src_y = (Y + args.padding_y) / args.stride_y;\n";
  c += "  int t_y = Y - (src_y * args.stride_y - args.padding_y);\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "  int src_z = (Z + args.padding_z) / args.stride_z;\n";
    c += "  int t_z = Z - (src_z * args.stride_z - args.padding_z);\n";
    c += "  int t_index = (t_y * args.kernel_size_x + t_x) * "
         "args.kernel_size_z + t_z;\n";
  } else {
    c += "  int t_index = t_y * args.kernel_size_x + t_x;\n";
  }
  std::string inbounds_check;
  if (!op_def.src_tensors[0].SupportsZeroClamp(Axis::WIDTH, gpu_info) ||
      !op_def.src_tensors[1].SupportsZeroClamp(Axis::WIDTH, gpu_info)) {
    c += "  bool inside_x = src_x >= 0 && src_x < args.src_tensor.Width();\n";
    c += "  src_x = clamp(src_x, 0, args.src_tensor.Width() - 1);\n";
    AppendConditionally("inside_x", " && ", &inbounds_check);
  }
  if (!op_def.src_tensors[0].SupportsZeroClamp(Axis::HEIGHT, gpu_info) ||
      !op_def.src_tensors[1].SupportsZeroClamp(Axis::HEIGHT, gpu_info)) {
    c += "  bool inside_y = src_y >= 0 && src_y < args.src_tensor.Height();\n";
    c += "  src_y = clamp(src_y, 0, args.src_tensor.Height() - 1);\n";
    AppendConditionally("inside_y", " && ", &inbounds_check);
  }
  if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    if (!op_def.src_tensors[0].SupportsZeroClamp(Axis::DEPTH, gpu_info) ||
        !op_def.src_tensors[1].SupportsZeroClamp(Axis::DEPTH, gpu_info)) {
      c += "  bool inside_z = src_z >= 0 && src_z < args.src_tensor.Depth();\n";
      c += "  src_z = clamp(src_z, 0, args.src_tensor.Depth() - 1);\n";
      AppendConditionally("inside_z", " && ", &inbounds_check);
    }
  }
  std::string src_args = op_def.dst_tensors[0].HasAxis(Axis::DEPTH)
                             ? "src_x, src_y, src_z, S"
                             : "src_x, src_y, S";
  c +=
      "  args.src_tensor::type src = args.src_tensor.Read(" + src_args + ");\n";
  c += "  int4 ind = args.src_indices.Read<int>(" + src_args + ");\n";
  if (!inbounds_check.empty()) {
    c += "  src *= INIT_FLT(" + inbounds_check + ");\n";
    c += "  ind *= INIT_INT(" + inbounds_check + ");\n";
  }
  c += "  args.src_tensor::type result;\n";
  c += "  result.x = t_index == ind.x ? src.x : INIT_FLT(0.0f);\n";
  c += "  result.y = t_index == ind.y ? src.y : INIT_FLT(0.0f);\n";
  c += "  result.z = t_index == ind.z ? src.z : INIT_FLT(0.0f);\n";
  c += "  result.w = t_index == ind.w ? src.w : INIT_FLT(0.0f);\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "  args.dst_tensor.Write(result, X, Y, Z, S);\n";
  } else {
    c += "  args.dst_tensor.Write(result, X, Y, S);\n";
  }
  c += "}\n";
  return c;
}
}  // namespace

GPUOperation CreateMaxUnpooling(const GpuInfo& gpu_info,
                                const OperationDef& definition,
                                const MaxUnpooling2DAttributes& attr) {
  GPUOperation op(definition);
  op.args_.AddInt("kernel_size_x", attr.kernel.w);
  op.args_.AddInt("padding_x", attr.padding.appended.w);
  op.args_.AddInt("stride_x", attr.strides.w);
  op.args_.AddInt("kernel_size_y", attr.kernel.h);
  op.args_.AddInt("padding_y", attr.padding.appended.h);
  op.args_.AddInt("stride_y", attr.strides.h);
  op.code_ = GetMaxUnpoolingKernelCode(gpu_info, definition, &op);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

GPUOperation CreateMaxUnpooling(const GpuInfo& gpu_info,
                                const OperationDef& definition,
                                const MaxUnpooling3DAttributes& attr) {
  GPUOperation op(definition);
  op.args_.AddInt("kernel_size_x", attr.kernel.w);
  op.args_.AddInt("padding_x", attr.padding.appended.w);
  op.args_.AddInt("stride_x", attr.strides.w);
  op.args_.AddInt("kernel_size_y", attr.kernel.h);
  op.args_.AddInt("padding_y", attr.padding.appended.h);
  op.args_.AddInt("stride_y", attr.strides.h);
  op.args_.AddInt("kernel_size_z", attr.kernel.d);
  op.args_.AddInt("padding_z", attr.padding.appended.d);
  op.args_.AddInt("stride_z", attr.strides.d);
  op.code_ = GetMaxUnpoolingKernelCode(gpu_info, definition, &op);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

}  // namespace gpu
}  // namespace tflite
