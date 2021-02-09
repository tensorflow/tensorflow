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

#include "tensorflow/lite/delegates/gpu/common/tasks/padding.h"

#include <string>

#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"

namespace tflite {
namespace gpu {

namespace {
std::string GetPaddingCode(const OperationDef& op_def,
                           const PadAttributes& attr, GPUOperation* op) {
  op->AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  op->AddDstTensor("dst_tensor", op_def.dst_tensors[0]);
  op->args_.AddInt("prepended_x", attr.prepended.w);
  op->args_.AddInt("prepended_y", attr.prepended.h);
  op->args_.AddInt("prepended_z", attr.prepended.c);
  op->args_.AddInt("prepended_w", attr.prepended.b);

  const std::string dst_batch =
      op_def.dst_tensors[0].HasAxis(Axis::BATCH) ? "B" : "0";
  std::string c;
  const std::string channels[] = {".x", ".y", ".z", ".w"};

  if (attr.type == PaddingContentType::REFLECT) {
    c += "int reflect(int x, int size) {\n";
    c += "  int t = abs(x) - size + 1;\n";
    c += "  return size - 1 - abs(t);\n";
    c += "}\n\n";
  }

  c += "MAIN_FUNCTION($0) {\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id = GLOBAL_ID_0;\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int X = GLOBAL_ID_0;\n";
  }
  c += "  int Y = GLOBAL_ID_1;\n";
  c += "  int Z = GLOBAL_ID_2;\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "Z >= args.dst_tensor.Slices()) { \n";
  c += "    return; \n";
  c += "  } \n";
  c += "  FLT4 result = INIT_FLT4(0.0);\n";
  c += "  int s_x = X - args.prepended_x;\n";
  c += "  int s_y = Y - args.prepended_y;\n";
  if (op_def.src_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int s_b = " + dst_batch + " - args.prepended_w;\n";
    c += "  args.src_tensor.SetBatchRef(s_b);\n";
  }
  if (attr.type == PaddingContentType::REFLECT) {
    c += "  s_x = reflect(s_x, args.src_tensor.Width());\n";
    c += "  s_y = reflect(s_y, args.src_tensor.Height());\n";
    if (op_def.src_tensors[0].HasAxis(Axis::BATCH)) {
      c += "  int s_b = reflect(s_b, args.src_tensor.Batch());\n";
    }
    if (attr.prepended.c == 0 && attr.appended.c == 0) {
      // optimized case
      c += "  result = args.src_tensor.Read(s_x, s_y, Z);\n";
    } else {
      c += "  int start_channel = Z * 4;\n";
      for (int i = 0; i < 4; ++i) {
        const auto& s = channels[i];
        c += "  {\n";
        c += "    int channel = start_channel + " + std::to_string(i) + ";\n";
        c += "    int s_z = channel - args.prepended_z;\n";
        // We need additional clamp for z, so that we use alignment for channels
        // and can proceed extra channels that can lead to reading out of
        // resource.
        c += "    s_z = clamp(reflect(s_z, args.src_tensor.Channels()), 0, "
             "args.src_tensor.Channels() - "
             "1);\n";
        c += "    FLT4 t = args.src_tensor.Read(s_x, s_y, s_z / 4);\n";
        c += "    FLT t_ar[4] = {t.x, t.y, t.z, t.w};\n";
        c += "    result" + s + " = t_ar[s_z % 4];\n";
        c += "  }\n";
      }
    }
  } else {
    c += "  bool inside_x = s_x >= 0 && s_x < args.src_tensor.Width();\n";
    c += "  bool inside_y = s_y >= 0 && s_y < args.src_tensor.Height();\n";
    if (op_def.src_tensors[0].HasAxis(Axis::BATCH)) {
      c += "  inside_y &= (s_b >= 0 && s_b < args.src_tensor.Batch());\n";
    }
    c += "  if (inside_x && inside_y) {\n";
    if (attr.prepended.c == 0 && attr.appended.c == 0) {
      // optimized case
      c += "    result = args.src_tensor.Read(s_x, s_y, Z);\n";
    } else if (attr.prepended.c % 4 == 0) {
      c += "    int s_z = Z - args.prepended_z / 4;\n";
      c += "    if (s_z >= 0 && s_z < args.src_tensor.Slices()) {\n";
      c += "      result = args.src_tensor.Read(s_x, s_y, s_z);\n";
      c += "    }\n";
    } else {
      c += "    int start_channel = Z * 4;\n";
      for (int i = 0; i < 4; ++i) {
        const auto& s = channels[i];
        c += "    {\n";
        c += "    int channel = start_channel + " + std::to_string(i) + ";\n";
        c += "    int s_z = channel - args.prepended_z;\n";
        c += "    if (s_z >= 0 && s_z < args.src_tensor.Channels()) {\n";
        c += "      FLT4 t = args.src_tensor.Read(s_x, s_y, s_z / 4);\n";
        c += "      FLT t_ar[4] = {t.x, t.y, t.z, t.w};\n";
        c += "      result" + s + " = t_ar[s_z % 4];\n";
        c += "    }\n";
        c += "    }\n";
      }
    }
    c += "  }\n";
  }
  c += "  args.dst_tensor.Write(result, X, Y, Z);\n";
  c += "}\n";

  return c;
}

}  // namespace

GPUOperation CreatePadding(const OperationDef& definition,
                           const PadAttributes& attr) {
  GPUOperation op(definition);
  op.code_ = GetPaddingCode(definition, attr, &op);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

}  // namespace gpu
}  // namespace tflite
