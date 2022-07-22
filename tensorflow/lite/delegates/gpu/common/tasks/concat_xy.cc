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

#include "tensorflow/lite/delegates/gpu/common/tasks/concat_xy.h"

#include <map>
#include <string>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace {
std::string GetConcatKernelCode(const OperationDef& op_def,
                                const ConcatAttributes& attr) {
  std::vector<std::string> tensor_names(op_def.src_tensors.size());
  for (int i = 0; i < op_def.src_tensors.size(); ++i) {
    tensor_names[i] = "src_tensor_" + std::to_string(i);
  }

  std::map<Axis, std::string> axis_to_selector = {
      {Axis::WIDTH, "Width"}, {Axis::HEIGHT, "Height"},
      {Axis::DEPTH, "Depth"}, {Axis::CHANNELS, "Channels"},
      {Axis::BATCH, "Batch"},
  };
  std::map<Axis, std::string> axis_to_coord = {
      {Axis::WIDTH, "X"},    {Axis::HEIGHT, "Y"}, {Axis::DEPTH, "D"},
      {Axis::CHANNELS, "S"}, {Axis::BATCH, "B"},
  };

  std::vector<std::string> src_coords;
  std::vector<std::string> dst_coords;
  for (auto axis :
       {Axis::WIDTH, Axis::HEIGHT, Axis::DEPTH, Axis::CHANNELS, Axis::BATCH}) {
    if (op_def.src_tensors[0].HasAxis(axis) && axis != Axis::BATCH) {
      if (axis == attr.axis) {
        src_coords.push_back("coord");
      } else {
        src_coords.push_back(axis_to_coord[axis]);
      }
    }
    if (op_def.dst_tensors[0].HasAxis(axis)) {
      dst_coords.push_back(axis_to_coord[axis]);
    }
  }
  std::string src_coord = src_coords[0];
  for (int i = 1; i < src_coords.size(); ++i) {
    src_coord += ", " + src_coords[i];
  }
  std::string dst_coord = dst_coords[0];
  for (int i = 1; i < dst_coords.size(); ++i) {
    dst_coord += ", " + dst_coords[i];
  }

  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id_0 = GLOBAL_ID_0;\n";
    c += "  int X = linear_id_0 / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id_0 % args.dst_tensor.Batch();\n";
  } else {
    c += "  int X = GLOBAL_ID_0;\n";
  }
  if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "  int linear_id_1 = GLOBAL_ID_1;\n";
    c += "  int Y = linear_id_1 / args.dst_tensor.Depth();\n";
    c += "  int D = linear_id_1 % args.dst_tensor.Depth();\n";
  } else {
    c += "  int Y = GLOBAL_ID_1;\n";
  }
  c += "  int S = GLOBAL_ID_2;\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "S >= args.dst_tensor.Slices()) { \n";
  c += "    return; \n";
  c += "  } \n";
  c += "  args.src_tensor_0::type result = args.src_tensor_0::zero_value;\n";
  c += "  int coord = " + axis_to_coord[attr.axis] + ";\n";
  for (int i = 0; i < op_def.src_tensors.size(); ++i) {
    const std::string field =
        "args." + tensor_names[i] + "." + axis_to_selector[attr.axis] + "()";
    c += "  if (coord >= 0 && coord < " + field + ") { \n";
    if (op_def.src_tensors[i].HasAxis(Axis::BATCH)) {
      if (attr.axis == Axis::BATCH) {
        c += "  args." + tensor_names[i] + ".SetBatchRef(coord);\n";
      } else {
        c += "  args." + tensor_names[i] + ".SetBatchRef(B);\n";
      }
    }
    c += "    result = args." + tensor_names[i] + ".Read(" + src_coord + ");\n";
    c += "  } \n";
    c += "  coord -= " + field + ";\n";
  }
  c += "  args.dst_tensor.Write(result, " + dst_coord + ");\n";
  c += "}\n";
  return c;
}
}  // namespace

GPUOperation CreateConcatXY(const OperationDef& definition,
                            const ConcatAttributes& attr) {
  GPUOperation op(definition);
  for (int i = 0; i < definition.src_tensors.size(); ++i) {
    const std::string name = "src_tensor_" + std::to_string(i);
    op.AddSrcTensor(name, definition.src_tensors[i]);
  }
  op.AddDstTensor("dst_tensor", definition.dst_tensors[0]);
  op.code_ = GetConcatKernelCode(definition, attr);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

}  // namespace gpu
}  // namespace tflite
