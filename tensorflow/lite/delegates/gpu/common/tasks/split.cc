/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/common/tasks/split.h"

#include <string>

#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"

namespace tflite {
namespace gpu {

Split::Split(const OperationDef& definition, const SplitAttributes& attr)
    : GPUOperation(definition), attr_(attr) {
  work_group_size_ = int3(8, 4, 1);
  code_ = attr.axis == Axis::CHANNELS ? GetSplitChannelsCode() : GetSplitCode();
}

std::string Split::GetSplitCode() {
  AddSrcTensor("src_tensor", definition_.src_tensors[0]);
  for (int i = 0; i < definition_.dst_tensors.size(); ++i) {
    AddDstTensor("dst_tensor_" + std::to_string(i), definition_.dst_tensors[i]);
  }
  const std::string task_width =
      attr_.axis == Axis::WIDTH ? "1" : "args.src_tensor.Width()";
  const std::string task_height =
      attr_.axis == Axis::HEIGHT ? "1" : "args.src_tensor.Height()";
  const std::string task_depth =
      attr_.axis == Axis::DEPTH ? "1" : "args.src_tensor.Depth()";
  const std::string task_batch =
      attr_.axis == Axis::BATCH ? "1" : "args.src_tensor.Batch()";
  const std::string task_slices =
      attr_.axis == Axis::CHANNELS ? "1" : "args.src_tensor.Slices()";

  std::map<Axis, std::string> axis_to_selector = {
      {Axis::WIDTH, "Width"}, {Axis::HEIGHT, "Height"},
      {Axis::DEPTH, "Depth"}, {Axis::CHANNELS, "Slices"},
      {Axis::BATCH, "Batch"},
  };
  std::map<Axis, std::string> axis_to_coord = {
      {Axis::WIDTH, "X"},    {Axis::HEIGHT, "Y"}, {Axis::DEPTH, "D"},
      {Axis::CHANNELS, "S"}, {Axis::BATCH, "B"},
  };

  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  if (definition_.src_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id = GLOBAL_ID_0;\n";
    c += "  int X = linear_id / " + task_batch + ";\n";
    c += "  int B = linear_id % " + task_batch + ";\n";
    c += "  if (X >= " + task_width + ") return;\n";
  } else {
    c += "  int X = GLOBAL_ID_0;\n";
    c += "  if (X >= " + task_width + ") return;\n";
  }
  if (definition_.src_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "  int linear_id = GLOBAL_ID_1;\n";
    c += "  int Y = linear_id % " + task_height + ";\n";
    c += "  int D = linear_id / " + task_height + ";\n";
    c += "  if (D >= " + task_depth + ") return;\n";
  } else {
    c += "  int Y = GLOBAL_ID_1;\n";
    c += "  if (Y >= " + task_height + ") return;\n";
  }
  c += "  int S = GLOBAL_ID_2;\n";
  c += "  if (S >= " + task_slices + ") return;\n";
  c += "  int src_counter = 0;\n";
  std::vector<std::string> src_coords;
  for (auto axis :
       {Axis::WIDTH, Axis::HEIGHT, Axis::DEPTH, Axis::CHANNELS, Axis::BATCH}) {
    if (definition_.src_tensors[0].HasAxis(axis)) {
      const std::string coord_name =
          attr_.axis == axis ? "src_counter" : axis_to_coord[axis];
      src_coords.push_back(coord_name);
    }
  }
  std::string src_coords_str = src_coords[0];
  for (int i = 1; i < src_coords.size(); ++i) {
    src_coords_str += ", " + src_coords[i];
  }
  for (int i = 0; i < definition_.dst_tensors.size(); ++i) {
    std::vector<std::string> dst_coords;
    for (auto axis : {Axis::WIDTH, Axis::HEIGHT, Axis::DEPTH, Axis::CHANNELS,
                      Axis::BATCH}) {
      if (definition_.dst_tensors[i].HasAxis(axis)) {
        const std::string coord_name =
            attr_.axis == axis ? "i" : axis_to_coord[axis];
        dst_coords.push_back(coord_name);
      }
    }
    std::string dst_coords_str = dst_coords[0];
    for (int j = 1; j < dst_coords.size(); ++j) {
      dst_coords_str += ", " + dst_coords[j];
    }
    const std::string dst_name = "args.dst_tensor_" + std::to_string(i);
    c += "  for (int i = 0; i < " + dst_name + "." +
         axis_to_selector[attr_.axis] + "(); ++i, src_counter++) {\n";
    c += "    FLT4 result = args.src_tensor.Read(" + src_coords_str + ");\n";
    c += "    " + dst_name + ".Write(result, " + dst_coords_str + ");\n";
    c += "  }\n";
  }
  c += "}\n";
  return c;
}

std::string Split::GetSplitChannelsCode() {
  AddSrcTensor("src_tensor", definition_.src_tensors[0]);
  for (int i = 0; i < definition_.dst_tensors.size(); ++i) {
    AddDstTensor("dst_tensor_" + std::to_string(i), definition_.dst_tensors[i]);
  }

  const std::string batch_coord =
      definition_.src_tensors[0].HasAxis(Axis::BATCH) ? ", B" : "";
  std::string coords = "X, Y";
  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  if (definition_.src_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id = GLOBAL_ID_0;\n";
    c += "  int X = linear_id / args.src_tensor.Batch();\n";
    c += "  int B = linear_id % args.src_tensor.Batch();\n";
    c += "  if (X >= args.src_tensor.Width()) return;\n";
  } else {
    c += "  int X = GLOBAL_ID_0;\n";
    c += "  if (X >= args.src_tensor.Width()) return;\n";
  }
  if (definition_.src_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "  int linear_id = GLOBAL_ID_1;\n";
    c += "  int Y = linear_id % args.src_tensor.Height();\n";
    c += "  int Z = linear_id / args.src_tensor.Height();\n";
    c += "  if (Z >= args.src_tensor.Depth()) return;\n";
    coords += ", Z";
  } else {
    c += "  int Y = GLOBAL_ID_1;\n";
    c += "  if (Y >= args.src_tensor.Height()) return;\n";
  }
  c += "  int src_channel = 0;\n";
  const std::string postfixes[] = {"x", "y", "z", "w"};
  for (int i = 0; i < definition_.dst_tensors.size(); ++i) {
    const std::string dst_name = "args.dst_tensor_" + std::to_string(i);
    c += "  for (int i = 0; i < " + dst_name + ".Slices(); ++i) {\n";
    c += "    FLT4 result = INIT_FLT4(0.0f);\n";
    for (int j = 0; j < 4; ++j) {
      c += "    if (i * 4 + " + std::to_string(j) + " < " + dst_name +
           ".Channels()) {\n";
      c += "      int src_slice = src_channel >> 2;\n";
      c += "      int src_sub_ch = src_channel & 3;\n";
      c += "      FLT4 t = args.src_tensor.Read(" + coords + ", src_slice" +
           batch_coord + ");\n";
      c += "      result." + postfixes[j] +
           " = SELECT_BY_INDEX_FROM_FLT4(t, src_sub_ch);\n";
      c += "      src_channel++;\n";
      c += "    }\n";
    }
    c += "    " + dst_name + ".Write(result, " + coords + ", i" + batch_coord +
         ");\n";
    c += "  }\n";
  }
  c += "}\n";
  return c;
}

int3 Split::GetGridSize() const {
  const int width = attr_.axis == Axis::WIDTH ? 1 : src_[0]->Width();
  const int height = attr_.axis == Axis::HEIGHT ? 1 : src_[0]->Height();
  const int depth = attr_.axis == Axis::DEPTH ? 1 : src_[0]->Depth();
  const int batch = attr_.axis == Axis::BATCH ? 1 : src_[0]->Batch();
  const int slices = attr_.axis == Axis::CHANNELS ? 1 : src_[0]->Slices();
  const int grid_x = width * batch;
  const int grid_y = height * depth;
  const int grid_z = slices;
  return int3(grid_x, grid_y, grid_z);
}

Split CreateSplit(const OperationDef& definition, const SplitAttributes& attr) {
  return Split(definition, attr);
}

}  // namespace gpu
}  // namespace tflite
