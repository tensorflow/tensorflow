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

#include "tensorflow/lite/delegates/gpu/metal/kernels/concat.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"

namespace tflite {
namespace gpu {
namespace metal {
namespace {

bool IsAllChannelsX4(const std::vector<int>& channels) {
  for (int channel : channels) {
    if (channel % 4 != 0) {
      return false;
    }
  }
  return true;
}

std::string GetConcatChannelsCode(const OperationDef& op_def,
                                  const std::vector<int>& channels) {
  std::vector<std::string> tensor_names(op_def.src_tensors.size());
  for (int i = 0; i < op_def.src_tensors.size(); ++i) {
    tensor_names[i] = "src_tensor_" + std::to_string(i);
  }

  std::string c = R"(
    kernel void ComputeFunction(
                                $0
                                uint3 ugid[[thread_position_in_grid]]) {
  int X = static_cast<int>(ugid.x);
  int Y = static_cast<int>(ugid.y);
)";

  std::string coords = "X, Y";
  if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "  int Z = static_cast<int>(ugid.z);\n";
    c += "  if (Z >= args.dst_tensor.Depth()) return;\n";
    coords = "X, Y, Z";
  }
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height()) "
       "return; \n";

  if (IsAllChannelsX4(channels)) {
    // When all channels % 4 == 0 we can read/assign/write FLT4 elements easily.
    // Also it is easy to write a loop in this case, to prevent long kernel
    // generation.
    c += "  int S = 0;\n";
    for (int i = 0; i < channels.size(); ++i) {
      std::string t_name = "args." + tensor_names[i];
      const int depth = DivideRoundUp(channels[i], 4);
      if (depth % 2 == 0) {
        // We can read more at once inside of loop in case depth % 2 == 0
        // it should be better for reading latency hiding
        c += "  for (int i = 0; i < " + t_name + ".Slices(); i += 2) {\n";
        c += "    FLT4 result0 = " + t_name + ".Read(" + coords + ", i);\n";
        c += "    FLT4 result1 = " + t_name + ".Read(" + coords + ", i + 1);\n";
        c += "    args.dst_tensor.Write(result0, " + coords + ", S);\n";
        c += "    args.dst_tensor.Write(result1, " + coords + ", S + 1);\n";
        c += "    S += 2;\n";
        c += "  }\n";
      } else {
        c += "  for (int i = 0; i < " + t_name + ".Slices(); ++i) {\n";
        c += "    FLT4 result = " + t_name + ".Read(" + coords + ", i);\n";
        c += "    args.dst_tensor.Write(result, " + coords + ", S);\n";
        c += "    S++;\n";
        c += "  }\n";
      }
    }
  } else {
    c += "  FLT4 value = FLT4(0.0);\n";
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
        c += "  FLT4 " + temp_name + " = " + tensor_name + ".Read(" + coords +
             ", " + std::to_string(d) + ");\n";
        for (int ch = 0; ch < channels_in_group; ++ch) {
          c += "  value" + postfix[out_channel] + " = ";
          c += temp_name + postfix[ch] + ";\n";
          out_channel++;
          if (out_channel == 4) {
            out_channel = 0;
            c += "  args.dst_tensor.Write(value, " + coords + ", " +
                 std::to_string(z) + ");\n";
            z++;
          }
        }
        read_index++;
      }
    }
    if (out_channel != 0) {
      c += "  args.dst_tensor.Write(value, " + coords + ", " +
           std::to_string(z) + ");\n";
    }
  }
  c += "}\n";
  return c;
}

}  // namespace

ComputeTaskDescriptor ConcatZ(const OperationDef& definition,
                              const ConcatAttributes& attr,
                              const std::vector<BHWC>& input_shapes) {
  std::vector<int> channels;
  channels.reserve(input_shapes.size());
  for (const auto& shape : input_shapes) {
    channels.push_back(shape.c);
  }
  ComputeTaskDescriptor desc(definition);
  desc.shader_source = GetConcatChannelsCode(definition, channels);

  for (int i = 0; i < definition.src_tensors.size(); ++i) {
    desc.AddSrcTensor("src_tensor_" + std::to_string(i),
                      definition.src_tensors[i]);
  }
  desc.AddDstTensor("dst_tensor", definition.dst_tensors[0]);


  desc.resize_function = [](const std::vector<BHWC>& src_shapes,
                            const std::vector<BHWC>& dst_shapes) {
    uint3 grid(dst_shapes[0].w, dst_shapes[0].h, 1);
    uint3 group_size{8u, 4u, 1u};
    uint3 groups;
    groups.x = DivideRoundUp(grid.x, group_size.x);
    groups.y = DivideRoundUp(grid.y, group_size.y);
    groups.z = DivideRoundUp(grid.z, group_size.z);
    return std::make_pair(group_size, groups);
  };

  return desc;
}

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

  std::string c = R"(
    kernel void ComputeFunction(
                                $0
                                uint3 ugid[[thread_position_in_grid]]) {
)";
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id_0 = static_cast<int>(ugid.x);\n";
    c += "  int X = linear_id_0 / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id_0 % args.dst_tensor.Batch();\n";
  } else {
    c += "  int X = static_cast<int>(ugid.x);\n";
  }
  if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "  int linear_id_1 = static_cast<int>(ugid.y);\n";
    c += "  int Y = linear_id_1 / args.dst_tensor.Depth();\n";
    c += "  int D = linear_id_1 % args.dst_tensor.Depth();\n";
  } else {
    c += "  int Y = static_cast<int>(ugid.y);\n";
  }
  c += "  int S = static_cast<int>(ugid.z);\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "S >= args.dst_tensor.Slices()) { \n";
  c += "    return; \n";
  c += "  } \n";
  c += "  FLT4 value = FLT4(0.0f);\n";
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
    c += "    value = args." + tensor_names[i] + ".Read(" + src_coord + ");\n";
    c += "  } \n";
    c += "  coord -= " + field + ";\n";
  }
  c += "  args.dst_tensor.Write(value, " + dst_coord + ");\n";
  c += "}\n";
  return c;
}
}  // namespace

ComputeTaskDescriptor ConcatX(const OperationDef& definition,
                              const ConcatAttributes& attr,
                              const std::vector<BHWC>& input_shapes) {
  ComputeTaskDescriptor desc(definition);
  desc.shader_source = GetConcatKernelCode(definition, attr);

  for (int i = 0; i < input_shapes.size(); ++i) {
    desc.AddSrcTensor("src_tensor_" + std::to_string(i),
                      definition.src_tensors[i]);
  }
  desc.AddDstTensor("dst_tensor", definition.dst_tensors[0]);

  desc.resize_function = [](const std::vector<BHWC>& src_shapes,
                            const std::vector<BHWC>& dst_shapes) {
    const uint3 groups_size{8, 4, 1};
    int groups_x = DivideRoundUp(dst_shapes[0].w, groups_size.x);
    int groups_y = DivideRoundUp(dst_shapes[0].h, groups_size.y);
    int groups_z = DivideRoundUp(dst_shapes[0].c, 4);
    return std::make_pair(groups_size, uint3{groups_x, groups_y, groups_z});
  };

  return desc;
}

ComputeTaskDescriptor ConcatY(const OperationDef& definition,
                              const ConcatAttributes& attr,
                              const std::vector<BHWC>& input_shapes) {
  ComputeTaskDescriptor desc(definition);
  desc.shader_source = GetConcatKernelCode(definition, attr);

  for (int i = 0; i < input_shapes.size(); ++i) {
    desc.AddSrcTensor("src_tensor_" + std::to_string(i),
                      definition.src_tensors[i]);
  }
  desc.AddDstTensor("dst_tensor", definition.dst_tensors[0]);

  desc.resize_function = [](const std::vector<BHWC>& src_shapes,
                            const std::vector<BHWC>& dst_shapes) {
    const uint3 groups_size{8, 4, 1};
    int groups_x = DivideRoundUp(dst_shapes[0].w, groups_size.x);
    int groups_y = DivideRoundUp(dst_shapes[0].h, groups_size.y);
    int groups_z = DivideRoundUp(dst_shapes[0].c, 4);
    return std::make_pair(groups_size, uint3{groups_x, groups_y, groups_z});
  };

  return desc;
}

ComputeTaskDescriptor Concat(const OperationDef& definition,
                             const ConcatAttributes& attr,
                             const std::vector<BHWC>& input_shapes) {
  if (attr.axis == Axis::CHANNELS) {
    return ConcatZ(definition, attr, input_shapes);
  } else if (attr.axis == Axis::WIDTH) {
    return ConcatX(definition, attr, input_shapes);
  } else {
    return ConcatY(definition, attr, input_shapes);
  }
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
