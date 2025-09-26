/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/delegates/gpu/common/tasks/cumsum.h"

#include <string>
#include <utility>

#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"

namespace tflite {
namespace gpu {

void Cumsum::GetCumsumCode(const OperationDef& op_def) {
  AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  AddDstTensor("dst_tensor", op_def.dst_tensors[0]);
  std::map<Axis, std::string> task_sizes = {
      {Axis::WIDTH, "args.src_tensor.Width()"},
      {Axis::HEIGHT, "args.src_tensor.Height()"},
      {Axis::DEPTH, "args.src_tensor.Depth()"},
      {Axis::CHANNELS, "args.src_tensor.Slices()"},
      {Axis::BATCH, "args.src_tensor.Batch()"},
  };
  std::string limit = task_sizes[axis_];
  task_sizes[axis_] = "1";
  std::map<Axis, std::string> index_name = {
      {Axis::WIDTH, "X"},    {Axis::HEIGHT, "Y"}, {Axis::DEPTH, "Z"},
      {Axis::CHANNELS, "S"}, {Axis::BATCH, "B"},
  };
  std::string indexes = "X, Y";
  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  if (definition_.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    indexes += ", Z";
    c += "  int linear_id = GLOBAL_ID_1;\n";
    c += "  int Y = linear_id % " + task_sizes[Axis::HEIGHT] + ";\n";
    c += "  int D = linear_id / " + task_sizes[Axis::HEIGHT] + ";\n";
    c += "  if (D >= " + task_sizes[Axis::DEPTH] + ") return;\n";
  } else {
    c += "  int Y = GLOBAL_ID_1;\n";
    c += "  if (Y >= " + task_sizes[Axis::HEIGHT] + ") return;\n";
  }
  indexes += ", S";
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    indexes += ", B";
    c += "  int linear_id = GLOBAL_ID_0;\n";
    c += "  int X = linear_id / " + task_sizes[Axis::BATCH] + ";\n";
    c += "  int B = linear_id % " + task_sizes[Axis::BATCH] + ";\n";
    c += "  if (X >= " + task_sizes[Axis::WIDTH] + ") return;\n";
  } else {
    c += "  int X = GLOBAL_ID_0;\n";
    c += "  if (X >= " + task_sizes[Axis::WIDTH] + ") return;\n";
  }
  c += "  int S = GLOBAL_ID_2;\n";
  c += "  if (S >= " + task_sizes[Axis::CHANNELS] + ") return;\n";
  if (definition_.precision == CalculationsPrecision::F16) {
    c += "  float4 res = TO_FLOAT4(args.src_tensor::zero_value);\n";
  } else {
    c += "  args.src_tensor::type res = args.src_tensor::zero_value;\n";
  }
  c += "  for (; " + index_name[axis_] + " < " + limit + "; " +
       index_name[axis_] + "++) {\n";
  if (definition_.precision == CalculationsPrecision::F16) {
    c +=
        "    float4 curr = TO_FLOAT4(args.src_tensor.Read(" + indexes + "));\n";
  } else {
    c += "    args.src_tensor::type curr = args.src_tensor.Read(" + indexes +
         ");\n";
  }

  if (axis_ == Axis::CHANNELS) {
    if (definition_.precision == CalculationsPrecision::F16) {
      c += "    res.x = res.w + curr.x;\n";
      c += "    res.y = res.x + curr.y;\n";
      c += "    res.z = res.y + curr.z;\n";
      c += "    res.w = res.z + curr.w;\n";
    } else {
      c += "    res.x = res.w + curr.x;\n";
      c += "    res.y = res.x + curr.y;\n";
      c += "    res.z = res.y + curr.z;\n";
      c += "    res.w = res.z + curr.w;\n";
    }
  } else {
    c += "    res += curr;\n";
  }
  if (definition_.precision == CalculationsPrecision::F16) {
    c += "    args.dst_tensor.Write(TO_FLT4(res), " + indexes + ");\n";
  } else {
    c += "    args.dst_tensor.Write(res, " + indexes + ");\n";
  }

  c += "  }\n";
  c += "}\n";
  code_ = c;
}

int3 Cumsum::GetGridSize() const {
  const int width = axis_ == Axis::WIDTH ? 1 : src_[0]->Width();
  const int height = axis_ == Axis::HEIGHT ? 1 : src_[0]->Height();
  const int depth = axis_ == Axis::DEPTH ? 1 : src_[0]->Depth();
  const int batch = axis_ == Axis::BATCH ? 1 : src_[0]->Batch();
  const int slices = axis_ == Axis::CHANNELS ? 1 : src_[0]->Slices();
  const int grid_x = width * batch;
  const int grid_y = height * depth;
  const int grid_z = slices;
  return int3(grid_x, grid_y, grid_z);
}

Cumsum::Cumsum(Cumsum&& operation)
    : GPUOperation(std::move(operation)), axis_(operation.axis_) {}

Cumsum& Cumsum::operator=(Cumsum&& operation) {
  if (this != &operation) {
    axis_ = operation.axis_;
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

Cumsum CreateCumsum(const OperationDef& definition,
                    const CumsumAttributes& attr) {
  Cumsum op(definition, attr.axis);
  op.GetCumsumCode(definition);
  return op;
}

}  // namespace gpu
}  // namespace tflite
