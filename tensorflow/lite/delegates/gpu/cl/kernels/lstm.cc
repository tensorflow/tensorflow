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

#include "tensorflow/lite/delegates/gpu/cl/kernels/lstm.h"

#include <string>

#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/work_group_picking.h"
#include "tensorflow/lite/delegates/gpu/cl/precision.h"

namespace tflite {
namespace gpu {
namespace cl {

LSTM::LSTM(const OperationDef& definition, const DeviceInfo& device_info)
    : GPUOperation(definition) {
  code_ = GetLSTMCode(definition_, device_info);
}

LSTM::LSTM(LSTM&& kernel) : GPUOperation(std::move(kernel)) {}

LSTM& LSTM::operator=(LSTM&& kernel) {
  if (this != &kernel) {
    GPUOperation::operator=(std::move(kernel));
  }
  return *this;
}

std::string LSTM::GetLSTMCode(const OperationDef& op_def,
                              const DeviceInfo& device_info) {
  AddSrcTensor("intermediate", op_def.src_tensors[0]);
  AddSrcTensor("prev_state", op_def.src_tensors[1]);
  AddDstTensor("new_state", op_def.dst_tensors[0]);
  AddDstTensor("activation", op_def.dst_tensors[1]);

  std::string c = GetCommonDefines(op_def.precision);
  c += "__kernel void main_function(\n";
  c += "$0) {\n";
  c += "  int B = get_global_id(0);\n";
  c += "  int Z = get_global_id(1);\n";
  c += "  if (Z >= args.activation.Slices() || B >= args.activation.Batch()) "
       "return;\n";
  c += "  FLT4 prev_st = args.prev_state.Read(0, 0, Z, B);\n";
  c += "  FLT4 r0 = args.intermediate.Read(0, 0, Z, B);\n";
  c += "  int state_stride = args.activation.Slices();\n";
  c += "  FLT4 r1 = args.intermediate.Read(0, 0, Z + state_stride, B);\n";
  c += "  FLT4 r2 = args.intermediate.Read(0, 0, Z + state_stride * 2, B);\n";
  c += "  FLT4 r3 = args.intermediate.Read(0, 0, Z + state_stride * 3, B);\n";
  if (op_def.precision != CalculationsPrecision::F32 &&
      device_info.IsAdreno()) {
    c += "  FLT4 input_gate;\n";
    c += "  FLT4 new_input;\n";
    c += "  FLT4 forget_gate;\n";
    c += "  FLT4 output_gate;\n";
    c += "  input_gate.x = native_recip(1.0h + native_exp(-r0.x));\n";
    c += "  input_gate.y = native_recip(1.0h + native_exp(-r0.y));\n";
    c += "  input_gate.z = native_recip(1.0h + native_exp(-r0.z));\n";
    c += "  input_gate.w = native_recip(1.0h + native_exp(-r0.w));\n";
    c += "  new_input.x = 1.0h - 2.0h * native_recip(1.0h + native_exp(2.0h * "
         "r1.x));\n";
    c += "  new_input.y = 1.0h - 2.0h * native_recip(1.0h + native_exp(2.0h * "
         "r1.y));\n";
    c += "  new_input.z = 1.0h - 2.0h * native_recip(1.0h + native_exp(2.0h * "
         "r1.z));\n";
    c += "  new_input.w = 1.0h - 2.0h * native_recip(1.0h + native_exp(2.0h * "
         "r1.w));\n";
    c += "  forget_gate.x = native_recip(1.0h + native_exp(-r2.x));\n";
    c += "  forget_gate.y = native_recip(1.0h + native_exp(-r2.y));\n";
    c += "  forget_gate.z = native_recip(1.0h + native_exp(-r2.z));\n";
    c += "  forget_gate.w = native_recip(1.0h + native_exp(-r2.w));\n";
    c += "  output_gate.x = native_recip(1.0h + native_exp(-r3.x));\n";
    c += "  output_gate.y = native_recip(1.0h + native_exp(-r3.y));\n";
    c += "  output_gate.z = native_recip(1.0h + native_exp(-r3.z));\n";
    c += "  output_gate.w = native_recip(1.0h + native_exp(-r3.w));\n";
  } else {
    c +=
        "  FLT4 input_gate  = (FLT4)(1.0f) / ((FLT4)(1.0f) + exp((FLT4)(-1.0f) "
        "* r0));\n";
    c += "  FLT4 new_input   = tanh(r1);\n";
    c +=
        "  FLT4 forget_gate = (FLT4)(1.0f) / ((FLT4)(1.0f) + exp((FLT4)(-1.0f) "
        "* r2));\n";
    c +=
        "  FLT4 output_gate = (FLT4)(1.0f) / ((FLT4)(1.0f) + exp((FLT4)(-1.0f) "
        "* r3));\n";
  }
  c += "  FLT4 new_st = input_gate * new_input + forget_gate * prev_st;\n";
  c += "  FLT4 act_value = output_gate * tanh(new_st);\n";
  c += "  args.activation.Write(act_value, 0, 0, Z, B);\n";
  c += "  args.new_state.Write(new_st, 0, 0, Z, B);\n";
  c += "}\n";
  return c;
}

int3 LSTM::GetGridSize() const {
  const int grid_x = dst_[0]->Batch();
  const int grid_y = dst_[0]->Slices();
  const int grid_z = 1;
  return int3(grid_x, grid_y, grid_z);
}

LSTM CreateLSTM(const OperationDef& definition, const DeviceInfo& device_info) {
  return LSTM(definition, device_info);
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
