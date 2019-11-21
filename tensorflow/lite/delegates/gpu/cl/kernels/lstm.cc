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
namespace {

std::string GetLSTMCode(const OperationDef& op_def, const CLDevice& device) {
  const TensorCodeGenerator::SizeVariablesNames state_size(
      "1", "1", "state_size.w", "BATCH_SIZE");
  const TensorCodeGenerator::SizeVariablesNames src_size("1", "1", "src_size.w",
                                                         "BATCH_SIZE");

  TensorCodeGenerator intermediate("src_data", src_size, op_def.src_tensors[0]);
  TensorCodeGenerator prev_state("prev_state", state_size,
                                 op_def.src_tensors[1]);

  TensorCodeGenerator activation("dst_data", state_size, op_def.dst_tensors[0]);
  TensorCodeGenerator new_state("new_state", state_size, op_def.dst_tensors[1]);

  std::string c = GetCommonDefines(op_def.precision);

  c += "__kernel void main_function(\n";
  c += intermediate.GetDeclaration(AccessType::READ) + ",\n";
  c += prev_state.GetDeclaration(AccessType::READ) + ",\n";
  c += new_state.GetDeclaration(AccessType::WRITE) + ",\n";
  c += activation.GetDeclaration(AccessType::WRITE) + ",\n";
  c += "    int4 src_size,             \n";
  c += "    int4 state_size,           \n";
  c += "    int BATCH_SIZE             \n";
  c += ") {\n";
  c += "  int B = get_global_id(0);\n";
  c += "  int Z = get_global_id(1);\n";
  c += "  if (Z >= state_size.w || B >= BATCH_SIZE) return;\n";
  c += "  FLT4 prev_st = " + prev_state.Read4D("0", "0", "Z", "B") + ";\n";
  c += "  FLT4 r0 = " + intermediate.Read4D("0", "0", "Z", "B") + ";\n";
  c += "  FLT4 r1 = " + intermediate.Read4D("0", "0", "Z + state_size.w", "B") +
       ";\n";
  c += "  FLT4 r2 = " +
       intermediate.Read4D("0", "0", "Z + state_size.w * 2", "B") + ";\n";
  c += "  FLT4 r3 = " +
       intermediate.Read4D("0", "0", "Z + state_size.w * 3", "B") + ";\n";
  if (op_def.precision != CalculationsPrecision::F32 && device.IsAdreno()) {
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
  c += "  FLT4 activation = output_gate * tanh(new_st);\n";
  c += "  " + activation.Write4D("activation", "0", "0", "Z", "B");
  c += "  " + new_state.Write4D("new_st", "0", "0", "Z", "B");
  c += "}\n";
  return c;
}
}  // namespace

LSTM::LSTM(const OperationDef& definition) : GPUOperation(definition) {}

LSTM::LSTM(LSTM&& kernel)
    : GPUOperation(std::move(kernel)),
      kernel_(std::move(kernel.kernel_)),
      work_group_size_(kernel.work_group_size_) {}

LSTM& LSTM::operator=(LSTM&& kernel) {
  if (this != &kernel) {
    kernel_ = std::move(kernel.kernel_);
    std::swap(work_group_size_, kernel.work_group_size_);
    GPUOperation::operator=(std::move(kernel));
  }
  return *this;
}

Status LSTM::Compile(const CreationContext& creation_context) {
  const auto code = GetLSTMCode(definition_, *creation_context.device);
  return creation_context.cache->GetOrCreateCLKernel(
      code, "main_function", *creation_context.context,
      *creation_context.device, &kernel_);
}

Status LSTM::BindArguments() {
  kernel_.ResetBindingCounter();
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(src_[0]->GetMemoryPtr()));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(src_[1]->GetMemoryPtr()));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(dst_[0]->GetMemoryPtrForWriting()));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(dst_[1]->GetMemoryPtrForWriting()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(src_[0]->GetSizeWithDepth()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(dst_[0]->GetSizeWithDepth()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(dst_[0]->Batch()));

  return OkStatus();
}

int3 LSTM::GetGridSize() const {
  const int grid_x = dst_[0]->Batch();
  const int grid_y = dst_[0]->Depth();
  const int grid_z = 1;
  return int3(grid_x, grid_y, grid_z);
}

Status LSTM::Tune(const TuningParameters& params) {
  RETURN_IF_ERROR(BindArguments());
  return GetBestWorkGroup(params, kernel_, GetGridSize(), &work_group_size_);
}

Status LSTM::AddToQueue(CLCommandQueue* queue) {
  RETURN_IF_ERROR(BindArguments());
  return queue->DispatchImplicit(kernel_, GetGridSize(), work_group_size_);
}

LSTM CreateLSTM(const OperationDef& definition) { return LSTM(definition); }

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
