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

#include "tensorflow/compiler/xla/service/dynamic_window_utils.h"

#include <string>

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"

namespace xla {
namespace {
// HloOp wraps an instuction pointer to do arithmetic based on operator
// overloading.
//
// TODO(yunxing): This is only used internally to this file to provide a
// convenient way to do operator overloadding.  Find out an idiom and merge this
// with hlo_creation_utils.
class HloOp {
 public:
  HloOp() = default;
  explicit HloOp(HloInstruction* inst) : inst_(inst) {}
  void SetName(const std::string& name) {
    inst_->SetAndSanitizeName(name);
    if (inst_->GetModule() != nullptr) {
      inst_->UniquifyName(&inst_->GetModule()->instruction_name_uniquer());
    }
  }
  HloInstruction* get() { return inst_; }

 private:
  HloInstruction* inst_ = nullptr;
};
HloOp BinaryOp(HloOp x, HloOp y, HloOpcode opcode,
               const std::string& name = "") {
  CHECK_EQ(x.get()->parent(), y.get()->parent());
  Shape binary_op_shape =
      ShapeInference::InferBinaryOpShape(opcode, x.get(), y.get()).value();
  return HloOp(x.get()->parent()->AddInstruction(
      HloInstruction::CreateBinary(binary_op_shape, opcode, x.get(), y.get()),
      name));
}
HloOp operator+(HloOp x, HloOp y) { return BinaryOp(x, y, HloOpcode::kAdd); }

HloOp operator-(HloOp x, HloOp y) {
  return BinaryOp(x, y, HloOpcode::kSubtract);
}

HloOp operator*(HloOp x, HloOp y) {
  return BinaryOp(x, y, HloOpcode::kMultiply);
}

HloOp operator/(HloOp x, HloOp y) { return BinaryOp(x, y, HloOpcode::kDivide); }

HloOp Maximum(HloOp x, HloOp y, const std::string& name = "") {
  return BinaryOp(x, y, HloOpcode::kMaximum, name);
}

template <typename NativeT>
HloOp ConstantR0(HloComputation* comp, NativeT value,
                 const std::string& name = "") {
  return HloOp(comp->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<NativeT>(value)),
      name));
}

template <typename NativeT>
HloOp One(HloComputation* comp) {
  return ConstantR0<NativeT>(comp, 1, "one");
}

template <typename NativeT>
HloOp Zero(HloComputation* comp) {
  return ConstantR0<NativeT>(comp, 0, "zero");
}

HloOp EffectiveFilterSize(HloComputation* comp, int64_t window_size,
                          int64_t window_dilation) {
  return ConstantR0<int32_t>(comp, (window_size - 1) * window_dilation + 1,
                             "effective_filter_size");
}
}  // namespace

DynamicWindowDims GetWindowedOutputSize(HloInstruction* input_size,
                                        int64_t window_size,
                                        int64_t window_dilation,
                                        int64_t window_stride,
                                        PaddingType padding_type) {
  HloComputation* comp = input_size->parent();
  DynamicWindowDims result;

  HloOp stride = ConstantR0<int32_t>(comp, window_stride, "stride");
  HloOp effective_filter_size =
      EffectiveFilterSize(comp, window_size, window_dilation);
  if (padding_type == PaddingType::PADDING_VALID) {
    HloOp output =
        (HloOp(input_size) + stride - effective_filter_size) / stride;
    result.output_size = output.get();
    result.padding_before = Zero<int32_t>(comp).get();
  } else if (padding_type == PaddingType::PADDING_SAME) {
    HloOp output = (HloOp(input_size) + stride - One<int32_t>(comp)) / stride;
    HloOp padding_needed = Maximum(
        Zero<int32_t>(comp), (output - One<int32_t>(comp)) * stride +
                                 effective_filter_size - HloOp(input_size));
    HloOp padding_before = padding_needed / ConstantR0<int32_t>(comp, 2);
    result.padding_before = padding_before.get();
    result.output_size = output.get();
  }

  return result;
}

DynamicWindowDims GetWindowedInputGradSize(HloInstruction* input_size,
                                           int64_t window_size,
                                           int64_t window_dilation,
                                           int64_t window_stride,
                                           PaddingType padding_type) {
  HloComputation* comp = input_size->parent();
  DynamicWindowDims result;
  HloOp effective_filter_size =
      ConstantR0<int32_t>(comp, (window_size - 1) * window_dilation + 1);
  HloOp stride = ConstantR0<int32_t>(comp, window_stride);
  DynamicWindowDims forward_dims = GetWindowedOutputSize(
      input_size, window_size, window_dilation, window_stride, padding_type);
  HloOp output_size =
      (HloOp(forward_dims.output_size) - One<int32_t>(comp)) * stride +
      One<int32_t>(comp);
  HloOp padding_before = effective_filter_size - One<int32_t>(comp) -
                         HloOp(forward_dims.padding_before);
  result.output_size = output_size.get();
  result.padding_before = padding_before.get();
  return result;
}
}  // namespace xla
