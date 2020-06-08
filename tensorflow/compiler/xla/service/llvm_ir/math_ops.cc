/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/llvm_ir/math_ops.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"

namespace xla {
namespace llvm_ir {

llvm::Value* EmitFastTanh(llvm::IRBuilder<>* b, llvm::Value* input) {
  llvm::Type* type = input->getType();

  // For small values of x, we can approximate tanh(x)=x. For extremely small
  // values of x (|x| < 1e-37), the other approximation evaluates tanh(x) = 0.
  const auto kCanUseApprox = 0.0004;
  auto abs_x =
      llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::fabs, {input}, {type}, b);
  auto use_aprox =
      b->CreateFCmpOLT(abs_x, llvm::ConstantFP::get(type, kCanUseApprox));

  // Clamp the input to [-9, 9].
  llvm::Value* input_clamped = llvm_ir::EmitFloatMin(
      llvm_ir::EmitFloatMax(input, llvm::ConstantFP::get(type, -9.0), b),
      llvm::ConstantFP::get(type, 9.0), b);

  static constexpr std::array<float, 7> numerator_coeffs{
      -2.76076847742355e-16f, 2.00018790482477e-13f, -8.60467152213735e-11f,
      5.12229709037114e-08f,  1.48572235717979e-05f, 6.37261928875436e-04f,
      4.89352455891786e-03f};

  static constexpr std::array<float, 4> denominator_coeffs{
      1.19825839466702e-06f, 1.18534705686654e-04f, 2.26843463243900e-03f,
      4.89352518554385e-03f};

  llvm::Value* input_squared = b->CreateFMul(input_clamped, input_clamped);
  llvm::Value* numerator = llvm::ConstantFP::get(type, numerator_coeffs[0]);
  for (int i = 1; i < numerator_coeffs.size(); i++) {
    numerator = b->CreateFAdd(b->CreateFMul(input_squared, numerator),
                              llvm::ConstantFP::get(type, numerator_coeffs[i]));
  }

  numerator = b->CreateFMul(input_clamped, numerator);

  llvm::Value* denominator = llvm::ConstantFP::get(type, denominator_coeffs[0]);
  for (int i = 1; i < denominator_coeffs.size(); i++) {
    denominator =
        b->CreateFAdd(b->CreateFMul(input_squared, denominator),
                      llvm::ConstantFP::get(type, denominator_coeffs[i]));
  }

  return b->CreateSelect(use_aprox, input,
                         b->CreateFDiv(numerator, denominator));
}

}  // namespace llvm_ir
}  // namespace xla
