/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/service/llvm_ir/math_ops.h"

#include "xla/service/llvm_ir/llvm_util.h"

namespace xla {
namespace llvm_ir {

llvm::Value* EmitFastTanh(llvm::IRBuilder<>* b, llvm::Value* input,
                          bool with_fma) {
  llvm::Type* type = input->getType();
  const float plus_clamp =
      with_fma ? 7.99881172180175781f : 7.90531110763549805f;
  const float minus_clamp = -plus_clamp;
  // Inputs in the range [plus_clamp, 9.0] may cause the output
  // of EmitFastTanh to be greater than 1, so we set the input to be at most
  // 'plus_clamp'. We choose 'plus_clamp' in a way that the
  // tanh approximation on that input is exactly 1.0. Similarly for
  // 'minus_clamp', where the tanh approximation will return exactly
  // -1.0.
  // Taken from tanh(Eigen/src/Core/MathFunctionsImpl.h).

  // For small values of x, we can approximate tanh(x)=x. For extremely small
  // values of x (|x| < 1e-37), the other approximation evaluates tanh(x) = 0.
  const auto kCanUseApprox = 0.0004;
  auto abs_x =
      llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::fabs, {input}, {type}, b);
  auto use_aprox =
      b->CreateFCmpOLT(abs_x, llvm::ConstantFP::get(type, kCanUseApprox));

  // To simplify the code base until it's an issue, don't have a slow min/max in
  // this approximation.
  llvm::Value* input_clamped = llvm_ir::EmitFloatMin(
      llvm_ir::EmitFloatMax(input, llvm::ConstantFP::get(type, minus_clamp), b,
                            /*enable_fast_min_max=*/true),
      llvm::ConstantFP::get(type, plus_clamp), b, /*enable_fast_min_max=*/true);

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

llvm::Value* EmitErfF32(llvm::IRBuilder<>* b, llvm::Value* x) {
  auto type = x->getType();
  constexpr float kErfInvOneMinusHalfULP = 3.832506856900711f;
  auto call_fabs = [b](llvm::Value* operand_value) {
    return llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::fabs, {operand_value},
                                        {operand_value->getType()}, b);
  };
  auto fcmp_le = [b](llvm::Value* lhs_value, llvm::Value* rhs_value) {
    return b->CreateFCmpOLE(lhs_value, rhs_value);
  };
  llvm::Value* const clamp = fcmp_le(
      llvm::ConstantFP::get(type, kErfInvOneMinusHalfULP), call_fabs(x));
  // The monomial coefficients of the numerator polynomial (odd).
  llvm::Value* const alpha_1 = llvm::ConstantFP::get(type, 1.128379143519084f);
  llvm::Value* const alpha_3 =
      llvm::ConstantFP::get(type, 0.18520832239976145f);
  llvm::Value* const alpha_5 =
      llvm::ConstantFP::get(type, 0.050955695062380861f);
  llvm::Value* const alpha_7 =
      llvm::ConstantFP::get(type, 0.0034082910107109506f);
  llvm::Value* const alpha_9 =
      llvm::ConstantFP::get(type, 0.00022905065861350646f);

  // The monomial coefficients of the denominator polynomial (even).
  llvm::Value* const beta_0 = llvm::ConstantFP::get(type, 1.0f);
  llvm::Value* const beta_2 = llvm::ConstantFP::get(type, 0.49746925110067538f);
  llvm::Value* const beta_4 = llvm::ConstantFP::get(type, 0.11098505178285362f);
  llvm::Value* const beta_6 =
      llvm::ConstantFP::get(type, 0.014070470171167667f);
  llvm::Value* const beta_8 =
      llvm::ConstantFP::get(type, 0.0010179625278914885f);
  llvm::Value* const beta_10 =
      llvm::ConstantFP::get(type, 0.000023547966471313185f);
  llvm::Value* const beta_12 =
      llvm::ConstantFP::get(type, -1.1791602954361697e-7f);

  // Since the polynomials are odd/even, we need x^2.
  llvm::Value* const x2 = b->CreateFMul(x, x);

  // Evaluate the numerator polynomial p.
  auto call_fma = [b](llvm::Value* multiplier, llvm::Value* multiplicand,
                      llvm::Value* addend) {
    return llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::fma,
                                        {multiplier, multiplicand, addend},
                                        {multiplier->getType()}, b);
  };
  llvm::Value* p = call_fma(x2, alpha_9, alpha_7);
  p = call_fma(x2, p, alpha_5);
  p = call_fma(x2, p, alpha_3);
  p = call_fma(x2, p, alpha_1);
  p = b->CreateFMul(x, p);

  // Evaluate the denominator polynomial p.
  llvm::Value* q = call_fma(x2, beta_12, beta_10);
  q = call_fma(x2, q, beta_8);
  q = call_fma(x2, q, beta_6);
  q = call_fma(x2, q, beta_4);
  q = call_fma(x2, q, beta_2);
  q = call_fma(x2, q, beta_0);

  // Divide the numerator by the denominator.
  auto call_copysign = [b](llvm::Value* mag, llvm::Value* sign) {
    return llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::copysign, {mag, sign},
                                        {mag->getType()}, b);
  };
  auto* result =
      b->CreateSelect(clamp, call_copysign(llvm::ConstantFP::get(type, 1.0), x),
                      b->CreateFDiv(p, q));
  return result;
}

}  // namespace llvm_ir
}  // namespace xla
