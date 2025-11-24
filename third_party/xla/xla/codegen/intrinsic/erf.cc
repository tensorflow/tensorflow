/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/codegen/intrinsic/erf.h"

#include "absl/log/check.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Casting.h"
#include "xla/codegen/intrinsic/intrinsic.h"
#include "xla/service/llvm_ir/llvm_util.h"

namespace xla::codegen::intrinsics {

// Emits an approximation of erf. The implementation uses the same rational
// interpolant as implemented in Eigen3.
static llvm::Value* EmitErfF32(llvm::IRBuilderBase* b, llvm::Value* x) {
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

absl::StatusOr<llvm::Function*> Erf::CreateDefinition(
    llvm::Module* module, const Type intrinsic_type) {
  llvm::Type* type = Type::TypeToIrType(intrinsic_type, module->getContext());
  CHECK(type != nullptr);
  CHECK(type->isFloatTy() ||
        (type->isVectorTy() && type->getScalarType()->isFloatTy()))
      << "Type must be a f32 or vector of f32.";

  llvm::LLVMContext& context = module->getContext();
  llvm::IRBuilder<> builder(context);

  int num_elements = 1;
  if (llvm::VectorType* vec_ty = llvm::dyn_cast<llvm::VectorType>(type)) {
    num_elements = vec_ty->getElementCount().getKnownMinValue();
  }

  llvm::FunctionType* function_type =
      llvm::FunctionType::get(type, {type}, false);
  llvm::Function* func = llvm::dyn_cast<llvm::Function>(
      module->getOrInsertFunction(Name(intrinsic_type), function_type)
          .getCallee());

  llvm::Argument* input_value = func->getArg(0);
  input_value->setName("input_value");

  llvm::BasicBlock* entry_bb = llvm::BasicBlock::Create(context, "entry", func);
  builder.SetInsertPoint(entry_bb);

  builder.CreateRet(EmitErfF32(&builder, input_value));

  return func;
}

}  // namespace xla::codegen::intrinsics
