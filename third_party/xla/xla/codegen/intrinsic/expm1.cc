/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/codegen/intrinsic/expm1.h"

#include <limits>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "xla/codegen/intrinsic/intrinsic.h"
#include "xla/xla_data.pb.h"

namespace xla::codegen::intrinsics {

absl::StatusOr<llvm::Function*> Expm1::CreateDefinition(llvm::Module* module,
                                                        Type type) {
  llvm::Type* input_type = Type::TypeToIrType(type, module->getContext());
  CHECK(input_type != nullptr);
  CHECK(input_type->isFloatingPointTy() || input_type->isVectorTy())
      << "Type must be a floating point or vector of floating point.";
  llvm::Type* scalar_type = input_type->getScalarType();
  CHECK(scalar_type->isFloatTy() || scalar_type->isDoubleTy())
      << "Only F32 and F64 are supported for xla.expm1.";

  const bool is_f64 = scalar_type->isDoubleTy();
  llvm::LLVMContext& context = module->getContext();
  llvm::IRBuilder<> builder(context);
  llvm::FastMathFlags fmf;
  fmf.setAllowContract(true);
  builder.setFastMathFlags(fmf);

  llvm::Type* int_type = is_f64 ? llvm::Type::getInt64Ty(context)
                                : llvm::Type::getInt32Ty(context);
  if (auto* vec_ty = llvm::dyn_cast<llvm::VectorType>(input_type)) {
    int_type = llvm::VectorType::get(int_type, vec_ty->getElementCount());
  }

  llvm::FunctionType* function_type =
      llvm::FunctionType::get(input_type, {input_type}, false);
  llvm::Function* func = llvm::dyn_cast<llvm::Function>(
      module->getOrInsertFunction(Expm1::Name(type), function_type)
          .getCallee());

  llvm::Argument* input_x = func->getArg(0);
  input_x->setName("input_x");

  llvm::BasicBlock* entry_bb = llvm::BasicBlock::Create(context, "entry", func);
  builder.SetInsertPoint(entry_bb);

  const auto v_const = [&](double val) {
    return llvm::ConstantFP::get(input_type, val);
  };

  llvm::Constant* kZero = v_const(0.0);
  llvm::Constant* kOne = v_const(1.0);
  llvm::Constant* kNegOne = v_const(-1.0);
  llvm::Constant* kHalf = v_const(0.5);

  // Operational range [kExpLo, kExpHi].
  llvm::Constant* kExpLo =
      v_const(is_f64 ? -0x1.2b708872320e2p+5 : -0x1.154246p+4f);
  llvm::Constant* kExpHi =
      v_const(is_f64 ? 0x1.62e42fefa39efp+9 : 0x1.62e43p+6f);

  llvm::Value* is_below_lo = builder.CreateFCmpOLT(input_x, kExpLo);
  llvm::Value* is_above_hi = builder.CreateFCmpOGT(input_x, kExpHi);
  llvm::Value* is_zero = builder.CreateFCmpOEQ(input_x, kZero);

  llvm::Function* minimum_fn = llvm::Intrinsic::getOrInsertDeclaration(
      module, llvm::Intrinsic::minimum, {input_type});
  llvm::Function* maximum_fn = llvm::Intrinsic::getOrInsertDeclaration(
      module, llvm::Intrinsic::maximum, {input_type});
  llvm::Value* x_clamped =
      builder.CreateCall(minimum_fn, {input_x, kExpHi}, "x_clamp_hi");
  x_clamped = builder.CreateCall(maximum_fn, {x_clamped, kExpLo}, "x_clamp_lo");

  // Argument reduction: x = n * ln(2) + r_hi + r_lo, where
  // n = round_to_nearest_even(x * log2(e)).
  llvm::Constant* kLog2e =
      v_const(is_f64 ? 0x1.71547652b82fep+0 : 0x1.715476p+0f);
  llvm::Function* rint_fn = llvm::Intrinsic::getOrInsertDeclaration(
      module, llvm::Intrinsic::rint, {input_type});
  llvm::Value* n = builder.CreateCall(
      rint_fn, {builder.CreateFMul(x_clamped, kLog2e)}, "n_rint");

  // Cody-Waite head-tail split of ln(2) = LN2_HI + LN2_LO.
  llvm::Constant* kLn2Hi =
      v_const(is_f64 ? 0x1.62e42fefa0000p-1 : 0x1.62e4p-1f);
  llvm::Constant* kLn2Lo =
      v_const(is_f64 ? 0x1.cf79abc9e3b3ap-40 : 0x1.7f7d1cp-20f);

  llvm::Value* n_hi = builder.CreateFMul(n, kLn2Hi);
  llvm::Value* r_hi = builder.CreateFSub(x_clamped, n_hi, "r_hi");
  llvm::Value* neg_n = builder.CreateFNeg(n);
  llvm::Value* r_lo = builder.CreateFMul(neg_n, kLn2Lo, "r_lo");
  llvm::Value* r = builder.CreateFAdd(r_hi, r_lo, "r");

  llvm::Value* q = nullptr;
  if (is_f64) {
    // Sollya degree-12 minimax polynomial for F64 on [-ln(2)/2, ln(2)/2].
    q = v_const(0x1.1f0413a6fc15ap-29);  // C12
    for (double c : {
             0x1.af5f1ac764956p-26,  // C11
             0x1.27e521cebe596p-22,  // C10
             0x1.71ddf816ea1afp-19,  // C9
             0x1.a01a01872c144p-16,  // C8
             0x1.a01a01b001948p-13,  // C7
             0x1.6c16c16c1c0d1p-10,  // C6
             0x1.111111110f657p-7,   // C5
             0x1.555555555554ap-5,   // C4
             0x1.5555555555559p-3,   // C3
         }) {
      q = builder.CreateFAdd(builder.CreateFMul(q, r), v_const(c));
    }
  } else {
    // Sollya degree-6 minimax polynomial for F32 on [-ln(2)/2, ln(2)/2].
    q = v_const(0x1.6c50aap-10f);  // C6
    q = builder.CreateFAdd(builder.CreateFMul(q, r),
                           v_const(0x1.1222d6p-7f));  // C5
    q = builder.CreateFAdd(builder.CreateFMul(q, r),
                           v_const(0x1.5555bep-5f));  // C4
    q = builder.CreateFAdd(builder.CreateFMul(q, r),
                           v_const(0x1.5554b8p-3f));  // C3
  }

  llvm::Value* h = builder.CreateFAdd(builder.CreateFMul(q, r), kHalf);
  llvm::Value* r2 = builder.CreateFMul(r, r);
  llvm::Value* r2_h = builder.CreateFMul(r2, h);
  llvm::Value* tail = builder.CreateFAdd(r_lo, r2_h);

  // Reconstruct expm1(x) = ((2^n - 1) + 2^n * r_hi) + 2^n * tail.
  // Split n = n1 + n2 where n1 = min(n_int, max_exp) and n2 = n_int - n1 to
  // avoid intermediate overflow when n == max_exp + 1 (128 for f32, 1024 for
  // f64).
  const int64_t max_exp = is_f64 ? 1023 : 127;
  const int64_t mantissa_bits = is_f64 ? 52 : 23;

  llvm::Value* n_int = builder.CreateFreeze(builder.CreateFPToSI(n, int_type));
  llvm::Value* bias_const = llvm::ConstantInt::get(int_type, max_exp);
  llvm::Value* n1_cmp = builder.CreateICmpSLT(n_int, bias_const);
  llvm::Value* n1 = builder.CreateSelect(n1_cmp, n_int, bias_const);
  llvm::Value* n2 = builder.CreateSub(n_int, n1);

  llvm::Value* shift = llvm::ConstantInt::get(int_type, mantissa_bits);
  llvm::Value* exp1_bits =
      builder.CreateShl(builder.CreateAdd(n1, bias_const), shift);
  llvm::Value* exp2_bits =
      builder.CreateShl(builder.CreateAdd(n2, bias_const), shift);
  llvm::Value* s1 = builder.CreateBitCast(exp1_bits, input_type, "two_pow_n1");
  llvm::Value* s2 = builder.CreateBitCast(exp2_bits, input_type, "two_pow_n2");

  // Regroup as ((s1 - 1) + s1 * r_hi) + s1 * tail so that (s1 - 1) + s1 * r_hi
  // is exact by Sterbenz's lemma across n = +/-1 binade drops.
  llvm::Value* s1_minus_one = builder.CreateFSub(s1, kOne);
  llvm::Value* s1_r_hi = builder.CreateFMul(s1, r_hi);
  llvm::Value* s1_tail = builder.CreateFMul(s1, tail);
  llvm::Value* head = builder.CreateFAdd(s1_minus_one, s1_r_hi);
  llvm::Value* result =
      builder.CreateFMul(builder.CreateFAdd(head, s1_tail), s2, "result");

  // Select special values:
  // - preserve signed zero (expm1(-0.0) = -0.0)
  // - x <= UNDERFLOW_THRESHOLD -> -1.0
  // - x >= OVERFLOW_THRESHOLD -> +inf
  llvm::Constant* kInf = v_const(std::numeric_limits<double>::infinity());
  result = builder.CreateSelect(is_zero, input_x, result);
  result = builder.CreateSelect(is_below_lo, kNegOne, result);
  result = builder.CreateSelect(is_above_hi, kInf, result);

  builder.CreateRet(result);
  return func;
}

}  // namespace xla::codegen::intrinsics
