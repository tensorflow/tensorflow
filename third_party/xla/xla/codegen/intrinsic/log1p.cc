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

#include "xla/codegen/intrinsic/log1p.h"

#include <array>

#include "absl/log/check.h"
#include "absl/types/span.h"
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
#include "xla/xla_data.pb.h"

namespace xla::codegen::intrinsics {

static llvm::Value* EvaluatePolynomial(llvm::Type* type, llvm::Value* x,
                                       absl::Span<const double> coefficients,
                                       llvm::IRBuilder<>& builder) {
  llvm::Value* poly = llvm::ConstantFP::get(type, 0.0);
  for (const double c : coefficients) {
    poly = builder.CreateFAdd(builder.CreateFMul(poly, x),
                              llvm::ConstantFP::get(type, c));
  }
  return poly;
}

absl::StatusOr<llvm::Function*> Log1p::CreateDefinition(llvm::Module* module,
                                                        Type intrinsic_type) {
  llvm::Type* type = Type::TypeToIrType(intrinsic_type, module->getContext());
  CHECK(type != nullptr);
  CHECK(type->isFloatingPointTy() || type->isVectorTy())
      << "Type must be a floating point or vector of floating point.";

  llvm::LLVMContext& context = module->getContext();
  llvm::IRBuilder<> builder(context);

  int num_elements = 1;
  if (llvm::VectorType* vec_ty = llvm::dyn_cast<llvm::VectorType>(type)) {
    num_elements = vec_ty->getElementCount().getKnownMinValue();
  }

  llvm::FunctionType* function_type =
      llvm::FunctionType::get(type, {type}, false);
  llvm::Function* func = llvm::dyn_cast<llvm::Function>(
      module->getOrInsertFunction(Log1p::Name(intrinsic_type), function_type)
          .getCallee());

  llvm::Argument* input_value = func->getArg(0);
  input_value->setName("input_value");

  llvm::BasicBlock* entry_bb = llvm::BasicBlock::Create(context, "entry", func);
  builder.SetInsertPoint(entry_bb);

  auto x = input_value;
  auto one = llvm::ConstantFP::get(type, 1.0);
  auto negative_half = llvm::ConstantFP::get(type, -0.5);
  // When x is large, the naive evaluation of ln(x + 1) is more
  // accurate than the Taylor series.
  auto for_large_x = llvm_ir::EmitCallToIntrinsic(
      llvm::Intrinsic::log, {builder.CreateFAdd(x, one)}, {type}, &builder);
  // When x is small, (defined to be less than sqrt(2) / 2), use a rational
  // approximation. The approximation below is based on one from the Cephes
  // Mathematical Library.
  //
  // sqrt(2) - 1.
  const auto kAntilogarithmIsSmallThreshold = 0.41421356237309504880;

  static const std::array<double, 7> kDenominatorCoeffs{
      1.,
      1.5062909083469192043167E1,
      8.3047565967967209469434E1,
      2.2176239823732856465394E2,
      3.0909872225312059774938E2,
      2.1642788614495947685003E2,
      6.0118660497603843919306E1,
  };

  static const std::array<double, 7> kNumeratorCoeffs{
      4.5270000862445199635215E-5, 4.9854102823193375972212E-1,
      6.5787325942061044846969E0,  2.9911919328553073277375E1,
      6.0949667980987787057556E1,  5.7112963590585538103336E1,
      2.0039553499201281259648E1,
  };

  auto x_squared = builder.CreateFMul(x, x);
  auto denominator = EvaluatePolynomial(type, x, kDenominatorCoeffs, builder);
  auto numerator = EvaluatePolynomial(type, x, kNumeratorCoeffs, builder);
  auto for_small_x = builder.CreateFDiv(numerator, denominator);
  for_small_x =
      builder.CreateFMul(builder.CreateFMul(x, x_squared), for_small_x);
  for_small_x = builder.CreateFAdd(builder.CreateFMul(negative_half, x_squared),
                                   for_small_x);
  for_small_x = builder.CreateFAdd(x, for_small_x);

  auto abs_x = llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::fabs,
                                            {input_value}, {type}, &builder);
  auto x_is_small = builder.CreateFCmpOLT(
      abs_x, llvm::ConstantFP::get(type, kAntilogarithmIsSmallThreshold));
  llvm::Value* result =
      builder.CreateSelect(x_is_small, for_small_x, for_large_x);

  builder.CreateRet(result);

  return func;
}
}  // namespace xla::codegen::intrinsics
