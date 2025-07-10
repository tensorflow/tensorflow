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

#include "xla/codegen/math/rsqrt.h"

#include <cstddef>
#include <string>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsX86.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/TypeSize.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/xla_data.pb.h"

namespace xla::codegen::math {

std::string RsqrtFunctionName(size_t num_elements, PrimitiveType type) {
  std::string vector_width =
      num_elements > 1 ? absl::StrCat("v", num_elements, ".") : "";
  return absl::StrCat("xla.rsqrt.", vector_width,
                      absl::AsciiStrToLower(PrimitiveType_Name(type)));
}

static llvm::Value* NewtonRaphsonRsqrtIteration(llvm::IRBuilder<>& builder,
                                                llvm::Value* x, llvm::Value* y,
                                                llvm::Type* type) {
  llvm::Module* module = builder.GetInsertBlock()->getModule();
  llvm::Function* fma_func = llvm::Intrinsic::getOrInsertDeclaration(
      module, llvm::Intrinsic::fma, {type});

  // This function implements the refinement step using the formula:
  // y_new = y * 0.5 * (3 - x * y^2)
  // It is implemented using an FMA for performance as:
  // refined_y = (y * 0.5) * fma(-x, y^2, 3.0)
  llvm::Constant* half = llvm::ConstantFP::get(type, 0.5);
  llvm::Constant* three = llvm::ConstantFP::get(type, 3.0);
  llvm::Value* y_squared = builder.CreateFMul(y, y, "y_squared");
  llvm::Value* neg_x = builder.CreateFNeg(x, "neg_x");
  llvm::Value* correction =
      builder.CreateCall(fma_func, {neg_x, y_squared, three}, "correction.fma");
  llvm::Value* half_y = builder.CreateFMul(half, y, "half_y");
  llvm::Value* refined_y = builder.CreateFMul(half_y, correction, "refined_y");
  return refined_y;
}

struct RsqrtIntrinsic {
  llvm::Intrinsic::ID id;
  int mask_bits;  // Some avx512 calls require masks.
  bool needs_insert_element;

  static RsqrtIntrinsic ForF32(size_t num_elements) {
    switch (num_elements) {
      case 1:
        return {llvm::Intrinsic::x86_sse_rsqrt_ss, 0, true};
      case 4:
        return {llvm::Intrinsic::x86_sse_rsqrt_ps, 0, false};
      case 8:
        return {llvm::Intrinsic::x86_avx_rsqrt_ps_256, 0, false};
      case 16:
        return {llvm::Intrinsic::x86_avx512_rsqrt14_ps_512, 16, false};
      default:
        LOG(FATAL) << "Unsupported vector width for rsqrt: " << num_elements;
    }
  }

  static RsqrtIntrinsic ForF64(size_t num_elements) {
    // We assume AVX512 is available for F64.
    switch (num_elements) {
      case 2:
        return {llvm::Intrinsic::x86_avx512_rsqrt14_pd_128, 8, false};
      case 4:
        return {llvm::Intrinsic::x86_avx512_rsqrt14_pd_256, 8, false};
      case 8:
        return {llvm::Intrinsic::x86_avx512_rsqrt14_pd_512, 8, false};
      default:
        LOG(FATAL) << "Unsupported vector width for rsqrt: " << num_elements;
    }
  }

  llvm::Value* CreateCall(llvm::IRBuilder<>& builder, llvm::Value* x) {
    llvm::Module* module = builder.GetInsertBlock()->getModule();
    llvm::Function* rsqrt_intrinsic =
        llvm::Intrinsic::getOrInsertDeclaration(module, id);

    llvm::Value* y_approx;
    if (needs_insert_element) {
      llvm::Type* sse_vec_type = llvm::VectorType::get(
          x->getType()->getScalarType(), llvm::ElementCount::getFixed(4));
      llvm::Value* vec_x = llvm::UndefValue::get(sse_vec_type);
      vec_x = builder.CreateInsertElement(vec_x, x, builder.getInt32(0));
      llvm::Value* approx_vec =
          builder.CreateCall(rsqrt_intrinsic, {vec_x}, "y_approx.vec");
      y_approx = builder.CreateExtractElement(approx_vec, builder.getInt32(0),
                                              "y_approx");
    } else if (mask_bits > 0) {
      llvm::Value* dest = llvm::ConstantFP::get(x->getType(), 0.0);
      llvm::Value* mask = llvm::ConstantInt::get(
          builder.getContext(), llvm::APInt(mask_bits, -1, true));
      y_approx =
          builder.CreateCall(rsqrt_intrinsic, {x, dest, mask}, "y_approx");

    } else {
      y_approx = builder.CreateCall(rsqrt_intrinsic, {x}, "y_approx");
    }
    return y_approx;
  }
};

llvm::Function* CreateRsqrtX86(llvm::Module* module, llvm::Type* input_type) {
  CHECK(input_type != nullptr);
  CHECK(input_type->isFloatingPointTy() || input_type->isVectorTy());
  CHECK(input_type->getScalarType()->isFloatTy() ||
        input_type->getScalarType()->isDoubleTy());

  llvm::LLVMContext& context = module->getContext();
  llvm::IRBuilder<> builder(context);

  int num_elements = 1;
  if (llvm::VectorType* vec_ty = llvm::dyn_cast<llvm::VectorType>(input_type)) {
    num_elements = vec_ty->getElementCount().getKnownMinValue();
  }

  llvm::FunctionType* function_type =
      llvm::FunctionType::get(input_type, {input_type}, false);
  llvm::Function* func = llvm::dyn_cast<llvm::Function>(
      module
          ->getOrInsertFunction(
              RsqrtFunctionName(num_elements, llvm_ir::PrimitiveTypeFromIrType(
                                                  input_type->getScalarType())),
              function_type)
          .getCallee());

  llvm::Argument* input_x_arg = func->getArg(0);
  input_x_arg->setName("x");
  llvm::BasicBlock* entry_bb = llvm::BasicBlock::Create(context, "entry", func);
  llvm::Value* x = input_x_arg;
  builder.SetInsertPoint(entry_bb);
  RsqrtIntrinsic rsqrt_intrinsic = input_type->getScalarType()->isFloatTy()
                                       ? RsqrtIntrinsic::ForF32(num_elements)
                                       : RsqrtIntrinsic::ForF64(num_elements);
  llvm::Value* y_approx = rsqrt_intrinsic.CreateCall(builder, x);

  llvm::Value* refined_result =
      NewtonRaphsonRsqrtIteration(builder, input_x_arg, y_approx, input_type);
  if (input_type->getScalarType()->isDoubleTy()) {
    // Do an additional refinement step for F64.
    refined_result = NewtonRaphsonRsqrtIteration(builder, input_x_arg,
                                                 refined_result, input_type);
  }

  // Create a mask for special cases (denormals and infinities) to fall back
  // to the intrinsic's result, matching Eigen's behavior.
  const llvm::fltSemantics& semantics =
      input_type->getScalarType()->getFltSemantics();
  llvm::Constant* flt_min = llvm::ConstantFP::get(
      input_type, llvm::APFloat::getSmallestNormalized(semantics));
  llvm::Constant* inf =
      llvm::ConstantFP::get(input_type, llvm::APFloat::getInf(semantics));

  llvm::Value* lt_min_mask = builder.CreateFCmpOLT(x, flt_min, "lt_min_mask");
  llvm::Value* inf_mask = builder.CreateFCmpOEQ(x, inf, "inf_mask");
  llvm::Value* use_hw_approx_mask =
      builder.CreateOr(lt_min_mask, inf_mask, "use_hw_approx_mask");

  // If input is normal and finite, use the refined result. Otherwise, use the
  // raw hardware approximation.
  llvm::Value* result = builder.CreateSelect(use_hw_approx_mask, y_approx,
                                             refined_result, "result");

  builder.CreateRet(result);
  return func;
}

}  // namespace xla::codegen::math
