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

#include "xla/codegen/intrinsic/rsqrt.h"

#include <cstddef>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
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
#include "llvm/Target/TargetMachine.h"
#include "xla/codegen/intrinsic/intrinsic.h"
#include "xla/xla_data.pb.h"

namespace xla::codegen::intrinsics {
namespace {
llvm::Value* PMAdd(llvm::IRBuilder<>& builder, llvm::Value* x, llvm::Value* y,
                   llvm::Value* z) {
  return builder.CreateFAdd(builder.CreateFMul(x, y), z);
}
}  // namespace

static llvm::Value* NewtonRaphsonRsqrtIteration(llvm::IRBuilder<>& builder,
                                                llvm::Value* x,
                                                llvm::Value* guess,
                                                llvm::Type* type, int steps) {
  // Based on https://libeigen.gitlab.io/docs/MathFunctionsImpl_8h_source.html
  llvm::Value* minus_half = llvm::ConstantFP::get(type, -0.5);
  llvm::Value* minus_one = llvm::ConstantFP::get(type, -1.0);
  llvm::Value* inv_sqrt = guess;
  for (int step = 0; step < steps; ++step) {
    // Refine the guess using one Newton-Raphson step.
    // h_n = (x * inv_sqrt) * inv_sqrt - 1 (so that h_n is nearly 0).
    // inv_sqrt = inv_sqrt - 0.5 * inv_sqrt * h_n
    llvm::Value* r2 = builder.CreateFMul(x, inv_sqrt);
    llvm::Value* half_r = builder.CreateFMul(inv_sqrt, minus_half);
    llvm::Value* h_n = PMAdd(builder, r2, inv_sqrt, minus_one);
    inv_sqrt = PMAdd(builder, half_r, h_n, inv_sqrt);
  }
  return inv_sqrt;
}

struct RsqrtIntrinsic {
  llvm::Intrinsic::ID id;
  int mask_bits;                  // Some avx512 calls require masks.
  int needs_insert_element_size;  // Some avx512 calls require padding.

  static RsqrtIntrinsic ForF32(size_t num_elements) {
    switch (num_elements) {
      case 1:
        return {llvm::Intrinsic::x86_sse_rsqrt_ss, 0, 4};
      case 4:
        return {llvm::Intrinsic::x86_sse_rsqrt_ps, 0, 0};
      case 8:
        return {llvm::Intrinsic::x86_avx_rsqrt_ps_256, 0, 0};
      case 16:
        return {llvm::Intrinsic::x86_avx512_rsqrt14_ps_512, 16, 0};
      default:
        LOG(FATAL) << "Unsupported vector width for rsqrt: " << num_elements;
    }
  }

  static RsqrtIntrinsic ForF64(size_t num_elements) {
    // We assume AVX512 is available for F64.
    switch (num_elements) {
      case 1:
        // Assuming AVX512 is available.
        // We don't use x86_avx512_rsqrt14_sd because it also requires padding
        // into <2 x double> vectors and it takes an additional source vector
        // for the upper bits of the result.
        return {llvm::Intrinsic::x86_avx512_rsqrt14_pd_128, 8, 2};
      case 2:
        return {llvm::Intrinsic::x86_avx512_rsqrt14_pd_128, 8, 0};
      case 4:
        return {llvm::Intrinsic::x86_avx512_rsqrt14_pd_256, 8, 0};
      case 8:
        return {llvm::Intrinsic::x86_avx512_rsqrt14_pd_512, 8, 0};
      default:
        LOG(FATAL) << "Unsupported vector width for rsqrt: " << num_elements;
    }
  }

  llvm::Value* CreateCall(llvm::IRBuilder<>& builder, llvm::Value* x) {
    llvm::Module* module = builder.GetInsertBlock()->getModule();
    llvm::Function* rsqrt_intrinsic =
        llvm::Intrinsic::getOrInsertDeclaration(module, id);

    llvm::Value* y_approx;
    std::vector<llvm::Value*> args = {x};
    if (needs_insert_element_size > 0) {
      // Pad into a vector of size `needs_insert_element_size`.
      llvm::Type* sse_vec_type = llvm::VectorType::get(
          x->getType()->getScalarType(),
          llvm::ElementCount::getFixed(needs_insert_element_size));
      llvm::Value* vec_x = llvm::UndefValue::get(sse_vec_type);
      vec_x = builder.CreateInsertElement(vec_x, x, builder.getInt32(0));
      args[0] = vec_x;
    }
    if (mask_bits > 0) {
      llvm::Value* src = llvm::ConstantFP::get(args[0]->getType(), 0.0);
      llvm::Value* mask = llvm::ConstantInt::get(
          builder.getContext(), llvm::APInt(mask_bits, -1, true));
      args.push_back(src);
      args.push_back(mask);
    }
    y_approx = builder.CreateCall(rsqrt_intrinsic, args, "y_approx");
    if (needs_insert_element_size > 0) {
      // Extract the result from the padded vector.
      y_approx = builder.CreateExtractElement(y_approx, builder.getInt32(0),
                                              "y_approx");
    }
    return y_approx;
  }
};

absl::StatusOr<llvm::Function*> Rsqrt::CreateDefinition(
    llvm::Module* module, const IntrinsicOptions& options, Type type) {
  CHECK(type.element_type() == F64 || type.element_type() == F32)
      << type.name();
  llvm::Type* input_type = Type::TypeToIrType(type, module->getContext());
  CHECK(input_type != nullptr);

  llvm::LLVMContext& context = module->getContext();
  llvm::IRBuilder<> builder(context);

  int num_elements = 1;
  if (llvm::VectorType* vec_ty = llvm::dyn_cast<llvm::VectorType>(input_type)) {
    num_elements = vec_ty->getElementCount().getKnownMinValue();
  }

  llvm::FunctionType* function_type =
      llvm::FunctionType::get(input_type, {input_type}, false);
  llvm::Function* func = llvm::dyn_cast<llvm::Function>(
      module->getOrInsertFunction(Rsqrt::Name(type), function_type)
          .getCallee());

  llvm::Argument* x = func->getArg(0);
  x->setName("x");
  llvm::BasicBlock* entry_bb = llvm::BasicBlock::Create(context, "entry", func);
  builder.SetInsertPoint(entry_bb);

  // Rsqrt is not portable across all CPUs, so we fall back to 1 / sqrt(x) if
  // 1. The user explicitly requested it, or
  // 2. The target CPU does not support AVX512F for F64, or
  // 3. The target CPU does not support AVX for F32.
  if (options.disable_platform_dependent_math ||
      (type.element_type() == F64 && !options.Contains("+avx512f")) ||
      !options.Contains("+avx")) {
    LOG_EVERY_N(INFO, 1000) << "Falling back to 1 / sqrt(x) for " << type.name()
                            << " " << options.disable_platform_dependent_math;
    // We can't use the same approximation algorithm for F64 without AVX512 or
    // anything non-x86 and without avx.
    llvm::Value* one = llvm::ConstantFP::get(input_type, 1.0);
    llvm::Value* sqrt_x =
        builder.CreateUnaryIntrinsic(llvm::Intrinsic::sqrt, x);
    llvm::Value* inv_sqrt_x = builder.CreateFDiv(one, sqrt_x, "inv_sqrt_x");
    builder.CreateRet(inv_sqrt_x);
    return func;
  }

  RsqrtIntrinsic rsqrt_intrinsic = input_type->getScalarType()->isFloatTy()
                                       ? RsqrtIntrinsic::ForF32(num_elements)
                                       : RsqrtIntrinsic::ForF64(num_elements);
  llvm::Value* y_approx = rsqrt_intrinsic.CreateCall(builder, x);

  // Eigen only does 1 step for F32, but that only gives within 2 ULPs and we
  // are targeting 1. AMD's SSE/AVX rsqrt intrinsics are more accurate; their
  // avx512f intrinsics have the same accuracy as Intel's avx512f intrinsics.
  const bool using_avx512 =
      options.Contains("+avx512f") &&
      (type.element_type() == F64 ||
       (type.element_type() == F32 && type.vector_width().value_or(1) > 8));
  const bool is_amd = options.device_type == DeviceType::kAmdCpu;
  const size_t num_steps = (is_amd && !using_avx512) ? 1 : 2;
  llvm::Value* refined_result =
      NewtonRaphsonRsqrtIteration(builder, x, y_approx, input_type, num_steps);

  const llvm::fltSemantics& semantics =
      input_type->getScalarType()->getFltSemantics();
  llvm::APFloat flt_min_val = llvm::APFloat::getSmallestNormalized(semantics);
  llvm::Constant* flt_min = llvm::ConstantFP::get(input_type, flt_min_val);

  llvm::Constant* inf =
      llvm::ConstantFP::get(input_type, llvm::APFloat::getInf(semantics));

  llvm::Value* lt_min_mask = builder.CreateFCmpOLT(x, flt_min, "lt_min_mask");
  llvm::Value* inf_mask = builder.CreateFCmpOEQ(x, inf, "inf_mask");
  llvm::Value* use_hw_approx_mask =
      builder.CreateOr(lt_min_mask, inf_mask, "use_hw_approx_mask");

  llvm::Value* result = builder.CreateSelect(use_hw_approx_mask, y_approx,
                                             refined_result, "result");

  // Hardware rsqrt may flush negative subnormals to -0/+0, returning +-inf
  // instead of NaN. Force NaN for any strictly negative input (x < -0.0).
  llvm::Constant* neg_zero = llvm::ConstantFP::get(input_type, -0.0);
  llvm::Value* neg_mask = builder.CreateFCmpOLT(x, neg_zero, "neg_mask");
  llvm::Constant* nan_val =
      llvm::ConstantFP::get(input_type, llvm::APFloat::getNaN(semantics));
  result = builder.CreateSelect(neg_mask, nan_val, result, "result_nan_fixup");

  builder.CreateRet(result);
  return func;
}

}  // namespace xla::codegen::intrinsics
