/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/cpu/llvm_ir_runtime.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace cpu {
namespace runtime {

const char* const kTanhV4F32SymbolName = "__xla_cpu_runtime_TanhV4F32";
const char* const kTanhV8F32SymbolName = "__xla_cpu_runtime_TanhV8F32";

namespace {
llvm::Value* EmitFMinOrMax(llvm::IRBuilder<>* ir_builder, llvm::Module* module,
                           llvm::Type* vector_type, llvm::Value* lhs,
                           llvm::Value* rhs, bool is_min,
                           bool enable_fast_math) {
  if (enable_fast_math) {
    // Using an unordered comparison lets LLVM generate a vminps / vmaxps
    // instruction on x86.  vminps/vmaxps choose the second operand if either
    // operand is a NaN and thus don't accurately implement the semantics of the
    // minnum and maxnum intrinsics, necessitating different IR emission.
    //
    // We can _probably_ do this even when fast math is disabled, but we can
    // certainly do this if fast math is enabled (and nnan applies).
    auto* compare = ir_builder->CreateFCmp(
        is_min ? llvm::FCmpInst::FCMP_ULE : llvm::FCmpInst::FCMP_UGE, lhs, rhs);
    return ir_builder->CreateSelect(compare, lhs, rhs);
  } else {
    llvm::Function* intrinsic = llvm::Intrinsic::getDeclaration(
        module, is_min ? llvm::Intrinsic::minnum : llvm::Intrinsic::maxnum,
        vector_type);
    return ir_builder->CreateCall(intrinsic, {lhs, rhs});
  }
}

llvm::Function* EmitVectorF32TanhIfNeeded(llvm::Module* module,
                                          llvm::StringRef function_name,
                                          int vector_width,
                                          bool enable_fast_math) {
  llvm::Function* vector_tanh_function = module->getFunction(function_name);
  if (vector_tanh_function == nullptr) {
    // If the function declaration is not present in the module, there can't be
    // any calls to resolve.  Don't emit the function in this case.
    return nullptr;
  }

  llvm::LLVMContext* context = &module->getContext();
  llvm::Type* float_type = llvm::Type::getFloatTy(*context);
  llvm::VectorType* vector_type =
      llvm::VectorType::get(float_type, vector_width);

  llvm::BasicBlock* vector_tanh_body =
      llvm::BasicBlock::Create(*context, "body", vector_tanh_function);

  llvm::IRBuilder<> ir_builder(vector_tanh_body);

  llvm::FastMathFlags fast_math_flags;
  fast_math_flags.setUnsafeAlgebra();
  ir_builder.setFastMathFlags(fast_math_flags);

  auto emit_fmin = [&](llvm::Value* lhs, llvm::Value* rhs) {
    return EmitFMinOrMax(&ir_builder, module, vector_type, lhs, rhs,
                         /*is_min=*/true,
                         /*enable_fast_math=*/enable_fast_math);
  };
  auto emit_fmax = [&](llvm::Value* lhs, llvm::Value* rhs) {
    return EmitFMinOrMax(&ir_builder, module, vector_type, lhs, rhs,
                         /*is_min=*/false,
                         /*enable_fast_math=*/enable_fast_math);
  };

  llvm::Value* input = &*vector_tanh_function->arg_begin();
  CHECK_EQ(input->getType(), vector_type);

  // This implements the same rational interpolant as implemented in Eigen3.
  llvm::Value* input_clamped =
      emit_fmin(emit_fmax(input, llvm::ConstantFP::get(vector_type, -9.0)),
                llvm::ConstantFP::get(vector_type, 9.0));

  std::array<float, 7> numerator_coeffs(
      {{-2.76076847742355e-16f, 2.00018790482477e-13f, -8.60467152213735e-11f,
        5.12229709037114e-08f, 1.48572235717979e-05f, 6.37261928875436e-04f,
        4.89352455891786e-03f}});

  std::array<float, 4> denominator_coeffs(
      {{1.19825839466702e-06f, 1.18534705686654e-04f, 2.26843463243900e-03f,
        4.89352518554385e-03f}});

  llvm::Value* input_squared =
      ir_builder.CreateFMul(input_clamped, input_clamped);
  llvm::Value* numerator =
      llvm::ConstantFP::get(vector_type, numerator_coeffs[0]);
  for (int i = 1; i < numerator_coeffs.size(); i++) {
    numerator = ir_builder.CreateFAdd(
        ir_builder.CreateFMul(input_squared, numerator),
        llvm::ConstantFP::get(vector_type, numerator_coeffs[i]));
  }
  numerator = ir_builder.CreateFMul(input_clamped, numerator);

  llvm::Value* denominator =
      llvm::ConstantFP::get(vector_type, denominator_coeffs[0]);
  for (int i = 1; i < denominator_coeffs.size(); i++) {
    denominator = ir_builder.CreateFAdd(
        ir_builder.CreateFMul(input_squared, denominator),
        llvm::ConstantFP::get(vector_type, denominator_coeffs[i]));
  }

  llvm::Value* result = ir_builder.CreateFDiv(numerator, denominator);
  ir_builder.CreateRet(result);

  DCHECK(!llvm::verifyFunction(*vector_tanh_function));
  return vector_tanh_function;
}
}  // namespace

void RewriteIRRuntimeFunctions(llvm::Module* module, bool enable_fast_math) {
  auto* tanh_v4f32 =
      EmitVectorF32TanhIfNeeded(module, kTanhV4F32SymbolName,
                                /*vector_width=*/4, enable_fast_math);
  auto* tanh_v8f32 =
      EmitVectorF32TanhIfNeeded(module, kTanhV8F32SymbolName,
                                /*vector_width=*/8, enable_fast_math);

  // Gather all the call sites, force inline them and then delete the vector
  // function bodies.

  std::vector<llvm::CallInst*> calls_to_inline;
  for (auto* function : {tanh_v4f32, tanh_v8f32}) {
    if (function != nullptr) {
      for (auto* user : function->users()) {
        calls_to_inline.push_back(llvm::cast<llvm::CallInst>(user));
      }
    }
  }

  for (auto* call_to_inline : calls_to_inline) {
    llvm::InlineFunctionInfo inline_function_info;
    CHECK(llvm::InlineFunction(call_to_inline, inline_function_info));
  }

  for (auto* function : {tanh_v4f32, tanh_v8f32}) {
    if (function != nullptr) {
      function->eraseFromParent();
    }
  }
}

}  // namespace runtime
}  // namespace cpu
}  // namespace xla
