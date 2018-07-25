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
#include "tensorflow/compiler/xla/service/cpu/vector_support_library.h"
#include "tensorflow/compiler/xla/service/llvm_ir/math_ops.h"
#include "tensorflow/core/lib/core/casts.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace cpu {
namespace runtime {

const char* const kTanhV4F32SymbolName = "__xla_cpu_runtime_TanhV4F32";
const char* const kTanhV8F32SymbolName = "__xla_cpu_runtime_TanhV8F32";
const char* const kExpV4F32SymbolName = "__xla_cpu_runtime_ExpV4F32";
const char* const kExpV8F32SymbolName = "__xla_cpu_runtime_ExpV8F32";
const char* const kLogV4F32SymbolName = "__xla_cpu_runtime_LogV4F32AVX";
const char* const kLogV8F32SymbolName = "__xla_cpu_runtime_LogV8F32AVX";

namespace {
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

  llvm::BasicBlock* vector_tanh_body =
      llvm::BasicBlock::Create(*context, "body", vector_tanh_function);

  llvm::IRBuilder<> b(vector_tanh_body);
  llvm::FastMathFlags fast_math_flags;
  fast_math_flags.setFast(enable_fast_math);
  b.setFastMathFlags(fast_math_flags);

  llvm::Value* input = &*vector_tanh_function->arg_begin();
  CHECK_EQ(vector_width, input->getType()->getVectorNumElements());
  b.CreateRet(llvm_ir::EmitFastTanh(&b, input));

  DCHECK(!llvm::verifyFunction(*vector_tanh_function));
  return vector_tanh_function;
}

llvm::Function* EmitVectorF32ExpIfNeeded(llvm::Module* module,
                                         llvm::StringRef function_name,
                                         int vector_width,
                                         bool enable_fast_math) {
  llvm::Function* vector_exp_function = module->getFunction(function_name);
  if (vector_exp_function == nullptr) {
    // If the function declaration is not present in the module, there can't be
    // any calls to resolve.  Don't emit the function in this case.
    return nullptr;
  }

  llvm::LLVMContext* context = &module->getContext();

  llvm::BasicBlock* vector_exp_body =
      llvm::BasicBlock::Create(*context, "body", vector_exp_function);

  llvm::IRBuilder<> b(vector_exp_body);
  llvm::FastMathFlags fast_math_flags;
  fast_math_flags.setFast();
  b.setFastMathFlags(fast_math_flags);

  VectorSupportLibrary vsl(F32, vector_width, &b, "exp_f32");

  // This implements the same polynomial approximation as implemented in Eigen3.

  const llvm::APFloat half = GetIeeeF32(0.5);
  const llvm::APFloat one = GetIeeeF32(1.0);

  const llvm::APFloat exp_hi = GetIeeeF32(88.3762626647950);
  const llvm::APFloat exp_lo = GetIeeeF32(-88.3762626647949);

  const llvm::APFloat cephes_LOG2EF = GetIeeeF32(1.44269504088896341);
  const llvm::APFloat cephes_exp_C1 = GetIeeeF32(0.693359375);
  const llvm::APFloat cephes_exp_C2 = GetIeeeF32(-2.12194440e-4);

  const llvm::APFloat cephes_exp_p0 = GetIeeeF32(1.9875691500E-4);
  const llvm::APFloat cephes_exp_p1 = GetIeeeF32(1.3981999507E-3);
  const llvm::APFloat cephes_exp_p2 = GetIeeeF32(8.3334519073E-3);
  const llvm::APFloat cephes_exp_p3 = GetIeeeF32(4.1665795894E-2);
  const llvm::APFloat cephes_exp_p4 = GetIeeeF32(1.6666665459E-1);
  const llvm::APFloat cephes_exp_p5 = GetIeeeF32(5.0000001201E-1);

  llvm::Value* input = &*vector_exp_function->arg_begin();
  llvm::Value* input_clamped =
      vsl.Clamp(input, /*low=*/exp_lo, /*high=*/exp_hi);
  llvm::Value* fx = vsl.Floor(vsl.MulAdd(input_clamped, cephes_LOG2EF, half));
  llvm::Value* tmp = vsl.Mul(cephes_exp_C1, fx);
  llvm::Value* z = vsl.Mul(cephes_exp_C2, fx);
  llvm::Value* x = vsl.Sub(input_clamped, tmp);
  x = vsl.Sub(x, z);
  z = vsl.Mul(x, x);

  llvm::Value* y = vsl.MulAdd(x, cephes_exp_p0, cephes_exp_p1);
  y = vsl.MulAdd(y, x, cephes_exp_p2);
  y = vsl.MulAdd(y, x, cephes_exp_p3);
  y = vsl.MulAdd(y, x, cephes_exp_p4);
  y = vsl.MulAdd(y, x, cephes_exp_p5);
  y = vsl.MulAdd(y, z, x);
  y = vsl.Add(one, y);

  // VectorSupportLibrary (intentionally) can't juggle more than one type at a
  // time so drop down to IRBuilder for this bit.
  llvm::Value* vector_constant_0x7f =
      b.CreateVectorSplat(vector_width, b.getInt32(0x7f));
  llvm::Value* vector_constant_23 =
      b.CreateVectorSplat(vector_width, b.getInt32(23));
  llvm::Type* i32_vector_type =
      llvm::VectorType::get(b.getInt32Ty(), vector_width);
  // fx is clamped so we don't have to worry about it being out of range for
  // i32.
  llvm::Value* emm0 = b.CreateFPToSI(fx, i32_vector_type);
  emm0 = b.CreateAdd(emm0, vector_constant_0x7f);
  emm0 = b.CreateShl(emm0, vector_constant_23);
  llvm::Value* emm0_f32 = b.CreateBitCast(emm0, vsl.vector_type());

  llvm::Value* result = vsl.Max(vsl.Mul(y, emm0_f32), input);

  b.CreateRet(result);

  DCHECK(!llvm::verifyFunction(*vector_exp_function));
  return vector_exp_function;
}

llvm::Function* EmitVectorF32LogIfNeeded(llvm::Module* module,
                                         llvm::StringRef function_name,
                                         int vector_width,
                                         bool enable_fast_math) {
  llvm::Function* vector_log_function = module->getFunction(function_name);
  if (vector_log_function == nullptr) {
    // If the function declaration is not present in the module, there can't be
    // any calls to resolve.  Don't emit the function in this case.
    return nullptr;
  }

  llvm::LLVMContext* context = &module->getContext();

  llvm::BasicBlock* vector_log_body =
      llvm::BasicBlock::Create(*context, "body", vector_log_function);

  llvm::IRBuilder<> b(vector_log_body);
  llvm::FastMathFlags fast_math_flags;
  fast_math_flags.setFast();
  b.setFastMathFlags(fast_math_flags);

  llvm::Value* input = &*vector_log_function->arg_begin();
  VectorSupportLibrary vsl(F32, vector_width, &b, "log_f32");

  const llvm::APFloat half = GetIeeeF32(0.5);
  const llvm::APFloat one = GetIeeeF32(1.0);

  // This implements the same polynomial approximation as implemented in Eigen3.
  // Returns NaN for x < 0, -INF for x = 0
  const llvm::APFloat cephes_SQRTHF = GetIeeeF32(0.707106781186547524);
  const llvm::APFloat cephes_log_p0 = GetIeeeF32(7.0376836292E-2);
  const llvm::APFloat cephes_log_p1 = GetIeeeF32(-1.1514610310E-1);
  const llvm::APFloat cephes_log_p2 = GetIeeeF32(1.1676998740E-1);
  const llvm::APFloat cephes_log_p3 = GetIeeeF32(-1.2420140846E-1);
  const llvm::APFloat cephes_log_p4 = GetIeeeF32(+1.4249322787E-1);
  const llvm::APFloat cephes_log_p5 = GetIeeeF32(-1.6668057665E-1);
  const llvm::APFloat cephes_log_p6 = GetIeeeF32(+2.0000714765E-1);
  const llvm::APFloat cephes_log_p7 = GetIeeeF32(-2.4999993993E-1);
  const llvm::APFloat cephes_log_p8 = GetIeeeF32(+3.3333331174E-1);
  const llvm::APFloat cephes_log_q1 = GetIeeeF32(-2.12194440e-4);
  const llvm::APFloat cephes_log_q2 = GetIeeeF32(0.693359375);

  // The smallest non denormalized float number.
  const llvm::APFloat min_norm_pos = GetIeeeF32FromBitwiseRep(0x00800000);
  const llvm::APFloat minus_inf = GetIeeeF32FromBitwiseRep(0xff800000);
  const llvm::APFloat inv_mant_mask = GetIeeeF32FromBitwiseRep(~0x7f800000);

  // invalid_mask is set if x is negative or NaN (and therefore output
  // must be NaN).
  llvm::Value* invalid_mask = vsl.FCmpULEMask(input, vsl.GetZeroVector());
  llvm::Value* iszero_mask = vsl.FCmpEQMask(input, vsl.GetZeroVector());

  // Cut off denormalized stuff.
  input = vsl.Max(min_norm_pos, input);

  // VectorSupportLibrary (intentionally) can't juggle more than one type at a
  // time so drop down to IRBuilder for this bit.
  llvm::Value* vector_constant_0x7f =
      b.CreateVectorSplat(vector_width, b.getInt32(0x7f));
  llvm::Value* vector_constant_23 =
      b.CreateVectorSplat(vector_width, b.getInt32(23));
  llvm::Type* i32_vector_type =
      llvm::VectorType::get(b.getInt32Ty(), vector_width);

  llvm::Value* emm0 =
      b.CreateLShr(b.CreateBitCast(input, i32_vector_type), vector_constant_23);

  // Keep only the fractional part.
  input = vsl.FloatAnd(input, inv_mant_mask);
  input = vsl.FloatOr(input, half);

  emm0 = b.CreateSub(emm0, vector_constant_0x7f);
  llvm::Value* e = vsl.Add(one, b.CreateSIToFP(emm0, vsl.vector_type()));

  // part2:
  //   if( x < SQRTHF ) {
  //     e -= 1;
  //     x = x + x - 1.0;
  //   } else { x = x - 1.0; }
  llvm::Value* mask = vsl.FCmpOLTMask(input, cephes_SQRTHF);
  llvm::Value* tmp = vsl.FloatAnd(input, mask);
  input = vsl.Sub(input, one);
  e = vsl.Sub(e, vsl.FloatAnd(mask, one));
  input = vsl.Add(input, tmp);

  llvm::Value* x2 = vsl.Mul(input, input);
  llvm::Value* x3 = vsl.Mul(x2, input);

  llvm::Value *y, *y1, *y2;
  y = vsl.MulAdd(input, cephes_log_p0, cephes_log_p1);
  y1 = vsl.MulAdd(input, cephes_log_p3, cephes_log_p4);
  y2 = vsl.MulAdd(input, cephes_log_p6, cephes_log_p7);
  y = vsl.MulAdd(y, input, cephes_log_p2);
  y1 = vsl.MulAdd(y1, input, cephes_log_p5);
  y2 = vsl.MulAdd(y2, input, cephes_log_p8);
  y = vsl.MulAdd(y, x3, y1);
  y = vsl.MulAdd(y, x3, y2);
  y = vsl.Mul(y, x3);

  y1 = vsl.Mul(cephes_log_q1, e);
  tmp = vsl.Mul(half, x2);
  y = vsl.Add(y, y1);
  input = vsl.Sub(input, tmp);
  y2 = vsl.Mul(cephes_log_q2, e);
  input = vsl.Add(input, y);
  input = vsl.Add(input, y2);

  // Negative arg will be NAN, 0 will be -INF.
  llvm::Value* or_lhs =
      vsl.FloatAndNot(iszero_mask, vsl.FloatOr(input, invalid_mask));
  llvm::Value* or_rhs = vsl.FloatAnd(iszero_mask, minus_inf);
  llvm::Value* result = vsl.FloatOr(or_lhs, or_rhs);

  b.CreateRet(result);

  DCHECK(!llvm::verifyFunction(*vector_log_function));
  return vector_log_function;
}
}  // namespace

void RewriteIRRuntimeFunctions(llvm::Module* module, bool enable_fast_math) {
  auto* tanh_v4f32 =
      EmitVectorF32TanhIfNeeded(module, kTanhV4F32SymbolName,
                                /*vector_width=*/4, enable_fast_math);
  auto* tanh_v8f32 =
      EmitVectorF32TanhIfNeeded(module, kTanhV8F32SymbolName,
                                /*vector_width=*/8, enable_fast_math);

  auto* exp_v4f32 =
      EmitVectorF32ExpIfNeeded(module, kExpV4F32SymbolName,
                               /*vector_width=*/4, enable_fast_math);
  auto* exp_v8f32 =
      EmitVectorF32ExpIfNeeded(module, kExpV8F32SymbolName,
                               /*vector_width=*/8, enable_fast_math);

  auto* log_v4f32 =
      EmitVectorF32LogIfNeeded(module, kLogV4F32SymbolName,
                               /*vector_width=*/4, enable_fast_math);
  auto* log_v8f32 =
      EmitVectorF32LogIfNeeded(module, kLogV8F32SymbolName,
                               /*vector_width=*/8, enable_fast_math);

  // Gather all the call sites, force inline them and then delete the vector
  // function bodies.
  //
  // TODO(b/73081976): Should we avoid inlining these intrinsics in some cases?

  std::vector<llvm::CallInst*> calls_to_inline;
  for (auto* function :
       {tanh_v4f32, tanh_v8f32, exp_v4f32, exp_v8f32, log_v4f32, log_v8f32}) {
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

  for (auto* function :
       {tanh_v4f32, tanh_v8f32, exp_v4f32, exp_v8f32, log_v4f32, log_v8f32}) {
    if (function != nullptr) {
      function->eraseFromParent();
    }
  }
}

}  // namespace runtime
}  // namespace cpu
}  // namespace xla
