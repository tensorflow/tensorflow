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

  llvm::IRBuilder<> ir_builder(vector_tanh_body);
  llvm::FastMathFlags fast_math_flags;
  fast_math_flags.setFast();
  ir_builder.setFastMathFlags(fast_math_flags);

  VectorSupportLibrary vsl(F32, vector_width, &ir_builder, "tanh_f32");

  llvm::Value* input = &*vector_tanh_function->arg_begin();
  CHECK_EQ(input->getType(), vsl.vector_type());

  // This implements the same rational interpolant as implemented in Eigen3.
  llvm::Value* input_clamped = vsl.Clamp(input, /*low=*/-9.0, /*high=*/9.0);

  std::array<float, 7> numerator_coeffs{
      -2.76076847742355e-16f, 2.00018790482477e-13f, -8.60467152213735e-11f,
      5.12229709037114e-08f,  1.48572235717979e-05f, 6.37261928875436e-04f,
      4.89352455891786e-03f};

  std::array<float, 4> denominator_coeffs{
      1.19825839466702e-06f, 1.18534705686654e-04f, 2.26843463243900e-03f,
      4.89352518554385e-03f};

  llvm::Value* input_squared = vsl.Mul(input_clamped, input_clamped);
  llvm::Value* numerator = vsl.SplatFloat(numerator_coeffs[0]);
  for (int i = 1; i < numerator_coeffs.size(); i++) {
    numerator = vsl.MulAdd(input_squared, numerator, numerator_coeffs[i]);
  }

  numerator = vsl.Mul(input_clamped, numerator);

  llvm::Value* denominator = vsl.SplatFloat(denominator_coeffs[0]);
  for (int i = 1; i < denominator_coeffs.size(); i++) {
    denominator = vsl.MulAdd(input_squared, denominator, denominator_coeffs[i]);
  }

  llvm::Value* result = vsl.Div(numerator, denominator);
  ir_builder.CreateRet(result);

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

  llvm::IRBuilder<> ir_builder(vector_exp_body);
  llvm::FastMathFlags fast_math_flags;
  fast_math_flags.setFast();
  ir_builder.setFastMathFlags(fast_math_flags);

  VectorSupportLibrary vsl(F32, vector_width, &ir_builder, "exp_f32");

  // This implements the same polynomial approximation as implemented in Eigen3.

  const float exp_hi = 88.3762626647950;
  const float exp_lo = -88.3762626647949;

  const float cephes_LOG2EF = 1.44269504088896341;
  const float cephes_exp_C1 = 0.693359375;
  const float cephes_exp_C2 = -2.12194440e-4;

  const float cephes_exp_p0 = 1.9875691500E-4;
  const float cephes_exp_p1 = 1.3981999507E-3;
  const float cephes_exp_p2 = 8.3334519073E-3;
  const float cephes_exp_p3 = 4.1665795894E-2;
  const float cephes_exp_p4 = 1.6666665459E-1;
  const float cephes_exp_p5 = 5.0000001201E-1;

  llvm::Value* input = &*vector_exp_function->arg_begin();
  llvm::Value* input_clamped =
      vsl.Clamp(input, /*low=*/exp_lo, /*high=*/exp_hi);
  llvm::Value* fx = vsl.Floor(vsl.MulAdd(input_clamped, cephes_LOG2EF, 0.5));
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
  y = vsl.Add(1.0f, y);

  // VectorSupportLibrary (intentionally) can't juggle more than one type at a
  // time so drop down to IRBuilder for this bit.
  llvm::Value* vector_constant_0x7f =
      ir_builder.CreateVectorSplat(vector_width, ir_builder.getInt32(0x7f));
  llvm::Value* vector_constant_23 =
      ir_builder.CreateVectorSplat(vector_width, ir_builder.getInt32(23));
  llvm::Type* i32_vector_type =
      llvm::VectorType::get(ir_builder.getInt32Ty(), vector_width);
  // fx is clamped so we don't have to worry about it being out of range for
  // i32.
  llvm::Value* emm0 = ir_builder.CreateFPToSI(fx, i32_vector_type);
  emm0 = ir_builder.CreateAdd(emm0, vector_constant_0x7f);
  emm0 = ir_builder.CreateShl(emm0, vector_constant_23);
  llvm::Value* emm0_f32 = ir_builder.CreateBitCast(emm0, vsl.vector_type());

  llvm::Value* result = vsl.Max(vsl.Mul(y, emm0_f32), input);

  ir_builder.CreateRet(result);

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

  llvm::IRBuilder<> ir_builder(vector_log_body);
  llvm::FastMathFlags fast_math_flags;
  fast_math_flags.setFast();
  ir_builder.setFastMathFlags(fast_math_flags);

  llvm::Value* input = &*vector_log_function->arg_begin();
  VectorSupportLibrary vsl(F32, vector_width, &ir_builder, "log_f32");

  const float half = 0.5;

  // This implements the same polynomial approximation as implemented in Eigen3.
  // Returns NaN for x < 0, -INF for x = 0
  const float cephes_SQRTHF = 0.707106781186547524;
  const float cephes_log_p0 = 7.0376836292E-2;
  const float cephes_log_p1 = -1.1514610310E-1;
  const float cephes_log_p2 = 1.1676998740E-1;
  const float cephes_log_p3 = -1.2420140846E-1;
  const float cephes_log_p4 = +1.4249322787E-1;
  const float cephes_log_p5 = -1.6668057665E-1;
  const float cephes_log_p6 = +2.0000714765E-1;
  const float cephes_log_p7 = -2.4999993993E-1;
  const float cephes_log_p8 = +3.3333331174E-1;
  const float cephes_log_q1 = -2.12194440e-4;
  const float cephes_log_q2 = 0.693359375;

  // The smallest non denormalized float number.
  const float min_norm_pos = tensorflow::bit_cast<float, int32>(0x00800000);
  const float minus_inf = tensorflow::bit_cast<float, int32>(0xff800000);

  // NB! This number is denormal and since TF sets the denormals-are-zero flag
  // (and if TF didn't, -ffast-math would) trying to operate on this float using
  // C++ operations (including, for instance, implicit conversion to double)
  // will coerce this to zero.
  const float inv_mant_mask = tensorflow::bit_cast<float, int32>(~0x7f800000);

  // invalid_mask is set if x is negative or NaN (and therefore output
  // must be NaN).
  llvm::Value* invalid_mask = vsl.FCmpULEMask(input, vsl.GetZeroVector());
  llvm::Value* iszero_mask = vsl.FCmpEQMask(input, vsl.GetZeroVector());

  // Cut off denormalized stuff.
  input = vsl.Max(min_norm_pos, input);

  // VectorSupportLibrary (intentionally) can't juggle more than one type at a
  // time so drop down to IRBuilder for this bit.
  llvm::Value* vector_constant_0x7f =
      ir_builder.CreateVectorSplat(vector_width, ir_builder.getInt32(0x7f));
  llvm::Value* vector_constant_23 =
      ir_builder.CreateVectorSplat(vector_width, ir_builder.getInt32(23));
  llvm::Type* i32_vector_type =
      llvm::VectorType::get(ir_builder.getInt32Ty(), vector_width);

  llvm::Value* emm0 = ir_builder.CreateLShr(
      ir_builder.CreateBitCast(input, i32_vector_type), vector_constant_23);

  // Keep only the fractional part.
  input = vsl.FloatAnd(input, inv_mant_mask);
  input = vsl.FloatOr(input, half);

  emm0 = ir_builder.CreateSub(emm0, vector_constant_0x7f);
  llvm::Value* e =
      vsl.Add(1.0f, ir_builder.CreateSIToFP(emm0, vsl.vector_type()));

  // part2:
  //   if( x < SQRTHF ) {
  //     e -= 1;
  //     x = x + x - 1.0;
  //   } else { x = x - 1.0; }
  llvm::Value* mask = vsl.FCmpOLTMask(input, cephes_SQRTHF);
  llvm::Value* tmp = vsl.FloatAnd(input, mask);
  input = vsl.Sub(input, 1.0);
  e = vsl.Sub(e, vsl.FloatAnd(mask, 1.0));
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

  ir_builder.CreateRet(result);

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
