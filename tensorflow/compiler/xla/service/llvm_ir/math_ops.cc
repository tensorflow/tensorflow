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

  auto cutoff_upper = llvm::ConstantFP::get(type, 15.6437711715698242f);
  //auto x = llvm_ir::EmitFloatMin(input, cutoff_upper, b);
  // Clamp the input to [-9, 9].
  //
  // To simplify the code base until it's an issue, don't have a slow min/max in
  // this approximation.
  llvm::Value* x = llvm_ir::EmitFloatMin(
      llvm_ir::EmitFloatMax(input, llvm::ConstantFP::get(type, -9.0), b,
                            /*enable_fast_min_max=*/true),
      llvm::ConstantFP::get(type, 9.0), b, /*enable_fast_min_max=*/true);



  auto alpha_1 = llvm::ConstantFP::get(type, 2.48287947061529e-01f);
  auto alpha_3 = llvm::ConstantFP::get(type, 8.51377133304701e-03f);
  auto alpha_5 = llvm::ConstantFP::get(type, 6.08574864600143e-05f);
  auto alpha_7 = llvm::ConstantFP::get(type, 1.15627324459942e-07f);
  auto alpha_9 = llvm::ConstantFP::get(type, 4.37031012579801e-11f);

  auto beta_0  = llvm::ConstantFP::get(type, 9.93151921023180e-01f);
  auto beta_2  = llvm::ConstantFP::get(type, 1.16817656904453e-01f);
  auto beta_4  = llvm::ConstantFP::get(type, 1.70198817374094e-03f);
  auto beta_6  = llvm::ConstantFP::get(type, 6.29106785017040e-06f);
  auto beta_8  = llvm::ConstantFP::get(type, 5.76102136993427e-09f);
  auto beta_10 = llvm::ConstantFP::get(type, 6.10247389755681e-13f);

  llvm::Value* x2 = b->CreateFMul(x, x);
  auto p = b->CreateFAdd(b->CreateFMul(x2, alpha_9), alpha_7);
  p = b->CreateFAdd(b->CreateFMul(x2, p), alpha_5);
  p = b->CreateFAdd(b->CreateFMul(x2, p), alpha_3);
  p = b->CreateFAdd(b->CreateFMul(x2, p), alpha_1);
  p = b->CreateFMul(x, p);

  auto q = b->CreateFAdd(b->CreateFMul(x2, beta_10), beta_8);
  q = b->CreateFAdd(b->CreateFMul(x2, q), beta_6);
  q = b->CreateFAdd(b->CreateFMul(x2, q), beta_4);
  q = b->CreateFAdd(b->CreateFMul(x2, q), beta_2);
  q = b->CreateFAdd(b->CreateFMul(x2, q), beta_0);

  auto ret = b->CreateFAdd(b->CreateFDiv(p,q), llvm::ConstantFP::get(type, 0.5f));

  return b->CreateSelect(use_aprox, input, ret);

}

}  // namespace llvm_ir
}  // namespace xla
