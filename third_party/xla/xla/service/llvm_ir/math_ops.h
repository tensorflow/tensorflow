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

#ifndef XLA_SERVICE_LLVM_IR_MATH_OPS_H_
#define XLA_SERVICE_LLVM_IR_MATH_OPS_H_

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"

namespace xla {
namespace llvm_ir {

enum TanhType {
  Double = 0,
  Float = 1,
};

static constexpr float kTanhInputUpperBounder = 9.0;
static constexpr float kTanhInputLowerBounder = -9.0;

// Inputs in the range [kTanhInputUpperBounderFloat, 9.0] may cause the output
// of EmitFastTanh to be greater than 1, so we set the input to be less than
// kUpperBounderFloat. 7.90531110763549805f by eigen float
// tanh(Eigen/src/Core/MathFunctionsImpl.h).
// We select 7.90531110763549805f because `EmitFastTanh` on GPU don't use FMA .
static constexpr float kTanhInputUpperBounderFloat = 7.90531110763549805f;
static constexpr float kTanhInputLowerBounderFloat = -7.90531110763549805f;

// Emits an approximation of tanh. The implementation uses the same rational
// interpolant as implemented in Eigen3.
llvm::Value* EmitFastTanh(llvm::IRBuilder<>* b, llvm::Value* input,
                          TanhType input_type = TanhType::Double);

}  // namespace llvm_ir
}  // namespace xla

#endif  // XLA_SERVICE_LLVM_IR_MATH_OPS_H_
