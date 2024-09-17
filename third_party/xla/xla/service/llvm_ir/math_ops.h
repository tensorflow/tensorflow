/* Copyright 2018 The OpenXLA Authors.

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

// Emits an approximation of tanh. The implementation uses the same rational
// interpolant as implemented in Eigen3. 'with_fma' should be set to true if FMA
// instructions are available.
llvm::Value* EmitFastTanh(llvm::IRBuilder<>* b, llvm::Value* input,
                          bool with_fma = false);
llvm::Value* EmitFastTanhF64(llvm::IRBuilder<>* b, llvm::Value* input,
                             bool with_fma = false);

// Emits an approximation of erf. The implementation uses the same rational
// interpolant as implemented in Eigen3.
llvm::Value* EmitErfF32(llvm::IRBuilder<>* b, llvm::Value* x);

}  // namespace llvm_ir
}  // namespace xla

#endif  // XLA_SERVICE_LLVM_IR_MATH_OPS_H_
