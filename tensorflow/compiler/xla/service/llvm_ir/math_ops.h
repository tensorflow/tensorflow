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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_MATH_OPS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_MATH_OPS_H_

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"

namespace xla {
namespace llvm_ir {

// Emits an approximation of tanh. The implementation uses the same rational
// interpolant as implemented in Eigen3.
llvm::Value* EmitFastTanh(llvm::IRBuilder<>* b, llvm::Value* input);

}  // namespace llvm_ir
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_MATH_OPS_H_
