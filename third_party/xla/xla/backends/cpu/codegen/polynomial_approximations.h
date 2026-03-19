/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_CODEGEN_POLYNOMIAL_APPROXIMATIONS_H_
#define XLA_BACKENDS_CPU_CODEGEN_POLYNOMIAL_APPROXIMATIONS_H_

#include <vector>

#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Module.h"
#include "xla/service/llvm_ir/llvm_util.h"

namespace xla::cpu {

// A library of XLA:CPU polynomial approximations for common math functions.
//
// By default LLVM compiles math functions (tanh, exp, etc.) as library calls
// to the `libm` library, and library calls inside XLA loops have unacceptable
// performance overheads (it's an external function call). In XLA:CPU we rewrite
// these library calls into the LLVM IR using polynomial approximations (often
// borrowed from Eigen), and as a result we get loops without any library calls
// and rely on LLVM to optimize loop bodies into efficient machine code.
//
// MLIR framework has a set of rewrite patterns that do similar rewrites on MLIR
// representation (see `PolynomialApproximation.cpp`). In contrast to MLIR
// approach these approximations applied late in the compilation pipeline, by
// invoking the rewrite explicitly from our custom `IRCompiler` layer which in
// plugged into our own JIT compiler built on top of LLVM ORC APIs.
//
// See: https://en.wikipedia.org/wiki/C_mathematical_functions

// Returns a vector of vectorization information for functions that have
// vectorized polynomial approximations. This enables LLVM vectorization passes
// to vectorize scalar math functions to custom function calls, that we later
// rewrite into LLVM IR, so we don't have any function calls in compiled code.
std::vector<llvm::VecDesc> PolynomialApproximationsVectorization();

// Rewrites supported math functions into LLVM IR polynomial approximations.
//
// This function rewrites function calls to the builtin LLVM math functions
// intrinsics (i.e. `llvm.tanh.f32`) and custom XLA:CPU vectorized math
// functions (see `PolynomialApproximationsVectorization` above) into lower
// level polynomial approximations LLVM IR.
void RewriteToPolynomialApproximations(llvm::Module* module,
                                       llvm::FastMathFlags fast_math_flags);

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_CODEGEN_POLYNOMIAL_APPROXIMATIONS_H_
