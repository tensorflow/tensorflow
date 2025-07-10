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

#ifndef XLA_CODEGEN_MATH_RSQRT_H_
#define XLA_CODEGEN_MATH_RSQRT_H_

#include <cstddef>
#include <string>

#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "xla/xla_data.pb.h"

namespace xla::codegen::math {

// Returns the function name for the reciprocal square root function based on
// the type and number of elements.
std::string RsqrtFunctionName(size_t num_elements, PrimitiveType type);

// Creates an LLVM function that computes the reciprocal square root (1/sqrt(x))
// with high precision (within 1 ULP).
// Uses the hardware rsqrt intrinsic as initial guess followed by some
// Newton-Raphson iterations.
// Based on Eigen's implementations, with some modifications:
// https://eigen.tuxfamily.org/dox-devel/arch_2AVX512_2MathFunctions_8h_source.html
// Assumes AVX512 is available for F64 and <16 x float> inputs.
llvm::Function* CreateRsqrtX86(llvm::Module* module, llvm::Type* input_type);

}  // namespace xla::codegen::math

#endif  // XLA_CODEGEN_MATH_RSQRT_H_
