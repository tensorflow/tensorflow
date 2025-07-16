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

#ifndef XLA_CODEGEN_MATH_LDEXP_H_
#define XLA_CODEGEN_MATH_LDEXP_H_

#include <cstddef>
#include <string>

#include "llvm/IR/Value.h"
#include "xla/codegen/math/intrinsic.h"

namespace xla::codegen {

class Intrinsic::Ldexp {
 public:
  static std::string Name(PrimitiveType type);
  static std::string Name(PrimitiveType type, size_t vector_width);
};

namespace math {

// Returns a fast bit-shifting f64 implementation of ldexp for F64 based on
// Eigen. Function is named `xla.ldexp.<vector_size>xf64`.
// Won't overflow even if 2^e doesn't fit in a double.
// N.B. A 7x faster implementation is available in Eigen that we could try
// in the future:
// https://gitlab.com/libeigen/eigen/-/blob/master/Eigen/src/Core/arch/Default/GenericPacketMathFunctions.h#L272
llvm::Function* CreateLdexpF64(llvm::Module* module, llvm::Type* vector_type);

}  // namespace math
}  // namespace xla::codegen

#endif  // XLA_CODEGEN_MATH_LDEXP_H_
