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

#ifndef XLA_CODEGEN_MATH_FPTRUNC_H_
#define XLA_CODEGEN_MATH_FPTRUNC_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/strings/str_cat.h"
#include "llvm/IR/Function.h"
#include "xla/codegen/math/intrinsic.h"
#include "xla/xla_data.pb.h"

namespace xla::codegen {

class Intrinsic::FpTrunc {
 public:
  static std::string Name(PrimitiveType t0, PrimitiveType t1) {
    return absl::StrCat("xla.fptrunc.", ScalarName(t0), ".to.", ScalarName(t1));
  }

  static std::string Name(PrimitiveType t0, PrimitiveType t1,
                          int64_t vector_width) {
    return absl::StrCat("xla.fptrunc.", VectorName(t0, vector_width), ".to.",
                        VectorName(t1, vector_width));
  }
};

namespace math {

// Return the XLA intrinsic name for the fptrunc function:
//
// `xla.fptrunc.v<num_elements><from_type>.v<num_elements><to_type>`
std::string FptruncFunctionName(size_t num_elements, PrimitiveType from,
                                PrimitiveType to, bool add_suffix = true);

llvm::Function* CreateFptruncF32ToBf16(llvm::Module* module,
                                       llvm::Type* input_type,
                                       bool add_suffix = true);

}  // namespace math
}  // namespace xla::codegen

#endif  // XLA_CODEGEN_MATH_FPTRUNC_H_
