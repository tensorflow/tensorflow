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

#ifndef XLA_CODEGEN_INTRINSIC_LDEXP_H_
#define XLA_CODEGEN_INTRINSIC_LDEXP_H_

#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "xla/codegen/intrinsic/intrinsic.h"
#include "xla/xla_data.pb.h"

namespace xla::codegen::intrinsics {

// Returns a fast bit-shifting f64 implementation of ldexp for F64 based on
// Eigen. Function is named `xla.ldexp.<vector_size>xf64`.
// Won't overflow even if 2^e doesn't fit in a double.
// N.B. A 7x faster implementation is available in Eigen that we could try
// in the future:
// https://gitlab.com/libeigen/eigen/-/blob/master/Eigen/src/Core/arch/Default/GenericPacketMathFunctions.h#L272
class Ldexp : public intrinsics::Intrinsic<Ldexp> {
 public:
  static constexpr absl::string_view kName = "ldexp";
  static constexpr int8_t kNumArgs = 2;
  static std::vector<std::vector<Type>> SupportedVectorTypes() {
    return {{Type::S(xla::F64), Type::S(xla::S32)},
            {Type::V(xla::F64, 1), Type::V(xla::S32, 1)},
            {Type::V(xla::F64, 2), Type::V(xla::S32, 2)},
            {Type::V(xla::F64, 4), Type::V(xla::S32, 4)},
            {Type::V(xla::F64, 8), Type::V(xla::S32, 8)}};
  }

  static absl::StatusOr<llvm::Function*> CreateDefinition(llvm::Module* module,
                                                          Type base, Type exp);
};
}  // namespace xla::codegen::intrinsics

#endif  // XLA_CODEGEN_INTRINSIC_LDEXP_H_
