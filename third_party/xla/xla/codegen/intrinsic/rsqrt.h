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

#ifndef XLA_CODEGEN_INTRINSIC_RSQRT_H_
#define XLA_CODEGEN_INTRINSIC_RSQRT_H_

#include <vector>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Target/TargetMachine.h"
#include "xla/codegen/intrinsic/intrinsic.h"
#include "xla/xla_data.pb.h"

namespace xla::codegen::intrinsics {

// XLA intrinsic for computing the reciprocal square root (1/sqrt(x)).
class Rsqrt : public Intrinsic<Rsqrt> {
 public:
  static constexpr absl::string_view kName = "rsqrt";
  static std::vector<std::vector<Type>> SupportedVectorTypes(
      absl::string_view features) {
    // Always include scalars so that we're able to vectorize from elemental
    // IR.
    std::vector<std::vector<Type>> supported_types = {{Type::S(F32)},
                                                      {Type::S(F64)}};
    if (absl::StrContains(features, "+avx")) {
      supported_types.push_back({Type::V(F32, 8)});
    }
    if (absl::StrContains(features, "+avx512f")) {
      supported_types.push_back({Type::V(F32, 16)});
      supported_types.push_back({Type::V(F64, 2)});
      supported_types.push_back({Type::V(F64, 4)});
      supported_types.push_back({Type::V(F64, 8)});
    }
    return supported_types;
  }

  // Creates an LLVM function that computes the reciprocal square root
  // (1/sqrt(x)) with high precision (within 1 ULP). Uses the hardware rsqrt
  // intrinsic as initial guess followed by some Newton-Raphson iterations.
  // Based on Eigen's implementations, with some modifications:
  // https://eigen.tuxfamily.org/dox-devel/arch_2AVX512_2MathFunctions_8h_source.html
  // Assumes AVX512 is available for F64 and <16 x float> inputs.
  static absl::StatusOr<llvm::Function*> CreateDefinition(
      llvm::Module* module, absl::string_view features, Type type);
};

}  // namespace xla::codegen::intrinsics

#endif  // XLA_CODEGEN_INTRINSIC_RSQRT_H_
