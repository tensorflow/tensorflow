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

#ifndef XLA_CODEGEN_INTRINSIC_FPTRUNC_H_
#define XLA_CODEGEN_INTRINSIC_FPTRUNC_H_

#include <cstdint>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Value.h"
#include "xla/codegen/intrinsic/intrinsic.h"
#include "xla/xla_data.pb.h"

namespace xla::codegen::intrinsics {

// XLA intrinsic for truncating floating point values (scalars and vectors).
class FpTrunc : public Intrinsic<FpTrunc> {
 public:
  static constexpr absl::string_view kName = "fptrunc";
  static constexpr bool kLastArgIsReturnType = true;
  static constexpr int8_t kNumArgs = 2;  // Second arg is the return type.
  static std::vector<std::vector<Type>> SupportedVectorTypes() {
    return {
        {Type::S(xla::F32), Type::S(xla::BF16)},
        {Type::V(xla::F32, 2), Type::V(xla::BF16, 2)},
        {Type::V(xla::F32, 4), Type::V(xla::BF16, 4)},
        {Type::V(xla::F32, 8), Type::V(xla::BF16, 8)},
        {Type::S(F8E5M2), Type::S(F16)},
        {Type::V(F8E5M2, 2), Type::V(F16, 2)},
        {Type::V(F8E5M2, 4), Type::V(F16, 4)},
        {Type::V(F8E5M2, 8), Type::V(F16, 8)},
    };
  }

  static absl::StatusOr<llvm::Function*> CreateDefinition(llvm::Module* module,
                                                          Type from, Type to);
};

}  // namespace xla::codegen::intrinsics

#endif  // XLA_CODEGEN_INTRINSIC_FPTRUNC_H_
