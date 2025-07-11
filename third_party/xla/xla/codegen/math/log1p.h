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

#ifndef XLA_CODEGEN_MATH_LOG1P_H_
#define XLA_CODEGEN_MATH_LOG1P_H_

#include <cstddef>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/Function.h"
#include "xla/codegen/math/intrinsic.h"
#include "xla/xla_data.pb.h"

namespace xla::codegen {

class Intrinsic::Log1p : public intrinsics::UnaryIntrinsic<Log1p> {
 public:
  static constexpr absl::string_view kName = "log1p";

  static absl::StatusOr<llvm::Function*> CreateDefinition(
      llvm::Module* module, PrimitiveType prim_type, size_t vector_width);
};

namespace math {

// Return the XLA intrinsic name for the log1p function:
//
// `xla.log1p.v<num_elements><type>`
std::string Log1pFunctionName(size_t num_elements, PrimitiveType type);

llvm::Function* CreateLog1p(llvm::Module* module, llvm::Type* type);

}  // namespace math
}  // namespace xla::codegen

#endif  // XLA_CODEGEN_MATH_LOG1P_H_
