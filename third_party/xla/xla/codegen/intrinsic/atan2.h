/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_CODEGEN_INTRINSIC_ATAN2_H_
#define XLA_CODEGEN_INTRINSIC_ATAN2_H_

#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "xla/codegen/intrinsic/intrinsic.h"
#include "xla/xla_data.pb.h"

namespace xla::codegen::intrinsics {

class Atan2 : public Intrinsic<Atan2> {
 public:
  static constexpr absl::string_view kName = "atan2";
  static constexpr int8_t kNumArgs = 2;

  static std::vector<std::vector<Type>> SupportedVectorTypes();

  static absl::StatusOr<llvm::Function*> CreateDefinition(llvm::Module* module,
                                                          Type y, Type x);
};

}  // namespace xla::codegen::intrinsics

#endif  // XLA_CODEGEN_INTRINSIC_ATAN2_H_
