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

#ifndef XLA_CODEGEN_INTRINSIC_TANH_H_
#define XLA_CODEGEN_INTRINSIC_TANH_H_

#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "xla/codegen/intrinsic/intrinsic.h"
#include "xla/xla_data.pb.h"

namespace xla::codegen::intrinsics {

class Tanh : public Intrinsic<Tanh> {
 public:
  static constexpr absl::string_view kName = "tanh";
  static std::vector<std::vector<Type>> SupportedVectorTypes() {
    // F16 via upcast to F32.
    return {
        {Type::S(xla::F16)},    {Type::V(xla::F16, 8)}, {Type::V(xla::F16, 16)},
        {Type::S(xla::F32)},

        {Type::V(xla::F32, 4)}, {Type::V(xla::F32, 8)}, {Type::V(xla::F32, 16)},
        {Type::S(xla::F64)},    {Type::V(xla::F64, 2)}, {Type::V(xla::F64, 4)},
        {Type::V(xla::F64, 8)},
    };
  }
  static absl::StatusOr<llvm::Function*> CreateDefinition(llvm::Module* module,
                                                          Type type);
};

}  // namespace xla::codegen::intrinsics

#endif  // XLA_CODEGEN_INTRINSIC_TANH_H_
