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

#ifndef XLA_CODEGEN_INTRINSIC_FUNCTION_H_
#define XLA_CODEGEN_INTRINSIC_FUNCTION_H_

#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "xla/codegen/intrinsic/intrinsic.h"
#include "xla/codegen/intrinsic/type.h"

namespace xla::codegen {

// Interface representing a single vectorized math function approximation.
// Each implementation may support multiple vector widths and primitive types,
// defined by the SupportedVectorTypes() method. To emit LLVM IR for a
// particular vector width and primitive type, call CreateDefinition() with the
// desired vector_width and primitive_type.
class IntrinsicFunction {
 public:
  virtual ~IntrinsicFunction() = default;
  // The name of the function being approximated.
  virtual absl::string_view FunctionName() const = 0;

  // Returns the vector types supported well by this approximation.
  virtual std::vector<std::vector<intrinsics::Type>> SupportedVectorTypes(
      absl::string_view features) const = 0;

  // Returns the LLVM IR function definition for the approximation.
  virtual llvm::Function* CreateDefinition(llvm::Module& module,
                                           intrinsics::IntrinsicOptions options,
                                           absl::string_view name) const = 0;

  // The vectorized function name, e.g. "xla.ldexp.v8f64.v8i32".
  virtual std::string GenerateVectorizedFunctionName(
      absl::Span<const intrinsics::Type> types) const = 0;

  // The LLVM mangled prefix for the vectorized function, e.g.
  // "_ZGV_LLVM_N8" used in llvm::VecDesc.
  virtual std::string GenerateMangledSimdPrefix(
      absl::Span<const intrinsics::Type> types) const = 0;
};

}  // namespace xla::codegen

#endif  // XLA_CODEGEN_INTRINSIC_FUNCTION_H_
