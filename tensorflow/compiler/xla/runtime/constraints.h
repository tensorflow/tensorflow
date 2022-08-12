/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_RUNTIME_CONSTRAINTS_H_
#define XLA_RUNTIME_CONSTRAINTS_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace xla {
namespace runtime {

// Constraints on the function argument can be specified with the function
// argument attributes.
//
// Example:
//
//   func @compute(
//     // Rank of the `%arg` must be known at compile time.
//     %arg: tensor<*xf32> { rt.constraint = "rank" }
//   ) -> tensor<?xf32> { ... }
//
// TODO(b/187114012): Add attribute verifier to `rt` dialect.
constexpr const char* kArgumentConstraintAttrName = "rt.constraint";

// Constraint on what argument information must be available at compile time in
// order to successfully compile the executable:
//
//   `rank`  : argument must have statically known rank.
//   `shape` : argument must have statically known shape.
//   `value` : argument must have statically known value, and such arguments
//             replaced with constants inside the compiled function body and
//             and all value constrained argument uses replaced with the sunk
//             constant value.
//
// For now these constraints are supported by arguments of shaped types (tensors
// or memrefs), but potentially can be extended to support open type hierarchy
// of user-defined types.
enum class ArgumentConstraint {
  // Constraint was resolved based on the static information in the function
  // signature type or it was never specified by the argument attribute.
  kResolved = 0,
  kRank = 1,
  kShape = 2,
  kValue = 3
};

llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                              const ArgumentConstraint& constraint);
llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                              llvm::ArrayRef<ArgumentConstraint> constraints);

// Converts argument constraint string to the corresponding enum class.
llvm::Expected<ArgumentConstraint> ParseArgumentConstraint(llvm::StringRef str);

}  // namespace runtime
}  // namespace xla

#endif  // XLA_RUNTIME_CONSTRAINTS_H_
