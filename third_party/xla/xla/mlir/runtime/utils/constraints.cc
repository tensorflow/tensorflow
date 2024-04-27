/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/mlir/runtime/utils/constraints.h"

#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Interfaces/FunctionInterfaces.h"  // from @llvm-project
#include "mlir/Support/DebugStringHelper.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "xla/runtime/constraints.h"

namespace xla {
namespace runtime {

using namespace mlir;  // NOLINT

using absl::InvalidArgumentError;
using absl::StatusOr;
using absl::StrCat;

using llvm::SmallVector;

StatusOr<SmallVector<ArgumentConstraint>> GetArgumentsConstraints(
    FunctionOpInterface func) {
  llvm::SmallVector<ArgumentConstraint> constraints;
  constraints.reserve(func.getNumArguments());

  auto parse = [](Attribute attr) -> StatusOr<ArgumentConstraint> {
    // If attribute is not defined it means that there is no constraint.
    if (!attr) return ArgumentConstraint::kResolved;

    // Otherwise try to parse constraint from the string attribute.
    auto str = mlir::dyn_cast_or_null<StringAttr>(attr);
    if (!str)
      return InvalidArgumentError(
          StrCat("unexpected ", kArgumentConstraintAttrName, " attribute"));
    return ParseArgumentConstraint(str.getValue());
  };

  for (int i = 0; i < func.getNumArguments(); ++i) {
    auto arg_type =
        llvm::cast<FunctionType>(func.getFunctionType()).getInput(i);

    auto constraint = parse(func.getArgAttr(i, kArgumentConstraintAttrName));
    if (!constraint.ok()) return constraint.status();

    auto resolved = ResolveArgumentConstraint(*constraint, arg_type);
    if (!resolved.ok()) return resolved.status();

    constraints.push_back(*resolved);
  }

  return constraints;
}

StatusOr<ArgumentConstraint> ResolveArgumentConstraint(
    ArgumentConstraint constraint, Type type) {
  // Skip already resolved constraints.
  if (constraint == ArgumentConstraint::kResolved) return constraint;

  // Operand must be a shaped type: memref or tensor.
  auto shaped = mlir::dyn_cast<ShapedType>(type);
  if (!shaped)
    return InvalidArgumentError(
        StrCat("unsupported operand type: ", debugString(type)));

  // Resolve `rank` constraint if rank is known at compile time.
  if (constraint == ArgumentConstraint::kRank && shaped.hasRank())
    return ArgumentConstraint::kResolved;

  // Resolve `shape` constraint if shape is known at compile time.
  if (constraint == ArgumentConstraint::kShape && shaped.hasStaticShape())
    return ArgumentConstraint::kResolved;

  // Leave the `value` constraint unmodified if the operand is sinkable.
  if (constraint == ArgumentConstraint::kValue) {
    if (SupportsValueSpecialization(shaped)) return constraint;
    return InvalidArgumentError(
        StrCat("cannot sink operand type: ", debugString(type)));
  }

  return constraint;
}

}  // namespace runtime
}  // namespace xla
