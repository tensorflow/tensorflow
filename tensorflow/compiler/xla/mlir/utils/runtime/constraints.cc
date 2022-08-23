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

#include "tensorflow/compiler/xla/mlir/utils/runtime/constraints.h"

#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "tensorflow/compiler/xla/runtime/errors.h"

namespace xla {
namespace runtime {

using namespace mlir;  // NOLINT

using llvm::Expected;
using llvm::SmallVector;

Expected<SmallVector<ArgumentConstraint>> GetArgumentsConstraints(
    func::FuncOp func) {
  llvm::SmallVector<ArgumentConstraint> constraints;
  constraints.reserve(func.getNumArguments());

  auto parse = [](Attribute attr) -> Expected<ArgumentConstraint> {
    // If attribute is not defined it means that there is no constraint.
    if (!attr) return ArgumentConstraint::kResolved;

    // Otherwise try to parse constraint from the string attribute.
    auto str = attr.dyn_cast_or_null<StringAttr>();
    if (!str)
      return MakeStringError("unexpected ", kArgumentConstraintAttrName,
                             " attribute");
    return ParseArgumentConstraint(str.getValue());
  };

  for (int i = 0; i < func.getNumArguments(); ++i) {
    auto arg_type = func.getFunctionType().getInput(i);

    auto constraint = parse(func.getArgAttr(i, kArgumentConstraintAttrName));
    if (auto err = constraint.takeError()) return std::move(err);

    auto resolved = ResolveArgumentConstraint(*constraint, arg_type);
    if (auto err = resolved.takeError()) return std::move(err);

    constraints.push_back(*resolved);
  }

  return constraints;
}

Expected<ArgumentConstraint> ResolveArgumentConstraint(
    ArgumentConstraint constraint, Type type) {
  // Skip already resolved constraints.
  if (constraint == ArgumentConstraint::kResolved) return constraint;

  // Operand must be a shaped type: memref or tensor.
  auto shaped = type.dyn_cast<ShapedType>();
  if (!shaped) return MakeStringError("unsupported operand type: ", type);

  // Resolve `rank` constraint if rank is known at compile time.
  if (constraint == ArgumentConstraint::kRank && shaped.hasRank())
    return ArgumentConstraint::kResolved;

  // Resolve `shape` constraint if shape is known at compile time.
  if (constraint == ArgumentConstraint::kShape && shaped.hasStaticShape())
    return ArgumentConstraint::kResolved;

  // Leave the `value` constraint unmodified if the operand is sinkable.
  if (constraint == ArgumentConstraint::kValue) {
    if (SupportsValueSpecialization(shaped)) return constraint;
    return MakeStringError("cannot sink operand type: ", type);
  }

  return constraint;
}

}  // namespace runtime
}  // namespace xla
