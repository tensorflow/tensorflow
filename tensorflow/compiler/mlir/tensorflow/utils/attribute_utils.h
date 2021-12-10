/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_ATTRIBUTE_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_ATTRIBUTE_UTILS_H_

#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project

namespace mlir {
namespace TF {

// Copies attributes that satisfy the given predicate from `from` to `to`.
template <typename Predicate>
void CopyAttributes(Operation *from, Operation *to, Predicate P) {
  for (const NamedAttribute &attr : from->getAttrs())
    if (P(attr)) to->setAttr(attr.getName(), attr.getValue());
}

// Copies attributes whose name begins with an _ from `from` to `to`.
inline void CopyUnderscoredAttributes(Operation *from, Operation *to) {
  CopyAttributes(from, to, [](const NamedAttribute &attr) {
    return attr.getName().strref().front() == '_';
  });
}

// Copies attributes that are either `device` or whose name begins with an _
// from `from` to `to`.
// TODO(b/158769932): This should be a general feature instead post some policy
// discussion.
inline void CopyDeviceAndUnderscoredAttributes(Operation *from, Operation *to) {
  auto device = mlir::Identifier::get("device", from->getContext());
  CopyAttributes(from, to, [&device](const NamedAttribute &attr) {
    return attr.getName().strref().front() == '_' || attr.getName() == device;
  });
}

// Forward declare these passthrough ops.
// TODO(jpienaar): Remove these and use trait instead.
class IdentityOp;
class IdentityNOp;

// Returns if a value corresponds to a constant, returns the matched constant
// as an attribute.
template <typename AttrT>
bool GetValueAsConstant(Value val, AttrT &attr) {
  while (auto result = val.dyn_cast<OpResult>()) {
    Operation *op = result.getOwner();
    if (!isa<IdentityOp>(op) && !isa<IdentityNOp>(op)) break;
    val = op->getOperand(result.getResultNumber());
  }
  return matchPattern(val, m_Constant(&attr));
}

}  // namespace TF
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_ATTRIBUTE_UTILS_H_
