//===- Traits.h - Common op traits shared by dialects -----------*- C++ -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file declares common op traits that are not core to MLIR but can be
// shared by multiple dialects.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TRAITS
#define MLIR_DIALECT_TRAITS

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace OpTrait {

// These functions are out-of-line implementations of the methods in the
// corresponding trait classes.  This avoids them being template
// instantiated/duplicated.
namespace impl {
bool verifyCompatibleOperandBroadcast(const OperationInst *op);
} // namespace impl

namespace util {
/// Returns the result broadcast composition type from the two given types by
/// following NumPy broadcast semantics. Returns null type if the two given
/// types are not broadcast-compatible.
Type getBroadcastedType(Type type1, Type type2);
} // namespace util

/// This class provides the API for ops that are known to have broadcast-
/// compatible operand and result types. Specifically,  starting from the
/// most varying dimension, each dimension pair of the two operands' types
/// should either be the same or one of them is one. Also, the result type
/// should be the same as the operand type with larger dimensions.
///
/// Ths trait assumes the op has two operands and one result, and it asserts
/// if the pre-condition is not satisfied.
template <typename ConcreteType>
class BroadcastableTwoOperandsOneResult
    : public TraitBase<ConcreteType, BroadcastableTwoOperandsOneResult> {
public:
  static bool verifyTrait(const OperationInst *op) {
    return impl::verifyCompatibleOperandBroadcast(op);
  }
};

} // end namespace OpTrait
} // end namespace mlir

#endif // MLIR_DIALECT_TRAITS
