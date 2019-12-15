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
LogicalResult verifyCompatibleOperandBroadcast(Operation *op);
} // namespace impl

namespace util {
/// Returns true and sets `resultShape` to the broadcasted shape from the two
/// given shapes if they are broadcast compatible. Returns false and clears
/// `resultShape` otherwise.
///
/// The rules for determining the result shape are:
///
/// Zip together the dimensions in the two given shapes by prepending the shape
/// with less dimensions with 1s. For each dimension pair, deduces the result
/// dimension according to the following order:
/// - If there are unknown dimensions, follows the TensorFlow behavior:
///   - If either dimension is greater than 1, we assume that the program is
///     correct, and the other dimension will be broadcast to match it.
///   - If either dimension is 1, the other dimension is the result.
///   - Otherwise, the result dimension is unknown dimension.
/// - If one of the dimension is 1, the other dimension is the result.
/// - If two dimensions are the same, that's the result.
/// - Otherwise, incompatible shape.
bool getBroadcastedShape(ArrayRef<int64_t> shape1, ArrayRef<int64_t> shape2,
                         SmallVectorImpl<int64_t> &resultShape);

/// Returns the result broadcast composition type from the two given types by
/// following NumPy broadcast semantics. Returned type may have dynamic shape if
/// either of the input types has dynamic shape. Returns null type if the two
/// given types are not broadcast-compatible.
Type getBroadcastedType(Type type1, Type type2);
} // namespace util

/// This class provides the API for ops that are known to have broadcast-
/// compatible operand and result types. Specifically,  starting from the
/// most varying dimension, each dimension pair of the two operands' types
/// should either be the same or one of them is one. Also, the result type
/// should have the corresponding dimension equal to the larger one, if known.
/// Shapes are checked partially if ranks or dimensions are not known. For
/// example, an op with tensor<? x 2 x f32> and tensor <2 x f32> as operand
/// types and tensor<3 x 2 x f32> as the result type is broadcast-compatible.
///
/// Ths trait assumes the op has two operands and one result, and it asserts
/// if the pre-condition is not satisfied.
template <typename ConcreteType>
class BroadcastableTwoOperandsOneResult
    : public TraitBase<ConcreteType, BroadcastableTwoOperandsOneResult> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifyCompatibleOperandBroadcast(op);
  }
};

} // end namespace OpTrait
} // end namespace mlir

#endif // MLIR_DIALECT_TRAITS
