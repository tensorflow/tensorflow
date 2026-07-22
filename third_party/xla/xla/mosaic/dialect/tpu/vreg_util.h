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

#ifndef XLA_MOSAIC_DIALECT_TPU_VREG_UTIL_H_
#define XLA_MOSAIC_DIALECT_TPU_VREG_UTIL_H_

#include <array>
#include <cstdint>

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "xla/mosaic/dialect/tpu/layout.h"
#include "xla/array.h"

namespace mlir::tpu {

// Returns the native vreg or vmask type for the given element type and target
// shape. The layout bitwidth is used for i1 (vmask) elements.
VectorType getNativeVregOrVmaskType(Type elem_ty, int8_t layout_bitwidth,
                                    ArrayRef<int64_t> target_shape);
VectorType getNativeVregType(Type elem_ty, ArrayRef<int64_t> target_shape);

// Returns a zero constant of the same type as `vty`.
TypedValue<VectorType> getZerosVector(ImplicitLocOpBuilder& builder,
                                      VectorType vty);
// Same as above, but takes a `vec` as input.
TypedValue<VectorType> getZerosLikeVector(ImplicitLocOpBuilder& builder,
                                          TypedValue<VectorType> vec);

// Returns a constant of the same type as `vty` with the given `value`.
TypedValue<VectorType> getFullVector(ImplicitLocOpBuilder& builder,
                                     VectorType vty, Attribute value);
// Same as above, but takes a `vec` as input.
TypedValue<VectorType> getFullLikeVector(ImplicitLocOpBuilder& builder,
                                         TypedValue<VectorType> vec,
                                         Attribute value);

// Same as above, but takes a `loc` as input, in case of an OpBuilder.
TypedValue<VectorType> getFullVector(OpBuilder& builder, Location loc,
                                     VectorType vty, Attribute value);

// Same as above, but takes a `vec` as input.
TypedValue<VectorType> getFullLikeVector(OpBuilder& builder, Location loc,
                                         TypedValue<VectorType> vec,
                                         Attribute value);

// Masks out the padding in the bottom and right of the `vregs`. Each vreg is
// expected to be tiled according to the given `tiling` and the masked vregs are
// mutated in `vregs`. `padding_bottom` and `padding_right` are the number of
// logical elements to pad in the bottom and right.
//
// For example, each tile in the following diagram represents a **vreg slice**
// and we want to pad both bottom and right.
//
// +----+----+----+
// | 0  | 1  | 2  |
// +----+----+----+
// | 3  | 4  | 5  |
// +----+----+----+
//
// The rightmost `padding_right` columns will be masked out in tile 2 and 5. The
// bottommost `padding_bottom` rows will be masked out in tile 3, 4 and 5.
LogicalResult maskTiledVregs(ImplicitLocOpBuilder& builder,
                             xla::Array<Value>& vregs,
                             std::array<int64_t, 2> target_shape,
                             std::array<int64_t, 2> tiling,
                             int64_t padding_bottom, int64_t padding_right,
                             int generation);

// Selects between values using the provided bounds.
//
// Arguments:
//   bounds:  An object specifying the bounds of the data to be masked.
//   in_bounds_vreg:     Vreg to select data that is in bounds.
//   out_of_bounds_vreg: Vreg to select data that is out of bounds. Must have
//                       the same type as in_bounds_vreg.
//
// Returns:
//   A vreg with its elements selected according to the provided bounds.
FailureOr<TypedValue<VectorType>> selectWithBounds(
    ImplicitLocOpBuilder& builder, const VRegDataBounds& bounds,
    TypedValue<VectorType> in_bounds_vreg,
    TypedValue<VectorType> out_of_bounds_vreg,
    std::array<int64_t, 2> target_shape, int generation);

// Selects between values using the provided bounds.
//
// Arguments:
//   bounds:  An object specifying the bounds of the data to be masked.
//   in_bounds_vreg:     Vreg to select data that is in bounds.
//   out_of_bounds_vreg: Vreg to select data that is out of bounds. Must have
//                       the same type as in_bounds_vreg.
//
// Returns:
//   A vreg with its elements selected according to the provided bounds.
FailureOr<TypedValue<VectorType>> selectWithBounds(
    ImplicitLocOpBuilder& builder, const VRegDataBounds& bounds,
    TypedValue<VectorType> in_bounds_vreg,
    TypedValue<VectorType> out_of_bounds_vreg,
    std::array<int64_t, 2> target_shape, int generation);

// Broadcasts the subelement at `subelement_idx` within each packed word.
// subelement_idx must be between 0 and packing.
FailureOr<TypedValue<VectorType>> broadcastSubelements(
    ImplicitLocOpBuilder& builder, TypedValue<VectorType> vec,
    int subelement_idx, ArrayRef<int64_t> target_shape);

}  // namespace mlir::tpu

#endif  // XLA_MOSAIC_DIALECT_TPU_VREG_UTIL_H_
