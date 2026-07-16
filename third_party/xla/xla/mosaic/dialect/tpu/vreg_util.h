/* Copyright 2024 The JAX Authors.

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

#ifndef THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_TPU_VREG_UTIL_H_
#define THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_TPU_VREG_UTIL_H_

#include <array>
#include <cstdint>

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "xla/array.h"

namespace mlir::tpu {

// Returns the native vreg or vmask type for the given element type and target
// shape. The layout bitwidth is used for i1 (vmask) elements.
VectorType getNativeVregOrVmaskType(Type elem_ty, int8_t layout_bitwidth,
                                    std::array<int64_t, 2> target_shape);
VectorType getNativeVregType(Type elem_ty, std::array<int64_t, 2> target_shape);

// Returns a zero constant of the same type as `vty`.
TypedValue<VectorType> getZerosVector(ImplicitLocOpBuilder &builder,
                                      VectorType vty);
// Same as above, but takes a `vec` as input.
TypedValue<VectorType> getZerosLikeVector(ImplicitLocOpBuilder &builder,
                                          TypedValue<VectorType> vec);

// Returns a constant of the same type as `vty` with the given `value`.
TypedValue<VectorType> getFullVector(ImplicitLocOpBuilder &builder,
                                     VectorType vty, Attribute value);
// Same as above, but takes a `vec` as input.
TypedValue<VectorType> getFullLikeVector(ImplicitLocOpBuilder &builder,
                                         TypedValue<VectorType> vec,
                                         Attribute value);

// Same as above, but takes a `loc` as input, in case of an OpBuilder.
TypedValue<VectorType> getFullVector(OpBuilder &builder, Location loc,
                                     VectorType vty, Attribute value);

// Same as above, but takes a `vec` as input.
TypedValue<VectorType> getFullLikeVector(OpBuilder &builder, Location loc,
                                         TypedValue<VectorType> vec,
                                         Attribute value);

// Creates a vmask with false flags to bottom (dim = 0)
// or right (dim = 1) where the flag count corresponds to the (dim_size -
// padding).
//
// For example, assume vmask shape is (4, 8)
//
// getX32VmaskByPaddingEnd(padding=3, dim=1) creates:
//  [T, T, T, T, T, F, F, F]
//  [T, T, T, T, T, F, F, F]
//  [T, T, T, T, T, F, F, F]
//  [T, T, T, T, T, F, F, F]
// TODO(b/385204135): Unify with getVmaskByPaddingEnd in tpu_rotate_rule, and
// improve the codegen.
FailureOr<TypedValue<VectorType>> getX32VmaskByPaddingEnd(
    ImplicitLocOpBuilder &builder, int64_t padding,
    std::array<int64_t, 2> target_shape, int64_t dim);

// Masks out the padding in the bottom and right of the vregs. vregs are
// expected to have native tiling, and the masked vregs are mutated in
// `vregs`. `padding_bottom` and `padding_right` is the number of elements to
// pad in the bottom and right.
LogicalResult maskNativeTilingVregs(ImplicitLocOpBuilder &builder,
                                    xla::Array<Value> &vregs,
                                    std::array<int64_t, 2> target_shape,
                                    int64_t padding_bottom,
                                    int64_t padding_right);

// Broadcasts the subelement at `subelement_idx` within each packed word.
// subelement_idx must be between 0 and packing.
FailureOr<TypedValue<VectorType>> broadcastSubelements(
    ImplicitLocOpBuilder &builder, TypedValue<VectorType> vec,
    int subelement_idx, std::array<int64_t, 2> target_shape);

}  // namespace mlir::tpu

#endif  // THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_TPU_VREG_UTIL_H_
