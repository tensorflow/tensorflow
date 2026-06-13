/* Copyright 2023 The JAX Authors.

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

#ifndef THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_TPU_TRANSFORMS_APPLY_VECTOR_LAYOUT_H_
#define THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_TPU_TRANSFORMS_APPLY_VECTOR_LAYOUT_H_

#include <array>
#include <cstdint>

#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/array.h"
#include "xla/mosaic/dialect/tpu/layout.h"
#include "xla/mosaic/dialect/tpu/tpu_dialect.h"

namespace mlir::tpu {

// TODO(tlongeri): Remove default values for use_implicit_shape.
RollVectorsOp assemble(OpBuilder &builder, VectorType vty,
                       const VectorLayout &layout,
                       const xla::Array<Value> &vals,
                       std::array<int64_t, 2> target_shape,
                       bool use_implicit_shape = false);
FailureOr<xla::Array<Value>> disassemble(OpBuilder &builder,
                                         const VectorLayout &layout,
                                         TypedValue<VectorType> val,
                                         std::array<int64_t, 2> target_shape,
                                         bool use_implicit_shape = false);

// Rewrites the operation according to its layout annotations.
//
// Args:
//   ctx: The context used for rewriting.
//   op: An MLIR operation to be rewritten.
//
// A valid op is expected to have a layout_in attribute unless it has no
// operands. The layout_in attribute must fulfill the following:
//   - All vector operands originate from an operation (not a BlockArgument)
//   and
//     have a valid layout (Layout1D or Layout2D)
//   - All non-vector operands must have NoLayout.
LogicalResult applyLayoutOp(ApplyVectorLayoutContext &ctx, Operation &op);

// Changes the layout of a vector value.
//
// Arguments:
//   ctx: The context used for rewriting.
//   builder: The builder used for rewriting.
//   v: The value to relayout. Must be of type VectorType.
//   src: The current layout of v.
//   dst: The target layout of v.
//
// Returns:
//   A new MLIR vector value, laid out as requested by dst.
FailureOr<TypedValue<VectorType>> relayout(ApplyVectorLayoutContext &ctx,
                                           OpBuilder &builder,
                                           TypedValue<VectorType> v,
                                           VectorLayout src, VectorLayout dst);

}  // namespace mlir::tpu

#endif  // THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_TPU_TRANSFORMS_APPLY_VECTOR_LAYOUT_H_
