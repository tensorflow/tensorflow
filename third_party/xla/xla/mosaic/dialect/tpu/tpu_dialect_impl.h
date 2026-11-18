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

#ifndef XLA_MOSAIC_DIALECT_TPU_TPU_DIALECT_IMPL_H_
#define XLA_MOSAIC_DIALECT_TPU_TPU_DIALECT_IMPL_H_

#include <cstdint>

#include "absl/log/check.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/layout.h"

namespace mlir::tpu::impl {

void printTiledLayout(ArrayRef<xla::Tile> tiles, ArrayRef<int64_t> tile_strides,
                      AsmPrinter& printer);

LogicalResult parseTiledLayout(AsmParser& parser,
                               SmallVectorImpl<xla::Tile>& tiles,
                               SmallVectorImpl<int64_t>& tile_strides);

AffineMap getTiledLayoutAffineMap(MLIRContext* context,
                                  ArrayRef<xla::Tile> tiles,
                                  ArrayRef<int64_t> tile_strides);

int64_t getUntiledRank(ArrayRef<xla::Tile> tiles, int64_t rank);

FailureOr<SmallVector<int64_t>> getExpandedShape(
    ArrayRef<int64_t> untiled_shape, ArrayRef<xla::Tile> tiles,
    bool require_alignment);

SmallVector<int64_t> getContiguousTileStrides(ArrayRef<xla::Tile> tiles,
                                              ArrayRef<int64_t> shape);

int64_t getNumTrailingDimsWithContiguousTiles(ArrayRef<xla::Tile> tiles,
                                              ArrayRef<int64_t> tile_strides,
                                              ArrayRef<int64_t> shape);

FailureOr<SmallVector<int64_t>> getExpandedStrides(
    ArrayRef<xla::Tile> tiles, ArrayRef<int64_t> tile_strides);

LogicalResult verifyTiledLayout(function_ref<InFlightDiagnostic()> emitError,
                                ArrayRef<xla::Tile> tiles,
                                ArrayRef<int64_t> tile_strides);

LogicalResult verifyTiledLayoutRank(
    int64_t layout_rank, ArrayRef<int64_t> shape,
    function_ref<InFlightDiagnostic()> emitError);

template <typename DialectT>
LogicalResult verifyOperationAttribute(Operation* const op,
                                       NamedAttribute attr) {
  CHECK(op != nullptr);
  if (attr.getName() == DialectT::GetMainKey()) {
    if (!isa<func::FuncOp>(op) || !isa<mlir::UnitAttr>(attr.getValue())) {
      return op->emitOpError() << DialectT::GetMainKey()
                               << " attribute is expected to be a unit "
                                  "attribute on a func.func operation";
    }
  }
  return success();
}

template <typename TiledLayoutAttrT>
TiledLayoutAttrT getContiguousTiledLayout(MLIRContext* context,
                                          ArrayRef<xla::Tile> tiles,
                                          ArrayRef<int64_t> shape) {
  return TiledLayoutAttrT::get(context, tiles,
                               getContiguousTileStrides(tiles, shape));
}

template <typename DotDimensionNumbersAttrT>
DotDimensionNumbersAttrT defaultDimensionNumbers(Builder& builder,
                                                 bool transpose_lhs,
                                                 bool transpose_rhs) {
  return DotDimensionNumbersAttrT::get(
      builder.getContext(),
      /*lhs_contracting_dims=*/{transpose_lhs ? 0 : 1},
      /*rhs_contracting_dims=*/{transpose_rhs ? 1 : 0},
      /*lhs_non_contracting_dims=*/{transpose_lhs ? 1 : 0},
      /*rhs_non_contracting_dims=*/{transpose_rhs ? 0 : 1},
      /*output_dim_order=*/{0, transpose_lhs ? 1 : 0, 1, transpose_rhs ? 0 : 1},
      /*lhs_batch_dims=*/{},
      /*rhs_batch_dims=*/{});
}

}  // namespace mlir::tpu::impl

#endif  // XLA_MOSAIC_DIALECT_TPU_TPU_DIALECT_IMPL_H_
