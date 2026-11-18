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
#include "llvm/ADT/STLExtras.h"
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

namespace mlir::tpu {

SmallVector<int64_t> getContiguousTileStrides(ArrayRef<xla::Tile> tiles,
                                              ArrayRef<int64_t> shape);

FailureOr<SmallVector<int64_t>> getExpandedStrides(
    ArrayRef<xla::Tile> tiles, ArrayRef<int64_t> tile_strides);

}  // namespace mlir::tpu

#endif  // XLA_MOSAIC_DIALECT_TPU_TPU_DIALECT_IMPL_H_
