//===- Intrinsics.h - Linalg intrinsics definitions -----------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_INTRINSICS_H_
#define MLIR_DIALECT_LINALG_INTRINSICS_H_

#include "mlir/EDSC/Intrinsics.h"

namespace mlir {
namespace linalg {
class CopyOp;
class FillOp;
class RangeOp;
class SliceOp;
namespace intrinsics {
using copy = mlir::edsc::intrinsics::OperationBuilder<CopyOp>;
using fill = mlir::edsc::intrinsics::OperationBuilder<FillOp>;
using range = mlir::edsc::intrinsics::ValueBuilder<RangeOp>;
using slice = mlir::edsc::intrinsics::ValueBuilder<SliceOp>;
} // namespace intrinsics
} // namespace linalg
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_INTRINSICS_H_
