//===- Intrinsics.h - MLIR EDSC Intrinsics for Linalg -----------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_LINALG_EDSC_INTRINSICS_H_
#define MLIR_DIALECT_LINALG_EDSC_INTRINSICS_H_

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/EDSC/Intrinsics.h"

namespace mlir {
namespace edsc {
namespace intrinsics {

using linalg_fill = OperationBuilder<linalg::FillOp>;
using linalg_yield = OperationBuilder<linalg::YieldOp>;

} // namespace intrinsics
} // namespace edsc
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_EDSC_INTRINSICS_H_
