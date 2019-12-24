//===- LinalgDoc.td - Linalg documentation -----------------*- tablegen -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This documentation files exists to circumvent limitations on mixing different
// .td files in cases one does not want to have all ops belong to the same
// logical unit. This file should only include other .td files only and be used
// for the purpose of generating documentation.
//
//===----------------------------------------------------------------------===//

#ifndef LINALG_DOC
#define LINALG_DOC

include "mlir/Dialect/Linalg/IR/LinalgBase.td"
include "mlir/Dialect/Linalg/IR/LinalgOps.td"
include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.td"

#endif // LINALG_DOC
