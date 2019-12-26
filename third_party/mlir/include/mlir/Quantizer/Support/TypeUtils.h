//===- TypeUtils.h - Helper function for manipulating types -----*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines various helper functions for manipulating types. The
// process of quantizing typically involves a number of type manipulations
// that are not very common elsewhere, and it is best to name them and define
// them here versus inline in the rest of the tool.
//
//===----------------------------------------------------------------------===//

#ifndef THIRD_PARTY_MLIR_EDGE_FXPSOLVER_SUPPORT_TYPEUTILS_H_
#define THIRD_PARTY_MLIR_EDGE_FXPSOLVER_SUPPORT_TYPEUTILS_H_

#include "mlir/IR/Types.h"

namespace mlir {
namespace quantizer {

/// Given an arbitrary container or primitive type, returns the element type,
/// where the element type is just the type for non-containers.
Type getElementOrPrimitiveType(Type t);

} // namespace quantizer
} // namespace mlir

#endif // THIRD_PARTY_MLIR_EDGE_FXPSOLVER_SUPPORT_TYPEUTILS_H_
