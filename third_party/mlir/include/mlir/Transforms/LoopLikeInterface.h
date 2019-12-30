//===- LoopLikeInterface.h - Loop-like operations interface ---------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the operation interface for loop like operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_LOOPLIKEINTERFACE_H_
#define MLIR_TRANSFORMS_LOOPLIKEINTERFACE_H_

#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {

#include "mlir/Transforms/LoopLikeInterface.h.inc"

} // namespace mlir

#endif // MLIR_TRANSFORMS_LOOPLIKEINTERFACE_H_
