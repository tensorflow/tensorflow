//===- Utils.h - General transformation utilities ---------------*- C++ -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This header file defines prototypes for various transformation utilities for
// memref's and non-loop IR structures. These are not passes by themselves but
// are used either by passes, optimization sequences, or in turn by other
// transformation utilities.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_UTILS_H
#define MLIR_TRANSFORMS_UTILS_H

#include "llvm/ADT/ArrayRef.h"

namespace mlir {

class AffineMap;
class MLValue;
class SSAValue;

/// Replace all uses of oldMemRef with newMemRef while optionally remapping the
/// old memref's indices using the supplied affine map and adding any additional
/// indices. The new memref could be of a different shape or rank. Returns true
/// on success and false if the replacement is not possible (whenever a memref
/// is used as an operand in a non-deferencing scenario).
/// Additional indices are added at the start.
// TODO(mlir-team): extend this for SSAValue / CFGFunctions. Can also be easily
// extended to add additional indices at any position.
bool replaceAllMemRefUsesWith(MLValue *oldMemRef, MLValue *newMemRef,
                              llvm::ArrayRef<SSAValue *> extraIndices,
                              AffineMap *indexRemap = nullptr);
} // end namespace mlir

#endif // MLIR_TRANSFORMS_UTILS_H
