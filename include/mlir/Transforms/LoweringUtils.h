//===- LoweringUtils.h ---- Utilities for Lowering Passes -------*- C++ -*-===//
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
// This file implements miscellaneous utility functions for lowering passes.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INCLUDE_MLIR_TRANSFORMS_LOWERINGUTILS_H
#define MLIR_INCLUDE_MLIR_TRANSFORMS_LOWERINGUTILS_H

namespace mlir {

class AffineApplyOp;

/// Convert the affine_apply operation `op` into a sequence of primitive
/// arithmetic instructions that have the same effect and insert them at the
/// current location of the `op`.  Erase the `op` from its parent.  Return true
/// if any errors happened during expansion.
bool expandAffineApply(AffineApplyOp *op);

} // namespace mlir

#endif // MLIR_INCLUDE_MLIR_TRANSFORMS_LOWERINGUTILS_H
