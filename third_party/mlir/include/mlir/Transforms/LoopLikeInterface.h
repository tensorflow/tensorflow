//===- LoopLikeInterface.h - Loop-like operations interface ---------------===//
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
