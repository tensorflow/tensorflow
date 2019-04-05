//===- Utils.h - Linalg dialect utility functions definitions -------------===//
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

#ifndef LINALG1_UTILS_H_
#define LINALG1_UTILS_H_

namespace mlir {
class Value;
} // namespace mlir

namespace linalg {
class ViewOp;

/// Asserts `view` is of ViewType and returns its rank.
unsigned getViewRank(mlir::Value *view);

/// Helper function to emit and return a new ViewOp from `memRef` that is
/// assumed to be of MemRefType. This needs to be called under a ScopedContext.
ViewOp emitAndReturnViewOpFromMemRef(mlir::Value *memRef);

} // namespace linalg

#endif // LINALG1_UTILS_H_
