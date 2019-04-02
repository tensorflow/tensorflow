//===- Analysis.h - Linalg dialect Analysis function definitions ----------===//
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

#ifndef LINALG1_ANALYSIS_H_
#define LINALG1_ANALYSIS_H_

#include "mlir/Support/LLVM.h"

namespace mlir {
class Value;
} // namespace mlir

namespace linalg {
class ViewOp;

/// Walks the chain of SliceOp until the unique base ViewOp.
ViewOp getViewBaseViewOp(mlir::Value *view);

/// Walks the chain of SliceOp until the unique base ViewOp and returns the
/// MemRef upon which the ViewOp is laid.
mlir::Value *getViewSupportingMemRef(mlir::Value *view);

/// Extract the indexing from the root ViewOp that this slice constrins along
/// `dim`. To achieve this, it walks back the chain of SliceOp and determine the
/// first slice that constrains `dim`.
/// Note that the dimension in the original ViewOp may shift due to
/// rank-reducing operations.
/// Returns a pair, with the indexing as the first element and the actual
/// dimension, in the root ViewOp, as the second element.
std::pair<mlir::Value *, unsigned> getViewRootIndexing(mlir::Value *view,
                                                       unsigned dim);

////////////////////////////////////////////////////////////////////////////////
/// Helper functions to avoid dispatching at all client sites.
////////////////////////////////////////////////////////////////////////////////
/// Asserts `view` is of ViewType and returns its rank.
unsigned getViewRank(mlir::Value *view);

} // namespace linalg

#endif // LINALG1_ANALYSIS_H_
