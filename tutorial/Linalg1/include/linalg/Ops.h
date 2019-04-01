//===- Ops.h - Linalg Ops forward declarations ------------------------===//
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

#ifndef LINALG_OPS_H_
#define LINALG_OPS_H_

#include "mlir/EDSC/Intrinsics.h"
#include "mlir/IR/OpDefinition.h"

namespace linalg {

class MatmulOp;
class RangeOp;
class SliceOp;
class ViewOp;
class ViewType;

struct ViewOrSliceOp {
public:
  ViewOrSliceOp(mlir::Value *v) : v(v) {}
  ViewOp view();
  SliceOp slice();
  operator bool();
  unsigned getRank();
  ViewType getViewType();
  /// Get the indexing at `dim` by recursing into the parent.
  /// Returns the indexing as well as its actual dimension, which may have
  /// shifted from the originally requested `dim`.
  std::pair<mlir::Value *, unsigned> getRootIndexing(unsigned dim);
  // Get all the indexings without recursing.
  mlir::Operation::operand_range getIndexings();
  mlir::Value *getSupportingMemRef();

private:
  mlir::Value *v;
};

namespace intrinsics {
using range = mlir::edsc::intrinsics::ValueBuilder<RangeOp>;
using slice = mlir::edsc::intrinsics::ValueBuilder<SliceOp>;
using view = mlir::edsc::intrinsics::ValueBuilder<ViewOp>;
} // namespace intrinsics
} // namespace linalg

#endif // LINALG_OPS_H_
