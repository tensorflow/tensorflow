//===- TensorOps.cpp - Implementation of the linalg TensorOps operation ---===//
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
// This file implements a simple IR operation to create new tensor computation
// operations in the linalg dialect.
//
//===----------------------------------------------------------------------===//

#include "linalg1/Analysis.h"
#include "linalg1/Common.h"
#include "linalg2/Intrinsics.h"
#include "linalg3/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::edsc;
using namespace linalg;
using namespace linalg::intrinsics;

// The body expression for matvec is: C(i) = scalarC + A(i, r_j) * B(r_j)
// The body expression for dot is: C() = A(r_i) * B(r_i);
// So we must drop the `i` loop from the matvec.
void linalg::MatvecOp::writeAsFinerGrainTensorContraction() {
  auto *op = getOperation();
  ScopedContext scope(FuncBuilder(op), op->getLoc());
  IndexHandle i;
  auto *vA(getInputView(0)), *vB(getInputView(1)), *vC(getOutputView(0));
  auto indexingPosPair = getViewRootIndexing(vA, 0);
  assert(indexingPosPair.first->getDefiningOp() &&
         indexingPosPair.first->getDefiningOp()->isa<RangeOp>());
  linalg::common::LoopNestRangeBuilder(&i, ValueHandle(indexingPosPair.first))({
      dot(slice(vA, i, 0), vB, slice(vC, i, 0)),
  });
}

// The body expression for matmul is: C(i, j) = scalarC + A(i, r_k) * B(r_k, j)
// The body expression for matvec is: C(i) = scalarC + A(i, r_j) * B(r_j)
// So we must drop the `j` loop from the matmul.
// This is fine because parallel dimensions permute: we can just do it
// declaratively.
void linalg::MatmulOp::writeAsFinerGrainTensorContraction() {
  auto *op = getOperation();
  ScopedContext scope(FuncBuilder(op), op->getLoc());
  IndexHandle j;
  auto *vA(getInputView(0)), *vB(getInputView(1)), *vC(getOutputView(0));
  auto indexingPosPair = getViewRootIndexing(vB, 1);
  assert(indexingPosPair.first->getDefiningOp() &&
         indexingPosPair.first->getDefiningOp()->isa<RangeOp>());
  linalg::common::LoopNestRangeBuilder(&j, ValueHandle(indexingPosPair.first))({
      matvec(vA, slice(vB, j, 1), slice(vC, j, 1)),
  });
}
