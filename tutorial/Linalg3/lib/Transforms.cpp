//===- Transforms.cpp - Implementation of the linalg Transformations ------===//
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
// This file implements analyses and transformations for the linalg dialect.
//
//===----------------------------------------------------------------------===//

#include "linalg3/Transforms.h"
#include "linalg2/Intrinsics.h"
#include "linalg3/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using namespace linalg;
using namespace linalg::intrinsics;

void linalg::composeSliceOps(mlir::Function *f) {
  f->walkPostOrder<SliceOp>([](SliceOp sliceOp) {
    auto *sliceResult = sliceOp.getResult();
    auto viewOp = createFullyComposedView(sliceResult);
    sliceResult->replaceAllUsesWith(viewOp.getResult());
    sliceOp.erase();
  });
}

void linalg::lowerToFinerGrainedTensorContraction(mlir::Function *f) {
  f->walkPostOrder([](Operation *op) {
    if (auto matmulOp = op->dyn_cast<linalg::MatmulOp>()) {
      matmulOp.writeAsFinerGrainTensorContraction();
    } else if (auto matvecOp = op->dyn_cast<linalg::MatvecOp>()) {
      matvecOp.writeAsFinerGrainTensorContraction();
    } else {
      return;
    }
    op->erase();
  });
}
