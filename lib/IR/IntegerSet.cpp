
//===- IntegerSet.cpp - MLIR Integer Set class ----------------------------===//
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

#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/AffineExpr.h"

using namespace mlir;

IntegerSet::IntegerSet(unsigned dimCount, unsigned symbolCount,
                       unsigned numConstraints,
                       ArrayRef<AffineExprRef> constraints,
                       ArrayRef<bool> eqFlags)
    : dimCount(dimCount), symbolCount(symbolCount),
      numConstraints(numConstraints), constraints(constraints),
      eqFlags(eqFlags) {}
