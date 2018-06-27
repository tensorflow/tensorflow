//===- AffineMap.cpp - MLIR Affine Map Classes ----------------------------===//
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

#include "mlir/IR/AffineMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

AffineMap::AffineMap(unsigned dimCount, unsigned symbolCount)
    : dimCount(dimCount), symbolCount(symbolCount) {
}

void AffineMap::print(raw_ostream &os) const {
  // TODO(andydavis) Print out affine map based on dimensionCount and
  // symbolCount: (d0, d1) [S0, S1] -> (d0 + S0, d1 + S1)
}
