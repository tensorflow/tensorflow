//===- AffineMap.h - MLIR Affine Map Class ----------------------*- C++ -*-===//
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
// Affine maps are mathematical functions which map a list of dimension
// identifiers and symbols, to multidimensional affine expressions.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_AFFINE_MAP_H
#define MLIR_IR_AFFINE_MAP_H

#include <vector>

#include "mlir/Support/LLVM.h"

namespace mlir {

class AffineExpr;

class AffineMap  {
 public:
  // Constructs an AffineMap with 'dimCount' dimension identifiers, and
  // 'symbolCount' symbols.
  // TODO(andydavis) Pass in ArrayRef<AffineExpr*> to populate list of exprs.
  AffineMap(unsigned dimCount, unsigned symbolCount);

  // Prints affine map to 'os'.
  void print(raw_ostream &os) const;

 private:
  // Number of dimensional indentifiers.
  const unsigned dimCount;
  // Number of symbols.
  const unsigned symbolCount;
  // TODO(andydavis) Do not use std::vector here (array size is not dynamic).
  std::vector<AffineExpr*> exprs;
};

} // end namespace mlir

#endif  // MLIR_IR_AFFINE_MAP_H
