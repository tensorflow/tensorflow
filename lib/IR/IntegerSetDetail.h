//===- IntegerSetDetail.h - MLIR IntegerSet storage details -----*- C++ -*-===//
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
// This holds implementation details of IntegerSet.
//
//===----------------------------------------------------------------------===//

#ifndef INTEGERSETDETAIL_H_
#define INTEGERSETDETAIL_H_

#include "mlir/IR/AffineExpr.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {
namespace detail {

struct IntegerSetStorage {
  unsigned dimCount;
  unsigned symbolCount;

  /// Array of affine constraints: a constraint is either an equality
  /// (affine_expr == 0) or an inequality (affine_expr >= 0).
  ArrayRef<AffineExpr> constraints;

  // Bits to check whether a constraint is an equality or an inequality.
  ArrayRef<bool> eqFlags;
};

} // end namespace detail
} // end namespace mlir
#endif // INTEGERSETDETAIL_H_
