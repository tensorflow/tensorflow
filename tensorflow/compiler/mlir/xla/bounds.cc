/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/mlir/xla/bounds.h"

#include <algorithm>
#include <climits>

namespace mlir {

Attribute addOrModifyUpperBound(MLIRContext* context, Attribute encoding,
                                int dimension, int64_t limit) {
  std::vector<AffineExpr> new_expr = {};
  std::vector<bool> new_iseq = {};
  int max_dimension = dimension;
  if (encoding && encoding.isa<IntegerSetAttr>()) {
    IntegerSet set = encoding.cast<IntegerSetAttr>().getValue();
    int i = 0;
    for (AffineExpr expr : set.getConstraints()) {
      bool has_dimension = false;
      expr.walk([&](AffineExpr e) {
        if (e.getKind() == AffineExprKind::DimId) {
          int dim = e.cast<AffineDimExpr>().getPosition();
          has_dimension |= (dim == dimension);
          max_dimension = std::max(max_dimension, dim);
        }
      });
      if (!has_dimension) {
        new_expr.push_back(expr);
        new_iseq.push_back(set.isEq(i));
      }
      i++;
    }
  }
  if (limit != LLONG_MAX) {
    // -dimension + limit >= 0
    AffineExpr expr = -getAffineDimExpr(dimension, context) +
                      getAffineConstantExpr(limit, context);
    new_expr.push_back(expr);
    new_iseq.push_back(false);
  }

  // vector<bool> doesn't cast to ArrayRef<bool>, so copy the contents.
  bool* iseq_contents = new bool[new_iseq.size()];
  std::copy(new_iseq.begin(), new_iseq.end(), iseq_contents);

  IntegerSet set =
      IntegerSet::get(max_dimension, 0, new_expr,
                      llvm::ArrayRef<bool>(iseq_contents, new_iseq.size()));

  delete[] iseq_contents;

  return IntegerSetAttr::get(set);
}

namespace {
llvm::Optional<std::pair<int, int64_t>> ParseAffineExpr(AffineExpr expr) {
  // An upper bound on a dimension is represented as
  //   dim_i * -1 + bound >= 0
  if (auto sum = expr.dyn_cast<AffineBinaryOpExpr>()) {
    if (auto dim_times_minus_one =
            sum.getLHS().dyn_cast<AffineBinaryOpExpr>()) {
      if (auto dimension_expr =
              dim_times_minus_one.getLHS().dyn_cast<AffineDimExpr>()) {
        int dimension = dimension_expr.getPosition();
        if (auto upper_bound_expr =
                sum.getRHS().dyn_cast<AffineConstantExpr>()) {
          int64_t upper_bound = upper_bound_expr.getValue();
          return std::pair<int, int64_t>(dimension, upper_bound);
        }
      }
    }
  }
  return {};
}
}  // namespace

int64_t getUpperBoundFromAttr(Attribute attr, int dimension) {
  if (!attr || !attr.isa<IntegerSetAttr>()) {
    // If we're not storing bounds yet, everything is unbounded.
    return -1;
  }
  IntegerSet set = attr.cast<IntegerSetAttr>().getValue();
  for (AffineExpr expr : set.getConstraints()) {
    auto parsed = ParseAffineExpr(expr);
    if (parsed) {
      int parsed_dimension = parsed->first;
      int64_t upper_bound = parsed->second;
      if (parsed_dimension == dimension) {
        return upper_bound;
      }
    }
  }
  return -1;
}

llvm::SmallVector<int64_t, 4> getUpperBoundsForTensor(Attribute attr,
                                                      RankedTensorType ty) {
  llvm::SmallVector<int64_t, 4> result(ty.getShape().begin(),
                                       ty.getShape().end());
  if (attr && attr.isa<IntegerSetAttr>()) {
    IntegerSet set = attr.cast<IntegerSetAttr>().getValue();
    for (AffineExpr expr : set.getConstraints()) {
      auto parsed = ParseAffineExpr(expr);
      if (parsed) {
        int dimension = parsed->first;
        int64_t upper_bound = parsed->second;
        if (dimension < 0 || dimension >= result.size()) {
          llvm::report_fatal_error("Invalid dimension for bound");
        }
        result[dimension] = upper_bound;
      }
    }
  }
  return result;
}

}  // namespace mlir
