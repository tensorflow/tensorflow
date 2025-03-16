/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_MODEL_SIZE_AND_STRIDE_EXPRESSION_H_
#define XLA_SERVICE_GPU_MODEL_SIZE_AND_STRIDE_EXPRESSION_H_

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "absl/types/span.h"
#include "mlir/IR/AffineExpr.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/service/gpu/model/constraint_expression.h"

namespace xla::gpu {

// Encapsulates expressions for size and stride and the corresponding
// constraints on the dimension values that need to be satisfied.
struct SizeAndStrideExpression {
  mlir::AffineExpr size;
  mlir::AffineExpr stride;
  ConstraintExpression constraints;

  explicit SizeAndStrideExpression(
      mlir::AffineExpr size, int64_t stride,
      ConstraintExpression constraints =
          ConstraintExpression::GetAlwaysSatisfied())
      : SizeAndStrideExpression(
            size, mlir::getAffineConstantExpr(stride, size.getContext()),
            std::move(constraints)) {}

  explicit SizeAndStrideExpression(
      mlir::AffineExpr size, mlir::AffineExpr stride,
      ConstraintExpression constraints =
          ConstraintExpression::GetAlwaysSatisfied())
      : size(size), stride(stride), constraints(std::move(constraints)) {}
};

// Given an expression `strided_indexing` that can be written as
// `stride_expr * index_expr`, attempts to produce stride and size expressions
// for the corresponding symbolic tile. A tile is a set of indices that can be
// represented as `offset + stride * index`, where `0 <= index < size`. A
// symbolic tile that corresponds to `strided_indexing` would map a tile of the
// "parameter space" (values plugged into `strided_indexing`) to a tile of the
// "value space" (with indices produced by `strided_indexing` when fed with the
// values from the "parameter space" tile). SizeAndStrideExpression also
// contains constraints that need to be satisfied to ensure that we only allow
// tiles from the "parameter space" that map to a set of indices in the "value
// space" that can be represented as a tile (with stride and size chosen
// according to the computed stride and size expressions from the
// SizeAndStrideExpression return value).
//
// `strided_indexing` should be an AffineExpr involving dimension ids between 0
// and `dimension_intervals.size() - 1`, and symbol ids between 0 and
// `symbol_intervals.size() - 1`. `dimension_intervals` specifies the valid
// range of values for the different dimension ids, `symbol_intervals` specifies
// the valid range of values for the different symbol ids. This method attempts
// to linearize the expression into `stride * index` and computes expressions
// for stride and tile size together with constraints on the values for the
// dimensions which need to be satisfied to make the expressions valid.
std::optional<SizeAndStrideExpression> ExtractSizeAndStride(
    mlir::AffineExpr strided_indexing,
    absl::Span<Interval const> dimension_intervals,
    absl::Span<Interval const> symbol_intervals);

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_MODEL_SIZE_AND_STRIDE_EXPRESSION_H_
