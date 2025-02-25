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

std::optional<SizeAndStrideExpression> ExtractSizeAndStride(
    mlir::AffineExpr strided_indexing,
    absl::Span<Interval const> dimension_intervals,
    absl::Span<Interval const> symbol_intervals);

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_MODEL_SIZE_AND_STRIDE_EXPRESSION_H_
