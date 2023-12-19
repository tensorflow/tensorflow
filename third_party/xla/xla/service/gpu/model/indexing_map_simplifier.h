/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_GPU_MODEL_INDEXING_MAP_SIMPLIFIER_H_
#define XLA_SERVICE_GPU_MODEL_INDEXING_MAP_SIMPLIFIER_H_

#include <cstdint>
#include <functional>
#include <optional>
#include <string>

#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project

namespace xla {
namespace gpu {

class IndexingMapSimplifier {
 public:
  explicit IndexingMapSimplifier(mlir::MLIRContext* mlir_context)
      : mlir_context_(mlir_context) {}

  // Sets the inclusive bounds for the given expression. It can be used to set
  // bounds for dimensions and symbols.
  void SetInclusiveBounds(mlir::AffineExpr expr, int64_t lower, int64_t upper);

  // Simplifies the map as much as possible.
  mlir::AffineMap Simplify(mlir::AffineMap affine_map);

  // Simplifies the expression as much as possible.
  mlir::AffineExpr Simplify(mlir::AffineExpr expr);

 private:
  struct Bounds {
    int64_t lower;
    int64_t upper;
  };
  Bounds GetInclusiveBounds(mlir::AffineExpr expr);

  std::optional<int64_t> GetConstantRhsMultiplier(mlir::AffineExpr expr);

  // Simplifier for mod.
  // - Rewrites (a * 100 + ...) % 100 to (...) % 100
  // - Rewrites a % b to a if a is known to be less than b.
  mlir::AffineExpr RewriteMod(mlir::AffineBinaryOpExpr mod);

  // Simplifier for floordiv.
  // - Rewrites (a * 100 + ...) / 100 to a + (...) / 100
  // - Rewrites a / 100 to 0 when a is known to be less than 100.
  mlir::AffineExpr RewriteFloorDiv(mlir::AffineBinaryOpExpr div);

  mlir::AffineExpr RewriteSumIf(
      mlir::AffineExpr expr, const std::function<bool(mlir::AffineExpr)>& pred);

  // Attempts to simplify the expression, but doesn't attempt to simplify the
  // result further.
  mlir::AffineExpr SimplifyOnce(mlir::AffineExpr expr);

  mlir::MLIRContext* mlir_context_;
  llvm::DenseMap<mlir::AffineExpr, Bounds> bounds_{};
};

std::string ToString(const mlir::AffineMap& affine_map);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_INDEXING_MAP_SIMPLIFIER_H_
