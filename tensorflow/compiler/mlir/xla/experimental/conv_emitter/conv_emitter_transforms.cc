/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/xla/experimental/conv_emitter/conv_emitter_transforms.h"

#include "absl/algorithm/container.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/Transforms/LoopUtils.h"  // from @llvm-project
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace experimental {

using mlir::OpBuilder;

BoundAffineMap GetBoundAffineMapFrom(mlir::Operation* op) {
  if (auto load = mlir::dyn_cast<mlir::AffineLoadOp>(op)) {
    return {load.getAffineMap(),
            std::vector<mlir::Value>(load.getMapOperands().begin(),
                                     load.getMapOperands().end())};
  } else if (auto store = mlir::dyn_cast<mlir::AffineStoreOp>(op)) {
    return {store.getAffineMap(),
            std::vector<mlir::Value>(store.getMapOperands().begin(),
                                     store.getMapOperands().end())};
  } else {
    CHECK(false);
  }
}

mlir::Operation* CloneWithNewAffineMap(mlir::Operation* op,
                                       BoundAffineMap new_affine,
                                       OpBuilder builder) {
  if (auto load = mlir::dyn_cast<mlir::AffineLoadOp>(op)) {
    return builder.create<mlir::AffineLoadOp>(
        builder.getUnknownLoc(), load.getMemRef(), new_affine.affine_map,
        new_affine.operands);
  } else if (auto store = mlir::dyn_cast<mlir::AffineStoreOp>(op)) {
    return builder.create<mlir::AffineStoreOp>(
        builder.getUnknownLoc(), store.getValueToStore(), store.getMemRef(),
        new_affine.affine_map, new_affine.operands);
  } else {
    CHECK(false);
  }
}

bool IsSimpleLoop(mlir::AffineForOp loop) {
  return loop.getLowerBoundMap().isSingleConstant() &&
         loop.getLowerBoundMap().getSingleConstantResult() == 0 &&
         loop.getStep() == 1 && loop.getUpperBoundMap().getNumResults() == 1 &&
         std::next(loop.region().begin()) == loop.region().end();
}

std::vector<mlir::AffineForOp> CreateNestedSimpleLoops(
    absl::Span<const int64_t> upper_bounds, OpBuilder builder) {
  std::vector<mlir::AffineForOp> loops;
  loops.reserve(upper_bounds.size());
  for (int64_t dim : upper_bounds) {
    auto loop =
        builder.create<mlir::AffineForOp>(builder.getUnknownLoc(), 0, dim);
    loops.push_back(loop);
    builder = OpBuilder::atBlockTerminator(loop.getBody());
  }
  return loops;
}

void SetBoundForSimpleLoop(mlir::AffineForOp loop, mlir::AffineExpr new_bound,
                           OpBuilder builder) {
  CHECK(IsSimpleLoop(loop));

  loop.setUpperBoundMap(mlir::AffineMap::get(
      loop.getUpperBoundMap().getNumDims(),
      loop.getUpperBoundMap().getNumSymbols(), {new_bound}));
}

mlir::AffineForOp TileLoop(mlir::AffineForOp loop, int64_t size,
                           mlir::AffineForOp target) {
  CHECK(IsSimpleLoop(loop));
  CHECK(IsSimpleLoop(target));
  {
    llvm::SmallVector<mlir::AffineForOp, 4> all_loops;
    getPerfectlyNestedLoops(all_loops, loop);
    CHECK(absl::c_linear_search(all_loops, target));
  }

  auto builder = OpBuilder::atBlockTerminator(target.getBody());

  auto inner_loop =
      builder.create<mlir::AffineForOp>(builder.getUnknownLoc(), 0, size);
  {
    auto& inner_operations = inner_loop.getBody()->getOperations();
    auto& target_operations = target.getBody()->getOperations();

    inner_operations.splice(inner_operations.begin(), target_operations,
                            target_operations.begin(),
                            std::prev(target_operations.end(), 2));

    mlir::AffineExpr length = loop.getUpperBoundMap().getResult(0);
    CHECK_EQ(0, length.cast<mlir::AffineConstantExpr>().getValue() % size);
    SetBoundForSimpleLoop(loop, length.ceilDiv(size), builder);
  }

  for (auto& use :
       llvm::make_early_inc_range(loop.getInductionVar().getUses())) {
    mlir::Operation* owner = use.getOwner();
    BoundAffineMap affine_map = GetBoundAffineMapFrom(owner);
    unsigned new_dim = affine_map.operands.size();
    affine_map.operands.push_back(inner_loop.getInductionVar());
    std::vector<mlir::AffineExpr> replacements;
    for (int i = 0; i < affine_map.affine_map.getNumDims(); i++) {
      if (affine_map.operands[i] == loop.getInductionVar()) {
        replacements.push_back(builder.getAffineDimExpr(i) * size +
                               builder.getAffineDimExpr(new_dim));
      } else {
        replacements.push_back(builder.getAffineDimExpr(i));
      }
    }
    affine_map.affine_map = affine_map.affine_map.replaceDimsAndSymbols(
        replacements, {}, affine_map.operands.size(), 0);
    auto new_op = CloneWithNewAffineMap(owner, affine_map, OpBuilder(owner));
    owner->replaceAllUsesWith(new_op);
    owner->erase();
  }
  return inner_loop;
}

void SinkPerfectlyNestedLoops(llvm::MutableArrayRef<mlir::AffineForOp> loops,
                              int rotate_amount) {
  CHECK_GE(rotate_amount, 0);
  std::vector<unsigned> permutation(loops.size());
  std::iota(permutation.begin(), permutation.end(), unsigned(0));
  std::rotate(permutation.begin(),
              permutation.begin() + loops.size() - rotate_amount,
              permutation.end());
  mlir::permuteLoops(loops, permutation);
}

}  // namespace experimental
}  // namespace xla
