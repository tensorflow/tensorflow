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

#include <functional>
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/tools/mlir_bisect/bisect_lib.h"
#include "tensorflow/compiler/xla/mlir_hlo/gml_st/IR/gml_st_ops.h"

namespace mlir {
namespace bisect {
namespace {

SmallVector<OwningOpRef<ModuleOp>> ReduceGmlStParallelBounds(
    BisectState&, gml_st::ParallelOp parallel_op) {
  SmallVector<OwningOpRef<ModuleOp>> result;
  for (int64_t i = 0; i < parallel_op.getUpperBound().size(); ++i) {
    if (!parallel_op.getUpperBound()[i]
             .getDefiningOp()
             ->hasTrait<OpTrait::ConstantLike>()) {
      continue;
    }

    auto [module, op] = CloneModuleFor(parallel_op);
    OpBuilder b(op);
    op.getUpperBoundMutable().slice(i, 1).assign(
        b.createOrFold<mlir::arith::SubIOp>(
            op->getLoc(), op.getUpperBound()[i],
            b.create<mlir::arith::ConstantIndexOp>(op->getLoc(), 1)));
    result.push_back(std::move(module));
  }
  return result;
}

REGISTER_MLIR_REDUCE_STRATEGY(ReduceGmlStParallelBounds);

}  // namespace
}  // namespace bisect
}  // namespace mlir
