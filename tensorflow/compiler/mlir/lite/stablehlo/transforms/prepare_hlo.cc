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

#include <cstdint>
#include <memory>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/conv_util.h"  // IWYU pragma: keep
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"  // IWYU pragma: keep

namespace mlir {
namespace odml {
namespace {

llvm::SmallVector<int64_t> UnrollSplat(DenseElementsAttr data) {
  if (!data.isSplat()) {
    return llvm::SmallVector<int64_t>(data.getValues<int64_t>());
  }
  return llvm::SmallVector<int64_t>(data.getType().getNumElements(),
                                    data.getSplatValue<int64_t>());
}

llvm::SmallVector<int64_t> SliceStartFromNegPadLows(DenseElementsAttr lows) {
  auto vals = UnrollSplat(lows);
  auto starts = llvm::map_range(
      vals, [](auto v) -> int64_t { return (v >= 0) ? 0 : -1 * v; });
  return llvm::to_vector(starts);
}

llvm::SmallVector<int64_t> SliceEndFromNegPadHighs(DenseElementsAttr highs,
                                                   ArrayRef<int64_t> shape) {
  auto vals = UnrollSplat(highs);
  auto zip = llvm::zip(vals, shape);
  auto ends = llvm::map_range(zip, [](auto it) -> int64_t {
    return (std::get<0>(it) >= 0) ? std::get<1>(it)
                                  : std::get<1>(it) + std::get<0>(it);
  });
  return llvm::to_vector(ends);
}

llvm::SmallVector<int64_t> ReplaceNegsWithZero(DenseElementsAttr data) {
  auto vals = UnrollSplat(data);
  auto res =
      llvm::map_range(vals, [](auto v) -> int64_t { return (v < 0) ? 0 : v; });
  return llvm::to_vector(res);
}

bool AnyNegativePads(DenseElementsAttr lows, DenseElementsAttr highs) {
  auto is_neg = [](int64_t v) { return v < 0; };
  auto lows_data = UnrollSplat(lows);
  auto highs_data = UnrollSplat(highs);
  return llvm::any_of(lows_data, is_neg) || llvm::any_of(highs_data, is_neg);
}

#define GEN_PASS_DEF_PREPAREHLOPASS
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/passes.h.inc"

class PrepareHloPass : public impl::PrepareHloPassBase<PrepareHloPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrepareHloPass);

  void runOnOperation() override;
};

#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/generated_prepare_hlo.inc"
void PrepareHloPass::runOnOperation() {
  MLIRContext* context = &getContext();
  auto func = getOperation();

  RewritePatternSet patterns(context);
  populateWithGenerated(patterns);

  if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
    func->dump();
    signalPassFailure();
  }
}
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreatePrepareHloPass() {
  return std::make_unique<PrepareHloPass>();
}

static PassRegistration<PrepareHloPass> pass;

}  // namespace odml
}  // namespace mlir
