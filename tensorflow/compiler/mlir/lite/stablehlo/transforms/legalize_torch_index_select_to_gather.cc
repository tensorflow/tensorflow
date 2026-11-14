/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

// StableHLO port of MHLO's `-mhlo-legalize-torch-index-select-to-gather` pass
// (xla/mlir_hlo/mhlo/transforms/legalize_torch_index_select_to_gather).
// Rewrites `stablehlo.torch_index_select` (emitted e.g. by the TF->StableHLO
// legalization of `tf.GatherV2`) into an equivalent `stablehlo.gather`, which
// the HLO->TFL legalization handles.

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/stablehlo_passes.h"

namespace mlir {
namespace odml {
namespace {

#define GEN_PASS_DEF_LEGALIZESTABLEHLOTORCHINDEXSELECTTOGATHERPASS
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/stablehlo_passes.h.inc"

struct TorchIndexSelectToGatherPattern
    : public OpRewritePattern<stablehlo::TorchIndexSelectOp> {
  using OpRewritePattern<stablehlo::TorchIndexSelectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::TorchIndexSelectOp op,
                                PatternRewriter& rewriter) const override {
    Value operand = op.getOperand();
    Value index = op.getIndex();
    auto operand_type = mlir::cast<RankedTensorType>(operand.getType());
    auto index_type = mlir::cast<RankedTensorType>(index.getType());
    if (!operand_type.hasStaticShape() || !index_type.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op, "operand and index must have static shapes");
    }

    int64_t dim = static_cast<int64_t>(op.getDim());
    int64_t batch_dims = static_cast<int64_t>(op.getBatchDims());
    if (dim < batch_dims) {
      return rewriter.notifyMatchFailure(
          op, "dim must be greater than or equal to the number of batch dims");
    }

    if (!mlir::isa<IntegerType>(index_type.getElementType())) {
      return rewriter.notifyMatchFailure(
          op, "index must have integer element type");
    }

    int64_t index_vector_dim = index_type.getRank();
    if (index_type.getElementType().getIntOrFloatBitWidth() == 64 &&
        operand_type.getShape()[dim] < std::numeric_limits<uint32_t>::max()) {
      index = stablehlo::ConvertOp::create(
          rewriter, op.getLoc(), index,
          rewriter.getIntegerType(32, /*isSigned=*/false));
    }

    if (batch_dims > 0) {
      llvm::SmallVector<int64_t> new_index_shape(index_type.getShape());
      new_index_shape.push_back(1);
      auto new_index_type = RankedTensorType::get(
          new_index_shape, getElementTypeOrSelf(index.getType()));

      llvm::SmallVector<Value> to_concat;
      for (int64_t batch_dim = 0; batch_dim < batch_dims; ++batch_dim) {
        to_concat.push_back(stablehlo::IotaOp::create(
            rewriter, op.getLoc(), new_index_type, batch_dim));
      }
      to_concat.push_back(stablehlo::ReshapeOp::create(rewriter, op.getLoc(),
                                                       new_index_type, index));
      index = stablehlo::ConcatenateOp::create(
          rewriter, op.getLoc(), ValueRange(to_concat), index_vector_dim);
    }

    llvm::SmallVector<int64_t> offset_dims;
    llvm::SmallVector<int64_t> collapsed_slice_dims;
    llvm::SmallVector<int64_t> start_index_map;
    llvm::SmallVector<int64_t> slice_sizes(operand_type.getShape());
    for (int64_t i = 0; i < operand_type.getRank(); ++i) {
      if (i < batch_dims || i == dim) {
        slice_sizes[i] = std::min(slice_sizes[i], static_cast<int64_t>(1));
        collapsed_slice_dims.push_back(i);
        start_index_map.push_back(i);
      } else if (i < dim) {
        offset_dims.push_back(i);
      } else {
        offset_dims.push_back(i + index_vector_dim - (1 + batch_dims));
      }
    }

    auto dimension_numbers = stablehlo::GatherDimensionNumbersAttr::get(
        rewriter.getContext(), offset_dims, collapsed_slice_dims,
        /*operandBatchingDims=*/{}, /*startIndicesBatchingDims=*/{},
        start_index_map, index_vector_dim);
    rewriter.replaceOpWithNewOp<stablehlo::GatherOp>(
        op, op.getType(), operand, index, dimension_numbers,
        rewriter.getDenseI64ArrayAttr(slice_sizes),
        /*indices_are_sorted=*/rewriter.getBoolAttr(false));
    return success();
  }
};

class LegalizeStablehloTorchIndexSelectToGatherPass
    : public impl::LegalizeStablehloTorchIndexSelectToGatherPassBase<
          LegalizeStablehloTorchIndexSelectToGatherPass> {
 public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<TorchIndexSelectToGatherPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
CreateLegalizeStablehloTorchIndexSelectToGatherPass() {
  return std::make_unique<LegalizeStablehloTorchIndexSelectToGatherPass>();
}

static PassRegistration<LegalizeStablehloTorchIndexSelectToGatherPass> pass;

}  // namespace odml
}  // namespace mlir
