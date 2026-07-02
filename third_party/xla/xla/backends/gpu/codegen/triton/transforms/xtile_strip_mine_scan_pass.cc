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

#include <algorithm>
#include <cstdint>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/codegen/xtile/ir/xtile_ops.h"

namespace mlir::triton::xla {

#define GEN_PASS_DEF_XTILESTRIPMINESCANPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {

struct TilingParameters {
  bool has_tiling = true;
  bool force_tiling = false;
  int64_t tile_size = -1;
  SmallVector<::xla::xtile::ExtractTileOp> extract_ops;
  SmallVector<::xla::xtile::InsertTileOp> insert_ops;
};

static TilingParameters resolveTilingParameters(::xla::xtile::ScanOp op) {
  int32_t axis = op.getDimension();
  int64_t total_size = op.getScanDimSize();
  int64_t resolved_tile_size = -1;
  TilingParameters params;

  for (Value input : op.getInputs()) {
    auto extract_op = input.getDefiningOp<::xla::xtile::ExtractTileOp>();
    if (!extract_op) {
      params.has_tiling = false;
      break;
    }
    params.extract_ops.push_back(extract_op);
    int64_t t_size =
        mlir::cast<RankedTensorType>(extract_op.getType()).getDimSize(axis);
    if (resolved_tile_size == -1) {
      resolved_tile_size = t_size;
    } else if (resolved_tile_size != t_size) {
      params.has_tiling = false;
      break;
    }
  }

  constexpr int64_t kMaxScanTileSize = 128;
  if (params.has_tiling && resolved_tile_size > 0) {
    params.tile_size = std::min(resolved_tile_size, kMaxScanTileSize);
    if (params.tile_size < total_size) {
      params.force_tiling = true;
    }
  }

  if (params.has_tiling && params.force_tiling) {
    for (int i = 0; i < op.getInputs().size(); ++i) {
      Value result = op.getResult(i);
      if (result.use_empty()) {
        params.insert_ops.push_back(nullptr);
        continue;
      }
      ::xla::xtile::InsertTileOp insert_op;
      for (auto user : result.getUsers()) {
        if (auto maybe_insert = dyn_cast<::xla::xtile::InsertTileOp>(user)) {
          insert_op = maybe_insert;
          break;
        }
      }
      if (!insert_op) {
        params.force_tiling = false;
        break;
      }
      params.insert_ops.push_back(insert_op);
    }
  }

  return params;
}

// Wraps a scan op in a for-loop that tiles the scan dimension, such that the
// inner scan has a shorter scan dimension.
class TileScanPattern : public mlir::OpRewritePattern<::xla::xtile::ScanOp> {
 public:
  using OpRewritePattern<::xla::xtile::ScanOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      ::xla::xtile::ScanOp op, mlir::PatternRewriter& rewriter) const override {
    TilingParameters params = resolveTilingParameters(op);
    if (!params.has_tiling || !params.force_tiling) {
      return mlir::failure();  // We only handle tiled scans here.
    }

    int32_t axis = op.getDimension();
    int64_t total_size = op.getScanDimSize();
    int64_t tile_size = params.tile_size;
    int num_operands = op.getInputs().size();
    bool reverse = op.getIsReverse();

    Value lb = arith::ConstantIndexOp::create(rewriter, op.getLoc(), 0);
    Value ub =
        arith::ConstantIndexOp::create(rewriter, op.getLoc(), total_size);
    Value step =
        arith::ConstantIndexOp::create(rewriter, op.getLoc(), tile_size);

    auto for_op =
        scf::ForOp::create(rewriter, op.getLoc(), lb, ub, step, op.getInits());

    {
      mlir::OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(for_op.getBody());
      Value iv = for_op.getInductionVar();
      ValueRange iter_args = for_op.getRegionIterArgs();

      Value current_offset = iv;
      if (reverse) {
        Value total_val =
            arith::ConstantIndexOp::create(rewriter, op.getLoc(), total_size);
        Value tile_val =
            arith::ConstantIndexOp::create(rewriter, op.getLoc(), tile_size);
        Value rem =
            arith::SubIOp::create(rewriter, op.getLoc(), total_val, tile_val);
        current_offset = arith::SubIOp::create(rewriter, op.getLoc(), rem, iv);
      }

      SmallVector<Value> new_extracted_tiles;
      for (int i = 0; i < num_operands; ++i) {
        auto extract_op = params.extract_ops[i];
        SmallVector<Value> offsets(extract_op.getOffsets().begin(),
                                   extract_op.getOffsets().end());
        offsets[axis] = current_offset;

        auto old_type = mlir::cast<RankedTensorType>(extract_op.getType());
        auto shape = llvm::to_vector(old_type.getShape());
        shape[axis] = tile_size;
        auto tiled_type =
            RankedTensorType::get(shape, old_type.getElementType());

        auto old_full_tile_shape = extract_op.getFullTileShape();
        auto new_full_tile_shape_vec = llvm::to_vector(old_full_tile_shape);
        new_full_tile_shape_vec[axis] = tile_size;
        auto new_full_tile_shape_attr =
            rewriter.getDenseI64ArrayAttr(new_full_tile_shape_vec);

        auto new_extract = ::xla::xtile::ExtractTileOp::create(
            rewriter, op.getLoc(), tiled_type, extract_op.getSource(), offsets,
            new_full_tile_shape_attr, extract_op.getStrides());
        new_extracted_tiles.push_back(new_extract);
      }

      SmallVector<Type> adjusted_result_types;
      for (auto result : op.getOutputs()) {
        auto old_type = mlir::cast<RankedTensorType>(result.getType());
        auto shape = llvm::to_vector(old_type.getShape());
        shape[axis] = tile_size;
        auto tiled_type =
            RankedTensorType::get(shape, old_type.getElementType());
        adjusted_result_types.push_back(tiled_type);
      }

      SmallVector<Type> adjusted_carry_types;
      for (auto carry : op.getCarries()) {
        adjusted_carry_types.push_back(carry.getType());
      }

      // inner scan op over tiles
      auto inner_scan_op = ::xla::xtile::ScanOp::create(
          rewriter, op.getLoc(), adjusted_result_types, adjusted_carry_types,
          new_extracted_tiles, iter_args, axis, tile_size, op.getIsReverse());

      // Inline the block logic
      rewriter.inlineRegionBefore(op.getRegion(), inner_scan_op.getRegion(),
                                  inner_scan_op.getRegion().end());

      SmallVector<Value> inner_results = inner_scan_op.getOutputs();

      for (int i = 0; i < num_operands; ++i) {
        auto insert_op = params.insert_ops[i];
        if (!insert_op) continue;
        SmallVector<Value> offsets(insert_op.getOffsets().begin(),
                                   insert_op.getOffsets().end());
        offsets[axis] = current_offset;

        auto old_full_tile_shape = insert_op.getFullTileShape();
        auto new_full_tile_shape_vec = llvm::to_vector(old_full_tile_shape);
        new_full_tile_shape_vec[axis] = tile_size;
        auto new_full_tile_shape_attr =
            rewriter.getDenseI64ArrayAttr(new_full_tile_shape_vec);

        ::xla::xtile::InsertTileOp::create(
            rewriter, op.getLoc(), inner_results[i], insert_op.getDestination(),
            offsets, new_full_tile_shape_attr, insert_op.getStrides());
      }

      scf::YieldOp::create(rewriter, op.getLoc(), inner_scan_op.getCarries());
    }

    for (auto insert_op : params.insert_ops) {
      if (insert_op) {
        rewriter.eraseOp(insert_op);
      }
    }

    for (int i = 0; i < op.getCarries().size(); ++i) {
      rewriter.replaceAllUsesWith(op.getCarries()[i], for_op.getResult(i));
    }
    rewriter.eraseOp(op);
    for (auto extract_op : params.extract_ops) {
      rewriter.eraseOp(extract_op);
    }

    return mlir::success();
  }
};

class XTileStripMineScanPass
    : public impl::XTileStripMineScanPassBase<XTileStripMineScanPass> {
 public:
  void runOnOperation() override {
    mlir::MLIRContext* mlir_context = &getContext();
    mlir::RewritePatternSet tile_patterns(mlir_context);
    tile_patterns.add<TileScanPattern>(mlir_context);
    if (mlir::failed(mlir::applyPatternsGreedily(getOperation(),
                                                 std::move(tile_patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace mlir::triton::xla
