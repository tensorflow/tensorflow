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

#include <cstddef>
#include <cstdint>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/gpu/codegen/triton/transforms/passes.h"  // IWYU pragma: keep
#include "xla/codegen/xtile/ir/xtile_ops.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::xla {

#define GEN_PASS_DEF_XTILESTRIPMINESCANPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {

struct StripMineScanPattern
    : public mlir::OpRewritePattern<::xla::xtile::InsertTileOp> {
  StripMineScanPattern(mlir::MLIRContext* context, int64_t tile_size)
      : OpRewritePattern<::xla::xtile::InsertTileOp>(context),
        tile_size(tile_size) {}

 private:
  mlir::LogicalResult matchAndRewrite(
      ::xla::xtile::InsertTileOp insert_op,
      mlir::PatternRewriter& rewriter) const override {
    auto cluster_or = IdentifyCluster(insert_op);
    if (mlir::failed(cluster_or)) {
      return mlir::failure();
    }
    auto [cluster, scan_op] = *std::move(cluster_or);

    if (mlir::failed(ValidateCluster(cluster, scan_op))) {
      return mlir::failure();
    }

    int64_t scan_dim_size = scan_op.getScanDimSize();
    if (scan_dim_size <= tile_size) {
      return mlir::failure();
    }

    RewriteCluster(scan_op, insert_op, cluster, tile_size, rewriter);
    return mlir::success();
  }

  static RankedTensorType GetTiledType(RankedTensorType type, int32_t axis,
                                       int64_t tile_size) {
    auto shape = llvm::to_vector(type.getShape());
    shape[axis] = tile_size;
    return RankedTensorType::get(shape, type.getElementType());
  }

  static SmallVector<Value> GetMappedValues(ValueRange values,
                                            IRMapping& mapping) {
    SmallVector<Value> mapped;
    mapped.reserve(values.size());
    for (Value v : values) {
      mapped.push_back(mapping.lookupOrDefault(v));
    }
    return mapped;
  }

  static mlir::FailureOr<std::pair<SetVector<Operation*>, ::xla::xtile::ScanOp>>
  IdentifyCluster(::xla::xtile::InsertTileOp insert_op) {
    // Allow xtile.extract_tile/scan/insert_tile and element-wise ops on tensors
    // inside the cluster.
    auto is_cluster_op = [&](Operation* op) {
      if (op->getBlock() != insert_op->getBlock()) {
        return false;
      }

      if (isa<::xla::xtile::ExtractTileOp, ::xla::xtile::InsertTileOp,
              ::xla::xtile::ScanOp>(op)) {
        return true;
      }

      if (!op->hasTrait<mlir::OpTrait::Elementwise>()) {
        return false;
      }

      return llvm::all_of(op->getResultTypes(), [](Type t) {
        return mlir::isa<RankedTensorType>(t);
      });
    };

    BackwardSliceOptions back_opts;
    back_opts.filter = is_cluster_op;
    back_opts.inclusive = true;

    SetVector<Operation*> cluster;
    if (mlir::failed(mlir::getBackwardSlice(insert_op.getSource(), &cluster,
                                            back_opts))) {
      return mlir::failure();
    }
    cluster.insert(insert_op);

    auto scan_op = dyn_cast_or_null<::xla::xtile::ScanOp>(
        llvm::find_singleton<Operation>(cluster, [](Operation* op, bool) {
          return dyn_cast<::xla::xtile::ScanOp>(op);
        }));
    if (!scan_op) {
      return mlir::failure();
    }

    // Do not include ops that are only used to produce scan inits.
    SetVector<Operation*> init_slice;
    for (Value init : scan_op.getInits()) {
      if (mlir::failed(mlir::getBackwardSlice(init, &init_slice, back_opts))) {
        return mlir::failure();
      }
    }
    for (Operation* init_op : init_slice) {
      cluster.remove(init_op);
    }

    // Also do not include ops that produce insert_op's destination.
    SetVector<Operation*> dest_slice;
    if (mlir::failed(mlir::getBackwardSlice(insert_op.getDestination(),
                                            &dest_slice, back_opts))) {
      return mlir::failure();
    }
    for (Operation* dest_op : dest_slice) {
      cluster.remove(dest_op);
    }

    SetVector<Operation*> sorted_cluster;
    for (Operation& op : *insert_op->getBlock()) {
      if (cluster.contains(&op)) {
        sorted_cluster.insert(&op);
      }
    }
    return std::make_pair(sorted_cluster, scan_op);
  }

  static mlir::LogicalResult ValidateCluster(
      const SetVector<Operation*>& cluster, ::xla::xtile::ScanOp scan_op) {
    bool has_insert = false, has_extract = false;

    for (Operation* op : cluster) {
      if (isa<::xla::xtile::InsertTileOp>(op)) {
        if (has_insert) {
          return mlir::failure();
        }
        has_insert = true;
        // The InsertTileOp is allowed to have uses outside the cluster.
        continue;
      }

      if (isa<::xla::xtile::ExtractTileOp>(op)) {
        has_extract = true;
      }

      if (!isa<::xla::xtile::ExtractTileOp, ::xla::xtile::ScanOp>(op)) {
        // Enforce all tensor inputs to elementwise ops are also inside the
        // cluster
        for (Value operand : op->getOperands()) {
          if (!mlir::isa<RankedTensorType>(operand.getType())) {
            continue;
          }

          Operation* def_op = operand.getDefiningOp();
          if (!def_op || !cluster.contains(def_op)) {
            return mlir::failure();
          }
        }
      }

      ResultRange results = op->getResults();
      if (op == scan_op) {
        // ScanOp carries are allowed to be used outside the cluster.
        results = scan_op.getOutputs();
      }

      for (auto result : results) {
        for (Operation* user : result.getUsers()) {
          if (!cluster.contains(user)) {
            return mlir::failure();
          }
        }
      }
    }

    if (!has_insert || !has_extract || !scan_op) {
      return mlir::failure();
    }

    return mlir::success();
  }

  static std::pair<SmallVector<Value>, mlir::DenseI64ArrayAttr>
  UpdateOffsetsAndShape(ValueRange old_offsets, ArrayRef<int64_t> old_shape,
                        mlir::PatternRewriter& rewriter, IRMapping& mapping,
                        int32_t axis, int64_t tile_size, Value current_offset,
                        Location loc) {
    SmallVector<Value> offsets = GetMappedValues(old_offsets, mapping);
    Value base_offset = offsets[axis];
    offsets[axis] =
        arith::AddIOp::create(rewriter, loc, base_offset, current_offset);

    auto new_shape_vec = llvm::to_vector(old_shape);
    new_shape_vec[axis] = tile_size;
    auto new_shape_attr = rewriter.getDenseI64ArrayAttr(new_shape_vec);

    return {offsets, new_shape_attr};
  }

  static void RewriteExtractOp(::xla::xtile::ExtractTileOp extract_op,
                               mlir::PatternRewriter& rewriter,
                               IRMapping& mapping, int32_t axis,
                               int64_t tile_size, Value current_offset) {
    auto [offsets, new_full_tile_shape_attr] = UpdateOffsetsAndShape(
        extract_op.getOffsets(), extract_op.getFullTileShape(), rewriter,
        mapping, axis, tile_size, current_offset, extract_op.getLoc());

    auto old_type = mlir::cast<RankedTensorType>(extract_op.getType());
    auto tiled_type = GetTiledType(old_type, axis, tile_size);

    auto new_extract = ::xla::xtile::ExtractTileOp::create(
        rewriter, extract_op.getLoc(), tiled_type,
        mapping.lookupOrDefault(extract_op.getSource()), offsets,
        new_full_tile_shape_attr, extract_op.getStrides());
    mapping.map(extract_op.getResult(), new_extract.getResult());
  }

  static ::xla::xtile::InsertTileOp RewriteInsertOp(
      ::xla::xtile::InsertTileOp insert_op, mlir::PatternRewriter& rewriter,
      IRMapping& mapping, int32_t axis, int64_t tile_size,
      Value current_offset) {
    auto [offsets, new_full_tile_shape_attr] = UpdateOffsetsAndShape(
        insert_op.getOffsets(), insert_op.getFullTileShape(), rewriter, mapping,
        axis, tile_size, current_offset, insert_op.getLoc());

    return ::xla::xtile::InsertTileOp::create(
        rewriter, insert_op.getLoc(),
        mapping.lookupOrDefault(insert_op.getSource()),
        mapping.lookupOrDefault(insert_op.getDestination()), offsets,
        new_full_tile_shape_attr, insert_op.getStrides());
  }

  static void RewriteScanOp(::xla::xtile::ScanOp original_scan,
                            mlir::PatternRewriter& rewriter, IRMapping& mapping,
                            int32_t axis, int64_t tile_size,
                            Value current_offset, int64_t total_size,
                            ValueRange active_inits) {
    SmallVector<Value> new_inputs =
        GetMappedValues(original_scan.getInputs(), mapping);
    SmallVector<Value> new_inits(active_inits.begin(), active_inits.end());
    SmallVector<Value> original_inits = original_scan.getInits();

    mlir::ImplicitLocOpBuilder b(original_scan.getLoc(), rewriter);
    auto i32_type = b.getI32Type();
    auto i32_row_type = RankedTensorType::get({tile_size}, i32_type);

    Value range = triton::MakeRangeOp::create(b, i32_row_type, 0, tile_size);
    Value offset_i32 = arith::IndexCastOp::create(b, i32_type, current_offset);
    Value splat_offset = triton::SplatOp::create(b, i32_row_type, offset_i32);
    Value indices = arith::AddIOp::create(b, range, splat_offset);

    Value bounds_val =
        arith::ConstantOp::create(b, b.getI32IntegerAttr(total_size));
    Value splat_bounds = triton::SplatOp::create(b, i32_row_type, bounds_val);
    Value mask_1d = arith::CmpIOp::create(b, arith::CmpIPredicate::slt, indices,
                                          splat_bounds);

    SmallVector<Value> padded_inputs;
    padded_inputs.reserve(new_inputs.size());
    for (size_t i = 0; i < new_inputs.size(); ++i) {
      Value input = new_inputs[i];
      Value init = original_inits[i];
      auto input_type = mlir::cast<RankedTensorType>(input.getType());

      Value broadcast_mask = mask_1d;
      for (int d = 0; d < input_type.getRank(); ++d) {
        if (d != axis) {
          broadcast_mask = triton::ExpandDimsOp::create(b, broadcast_mask, d);
        }
      }
      auto mask_type =
          RankedTensorType::get(input_type.getShape(), b.getI1Type());
      broadcast_mask =
          triton::BroadcastOp::create(b, mask_type, broadcast_mask);

      Value padding_value;
      if (!mlir::isa<RankedTensorType>(init.getType())) {
        padding_value = triton::SplatOp::create(b, input_type, init);
      } else if (init.getType() != input_type) {
        padding_value = triton::ExpandDimsOp::create(b, init, axis);
        padding_value =
            triton::BroadcastOp::create(b, input_type, padding_value);
      } else {
        padding_value = init;
      }

      padded_inputs.push_back(
          arith::SelectOp::create(b, broadcast_mask, input, padding_value));
    }

    SmallVector<Type> adjusted_result_types;
    for (auto result : original_scan.getOutputs()) {
      auto old_type = mlir::cast<RankedTensorType>(result.getType());
      adjusted_result_types.push_back(GetTiledType(old_type, axis, tile_size));
    }

    auto carries_type_range = original_scan.getCarries().getTypes();
    SmallVector<Type> adjusted_carry_types(carries_type_range.begin(),
                                           carries_type_range.end());

    auto inner_scan_op = ::xla::xtile::ScanOp::create(
        rewriter, original_scan.getLoc(), adjusted_result_types,
        adjusted_carry_types, padded_inputs, new_inits, axis, tile_size,
        original_scan.getIsReverse());

    rewriter.inlineRegionBefore(original_scan.getRegion(),
                                inner_scan_op.getRegion(),
                                inner_scan_op.getRegion().end());

    for (auto [old_res, new_res] :
         llvm::zip(original_scan.getOutputs(), inner_scan_op.getOutputs())) {
      mapping.map(old_res, new_res);
    }
    for (auto [old_res, new_res] :
         llvm::zip(original_scan.getCarries(), inner_scan_op.getCarries())) {
      mapping.map(old_res, new_res);
    }
  }

  static void RewriteElementwiseOp(Operation* op,
                                   mlir::PatternRewriter& rewriter,
                                   IRMapping& mapping, int32_t axis,
                                   int64_t tile_size) {
    Operation* cloned = rewriter.clone(*op, mapping);
    for (auto result : cloned->getResults()) {
      if (auto old_type = mlir::dyn_cast<RankedTensorType>(result.getType())) {
        result.setType(GetTiledType(old_type, axis, tile_size));
      }
    }
    for (auto [old_res, new_res] :
         llvm::zip(op->getResults(), cloned->getResults())) {
      mapping.map(old_res, new_res);
    }
  }

  static void RewriteCluster(::xla::xtile::ScanOp scan_op,
                             ::xla::xtile::InsertTileOp root_insert_op,
                             const SetVector<Operation*>& cluster,
                             int64_t tile_size,
                             mlir::PatternRewriter& rewriter) {
    int32_t axis = scan_op.getDimension();
    int64_t total_size = scan_op.getScanDimSize();

    Value lb = arith::ConstantIndexOp::create(rewriter, scan_op.getLoc(), 0);
    Value ub =
        arith::ConstantIndexOp::create(rewriter, scan_op.getLoc(), total_size);
    Value step =
        arith::ConstantIndexOp::create(rewriter, scan_op.getLoc(), tile_size);

    int64_t num_tiles = (total_size + tile_size - 1) / tile_size;
    Value max_iv;
    if (scan_op.getIsReverse()) {
      max_iv = arith::ConstantIndexOp::create(rewriter, scan_op.getLoc(),
                                              (num_tiles - 1) * tile_size);
    }

    bool has_tensor_result = root_insert_op->getNumResults() > 0;
    SmallVector<Value> iter_args(scan_op.getInits());
    if (has_tensor_result) {
      iter_args.push_back(root_insert_op.getDestination());
    }

    auto for_op =
        scf::ForOp::create(rewriter, scan_op.getLoc(), lb, ub, step, iter_args);

    {
      mlir::OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(for_op.getBody());
      Value iv = for_op.getInductionVar();

      Value current_offset = iv;
      if (scan_op.getIsReverse()) {
        current_offset =
            arith::SubIOp::create(rewriter, scan_op.getLoc(), max_iv, iv);
      }

      IRMapping mapping;
      auto block_args = for_op.getRegionIterArgs();

      if (has_tensor_result) {
        mapping.map(root_insert_op.getDestination(), block_args.back());
      }

      ::xla::xtile::InsertTileOp new_insert;

      for (Operation* op : cluster) {
        if (auto extract_op = dyn_cast<::xla::xtile::ExtractTileOp>(op)) {
          RewriteExtractOp(extract_op, rewriter, mapping, axis, tile_size,
                           current_offset);
        } else if (auto insert_op = dyn_cast<::xla::xtile::InsertTileOp>(op)) {
          new_insert = RewriteInsertOp(insert_op, rewriter, mapping, axis,
                                       tile_size, current_offset);
        } else if (auto original_scan = dyn_cast<::xla::xtile::ScanOp>(op)) {
          RewriteScanOp(original_scan, rewriter, mapping, axis, tile_size,
                        current_offset, total_size,
                        block_args.take_front(scan_op.getInits().size()));
        } else {
          RewriteElementwiseOp(op, rewriter, mapping, axis, tile_size);
        }
      }

      SmallVector<Value> yielded_values =
          GetMappedValues(scan_op.getCarries(), mapping);
      if (has_tensor_result) {
        yielded_values.push_back(new_insert->getResult(0));
      }
      scf::YieldOp::create(rewriter, scan_op.getLoc(), yielded_values);
    }

    if (has_tensor_result) {
      rewriter.replaceOp(root_insert_op, for_op.getResults().back());
    } else {
      rewriter.eraseOp(root_insert_op);
    }

    for (Operation* op : llvm::reverse(cluster)) {
      if (op == root_insert_op) {
        continue;
      }
      if (auto scan_op_to_erase = dyn_cast<::xla::xtile::ScanOp>(op)) {
        for (auto [idx, carry] :
             llvm::enumerate(scan_op_to_erase.getCarries())) {
          rewriter.replaceAllUsesWith(carry, for_op.getResult(idx));
        }
      }
      rewriter.eraseOp(op);
    }
  }

 private:
  int64_t tile_size;
};

class XTileStripMineScanPass
    : public impl::XTileStripMineScanPassBase<XTileStripMineScanPass> {
 public:
  using impl::XTileStripMineScanPassBase<
      XTileStripMineScanPass>::XTileStripMineScanPassBase;

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<StripMineScanPattern>(&getContext(), tile_size_);
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace mlir::triton::xla
