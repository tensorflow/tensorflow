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

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Inliner.h"
#include "mlir/Transforms/InliningUtils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"
#include "xla/codegen/xtile/codegen/emitter_helpers.h"
#include "xla/codegen/xtile/ir/xtile_ops.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

namespace mlir::triton::xla {

#define GEN_PASS_DEF_TRITONXLALOWERXTILEPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {

namespace ttir = ::mlir::triton;
namespace ma = ::mlir::arith;

// Get the new arg types of the lowered function by translating memrefs to the
// corresponding pointer types.
llvm::SmallVector<mlir::Type> GetTransformedArgTypes(
    ::xla::xtile::EntryFuncOp& entry_op) {
  llvm::SmallVector<mlir::Type> arg_types;
  // Tile id is not carried over hence -1.
  arg_types.reserve(entry_op.getNumArguments() - 1U);
  for (const auto& arg : entry_op.getBufferArgs()) {
    mlir::MemRefType memref_type = mlir::cast<mlir::MemRefType>(arg.getType());
    arg_types.push_back(
        ttir::getPointerTypeToElement(memref_type.getElementType()));
  }
  mlir::TypeRange opaque_args(entry_op.getOpaqueArgs());
  arg_types.append(opaque_args.begin(), opaque_args.end());
  return arg_types;
}

MemrefToPtrOp CreateMemrefToPtr(mlir::OpBuilder& builder,
                                mlir::TypedValue<mlir::MemRefType> memref) {
  mlir::Type ptr_type =
      ttir::getPointerTypeToElement(memref.getType().getElementType());
  return MemrefToPtrOp::create(builder, memref.getLoc(), ptr_type, memref);
}

// Rewrite a xtile entry to a func.func with the same body, but with memref
// arguments replaced by pointers.
class XTileEntryToTriton
    : public mlir::OpRewritePattern<::xla::xtile::EntryFuncOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      ::xla::xtile::EntryFuncOp entry_op,
      mlir::PatternRewriter& rewriter) const override {
    mlir::ModuleOp module = entry_op->getParentOfType<mlir::ModuleOp>();
    mlir::ImplicitLocOpBuilder builder(module->getLoc(), module);
    builder.setInsertionPointToStart(module.getBody());

    const int64_t num_buffer_args = entry_op.getBufferArgs().size();
    auto new_arg_types = GetTransformedArgTypes(entry_op);
    auto new_func_op =
        mlir::func::FuncOp::create(builder, entry_op.getName(),
                                   builder.getFunctionType(new_arg_types, {}));

    // Move the old function's body to the new function
    rewriter.inlineRegionBefore(
        entry_op.getBody(), new_func_op.getFunctionBody(), new_func_op.end());

    Block& entry_block = new_func_op.front();
    builder.setInsertionPointToStart(&entry_block);

    SmallVector<BlockArgument> old_args(entry_block.getArguments());
    SmallVector<BlockArgument> new_args(entry_block.addArguments(
        new_arg_types,
        SmallVector<Location>(new_arg_types.size(), entry_op.getLoc())));

    BlockArgument tile_id_arg = old_args.back();

    auto pid = ttir::GetProgramIdOp::create(builder, ttir::ProgramIDDim::X);
    Value pid_idx =
        ma::IndexCastOp::create(builder, builder.getIndexType(), pid);
    rewriter.replaceAllUsesWith(tile_id_arg, pid_idx);

    // Handle memref arguments.
    for (auto [old_arg, new_arg] :
         llvm::zip(old_args,
                   mlir::ValueRange(new_args).take_front(num_buffer_args))) {
      mlir::MemRefType memref_type =
          mlir::cast<mlir::MemRefType>(old_arg.getType());

      mlir::Value memref_cast =
          PtrToMemrefOp::create(builder, memref_type, new_arg);

      // Replace all uses of the old argument with the result of the cast.
      rewriter.replaceAllUsesWith(old_arg, memref_cast);
    }
    // For opaque arguments, we can simply replace all uses with the new
    // argument.
    for (auto [old_arg, new_arg] :
         llvm::zip(mlir::ValueRange(old_args).drop_front(num_buffer_args),
                   mlir::ValueRange(new_args).drop_front(num_buffer_args))) {
      rewriter.replaceAllUsesWith(old_arg, new_arg);
    }

    entry_block.eraseArguments(0, old_args.size());

    rewriter.setInsertionPointToEnd(&entry_block);

    rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(
        entry_block.getTerminator());

    rewriter.eraseOp(entry_op);
    return success();
  }
};

// Rewrite a xtile extract to a triton_xla extract.
class XTileExtractToTriton
    : public mlir::OpRewritePattern<::xla::xtile::ExtractTileOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      ::xla::xtile::ExtractTileOp extract_op,
      mlir::PatternRewriter& rewriter) const override {
    mlir::MemRefType source_type = extract_op.getSource().getType();
    mlir::RankedTensorType result_type = extract_op.getType();

    mlir::Value memref_to_ptr =
        CreateMemrefToPtr(rewriter, extract_op.getSource());

    if (result_type.getRank() == 0) {
      mlir::Value scalar_value = ttir::LoadOp::create(
          rewriter, extract_op->getLoc(), memref_to_ptr,
          ttir::CacheModifier::NONE, ttir::EvictionPolicy::NORMAL,
          /*isVolatile=*/false);

      rewriter.replaceOpWithNewOp<mlir::tensor::FromElementsOp>(
          extract_op, result_type, scalar_value);
      return mlir::success();
    }

    absl::StatusOr<SmallVector<int64_t>> minor_to_major_or =
        ::xla::xtile::GetPermutationMinorToMajor(source_type);
    if (!minor_to_major_or.ok()) {
      return rewriter.notifyMatchFailure(extract_op,
                                         minor_to_major_or.status().ToString());
    }
    const SmallVector<int64_t>& minor_to_major = *minor_to_major_or;
    auto triton_extract_op = ExtractOp::create(
        rewriter, extract_op.getLoc(), result_type, memref_to_ptr,
        extract_op.getOffsets(), extract_op.getFullTileShape(),
        extract_op.getStrides(), source_type.getShape(), minor_to_major);

    rewriter.replaceOp(extract_op, triton_extract_op);

    return mlir::success();
  }
};

// Rewrite a xtile insert to a triton_xla insert.
class XTileInsertToTriton
    : public mlir::OpRewritePattern<::xla::xtile::InsertTileOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      ::xla::xtile::InsertTileOp insert_op,
      mlir::PatternRewriter& rewriter) const override {
    mlir::MemRefType destination_type = insert_op.getDestination().getType();

    mlir::Value memref_to_ptr =
        CreateMemrefToPtr(rewriter, insert_op.getDestination());

    if (insert_op.getSource().getType().getRank() == 0) {
      mlir::Value scalar_value = mlir::tensor::ExtractOp::create(
          rewriter, insert_op.getLoc(), insert_op.getSource());

      rewriter.replaceOpWithNewOp<ttir::StoreOp>(
          insert_op, memref_to_ptr, scalar_value, /*mask=*/nullptr);
      return mlir::success();
    }

    absl::StatusOr<SmallVector<int64_t>> minor_to_major_or =
        ::xla::xtile::GetPermutationMinorToMajor(destination_type);
    if (!minor_to_major_or.ok()) {
      return rewriter.notifyMatchFailure(insert_op,
                                         minor_to_major_or.status().ToString());
    }
    const SmallVector<int64_t>& minor_to_major = *minor_to_major_or;
    auto triton_insert_op = InsertOp::create(
        rewriter, insert_op.getLoc(), insert_op.getSource(), memref_to_ptr,
        insert_op.getOffsets(), insert_op.getFullTileShape(),
        insert_op.getStrides(), destination_type.getShape(), minor_to_major);

    rewriter.replaceOp(insert_op, triton_insert_op);

    return mlir::success();
  }
};

class XTileMaskToTriton : public mlir::OpRewritePattern<::xla::xtile::MaskOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      ::xla::xtile::MaskOp op, mlir::PatternRewriter& rewriter) const override {
    llvm::SmallVector<int64_t> masked_dimensions = op.getMaskedDimensions();
    if (masked_dimensions.size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "triton masking only supports masking over a single dimension");
    }

    int64_t mask_dimension = masked_dimensions.front();
    int64_t mask_bound = op.getBounds()[mask_dimension];
    int64_t masked_dim_size = op.getType().getDimSize(mask_dimension);
    auto iota_type =
        mlir::RankedTensorType::get(masked_dim_size, rewriter.getI32Type());
    auto range = stablehlo::IotaOp::create(rewriter, op.getLoc(), iota_type, 0);
    auto bcast_type = mlir::RankedTensorType::get(op.getType().getShape(),
                                                  iota_type.getElementType());
    auto bcast = stablehlo::BroadcastInDimOp::create(
        rewriter, op.getLoc(), bcast_type, range, {mask_dimension});
    auto constant = mlir::arith::ConstantOp::create(
        rewriter, op.getLoc(),
        mlir::DenseElementsAttr::get(bcast_type,
                                     rewriter.getI32IntegerAttr(mask_bound)));
    Value mask = arith::CmpIOp::create(
        rewriter, op.getLoc(), arith::CmpIPredicate::slt, bcast, constant);

    auto mask_value_tensor = mlir::tensor::FromElementsOp::create(
        rewriter, op.getLoc(),
        mlir::RankedTensorType::get({}, op.getValue().getType()),
        op.getValue());
    auto neutral = stablehlo::BroadcastInDimOp::create(
        rewriter, op.getLoc(), op.getType(), mask_value_tensor,
        ArrayRef<int64_t>{});

    rewriter.replaceOpWithNewOp<arith::SelectOp>(op, mask, op.getSource(),
                                                 neutral);

    return mlir::success();
  }
};

class FoldIntoMemrefToPtr : public mlir::OpRewritePattern<MemrefToPtrOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      MemrefToPtrOp op, mlir::PatternRewriter& rewriter) const override {
    // As a transpose doesn't add any offset we can simply fold it into the
    // memref_to_ptr.
    auto transpose = op.getSrc().getDefiningOp<mlir::memref::TransposeOp>();
    if (!transpose) {
      return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<MemrefToPtrOp>(op, op.getType(),
                                               transpose.getIn());
    return mlir::success();
  }
};

class TritonXLALowerXTilePass
    : public impl::TritonXLALowerXTilePassBase<TritonXLALowerXTilePass> {
 public:
  using TritonXLALowerXTilePassBase::TritonXLALowerXTilePassBase;

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext* context = &getContext();

    mlir::RewritePatternSet patterns(context);

    patterns.add<XTileEntryToTriton, XTileExtractToTriton, XTileInsertToTriton,
                 XTileMaskToTriton, FoldIntoMemrefToPtr>(context);
    if (mlir::failed(
            mlir::applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

}  // namespace

std::unique_ptr<Pass> CreateTritonXLALowerXTilePass() {
  return std::make_unique<TritonXLALowerXTilePass>();
}

}  // namespace mlir::triton::xla
