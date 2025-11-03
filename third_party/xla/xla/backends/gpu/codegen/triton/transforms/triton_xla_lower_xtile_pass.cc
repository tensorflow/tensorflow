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

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
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
#include "xla/backends/gpu/codegen/triton/emitter_helpers.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"
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
llvm::SmallVector<mlir::Type> GetPtrArgTypes(mlir::ValueRange args) {
  llvm::SmallVector<mlir::Type> arg_types;
  arg_types.reserve(args.size());
  for (auto arg : args) {
    mlir::MemRefType memref_type = mlir::cast<mlir::MemRefType>(arg.getType());
    arg_types.push_back(
        ::xla::gpu::triton::GetGlobalPointerType(memref_type.getElementType()));
  }
  return arg_types;
}

// Function to get the permutation vector from a MemRefType.
// The motivation for extracting it from getStridesAndOffset vs directly from
// triton_xla.layout is that when we fold memrefs (such as in a transpose) it
// will have a generic strided layout that does not directly encode the
// permutation.
absl::StatusOr<llvm::SmallVector<int64_t>> getPermutationMinorToMajor(
    mlir::MemRefType memref) {
  llvm::SmallVector<int64_t> strides;
  int64_t offset;
  if (memref.getStridesAndOffset(strides, offset).failed()) {
    // This can fail if the layout is not strided (e.g., has dynamic strides).
    return absl::InvalidArgumentError("Failed to get strides and offset");
  }

  llvm::SmallVector<int64_t> permutation;
  permutation.resize(strides.size());
  absl::c_iota(permutation, 0);

  absl::c_sort(permutation, [&](int64_t lhs_dim, int64_t rhs_dim) {
    int64_t lhs_stride = strides[lhs_dim];
    int64_t rhs_stride = strides[rhs_dim];
    if (lhs_stride != rhs_stride) {
      return lhs_stride < rhs_stride;
    }

    // If the strides are the same, we need to ensure that the unit dimension is
    // the more minor.
    int64_t lhs_size = memref.getDimSize(lhs_dim);
    int64_t rhs_size = memref.getDimSize(rhs_dim);
    if (lhs_size != rhs_size) {
      return lhs_size < rhs_size;
    }

    // If all else fails just sort in the canonical order.
    return lhs_dim > rhs_dim;
  });

  // Check that the strides actually represent a permutation,
  // this could happen for example with padded buffers.
  int64_t size_product = 1;
  for (int64_t dim : permutation) {
    if (strides[dim] != size_product) {
      return absl::InvalidArgumentError("Layout is not a valid permutation");
    }
    size_product *= memref.getDimSize(dim);
  }

  return permutation;
}

MemrefToPtrOp CreateMemrefToPtr(mlir::OpBuilder& builder,
                                mlir::TypedValue<mlir::MemRefType> memref) {
  PointerType ptr_type = ::xla::gpu::triton::GetGlobalPointerType(
      memref.getType().getElementType());
  return builder.create<MemrefToPtrOp>(memref.getLoc(), ptr_type, memref);
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

    auto new_arg_types = GetPtrArgTypes(entry_op.getBufferArgs());
    auto new_func_op = builder.create<mlir::func::FuncOp>(
        entry_op.getName(), builder.getFunctionType(new_arg_types, {}));

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

    auto pid = builder.create<ttir::GetProgramIdOp>(ttir::ProgramIDDim::X);
    Value pid_idx =
        builder.create<ma::IndexCastOp>(builder.getIndexType(), pid);
    rewriter.replaceAllUsesWith(tile_id_arg, pid_idx);

    // Handle memeref arguments.
    for (auto [old_arg, new_arg] : llvm::zip(old_args, new_args)) {
      mlir::MemRefType memref_type =
          mlir::cast<mlir::MemRefType>(old_arg.getType());

      mlir::Value memref_cast =
          builder.create<PtrToMemrefOp>(memref_type, new_arg);

      // Replace all uses of the old argument with the result of the cast.
      rewriter.replaceAllUsesWith(old_arg, memref_cast);
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

    if (extract_op.getType().getRank() == 0) {
      mlir::Value scalar_value = rewriter.create<ttir::LoadOp>(
          extract_op->getLoc(), memref_to_ptr, ttir::CacheModifier::NONE,
          ttir::EvictionPolicy::NORMAL, /*isVolatile=*/false);

      rewriter.replaceOpWithNewOp<::xla::xtile::ToTensorOp>(extract_op,
                                                            scalar_value);
      return mlir::success();
    }

    absl::StatusOr<SmallVector<int64_t>> minor_to_major_or =
        getPermutationMinorToMajor(source_type);
    if (!minor_to_major_or.ok()) {
      return rewriter.notifyMatchFailure(extract_op,
                                         minor_to_major_or.status().ToString());
    }
    const SmallVector<int64_t>& minor_to_major = *minor_to_major_or;
    auto triton_extract_op = rewriter.create<ExtractOp>(
        extract_op.getLoc(), result_type, memref_to_ptr,
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
      mlir::Value scalar_value = ::xla::xtile::ToScalarOp::create(
          rewriter, insert_op.getLoc(), insert_op.getSource());

      rewriter.replaceOpWithNewOp<ttir::StoreOp>(
          insert_op, memref_to_ptr, scalar_value, /*mask=*/nullptr);
      return mlir::success();
    }

    absl::StatusOr<SmallVector<int64_t>> minor_to_major_or =
        getPermutationMinorToMajor(destination_type);
    if (!minor_to_major_or.ok()) {
      return rewriter.notifyMatchFailure(insert_op,
                                         minor_to_major_or.status().ToString());
    }
    const SmallVector<int64_t>& minor_to_major = *minor_to_major_or;
    auto triton_insert_op = rewriter.create<InsertOp>(
        insert_op.getLoc(), insert_op.getSource(), memref_to_ptr,
        insert_op.getOffsets(), insert_op.getFullTileShape(),
        insert_op.getStrides(), destination_type.getShape(), minor_to_major);

    rewriter.replaceOp(insert_op, triton_insert_op);

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
                 FoldIntoMemrefToPtr>(context);
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
