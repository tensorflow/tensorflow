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
#include <memory>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include "triton/Dialect/Triton/IR/Dialect.h"
// Above header needs to be included first to avoid 'major' macro collision.

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"
#include "xla/backends/gpu/codegen/triton/lowering_util.h"
#include "xla/backends/gpu/codegen/triton/transforms/passes.h"
#include "xla/codegen/emitters/ir/xla_ops.h"  // IWYU pragma: keep
#include "xla/codegen/xtile/codegen/emitter_helpers.h"
#include "xla/permutation_util.h"
#include "xla/stream_executor/gpu/tma_metadata.h"
#include "triton/Dialect/Triton/IR/Types.h"

namespace mlir::triton::xla {

#define GEN_PASS_DEF_TRITONXLAEXTRACTINSERTTOTRITONPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace xtile = ::xla::xtile;
namespace xtriton = ::xla::gpu::triton;

namespace {

bool HasBroadcastConsumer(Operation* op) {
  llvm::SetVector<Operation*> slice;
  mlir::getForwardSlice(op, &slice);
  for (Operation* sliced_op : slice) {
    if (llvm::isa<triton::BroadcastOp>(sliced_op)) {
      return true;
    }
  }
  return false;
}

PointerType GetTensorPtrType(Type type) {
  return PointerType::get(
      xtile::StorageType(type),
      static_cast<unsigned>(mlir::NVVM::NVVMMemorySpace::Global));
}

// Canonicalizes tile strides. Currently this converts zero strides to 1.
// If validation is requested and a tile stride is 0:
// If the corresponding tile shape or original shape value at the same index is
// 1, then the tile stride is set to 1. Otherwise, it returns an error.
absl::Status CanonicalizeTileStrides(SmallVector<int64_t>& tile_strides,
                                     const ArrayRef<int64_t>& tile_shape,
                                     const ArrayRef<int64_t>& original_shape,
                                     bool validate = true) {
  for (int64_t i = 0; i < tile_strides.size(); ++i) {
    if (tile_strides[i] == 0) {
      if (validate && tile_shape[i] != 1 && original_shape[i] != 1) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "tile_stride at index %d is 0, but tile_shape at the same "
            "index is %d, and original_shape at the same index is %d. Expected "
            "tile_shape or original_shape to be 1 at that index.",
            i, tile_shape[i], original_shape[i]));
      }
      tile_strides[i] = 1;
    }
  }
  return absl::OkStatus();
}

// Check if the offset is divisible by 16 bytes:
//  - If the offset is a constant, we can check this directly.
//  - If the offset is the result of an apply indexing op, we can check if the
//    indexing map is divisible.
// TODO(b/435099668): Make the filter cover more cases. E.g.:
//  - Offsets from other operations like add, mul, etc.
//  - Potentially trace back beyond apply_indexing to prune the domain.
bool IsOffsetDivisibilityGuaranteed(mlir::Value offset_val,
                                    int64_t element_byte_size) {
  const int64_t kByteDivisibilityFactor = 16;
  int64_t divisor = kByteDivisibilityFactor /
                    std::gcd(kByteDivisibilityFactor, element_byte_size);
  return xtriton::IsGuaranteedDivisible(offset_val, divisor);
}

// Limitations of TMA are documented in IsTmaCompatible
// in third_party/tensorflow/compiler/xla/stream_executor/gpu/tma_metadata.h.
// Additionally:
// - UNDOCUMENTED LIMITATION (informed by Nvidia in chat):
//      - The address we load/store from (base + offset) must be divisible
//      by 16. Since we already check that both the global strides and most
//      minor tile dimension (in bytes) must be divisible by 16, it is
//      sufficient to check that the offset in the minor dimension (in bytes) is
//      divisible by 16.
bool CanUseTma(Operation* op, bool allow_tma, int num_stages,
               const ArrayRef<int64_t>& original_shape,
               const ArrayRef<int64_t>& tile_shape,
               const ArrayRef<int64_t>& tile_strides, ValueRange offsets,
               const TypedValue<PointerType>& pointer,
               const ArrayRef<int64_t>& minor_to_major_layout) {
  if (!allow_tma) {
    return false;
  }

  // We only enable TMA for inputs that have one use only.
  auto block_arg = mlir::dyn_cast<BlockArgument>(pointer);
  if (!block_arg || !block_arg.hasOneUse()) {
    return false;
  }
  auto func_op =
      mlir::dyn_cast<func::FuncOp>(block_arg.getOwner()->getParentOp());
  if (!func_op) {
    return false;
  }

  // TODO(b/421858850): CUDA_ERROR_MISALIGNED_ADDRESS errors are
  // happening for some cases when pipelining stages are > 2. The pattern
  // observed is that these happen in the presence of a broadcast.
  // This is a temporary solution. We should remove this once we have a fix for
  // the error.
  if (num_stages > 2 && HasBroadcastConsumer(op)) {
    return false;
  }

  // Some TMA constraints can't be validated if tile strides are dynamic.
  if (mlir::ShapedType::isDynamicShape(tile_strides)) {
    return false;
  }

  // Canonicalize without validation since we are still not sure if TMA will be
  // used or not.
  SmallVector<int64_t> canonical_tile_strides(tile_strides.begin(),
                                              tile_strides.end());
  // TODO(csigg): canonicalize_status is ignored.
  auto canonicalize_status = CanonicalizeTileStrides(canonical_tile_strides,
                                                     tile_shape, original_shape,
                                                     /*validate=*/false);

  uint64_t element_byte_size =
      pointer.getType().getPointeeType().getIntOrFloatBitWidth() / 8;

  auto tma_compatibilty_status = stream_executor::gpu::IsTmaCompatible(
      absl::MakeSpan(original_shape.data(), original_shape.size()),
      absl::MakeSpan(tile_shape.data(), tile_shape.size()),
      absl::MakeSpan(canonical_tile_strides.data(),
                     canonical_tile_strides.size()),
      absl::MakeSpan(minor_to_major_layout.data(),
                     minor_to_major_layout.size()),
      element_byte_size);
  if (!tma_compatibilty_status.ok()) {
    VLOG(1) << "TMA is not compatible for this argument. Reason: "
            << tma_compatibilty_status.message();
    return false;
  }

  // Validate minor dimension offset.
  if (!IsOffsetDivisibilityGuaranteed(offsets[minor_to_major_layout[0]],
                                      element_byte_size)) {
    return false;
  }
  return true;
}

// Add TMA attributes to the corresponding argument in the function.
void AddTmaAttributes(ImplicitLocOpBuilder& builder,
                      const TypedValue<PointerType>& pointer,
                      const ArrayRef<int64_t>& original_shape,
                      const ArrayRef<int64_t>& layout,
                      const ArrayRef<int64_t>& tile_shape,
                      const ArrayRef<int64_t>& tile_strides) {
  auto block_arg = mlir::dyn_cast<BlockArgument>(pointer);
  auto func_op =
      mlir::dyn_cast<func::FuncOp>(block_arg.getOwner()->getParentOp());
  func_op.setArgAttr(block_arg.getArgNumber(), "tt.nv_tma_desc",
                     builder.getI32IntegerAttr(1));
  // Prefixing the attribute name with "tt", otherwise tt.func will
  // complain that it is not part of the dialect. Not the best way to
  // do this, but it works for now.
  func_op.setArgAttr(
      block_arg.getArgNumber(), "tt.tma_descriptor",
      builder.getAttr<TmaDescriptorAttr>(
          original_shape, tile_shape, tile_strides, layout,
          pointer.getType().getPointeeType().getIntOrFloatBitWidth() / 8));
}

// Checks whether 'layout' is the default HLO layout in major-to-minor order,
// i.e. iff it's [N-1, N-2, ... 1, 0].
bool IsMajorToMinorLayout(ArrayRef<int64_t> layout) {
  for (auto [i, value] : llvm::enumerate(layout)) {
    if (value != layout.size() - 1 - i) {
      return false;
    }
  }
  return true;
}

// Returns 'values' in major-to-minor order given minor-to-major 'layout'.
template <typename T>
SmallVector<T> GetMajorToMinorOrder(ArrayRef<T> values,
                                    ArrayRef<int64_t> layout) {
  if (IsMajorToMinorLayout(layout)) {
    return llvm::to_vector(values);
  }

  auto reversed_layout = llvm::to_vector(layout);
  std::reverse(reversed_layout.begin(), reversed_layout.end());
  std::vector<T> vector = ::xla::Permute(values, reversed_layout);
  return SmallVector<T>(vector.begin(), vector.end());
}

// Returns 'values' in major-to-minor order given minor-to-major 'layout'.
SmallVector<Value> GetMajorToMinorOrder(ValueRange values,
                                        ArrayRef<int64_t> layout) {
  return GetMajorToMinorOrder(ArrayRef<Value>(llvm::to_vector(values)), layout);
}

// Given the layout of a tensor, return the inverse permutation required to
// transpose an already major-to-minor tensor to the original tensor.
SmallVector<int32_t> GetInverseLayoutPermutation(ArrayRef<int64_t> layout) {
  auto reversed_layout = llvm::to_vector(layout);
  std::reverse(reversed_layout.begin(), reversed_layout.end());
  auto permutation =
      llvm::to_vector_of<int32_t>(::xla::InversePermutation(reversed_layout));
  return SmallVector<int32_t>(permutation.begin(), permutation.end());
}

// Rewrite func.func to tt.func.
class RewriteFuncOp : public mlir::OpRewritePattern<func::FuncOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

 private:
  mlir::LogicalResult matchAndRewrite(
      func::FuncOp op, mlir::PatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder builder(op.getLoc(), rewriter);

    auto input_types = op.getFunctionType().getInputs();

    SmallVector<Type> new_operand_types(input_types);
    for (auto&& [index, operand_type] : llvm::enumerate(new_operand_types)) {
      auto attr = op.getArgAttr(index, "tt.tma_descriptor");
      if (!attr) {
        continue;
      }
      mlir::BlockArgument func_arg = op.getArgument(index);
      auto element_type =
          mlir::cast<PointerType>(operand_type).getPointeeType();
      auto tma_descriptor = mlir::cast<TmaDescriptorAttr>(attr);
      auto layout = tma_descriptor.getLayout();
      auto block_shape = tma_descriptor.getTileShape();
      SmallVector<int64_t> ordered_block_shape =
          GetMajorToMinorOrder(block_shape, layout);

      operand_type = TensorDescType::get(
          builder.getContext(),
          RankedTensorType::get(ordered_block_shape, element_type));
      // !tt.tensordesc<tensor<block_shape x element_type>> -> !tt.ptr<>
      auto cast_to_orig_type = mlir::UnrealizedConversionCastOp::create(
          builder, operand_type, func_arg);
      func_arg.replaceAllUsesExcept(cast_to_orig_type.getResult(0),
                                    cast_to_orig_type);
    }

    // Replace the function arguments with the new types.
    mlir::Block* entry_block = &op.getBody().front();
    for (auto [arg, arg_type] :
         llvm::zip(entry_block->getArguments(), new_operand_types)) {
      arg.setType(arg_type);
    }

    auto new_function_type = FunctionType::get(
        op.getContext(), new_operand_types, /*result_types=*/{});

    // Transfer the argument attributes from the old function to the new one.
    SmallVector<DictionaryAttr> arg_attrs;
    if (op.getArgAttrs().has_value()) {
      auto oldArgAttrsArray = op.getArgAttrs().value();
      for (int i = 0; i < oldArgAttrsArray.size(); ++i) {
        arg_attrs.push_back(
            mlir::cast<mlir::DictionaryAttr>(oldArgAttrsArray[i]));
      }
    }

    // Currently not propagating any function attributes to the new function.
    ArrayRef<NamedAttribute> attrs;
    auto new_func = triton::FuncOp::create(builder, op.getName(),
                                           new_function_type, attrs, arg_attrs);

    for (int i = 0; i < new_func.getNumArguments(); ++i) {
      // TMA arguments don't require tt.divisibility.
      if (op.getArgAttr(i, "tt.nv_tma_desc")) {
        continue;
      }
      const mlir::NamedAttribute attr = xtile::GetDivisibilityAttr(builder);
      new_func.setArgAttr(i, attr.getName(), attr.getValue());
    }

    rewriter.inlineRegionBefore(op.getRegion(), new_func.getFunctionBody(),
                                new_func.end());
    rewriter.replaceOp(op, new_func);

    auto terminator = new_func.getBody().front().getTerminator();
    rewriter.setInsertionPoint(terminator);
    triton::ReturnOp::create(rewriter, new_func.getLoc());
    rewriter.eraseOp(terminator);

    return mlir::success();
  }
};

class RewriteExtract : public mlir::OpRewritePattern<ExtractOp> {
 public:
  RewriteExtract(mlir::MLIRContext* context, bool allow_tma, int num_stages)
      : OpRewritePattern(context),
        allow_tma_(allow_tma),
        num_stages_(num_stages) {}
  using OpRewritePattern::OpRewritePattern;

 private:
  // Rewriting ExtractOp as:
  // Without TMA:
  // tt.addptr + tt.make_tensor_ptr + tt.load.
  // Offsets are resolved in tt.addptr.
  //
  // With TMA:
  // tt.descriptor_load.
  // Offsets are resolved in tt.descriptor_load.
  // If the layout is not major-to-minor, we insert a transpose to ensure that
  // the tile loaded in both TMA and non-TMA cases is the same:
  // tt.descriptor_load + tt.transpose.
  mlir::LogicalResult matchAndRewrite(
      ExtractOp op, mlir::PatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
    RankedTensorType tile_type = op.getType();
    ArrayRef<int64_t> tile_shape = tile_type.getShape();
    ArrayRef<int64_t> src_shape = op.getSrcShape();
    ArrayRef<int64_t> src_layout = op.getSrcLayout();

    auto offsets = op.getOffsetsAsValues(builder);
    auto sizes = op.getStaticSizes();
    auto strides = to_vector(op.getStaticStrides());

    if (CanUseTma(op, allow_tma_, num_stages_, src_shape, sizes, strides,
                  offsets, op.getSrc(), src_layout)) {
      if (auto result = CanonicalizeTileStrides(strides, sizes, src_shape);
          !result.ok()) {
        return rewriter.notifyMatchFailure(op, result.message());
      }

      AddTmaAttributes(builder, op.getSrc(), src_shape, src_layout, sizes,
                       strides);

      auto ordered_offsets = GetMajorToMinorOrder(offsets, src_layout);
      auto ordered_sizes = GetMajorToMinorOrder(sizes, src_layout);
      auto ordered_type =
          tile_type.clone(GetMajorToMinorOrder(sizes, src_layout));

      // ptr -> !tt.tensordesc<tile_type>
      auto desc_type = TensorDescType::get(builder.getContext(), ordered_type);
      auto cast_to_tensor_desc = mlir::UnrealizedConversionCastOp::create(
          builder, desc_type, op.getSrc());

      Value result = DescriptorLoadOp::create(
          builder, ordered_type, cast_to_tensor_desc.getResult(0),
          xtriton::IndexCast(builder, builder.getI32Type(), ordered_offsets));

      // Insert a transpose if the layout is not major-to-minor.
      if (!IsMajorToMinorLayout(src_layout)) {
        result = TransOp::create(builder, result,
                                 GetInverseLayoutPermutation(src_layout));
      }
      // Insert a reshape if the result is rank-reduced.
      if (sizes.size() != tile_shape.size()) {
        result = ReshapeOp::create(builder, tile_shape, result,
                                   /*allowReorder=*/false);
      }

      rewriter.replaceOp(op, result);
      return mlir::success();
    }

    // Compute the set of reduced dimensions.
    auto reduction_mask = mlir::computeRankReductionMask(sizes, tile_shape);
    if (!reduction_mask) {
      return rewriter.notifyMatchFailure(op, "Unsupported rank reduction.");
    }
    SmallVector<unsigned> reduced_dims = to_vector(*reduction_mask);
    absl::c_sort(reduced_dims);

    auto [ptr, mask] = xtriton::CreateTensorOfPointersAndMask(
        builder, op.getSrc(), src_shape, src_layout, offsets, sizes, strides,
        reduced_dims, tile_shape);
    Value other;
    if (mask) {
      other = arith::ConstantOp::create(
          builder, builder.getZeroAttr(RankedTensorType::get(
                       tile_shape, tile_type.getElementType())));
    }
    auto load = LoadOp::create(builder, ptr, mask, other, CacheModifier::NONE,
                               EvictionPolicy::NORMAL,
                               /*isVolatile=*/false);
    rewriter.replaceOp(op, load);
    return mlir::success();
  }

  const bool allow_tma_;
  const int num_stages_;
};

class RewriteInsert : public mlir::OpRewritePattern<InsertOp> {
 public:
  RewriteInsert(mlir::MLIRContext* context, bool allow_tma, int num_stages)
      : OpRewritePattern(context),
        allow_tma_(allow_tma),
        num_stages_(num_stages) {}
  using OpRewritePattern::OpRewritePattern;

 private:
  // Rewriting InsertOp as:
  // Without TMA:
  // tt.addptr + tt.make_tensor_ptr + tt.store.
  // Offsets are resolved in tt.addptr.
  //
  // With TMA:
  // tt.descriptor_store.
  // Offsets are resolved in tt.descriptor_store.
  // If the layout is not major-to-minor, we insert a transpose to to be
  // compatible with TMA's physical restrictions. tt.transpose +
  // tt.descriptor_store.
  mlir::LogicalResult matchAndRewrite(
      InsertOp op, mlir::PatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
    RankedTensorType tile_type = op.getSrc().getType();
    ArrayRef<int64_t> tile_shape = tile_type.getShape();
    ArrayRef<int64_t> dst_shape = op.getDstShape();
    ArrayRef<int64_t> dst_layout = op.getDstLayout();

    auto offsets = op.getOffsetsAsValues(builder);
    auto sizes = op.getStaticSizes();
    auto strides = to_vector(op.getStaticStrides());

    // Compute the set of reduced dimensions.
    auto reduction_mask = mlir::computeRankReductionMask(sizes, tile_shape);
    if (!reduction_mask) {
      return rewriter.notifyMatchFailure(op, "Unsupported rank reduction.");
    }
    SmallVector<unsigned> reduced_dims = to_vector(*reduction_mask);
    absl::c_sort(reduced_dims);

    if (CanUseTma(op, allow_tma_, num_stages_, dst_shape, sizes, strides,
                  offsets, op.getDst(), dst_layout)) {
      if (auto result = CanonicalizeTileStrides(strides, sizes, dst_shape);
          !result.ok()) {
        return rewriter.notifyMatchFailure(op, result.message());
      }

      AddTmaAttributes(builder, op.getDst(), dst_shape, dst_layout, sizes,
                       strides);

      // ptr -> !tt.tensordesc<tile_type>
      auto desc_type = TensorDescType::get(
          builder.getContext(),
          tile_type.clone(GetMajorToMinorOrder(sizes, dst_layout)));
      auto cast_to_tensor_desc = mlir::UnrealizedConversionCastOp::create(
          builder, desc_type, op.getDst());

      Value src = op.getSrc();
      // Insert a expand_dims if the source is rank-reduced.
      for (auto dim : reduced_dims) {
        src = ExpandDimsOp::create(builder, src, dim);
      }
      // Insert a transpose if the layout is not major-to-minor.
      if (!IsMajorToMinorLayout(dst_layout)) {
        // Transpose to a major-to-minor tensor by simply reversing the layout.
        auto transpose_order = llvm::to_vector_of<int32_t>(dst_layout);
        std::reverse(transpose_order.begin(), transpose_order.end());
        src = TransOp::create(builder, src, transpose_order);
      }

      auto ordered_offsets = GetMajorToMinorOrder(offsets, dst_layout);
      DescriptorStoreOp::create(
          builder, cast_to_tensor_desc.getResult(0), src,
          xtriton::IndexCast(builder, builder.getI32Type(), ordered_offsets));
    } else {
      auto [ptr, mask] = xtriton::CreateTensorOfPointersAndMask(
          builder, op.getDst(), dst_shape, dst_layout, offsets, sizes, strides,
          reduced_dims, tile_shape);
      StoreOp::create(builder, ptr, op.getSrc(), mask, CacheModifier::NONE,
                      EvictionPolicy::NORMAL);
    }
    rewriter.eraseOp(op);
    return mlir::success();
  }

  const bool allow_tma_;
  const int num_stages_;
};

// Rewriting tensor::InsertOp as tt.store.
class RewriteScalarInsert : public mlir::OpRewritePattern<tensor::InsertOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

 private:
  mlir::LogicalResult matchAndRewrite(
      tensor::InsertOp op, mlir::PatternRewriter& rewriter) const override {
    if (op.getDest().getType().getRank() != 0) {
      return rewriter.notifyMatchFailure(op, "Expected dest to be scalar.");
    }
    ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
    auto ptr_type = GetTensorPtrType(op.getScalar().getType());
    auto cast_dst_to_tensor_ptr_type = mlir::UnrealizedConversionCastOp::create(
                                           builder, ptr_type, op.getDest())
                                           .getResult(0);
    StoreOp::create(builder, cast_dst_to_tensor_ptr_type, op.getScalar(),
                    /*boundary_checks=*/std::vector<int32_t>{},
                    CacheModifier::NONE, EvictionPolicy::NORMAL);
    rewriter.replaceOp(op, op.getDest());
    return mlir::success();
  }
};

class RewriteScalarExtract : public mlir::OpRewritePattern<tensor::ExtractOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

 private:
  // Rewriting ExtractOp as tt.advance + tt.store.
  mlir::LogicalResult matchAndRewrite(
      tensor::ExtractOp op, mlir::PatternRewriter& rewriter) const override {
    if (op.getTensor().getType().getRank() != 0) {
      return rewriter.notifyMatchFailure(op, "Expected src to be scalar.");
    }
    ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
    auto ptr_type = GetTensorPtrType(op.getType());
    auto cast_src_to_tensor_ptr_type = mlir::UnrealizedConversionCastOp::create(
                                           builder, ptr_type, op.getTensor())
                                           .getResult(0);
    auto scalar = LoadOp::create(builder, cast_src_to_tensor_ptr_type,
                                 CacheModifier::NONE, EvictionPolicy::NORMAL,
                                 /*isVolatile=*/false);
    rewriter.replaceOp(op, scalar.getResult());
    return mlir::success();
  }
};

class TritonXLAExtractInsertToTritonPass
    : public impl::TritonXLAExtractInsertToTritonPassBase<
          TritonXLAExtractInsertToTritonPass> {
 public:
  using Base::Base;

 private:
  void runOnOperation() override {
    mlir::MLIRContext* mlir_context = &getContext();
    mlir::RewritePatternSet patterns(mlir_context);
    patterns.add<RewriteExtract, RewriteInsert>(
        mlir_context, allow_tma_.getValue(), num_stages_.getValue());
    patterns.add<RewriteScalarExtract, RewriteScalarInsert>(mlir_context);
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }

    if (mlir::failed(mlir::applyPatternsGreedily(
            getOperation(), mlir::RewritePatternSet(
                                mlir_context, std::make_unique<RewriteFuncOp>(
                                                  mlir_context))))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateTritonXLAExtractInsertToTritonPass() {
  return std::make_unique<TritonXLAExtractInsertToTritonPass>();
}

std::unique_ptr<mlir::Pass> CreateTritonXLAExtractInsertToTritonPass(
    bool allow_tma, int num_stages) {
  return std::make_unique<TritonXLAExtractInsertToTritonPass>(
      TritonXLAExtractInsertToTritonPassOptions{allow_tma, num_stages});
}

}  // namespace mlir::triton::xla
