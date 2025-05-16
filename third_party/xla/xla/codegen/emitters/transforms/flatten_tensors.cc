/* Copyright 2024 The OpenXLA Authors.

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
#include <optional>
#include <utility>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/cpu/codegen/emitters/ir/xla_cpu_ops.h"
#include "xla/backends/gpu/codegen/emitters/ir/xla_gpu_ops.h"
#include "xla/hlo/analysis/indexing_analysis.h"
#include "xla/layout_util.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace emitters {
namespace {

#define GEN_PASS_DEF_FLATTENTENSORSPASS
#include "xla/codegen/emitters/transforms/passes.h.inc"

using mlir::Attribute;
using mlir::Location;
using mlir::LogicalResult;
using mlir::MLIRContext;
using mlir::OpRewritePattern;
using mlir::PatternRewriter;
using mlir::RankedTensorType;
using mlir::ShapedType;
using mlir::SmallVector;
using mlir::Type;
using mlir::TypedValue;
using mlir::TypeRange;
using mlir::UnrealizedConversionCastOp;
using mlir::Value;
using mlir::ValueRange;
using mlir::VectorType;
using mlir::func::FuncOp;
using mlir::func::ReturnOp;
using mlir::scf::ForOp;
using mlir::scf::IfOp;
using mlir::scf::IndexSwitchOp;
using mlir::tensor::ExtractOp;
using mlir::tensor::InsertOp;
namespace mv = mlir::vector;

RankedTensorType GetFlattenedType(RankedTensorType tensor_type) {
  return RankedTensorType::get({tensor_type.getNumElements()},
                               tensor_type.getElementType());
}

VectorType GetFlattenedType(VectorType vector_type) {
  return VectorType::get({vector_type.getNumElements()},
                         vector_type.getElementType());
}

ShapedType GetFlattenedType(Type type) {
  if (auto vector_type = mlir::dyn_cast<VectorType>(type)) {
    return GetFlattenedType(vector_type);
  }
  return GetFlattenedType(mlir::cast<RankedTensorType>(type));
}

bool IsScalarOrFlat(Type type) {
  if (auto shaped_type = mlir::dyn_cast<ShapedType>(type)) {
    return shaped_type.getRank() < 2;
  }
  return true;
}

bool HasOnlyFlatTensorsFlatVectorsOrScalars(TypeRange types) {
  return llvm::all_of(types, IsScalarOrFlat);
}

Value Flatten(Value value, PatternRewriter& rewriter) {
  if (IsScalarOrFlat(value.getType())) return value;
  auto flat_type = GetFlattenedType(value.getType());
  return rewriter
      .create<UnrealizedConversionCastOp>(value.getLoc(), flat_type, value)
      .getResult(0);
}

struct RewriteFunctionSignatures : OpRewritePattern<FuncOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(FuncOp op,
                                PatternRewriter& rewriter) const override {
    auto input_types = op.getFunctionType().getInputs();
    auto result_types = op.getFunctionType().getResults();
    if (HasOnlyFlatTensorsFlatVectorsOrScalars(input_types) &&
        HasOnlyFlatTensorsFlatVectorsOrScalars(result_types)) {
      return rewriter.notifyMatchFailure(op, "nothing to flatten");
    }

    auto loc = op.getLoc();
    mlir::Block* entry_block = &op.getBody().front();
    SmallVector<Type> new_result_types;
    SmallVector<Value> new_results;

    // If some results are tensors or vectors, we need to flatten them.
    auto terminator = entry_block->getTerminator();
    rewriter.setInsertionPoint(terminator);

    for (Value result : terminator->getOperands()) {
      Value flattened = Flatten(result, rewriter);
      new_results.push_back(flattened);
      new_result_types.push_back(flattened.getType());
    }
    rewriter.replaceOpWithNewOp<ReturnOp>(terminator, new_results);

    // Cast all function arguments to the original type.
    SmallVector<Type> new_operand_types(input_types);
    rewriter.setInsertionPointToStart(entry_block);
    for (auto&& [index, operand_type] : llvm::enumerate(new_operand_types)) {
      if (IsScalarOrFlat(operand_type)) continue;
      mlir::BlockArgument func_argument = op.getArgument(index);
      auto cast_to_orig_type = rewriter.create<UnrealizedConversionCastOp>(
          loc, operand_type, func_argument);
      func_argument.replaceAllUsesExcept(cast_to_orig_type.getResult(0),
                                         cast_to_orig_type);
      operand_type = GetFlattenedType(operand_type);
    }
    // Replace the function arguments with the new types.
    for (auto [arg, arg_type] :
         llvm::zip(entry_block->getArguments(), new_operand_types)) {
      arg.setType(arg_type);
    }
    // Update function signature.
    op.setType(rewriter.getFunctionType(new_operand_types, new_result_types));
    return mlir::success();
  }
};

struct RewritePureCall : OpRewritePattern<PureCallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PureCallOp op,
                                PatternRewriter& rewriter) const override {
    if (HasOnlyFlatTensorsFlatVectorsOrScalars(op.getOperandTypes()) &&
        HasOnlyFlatTensorsFlatVectorsOrScalars(op.getResultTypes())) {
      return rewriter.notifyMatchFailure(op, "nothing to flatten");
    }
    SmallVector<Value> flat_operands;
    flat_operands.reserve(op.getNumOperands());
    for (Value operand : op.getOperands()) {
      flat_operands.push_back(Flatten(operand, rewriter));
    }
    SmallVector<Type> flat_result_types;
    flat_result_types.reserve(op.getNumResults());
    llvm::SmallBitVector results_to_update(op.getNumResults(), false);
    for (auto [index, result_type] : llvm::enumerate(op.getResultTypes())) {
      if (IsScalarOrFlat(result_type)) {
        flat_result_types.push_back(result_type);
        continue;
      }
      results_to_update.set(index);
      flat_result_types.push_back(GetFlattenedType(result_type));
    }
    Location loc = op.getLoc();
    auto new_call_op = rewriter.create<PureCallOp>(
        loc, flat_result_types, op.getCalleeAttr(), flat_operands);
    SmallVector<Value> new_results;
    new_results.reserve(op.getNumResults());
    for (auto [index, new_result] : llvm::enumerate(new_call_op.getResults())) {
      if (results_to_update.test(index)) {
        new_results.push_back(new_result);
        continue;
      }
      auto cast_to_orig_type = rewriter.create<UnrealizedConversionCastOp>(
          loc, op.getResult(index).getType(), new_result);
      new_results.push_back(cast_to_orig_type.getResult(0));
    }
    rewriter.replaceOp(op, new_results);
    return mlir::success();
  }
};

// Returns the linearized index.
Value LinearizeIndex(Location loc, ShapedType type, ValueRange indices,
                     PatternRewriter& rewriter, Attribute encoding = nullptr) {
  auto byte_shape = ShapeUtil::MakeShape(U8, type.getShape());
  if (encoding) {
    *byte_shape.mutable_layout() = LayoutUtil::MakeLayout(llvm::to_vector(
        mlir::cast<mlir::DenseElementsAttr>(encoding).getValues<int64_t>()));
  }
  auto linear_shape =
      ShapeUtil::MakeShape(U8, {ShapeUtil::ElementsIn(byte_shape)});
  auto linearized_map =
      GetBitcastMap(byte_shape, linear_shape, rewriter.getContext());
  mlir::SmallVector<Value> result;
  rewriter.createOrFold<ApplyIndexingOp>(result, loc, indices, ValueRange{},
                                         linearized_map);
  return result.front();
}

struct RewriteAllocateShared : OpRewritePattern<gpu::AllocateSharedOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::AllocateSharedOp op,
                                PatternRewriter& rewriter) const override {
    auto tensor_type = op.getResult().getType();
    if (IsScalarOrFlat(tensor_type)) {
      return rewriter.notifyMatchFailure(op, "the tensor is already flat");
    }
    auto flat_type = GetFlattenedType(tensor_type);
    Location loc = op.getLoc();
    Value new_op =
        rewriter.create<gpu::AllocateSharedOp>(op.getLoc(), flat_type);
    auto cast_to_orig_type =
        rewriter.create<UnrealizedConversionCastOp>(loc, tensor_type, new_op);
    rewriter.replaceOp(op, cast_to_orig_type.getResult(0));
    return mlir::success();
  }
};

struct RewriteCpuLoad : OpRewritePattern<cpu::LoadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cpu::LoadOp op,
                                PatternRewriter& rewriter) const override {
    auto tensor_type = op.getResult().getType();
    if (IsScalarOrFlat(tensor_type)) {
      return rewriter.notifyMatchFailure(op, "the tensor is already flat");
    }
    auto flat_type = GetFlattenedType(tensor_type);
    Location loc = op.getLoc();
    Value new_op = rewriter.create<cpu::LoadOp>(
        op.getLoc(), flat_type, op.getCallFrame(), op.getIndex());
    auto cast_to_orig_type =
        rewriter.create<UnrealizedConversionCastOp>(loc, tensor_type, new_op);
    rewriter.replaceOp(op, cast_to_orig_type.getResult(0));
    return mlir::success();
  }
};

struct RewriteConstant : OpRewritePattern<mlir::arith::ConstantOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::arith::ConstantOp op,
                                PatternRewriter& rewriter) const override {
    if (IsScalarOrFlat(op.getType())) {
      return rewriter.notifyMatchFailure(
          op, "the tensor or vector is already flat");
    }
    auto dense_attr = mlir::dyn_cast<mlir::DenseElementsAttr>(op.getValue());
    auto new_type = GetFlattenedType(op.getType());
    Value new_constant = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), dense_attr.reshape(new_type));
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(op, op.getType(),
                                                            new_constant);
    return mlir::success();
  }
};

struct RewriteTensorExtract : OpRewritePattern<ExtractOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ExtractOp op,
                                PatternRewriter& rewriter) const override {
    auto tensor = op.getTensor();
    auto tensor_type = tensor.getType();
    if (tensor_type.getRank() < 2) {
      return rewriter.notifyMatchFailure(op, "the tensor is already flat");
    }
    auto loc = op.getLoc();
    auto linear_index = LinearizeIndex(loc, tensor_type, op.getIndices(),
                                       rewriter, tensor_type.getEncoding());
    auto tensor_1D = rewriter
                         .create<UnrealizedConversionCastOp>(
                             loc, GetFlattenedType(tensor_type), tensor)
                         .getResult(0);
    rewriter.replaceOpWithNewOp<ExtractOp>(op, tensor_1D, linear_index);
    return mlir::success();
  }
};

struct RewriteVectorTransferRead : OpRewritePattern<mv::TransferReadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mv::TransferReadOp op,
                                PatternRewriter& rewriter) const override {
    auto vector = op.getVector();
    auto vector_type = vector.getType();
    if (vector_type.getRank() != 1) {
      return rewriter.notifyMatchFailure(op, "the vector should be 1D");
    }
    auto tensor = op.getBase();
    auto tensor_type = tensor.getType();
    if (tensor_type.getRank() < 2) {
      return rewriter.notifyMatchFailure(op,
                                         "the source tensore is already flat");
    }
    auto loc = op.getLoc();
    auto linear_index =
        LinearizeIndex(loc, tensor_type, op.getIndices(), rewriter);
    auto tensor_1D = rewriter
                         .create<UnrealizedConversionCastOp>(
                             loc, GetFlattenedType(tensor_type), tensor)
                         .getResult(0);
    rewriter.replaceOpWithNewOp<mv::TransferReadOp>(
        op, vector_type, tensor_1D, linear_index, llvm::ArrayRef<bool>{true});
    return mlir::success();
  }
};
struct RewriteVectorExtract : OpRewritePattern<mv::ExtractOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mv::ExtractOp op,
                                PatternRewriter& rewriter) const override {
    auto vector = op.getVector();
    auto vector_type = vector.getType();
    if (vector_type.getRank() < 2) {
      return rewriter.notifyMatchFailure(op, "the vector is already flat");
    }
    auto indices =
        mv::getAsValues(rewriter, op.getLoc(), op.getMixedPosition());
    auto linear_index =
        LinearizeIndex(op.getLoc(), vector_type, indices, rewriter);
    auto vector_1D = rewriter
                         .create<UnrealizedConversionCastOp>(
                             op.getLoc(), GetFlattenedType(vector_type), vector)
                         .getResult(0);
    rewriter.replaceOpWithNewOp<mv::ExtractOp>(op, vector_1D, linear_index);
    return mlir::success();
  }
};

struct RewriteTensorInsert : OpRewritePattern<InsertOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(InsertOp op,
                                PatternRewriter& rewriter) const override {
    auto tensor = op.getDest();
    auto tensor_type = tensor.getType();
    if (tensor_type.getRank() < 2) {
      return rewriter.notifyMatchFailure(op, "the tensor is already flat");
    }
    auto loc = op.getLoc();
    auto linear_index = LinearizeIndex(loc, tensor_type, op.getIndices(),
                                       rewriter, tensor_type.getEncoding());
    mlir::ImplicitLocOpBuilder b(loc, rewriter);
    auto tensor_1D = b.create<UnrealizedConversionCastOp>(
                          GetFlattenedType(tensor_type), tensor)
                         .getResult(0);
    auto new_insert =
        b.create<InsertOp>(op.getScalar(), tensor_1D, linear_index);
    auto cast_to_orig_type = b.create<UnrealizedConversionCastOp>(
        tensor_type, new_insert.getResult());
    rewriter.replaceOp(op, cast_to_orig_type.getResult(0));
    return mlir::success();
  }
};

struct RewriteVectorInsert : OpRewritePattern<mv::InsertOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mv::InsertOp op,
                                PatternRewriter& rewriter) const override {
    auto vector = op.getDest();
    auto vector_type = vector.getType();
    if (vector_type.getRank() < 2) {
      return rewriter.notifyMatchFailure(op, "the vector is already flat");
    }
    auto loc = op.getLoc();
    auto indices = mv::getAsValues(rewriter, loc, op.getMixedPosition());
    auto linear_index = LinearizeIndex(loc, vector_type, indices, rewriter);
    mlir::ImplicitLocOpBuilder b(loc, rewriter);
    auto vector_1D = b.create<UnrealizedConversionCastOp>(
                          GetFlattenedType(vector_type), vector)
                         .getResult(0);
    auto new_insert =
        b.create<mv::InsertOp>(op.getValueToStore(), vector_1D, linear_index);
    auto cast_to_orig_type = b.create<UnrealizedConversionCastOp>(
        vector_type, new_insert.getResult());
    rewriter.replaceOp(op, cast_to_orig_type.getResult(0));
    return mlir::success();
  }
};

struct RewriteAtomicRMW : OpRewritePattern<AtomicRMWOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AtomicRMWOp op,
                                PatternRewriter& rewriter) const override {
    auto tensor = op.getInput();
    auto tensor_type = tensor.getType();
    if (tensor_type.getRank() < 2) {
      return rewriter.notifyMatchFailure(op, "the tensor is already flat");
    }
    auto loc = op.getLoc();
    auto linear_index = LinearizeIndex(loc, tensor_type, op.getIndices(),
                                       rewriter, tensor_type.getEncoding());
    mlir::ImplicitLocOpBuilder b(loc, rewriter);
    auto tensor_1D = b.create<UnrealizedConversionCastOp>(
                          GetFlattenedType(tensor_type), tensor)
                         .getResult(0);
    auto new_atomic_rmw = b.create<AtomicRMWOp>(tensor_1D, linear_index);
    rewriter.inlineRegionBefore(op.getRegion(),
                                &new_atomic_rmw.getRegion().front());
    auto cast_to_orig_type = b.create<UnrealizedConversionCastOp>(
        tensor_type, new_atomic_rmw.getResult());
    rewriter.replaceOp(op, cast_to_orig_type.getResult(0));
    return mlir::success();
  }
};

// Returns the rank of the tensor or vector or nullopt if it is of neither type.
std::optional<int64_t> GetRankOfTensorOrVector(Type type) {
  if (auto shaped_type = mlir::dyn_cast<ShapedType>(type)) {
    return shaped_type.getRank();
  }
  return std::nullopt;
}

// Checks that the value is produced by an unrealized conversion cast from 1D
// tensor or vector to ND. Returns the 1D tensor or vector if so.
std::optional<Value> GetDelinearizedTensorOrVector(Value value) {
  auto rank = GetRankOfTensorOrVector(value.getType());
  if (!rank.has_value() || *rank < 2) {
    return std::nullopt;
  }
  auto cast = value.getDefiningOp<UnrealizedConversionCastOp>();
  if (!cast || cast->getNumResults() != 1 || cast->getNumOperands() != 1) {
    return std::nullopt;
  }
  auto rank_before_linearization =
      GetRankOfTensorOrVector(cast->getOperand(0).getType());
  if (!rank_before_linearization.has_value() ||
      *rank_before_linearization != 1) {
    return std::nullopt;
  }
  return cast->getOperand(0);
}

struct RewriteFor : public OpRewritePattern<ForOp> {
  using OpRewritePattern<ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ForOp op,
                                PatternRewriter& rewriter) const override {
    llvm::SmallBitVector args_to_update(op.getNumResults(), false);
    mlir::SmallVector<Value> new_init_args;
    new_init_args.reserve(op.getNumResults());
    for (auto [index, arg] : llvm::enumerate(op.getInitArgs())) {
      auto type_before_linearization = GetDelinearizedTensorOrVector(arg);
      if (!type_before_linearization.has_value()) {
        new_init_args.push_back(arg);
        continue;
      }
      new_init_args.push_back(*type_before_linearization);
      args_to_update.set(index);
    }
    if (args_to_update.none()) {
      return rewriter.notifyMatchFailure(op, "no args to update");
    }
    // Create new ForOp with updated init args.
    Location loc = op.getLoc();
    auto new_for_op =
        rewriter.create<ForOp>(loc, op.getLowerBound(), op.getUpperBound(),
                               op.getStep(), new_init_args);
    new_for_op->setAttrs(op->getAttrs());

    // Insert casts for the block arguments.
    mlir::Block* new_body = new_for_op.getBody();
    mlir::Block* old_body = op.getBody();
    rewriter.setInsertionPoint(new_body, new_body->begin());
    SmallVector<Value, 4> updated_block_args{new_body->getArguments().begin(),
                                             new_body->getArguments().end()};
    for (auto [index, arg] :
         llvm::enumerate(new_body->getArguments().drop_front())) {
      if (!args_to_update.test(index)) continue;
      updated_block_args[index + 1] =
          rewriter
              .create<UnrealizedConversionCastOp>(
                  loc, old_body->getArgument(index + 1).getType(), arg)
              .getResult(0);
    }

    // Move the body of the old ForOp to the new one.
    rewriter.mergeBlocks(old_body, new_body, updated_block_args);

    // Update the terminator.
    auto new_terminator =
        mlir::cast<mlir::scf::YieldOp>(new_body->getTerminator());
    rewriter.setInsertionPoint(new_terminator);
    for (auto&& [index, yielded_value] :
         llvm::enumerate(new_terminator.getResultsMutable())) {
      if (!args_to_update.test(index)) continue;
      yielded_value.assign(
          rewriter
              .create<UnrealizedConversionCastOp>(
                  loc, new_init_args[index].getType(), yielded_value.get())
              .getResult(0));
    }

    // Cast back the results.
    rewriter.setInsertionPointAfter(new_for_op);
    SmallVector<Value> new_results(new_for_op.getResults());
    for (auto&& [index, result] : llvm::enumerate(new_results)) {
      if (!args_to_update.test(index)) continue;
      result = rewriter
                   .create<UnrealizedConversionCastOp>(
                       loc, op->getResult(index).getType(), result)
                   .getResult(0);
    }
    rewriter.replaceOp(op, new_results);
    return mlir::success();
  }
};

struct RewriteIf : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IfOp op,
                                PatternRewriter& rewriter) const override {
    auto result_types = op.getResultTypes();
    if (HasOnlyFlatTensorsFlatVectorsOrScalars(result_types)) {
      return rewriter.notifyMatchFailure(op, "nothing to flatten");
    }
    mlir::scf::YieldOp then_yield = op.thenYield();
    SmallVector<Type> new_result_types;
    new_result_types.reserve(then_yield.getNumOperands());
    bool found_cast = false;
    for (auto& result : then_yield->getOpOperands()) {
      auto delinearized_tensor = GetDelinearizedTensorOrVector(result.get());
      if (!delinearized_tensor.has_value()) {
        new_result_types.push_back(result.get().getType());
        continue;
      }
      new_result_types.push_back(delinearized_tensor->getType());
      result.set(*delinearized_tensor);
      found_cast = true;
    }
    if (!found_cast) {
      return rewriter.notifyMatchFailure(op, "no cast found");
    }
    Location loc = op.getLoc();
    // Update the else branch if present.
    bool has_else_region = !op.getElseRegion().empty();
    if (has_else_region) {
      mlir::scf::YieldOp else_yield = op.elseYield();
      mlir::OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(else_yield);
      for (auto&& [result, type] :
           llvm::zip(else_yield->getOpOperands(), new_result_types)) {
        if (result.get().getType() == type) continue;
        result.set(
            rewriter.create<UnrealizedConversionCastOp>(loc, type, result.get())
                .getResult(0));
      }
    }
    // Create new IfOp and move the old op's regions to the new one.
    auto new_if_op = rewriter.create<IfOp>(loc, new_result_types,
                                           op.getCondition(), has_else_region);
    rewriter.inlineRegionBefore(op.getThenRegion(),
                                &new_if_op.getThenRegion().back());
    rewriter.eraseBlock(&new_if_op.getThenRegion().back());
    if (has_else_region) {
      rewriter.inlineRegionBefore(op.getElseRegion(),
                                  &new_if_op.getElseRegion().back());
      rewriter.eraseBlock(&new_if_op.getElseRegion().back());
    }

    // Update the results.
    rewriter.setInsertionPointAfter(new_if_op);
    SmallVector<Value> new_results(new_if_op.getResults());
    for (auto&& [index, result] : llvm::enumerate(new_results)) {
      Type old_type = op->getResult(index).getType();
      if (result.getType() == old_type) continue;
      result =
          rewriter.create<UnrealizedConversionCastOp>(loc, old_type, result)
              .getResult(0);
    }
    rewriter.replaceOp(op, new_results);
    return mlir::success();
  }
};

struct RewriteIndexSwitch : public OpRewritePattern<IndexSwitchOp> {
  using OpRewritePattern<IndexSwitchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IndexSwitchOp op,
                                PatternRewriter& rewriter) const override {
    auto result_types = op.getResultTypes();
    if (HasOnlyFlatTensorsFlatVectorsOrScalars(result_types)) {
      return rewriter.notifyMatchFailure(op, "nothing to flatten");
    }
    auto default_yield =
        mlir::cast<mlir::scf::YieldOp>(op.getDefaultBlock().getTerminator());
    SmallVector<Type> new_result_types;
    new_result_types.reserve(default_yield.getNumOperands());
    bool found_cast = false;
    for (auto& result : default_yield->getOpOperands()) {
      auto delinearized_tensor = GetDelinearizedTensorOrVector(result.get());
      if (!delinearized_tensor.has_value()) {
        new_result_types.push_back(result.get().getType());
        continue;
      }
      new_result_types.push_back(delinearized_tensor->getType());
      result.set(*delinearized_tensor);
      found_cast = true;
    }
    if (!found_cast) {
      return rewriter.notifyMatchFailure(op, "no cast found");
    }
    Location loc = op.getLoc();
    // Update the "case" regions.
    for (auto& case_region : op.getCaseRegions()) {
      auto yield = mlir::cast<mlir::scf::YieldOp>(
          case_region.getBlocks().front().getTerminator());
      mlir::OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(yield);
      for (auto&& [result, type] :
           llvm::zip(yield->getOpOperands(), new_result_types)) {
        if (result.get().getType() == type) continue;
        result.set(
            rewriter.create<UnrealizedConversionCastOp>(loc, type, result.get())
                .getResult(0));
      }
    }
    // Create new IndexSwitchOp and move the old op's regions to the new one.
    auto new_index_switch = rewriter.create<IndexSwitchOp>(
        loc, new_result_types, op.getArg(), op.getCases(), op.getNumCases());
    for (auto&& [old_region, new_region] :
         llvm::zip(op.getRegions(), new_index_switch.getRegions())) {
      rewriter.inlineRegionBefore(*old_region, *new_region, new_region->end());
    }
    // Update the results.
    rewriter.setInsertionPointAfter(new_index_switch);
    SmallVector<Value> new_results(new_index_switch.getResults());
    for (auto&& [index, result] : llvm::enumerate(new_results)) {
      Type old_type = op->getResult(index).getType();
      if (result.getType() == old_type) continue;
      result =
          rewriter.create<UnrealizedConversionCastOp>(loc, old_type, result)
              .getResult(0);
    }
    rewriter.replaceOp(op, new_results);
    return mlir::success();
  }
};

struct RewriteSyncThreads : OpRewritePattern<gpu::SyncThreadsOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::SyncThreadsOp op,
                                PatternRewriter& rewriter) const override {
    auto types = op.getResultTypes();
    if (HasOnlyFlatTensorsFlatVectorsOrScalars(types)) {
      return rewriter.notifyMatchFailure(op, "nothing to flatten");
    }

    auto loc = op.getLoc();

    SmallVector<Value> new_operands;
    new_operands.reserve(op.getNumOperands());
    llvm::SmallBitVector results_to_update(op.getNumResults(), false);
    for (auto& operand : op->getOpOperands()) {
      auto tensor_type = mlir::cast<RankedTensorType>(operand.get().getType());
      if (tensor_type.getRank() < 2) continue;
      results_to_update.set(operand.getOperandNumber());
      new_operands.push_back(
          rewriter
              .create<UnrealizedConversionCastOp>(
                  loc, GetFlattenedType(tensor_type), operand.get())
              .getResult(0));
    }
    auto new_op = rewriter.create<gpu::SyncThreadsOp>(
        loc, TypeRange(new_operands), new_operands);
    SmallVector<Value> new_results;
    new_results.reserve(op.getNumResults());
    for (auto [index, result] : llvm::enumerate(new_op.getResults())) {
      if (!results_to_update.test(index)) {
        new_results.push_back(result);
        continue;
      }
      auto cast_to_orig_type = rewriter.create<UnrealizedConversionCastOp>(
          loc, result.getType(), result);
      new_results.push_back(cast_to_orig_type.getResult(0));
    }
    rewriter.replaceOp(op, new_results);
    return mlir::success();
  }
};

class FlattenTensorsPass
    : public impl::FlattenTensorsPassBase<FlattenTensorsPass> {
 public:
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    MLIRContext* mlir_context = &getContext();
    mlir::RewritePatternSet patterns(mlir_context);
    // clang-format off
    patterns.add<
        RewriteAllocateShared,
        RewriteAtomicRMW,
        RewriteConstant,
        RewriteFor,
        RewriteFunctionSignatures,
        RewriteIf,
        RewriteIndexSwitch,
        RewritePureCall,
        RewriteSyncThreads,
        RewriteTensorExtract,
        RewriteTensorInsert,
        RewriteVectorExtract,
        RewriteVectorInsert,
        RewriteVectorTransferRead,
        RewriteCpuLoad
    >(mlir_context);
    // clang-format on
    ApplyIndexingOp::getCanonicalizationPatterns(patterns, mlir_context);
    if (mlir::failed(
            mlir::applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
    // Check if there are no unrealized_conversion_casts.
    bool module_has_casts = module
                                .walk([](UnrealizedConversionCastOp op) {
                                  return mlir::WalkResult::interrupt();
                                })
                                .wasInterrupted();
    if (module_has_casts) {
      llvm::outs() << "FlattenTensorsPass failed to converge";
      signalPassFailure();
      return;
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateFlattenTensorsPass() {
  return std::make_unique<FlattenTensorsPass>();
}

}  // namespace emitters
}  // namespace xla
