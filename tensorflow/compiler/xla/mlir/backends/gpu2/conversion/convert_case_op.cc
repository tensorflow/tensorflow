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

#include "tensorflow/compiler/xla/mlir/backends/gpu2/conversion/convert_case_op.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <utility>

#include "iree-dialects/Dialect/Input/InputOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/backends/gpu2/conversion/de_bufferization.h"
#include "tensorflow/compiler/xla/mlir_hlo/lhlo/IR/lhlo_ops.h"

namespace xla {
namespace gpu {

namespace {
using namespace mlir;                 // NOLINT
using namespace mlir::iree_compiler;  // NOLINT

// Keep track of converted case operations to correctly lower terminators in
// the branch regions before and after regions (condition and body regions).
struct ConvertedCaseOp {
  UsedBuffers buffers;
};

using ConvertedCaseOps = llvm::DenseMap<Operation *, ConvertedCaseOp>;

//===----------------------------------------------------------------------===//
// Converts lmhlo.case op to a scf.if or scf.index_switch
//===----------------------------------------------------------------------===//

struct ConvertCaseOpToHal : public OpConversionPattern<lmhlo::CaseOp> {
  ConvertCaseOpToHal(TypeConverter &converter, MLIRContext *ctx,
                     DeBufferization &state,
                     std::shared_ptr<ConvertedCaseOps> converted)
      : OpConversionPattern(converter, ctx),
        state(state),
        converted(std::move(converted)) {}

  LogicalResult matchAndRewrite(
      lmhlo::CaseOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Block *block = op->getBlock();

    // Collect all buffers accessed in all branches.
    SmallVector<Block *> blocks;
    for (auto &branch : op.getBranches()) {
      for (auto &block : branch.getBlocks()) blocks.push_back(&block);
    }
    auto bufs = getUsedBuffers(blocks);

    // Tensors updated in all case op branches.
    SmallVector<Value> updated_tensors =
        llvm::to_vector(llvm::map_range(bufs.write, [&](auto memref) -> Value {
          return state.remapped[block][memref];
        }));

    // Sets up buffer to tensor remapping inside branch regions.
    auto remap_branch_tensors = [&](Block *branch_block) {
      for (auto r : bufs.read)
        state.remapped[branch_block][r] = state.remapped[block][r];
      for (auto w : bufs.write)
        state.remapped[branch_block][w] = state.remapped[block][w];
    };

    // Load branch index from the argument.
    Value offset = b.create<arith::ConstantIndexOp>(0);
    Value index = b.create<IREE::Input::TensorLoadOp>(
        state.remapped[block][op.getIndex()], /*source_dims=*/ValueRange(),
        /*indices=*/offset);

    bool is_predicate = index.getType().isInteger(1);
    int64_t num_branches = op.getNumRegions();

    // Converts `i1` predicate to a branch index.
    auto predicate_index = [&](Value index) -> Value {
      Value c0 = b.create<arith::ConstantIndexOp>(0);
      Value c1 = b.create<arith::ConstantIndexOp>(1);
      return b.create<arith::SelectOp>(index, c0, c1);
    };

    // If integer index it outside of [0, num_branches) range, according to the
    // HLO specification we will execute the last branch.
    auto integer_index = [&](Value index) -> Value {
      index = b.createOrFold<arith::IndexCastOp>(b.getIndexType(), index);
      Value c0 = b.create<arith::ConstantIndexOp>(0);
      Value cN = b.create<arith::ConstantIndexOp>(num_branches - 1);

      Value too_small = b.create<arith::CmpIOp>(
          b.getI1Type(), arith::CmpIPredicate::slt, index, c0);
      Value too_large = b.create<arith::CmpIOp>(
          b.getI1Type(), arith::CmpIPredicate::sgt, index, cN);

      Value out_of_range = b.create<arith::OrIOp>(too_small, too_large);
      return b.create<arith::SelectOp>(out_of_range, cN, index);
    };

    // Check if branch index is zero.
    auto is_zero = [&](Value index) -> Value {
      Value c0 = b.create<arith::ConstantIndexOp>(0);
      return b.create<arith::CmpIOp>(b.getI1Type(), arith::CmpIPredicate::eq,
                                     index, c0);
    };

    // Converts case operation to scf.if.
    auto convert_to_cond = [&]() -> scf::IfOp {
      assert(op.getBranches().size() == 2 && "expected two branches");
      index = is_predicate ? predicate_index(index) : integer_index(index);

      auto cond = b.create<scf::IfOp>(TypeRange(updated_tensors),
                                      is_zero(index), true, true);

      Block *then_block = &cond.getThenRegion().front();
      remap_branch_tensors(then_block);
      rewriter.mergeBlocks(&op.getBranches()[0].front(), then_block);

      Block *else_block = &cond.getElseRegion().front();
      remap_branch_tensors(else_block);
      rewriter.mergeBlocks(&op.getBranches()[1].front(), else_block);

      return cond;
    };

    // Converts case operation to scf.index_switch
    auto convert_to_index_switch = [&]() -> scf::IndexSwitchOp {
      SmallVector<int64_t> cases = llvm::to_vector(llvm::seq(num_branches));

      // Create an `scf.index_switch` op in place of `lmhlo.case`.
      auto index_switch = b.create<scf::IndexSwitchOp>(
          TypeRange(updated_tensors), integer_index(index), cases,
          cases.size());

      // Add a default block that will forward all updated tensors to results.
      // We rely on terminator lowering defined below to do it automatically.
      Block *default_block = &index_switch.getDefaultRegion().emplaceBlock();
      remap_branch_tensors(default_block);
      b.setInsertionPointToEnd(default_block);
      b.create<lmhlo::TerminatorOp>();

      // Move all branches into the index switch operation.
      for (auto branch : llvm::enumerate(op.getBranches())) {
        Block *case_block =
            &index_switch.getCaseRegions()[branch.index()].emplaceBlock();
        remap_branch_tensors(case_block);
        rewriter.mergeBlocks(&branch.value().front(), case_block);
      }

      return index_switch;
    };

    // Replace lmhlo.case operation with a structured op.
    Operation *replacement =
        num_branches == 2 ? convert_to_cond() : convert_to_index_switch();

    // Use replacement op results to remap buffers in the parent block.
    for (auto [from, to] :
         llvm::zip_equal(bufs.write, replacement->getResults()))
      state.remapped[block][from] = cast<TypedValue<TensorType>>(to);

    (*converted)[replacement] = ConvertedCaseOp{std::move(bufs)};

    // Erase the original case op.
    rewriter.eraseOp(op);

    return success();
  }

  DeBufferization &state;
  std::shared_ptr<ConvertedCaseOps> converted;
};

//===----------------------------------------------------------------------===//
// Converts lmhlo.terminator in the scf.case branches
//===----------------------------------------------------------------------===//

struct ConvertTerminatorOpToHal
    : public OpConversionPattern<lmhlo::TerminatorOp> {
  ConvertTerminatorOpToHal(TypeConverter &converter, MLIRContext *ctx,
                           DeBufferization &state,
                           std::shared_ptr<ConvertedCaseOps> converted)
      : OpConversionPattern(converter, ctx),
        state(state),
        converted(std::move(converted)) {}

  LogicalResult matchAndRewrite(
      lmhlo::TerminatorOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Operation *parent = op->getParentOp();

    // Check that we are inside one of the supported structured operations.
    if (!isa<scf::IfOp, scf::IndexSwitchOp>(parent))
      return rewriter.notifyMatchFailure(
          op, "not a terminator inside scf.if or scf.index_switch operation");

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto it = converted->find(parent);
    assert(it != converted->end() && "case conversion state was not found");

    auto updated_tensors = llvm::to_vector(
        llvm::map_range(it->second.buffers.write, [&](auto memref) -> Value {
          return state.remapped[op->getBlock()][memref];
        }));

    // Convert lmhlo.terminator in the branch block to scf.yield operation
    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, TypeRange(), updated_tensors);

    return success();
  }

  DeBufferization &state;
  std::shared_ptr<ConvertedCaseOps> converted;
};

}  // namespace

//===----------------------------------------------------------------------===//

void populateCaseOpConversionPatterns(mlir::RewritePatternSet &patterns,
                                      mlir::TypeConverter &converter,
                                      DeBufferization &state) {
  auto *ctx = patterns.getContext();
  auto converted = std::make_shared<ConvertedCaseOps>();
  patterns.insert<ConvertCaseOpToHal, ConvertTerminatorOpToHal>(
      converter, ctx, state, converted);
}

}  // namespace gpu
}  // namespace xla
