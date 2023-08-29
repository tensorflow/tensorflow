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

#include "tensorflow/compiler/xla/mlir/backends/gpu2/conversion/convert_while_op.h"

#include <cassert>
#include <memory>
#include <utility>

#include "iree-dialects/Dialect/Input/InputOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/backends/gpu2/conversion/de_bufferization.h"
#include "tensorflow/compiler/xla/mlir/backends/gpu2/conversion/xla_gpu_api.h"
#include "tensorflow/compiler/xla/mlir/backends/gpu2/ir/xla_gpu_dialect.h"
#include "tensorflow/compiler/xla/mlir_hlo/lhlo/IR/lhlo_ops.h"

namespace xla {
namespace gpu {

namespace {
using namespace mlir;                 // NOLINT
using namespace mlir::iree_compiler;  // NOLINT

// TODO(ezhulenev): Rewrite while loops with statically known trip count to
// scf.for loops (see `op.getTripCount()` attribute).

// Keep track of converted while operations to correctly lower terminators in
// the loop before and after regions (condition and body regions).
struct ConvertedWhileOp {
  TypedValue<MemRefType> predicate;
  UsedBuffers buffers;
};

using ConvertedWhileOps = llvm::DenseMap<scf::WhileOp, ConvertedWhileOp>;

//===----------------------------------------------------------------------===//
// Converts lmhlo.while op to a scf.while + iree_input.tensor.load
//===----------------------------------------------------------------------===//

struct ConvertWhileOpToHal : public OpConversionPattern<lmhlo::WhileOp> {
  ConvertWhileOpToHal(TypeConverter &converter, MLIRContext *ctx,
                      DeBufferization &state,
                      std::shared_ptr<ConvertedWhileOps> converted)
      : OpConversionPattern(converter, ctx),
        state(state),
        converted(std::move(converted)) {}

  LogicalResult matchAndRewrite(
      lmhlo::WhileOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // Collect all buffers accessed in the loop condition and loop body.
    auto bufs = getUsedBuffers({&op.getCond().front(), &op.getBody().front()});

    Block *block = op->getBlock();

    // Pass updated tensors as loop iteration argument.
    SmallVector<Value> iter_args =
        llvm::to_vector(llvm::map_range(bufs.write, [&](auto memref) -> Value {
          return state.remapped[block][memref];
        }));

    // Set up buffer to tensor remapping inside nested regions.
    auto remap_iteration_args = [&](Block *nested_block, ValueRange iter_args) {
      // Read-only buffers remapped to tensors defined in the parent block.
      for (auto r : bufs.read)
        state.remapped[nested_block][r] = state.remapped[block][r];

      // Written-to buffers remapped to iteration arguments.
      for (auto [from, to] : llvm::zip_equal(bufs.write, iter_args))
        state.remapped[nested_block][from] = cast<TypedValue<TensorType>>(to);
    };

    // Create an `scf.while` loop in place of `lmhlo.while` loop.
    auto loop = b.create<scf::WhileOp>(
        TypeRange(iter_args), iter_args,
        [&](OpBuilder &before_builder, Location before_loc, ValueRange args) {
          Block *cond = before_builder.getBlock();
          rewriter.mergeBlocks(&op.getCond().front(), cond);
          remap_iteration_args(cond, args);
        },
        [&](OpBuilder &after_builder, Location after_loc, ValueRange args) {
          Block *body = after_builder.getBlock();
          rewriter.mergeBlocks(&op.getBody().front(), body);
          remap_iteration_args(body, args);
        });

    // Use loop results to remap buffers in the parent block.
    for (auto [from, to] : llvm::zip_equal(bufs.write, loop.getResults()))
      state.remapped[block][from] = cast<TypedValue<TensorType>>(to);

    // Predicate buffer placed on the device.
    auto predicate = cast<TypedValue<MemRefType>>(op.getOperand(0));
    (*converted)[loop] = ConvertedWhileOp{predicate, std::move(bufs)};

    // Erase the original while loop.
    rewriter.eraseOp(op);

    return success();
  }

  DeBufferization &state;
  std::shared_ptr<ConvertedWhileOps> converted;
};

//===----------------------------------------------------------------------===//
// Converts lmhlo.terminator in the scf.while regions and HAL backend
//===----------------------------------------------------------------------===//

struct ConvertTerminatorOpToHal
    : public OpConversionPattern<lmhlo::TerminatorOp> {
  ConvertTerminatorOpToHal(TypeConverter &converter, MLIRContext *ctx,
                           DeBufferization &state,
                           std::shared_ptr<ConvertedWhileOps> converted)
      : OpConversionPattern(converter, ctx),
        state(state),
        converted(std::move(converted)) {}

  LogicalResult matchAndRewrite(
      lmhlo::TerminatorOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loop = dyn_cast<scf::WhileOp>(op->getParentOp());
    if (!loop)
      return rewriter.notifyMatchFailure(
          op, "not a terminator inside scf.while operation");

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto it = converted->find(loop);
    assert(it != converted->end() && "loop conversion state was not found");

    auto iter_args = llvm::to_vector(llvm::map_range(
        (*converted)[loop].buffers.write, [&](auto memref) -> Value {
          return state.remapped[op->getBlock()][memref];
        }));

    // Convert lmhlo.terminator in the before block to scf.condition operation
    if (auto *cond = op->getBlock(); cond == &loop.getBefore().front()) {
      Value offset = b.create<arith::ConstantIndexOp>(0);
      auto predicate = b.create<IREE::Input::TensorLoadOp>(
          state.remapped[cond][it->second.predicate],
          /*source_dims=*/ValueRange(), /*indices=*/offset);

      rewriter.replaceOpWithNewOp<scf::ConditionOp>(op, predicate, iter_args);
      return success();
    }

    // Convert lmhlo.terminator in the after block to scf.yield operation
    if (auto *body = op->getBlock(); body == &loop.getAfter().front()) {
      rewriter.replaceOpWithNewOp<scf::YieldOp>(op, TypeRange(), iter_args);
      return success();
    }

    return failure();
  }

  DeBufferization &state;
  std::shared_ptr<ConvertedWhileOps> converted;
};

//===----------------------------------------------------------------------===//
// Converts lmhlo.while op to a scf.while + @xla_gpu.memcpy.load.i1
//===----------------------------------------------------------------------===//

struct ConvertWhileOpToApiCall : public OpConversionPattern<lmhlo::WhileOp> {
  ConvertWhileOpToApiCall(TypeConverter &converter, MLIRContext *ctx,
                          DeBufferization &state, XlaGpuApi &api,
                          std::shared_ptr<ConvertedWhileOps> converted)
      : OpConversionPattern(converter, ctx),
        state(state),
        api(api),
        converted(std::move(converted)) {}

  LogicalResult matchAndRewrite(
      lmhlo::WhileOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // Collect all buffers accessed in the loop condition and loop body.
    auto bufs = getUsedBuffers({&op.getCond().front(), &op.getBody().front()});

    Block *block = op->getBlock();

    // Set up buffer to tensor remapping inside nested regions.
    auto remap_iteration_args = [&](Block *nested_block) {
      for (auto r : bufs.read)
        state.remapped[nested_block][r] = state.remapped[block][r];
      for (auto w : bufs.write)
        state.remapped[nested_block][w] = state.remapped[block][w];
    };

    // Create an `scf.while` loop in place of `lmhlo.while` loop.
    auto loop = rewriter.replaceOpWithNewOp<scf::WhileOp>(
        op, TypeRange(), ValueRange(),
        [&](OpBuilder &before_builder, Location before_loc, ValueRange args) {
          Block *cond = before_builder.getBlock();
          rewriter.mergeBlocks(&op.getCond().front(), cond);
          remap_iteration_args(cond);
        },
        [&](OpBuilder &after_builder, Location after_loc, ValueRange args) {
          Block *body = after_builder.getBlock();
          rewriter.mergeBlocks(&op.getBody().front(), body);
          remap_iteration_args(body);
        });

    // Predicate buffer placed on the device.
    auto predicate = cast<TypedValue<MemRefType>>(op.getOperand(0));
    (*converted)[loop] = ConvertedWhileOp{predicate, std::move(bufs)};

    return success();
  }

  DeBufferization &state;
  XlaGpuApi &api;
  std::shared_ptr<ConvertedWhileOps> converted;
};

//===----------------------------------------------------------------------===//
// Converts lmhlo.terminator in the scf.while regions and StreamExecutor backend
//===----------------------------------------------------------------------===//

TypedValue<ExecutionContextType> getExecutionContext(Operation *op) {
  auto func = op->getParentOfType<func::FuncOp>();
  return func.getArguments().front().cast<TypedValue<ExecutionContextType>>();
}

struct ConvertTerminatorOpToApiCall
    : public OpConversionPattern<lmhlo::TerminatorOp> {
  ConvertTerminatorOpToApiCall(TypeConverter &converter, MLIRContext *ctx,
                               DeBufferization &state, XlaGpuApi &api,
                               std::shared_ptr<ConvertedWhileOps> converted)
      : OpConversionPattern(converter, ctx),
        state(state),
        api(api),
        converted(std::move(converted)) {}

  LogicalResult matchAndRewrite(
      lmhlo::TerminatorOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loop = dyn_cast<scf::WhileOp>(op->getParentOp());
    if (!loop) return failure();

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    assert(converted->contains(loop) && "loop conversion state was not found");

    auto module = op->getParentOfType<ModuleOp>();

    // Convert lmhlo.terminator in the before block to scf.condition operation
    if (auto *cond = op->getBlock(); cond == &loop.getBefore().front()) {
      auto predicate = state.remapped[cond][(*converted)[loop].predicate];
      SmallVector<Value> args = {getExecutionContext(op),
                                 api.getBufferView(b, predicate),
                                 b.create<arith::ConstantIntOp>(0, 32)};

      auto api_func = api.getLoadI1Memcpy(b, module);
      auto call = b.create<func::CallOp>(api_func.getSymName(),
                                         api_func.getResultTypes(), args);

      rewriter.replaceOpWithNewOp<scf::ConditionOp>(op, call.getResult(0),
                                                    ValueRange());
      return success();
    }

    // Convert lmhlo.terminator in the after block to scf.yield operation
    if (auto *body = op->getBlock(); body == &loop.getAfter().front()) {
      rewriter.replaceOpWithNewOp<scf::YieldOp>(op, TypeRange(), ValueRange());
      return success();
    }

    return success();
  }

  DeBufferization &state;
  XlaGpuApi &api;
  std::shared_ptr<ConvertedWhileOps> converted;
};

}  // namespace

//===----------------------------------------------------------------------===//

void populateWhileOpConversionPatterns(mlir::RewritePatternSet &patterns,
                                       mlir::TypeConverter &converter,
                                       DeBufferization &state) {
  auto *ctx = patterns.getContext();
  auto converted = std::make_shared<ConvertedWhileOps>();
  patterns.insert<ConvertWhileOpToHal, ConvertTerminatorOpToHal>(
      converter, ctx, state, converted);
}

void populateWhileOpConversionPatterns(mlir::RewritePatternSet &patterns,
                                       mlir::TypeConverter &converter,
                                       DeBufferization &state, XlaGpuApi &api) {
  auto *ctx = patterns.getContext();
  auto converted = std::make_shared<ConvertedWhileOps>();
  patterns.insert<ConvertWhileOpToApiCall, ConvertTerminatorOpToApiCall>(
      converter, ctx, state, api, converted);
}

}  // namespace gpu
}  // namespace xla
