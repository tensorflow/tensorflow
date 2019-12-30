/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This file implements logic for lowering HLO dialect to LHLO dialect.

#include "absl/memory/memory.h"
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:llvm-project
#include "mlir/IR/Attributes.h"  // TF:llvm-project
#include "mlir/IR/BlockAndValueMapping.h"  // TF:llvm-project
#include "mlir/IR/Builders.h"  // TF:llvm-project
#include "mlir/IR/Function.h"  // TF:llvm-project
#include "mlir/IR/Location.h"  // TF:llvm-project
#include "mlir/IR/MLIRContext.h"  // TF:llvm-project
#include "mlir/IR/Operation.h"  // TF:llvm-project
#include "mlir/IR/PatternMatch.h"  // TF:llvm-project
#include "mlir/IR/StandardTypes.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "mlir/Transforms/DialectConversion.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/ir/lhlo_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"
#include "tensorflow/compiler/mlir/xla/transforms/rewriters.h"

namespace mlir {
namespace xla_hlo {
namespace {

constexpr StringRef kTempBufferAttr = "temp";

Value GetTensorStoreOrReturnMemRef(Value value) {
  for (const auto& user : value->getUsers()) {
    if (auto tensor_store = dyn_cast<TensorStoreOp>(user)) {
      if (tensor_store.getOperand(0) == value) {
        return tensor_store.getOperand(1);
      }
    }
    if (auto return_op = dyn_cast<xla_hlo::ReturnOp>(user)) {
      if (return_op.getOperand(0) == value) {
        auto block = return_op.getOperation()->getBlock();
        return *block->args_rbegin();
      }
    }
  }
  return nullptr;
}

Operation* GetLastUse(Value value) {
  Operation* last = value->getDefiningOp();
  for (auto& user : value->getUses()) {
    Operation* user_op = user.getOwner();
    if (!user_op->isBeforeInBlock(last)) {
      last = user_op;
    }
  }
  return last;
}

Value InsertAllocAndDealloc(Location loc, Value result,
                            ConversionPatternRewriter* rewriter) {
  auto result_type = result->getType().dyn_cast<ShapedType>();
  if (!result_type || !result_type.hasStaticShape()) {
    emitError(loc,
              "tensor to buffer conversion expects statically shaped results");
  }
  auto memref_type =
      MemRefType::get(result_type.getShape(), result_type.getElementType());

  Operation* last = GetLastUse(result);

  Operation* op = result->getDefiningOp();
  OpBuilder allocBuilder(op);
  auto alloc = allocBuilder.create<AllocOp>(loc, memref_type);
  alloc.setAttr(kTempBufferAttr, rewriter->getBoolAttr(true));

  allocBuilder.setInsertionPoint(op->getBlock(),
                                 std::next(Block::iterator(last)));
  allocBuilder.create<DeallocOp>(loc, alloc);
  return alloc;
}

/// For every tensor-type value that is produced in the original function,
/// this function returns the buffer that can be used in the converted
/// function to store that values held in the tensor.
Value GetBufferForResultValue(Location loc, Value result,
                              ConversionPatternRewriter* rewriter) {
  if (auto existing_memref = GetTensorStoreOrReturnMemRef(result)) {
    return existing_memref;
  }
  return InsertAllocAndDealloc(loc, result, rewriter);
}

template <typename HloOpTy, typename LhloOpTy>
class HloToLhloOpConverter : public ConversionPattern {
 public:
  explicit HloToLhloOpConverter(MLIRContext* context)
      : ConversionPattern(HloOpTy::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(
      Operation* op, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const final {
    if (op->getParentRegion()->getBlocks().size() != 1) {
      emitError(op->getLoc(),
                "tensor to buffer conversion expects a single block in the "
                "region containing the operation");
    }
    const auto& original_results = op->getResults();
    SmallVector<Value, 4> buffer_args(operands.begin(), operands.end());
    for (auto result : original_results) {
      buffer_args.push_back(
          GetBufferForResultValue(op->getLoc(), result, &rewriter));
    }
    rewriter.create<LhloOpTy>(op->getLoc(), llvm::None, buffer_args,
                              op->getAttrs());
    rewriter.replaceOp(op, ArrayRef<Value>(buffer_args).slice(operands.size()),
                       original_results);
    return matchSuccess();
  }
};

struct HloToLHloReduceConverter
    : public OpConversionPattern<xla_hlo::ReduceOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  PatternMatchResult matchAndRewrite(
      xla_hlo::ReduceOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const final {
    auto loc = op.getLoc();
    // TODO(b/137624192) Implement variadic reduce.
    if (op.getNumResults() != 1) return matchFailure();
    if (op.getParentRegion()->getBlocks().size() != 1) {
      emitError(loc,
                "tensor to buffer conversion expects a single block in the "
                "region containing the operation");
    }
    const auto& original_results = op.getResults();
    SmallVector<Value, 4> buffer_args(operands.begin(), operands.end());
    for (auto result : original_results) {
      buffer_args.push_back(GetBufferForResultValue(loc, result, &rewriter));
    }
    auto new_op = rewriter.create<xla_lhlo::ReduceOp>(
        loc, llvm::None, buffer_args, op.getAttrs());

    // Copy over the operations inside the region.
    rewriter.inlineRegionBefore(op.body(), new_op.body(), new_op.body().end());

    // Create new block arguments with correct type.
    auto& entry_block = new_op.body().front();
    int original_arg_count = entry_block.getNumArguments();
    for (int i = 0; i < original_arg_count; ++i) {
      auto old_arg = entry_block.getArgument(i);
      auto old_type = old_arg->getType().cast<TensorType>();
      auto new_type =
          MemRefType::get(old_type.getShape(), old_type.getElementType());
      auto new_arg = entry_block.addArgument(new_type);
      rewriter.replaceUsesOfBlockArgument(old_arg, new_arg);
    }
    // Add an argument for the result.
    entry_block.addArgument(
        entry_block.getArgument(original_arg_count)->getType());
    // Remove the old arguments.
    for (int i = original_arg_count - 1; i >= 0; --i) {
      entry_block.eraseArgument(i);
    }
    // Insert terminator at the end.
    rewriter.setInsertionPointToEnd(&entry_block);
    rewriter.create<xla_lhlo::TerminatorOp>(loc);

    rewriter.replaceOp(op, ArrayRef<Value>(buffer_args).slice(operands.size()),
                       original_results);

    return matchSuccess();
  }
};

class HloToLhloTensorLoadConverter : public ConversionPattern {
 public:
  explicit HloToLhloTensorLoadConverter(MLIRContext* context)
      : ConversionPattern(TensorLoadOp::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(
      Operation* op, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const final {
    rewriter.replaceOp(op, operands, op->getResults());
    return matchSuccess();
  }
};

// TODO(b/137624192): Rewrite into a copy and elide copy if possible.
class HloToLhloTensorStoreConverter : public ConversionPattern {
 public:
  explicit HloToLhloTensorStoreConverter(MLIRContext* context)
      : ConversionPattern(TensorStoreOp::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(
      Operation* op, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const final {
    rewriter.eraseOp(op);
    return matchSuccess();
  }
};

// TODO(b/137624192): Rewrite into a copy and elide copy if possible.
class HloToLhloReturnConverter : public OpConversionPattern<xla_hlo::ReturnOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  PatternMatchResult matchAndRewrite(
      xla_hlo::ReturnOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const final {
    rewriter.eraseOp(op);
    return matchSuccess();
  }
};

// Lowers from HLO dialect to LHLO dialect allocating/deallocating temporary
// buffers if necessary.
//
// Example fusion with HLO ops.
//
// func @fusion(%arg0: memref<2x2xf32>,
//              %arg1: memref<2x2xf32>,
//              %arg2: memref<2x2xf32>,
//              %arg3: memref<2x2xf32>) {
//   "xla_lhlo.fusion"() ({
//     %0 = tensor_load %arg1 : memref<2x2xf32>
//     %1 = tensor_load %arg2 : memref<2x2xf32>
//     %2 = "xla_hlo.add"(%0, %1) {name = "add"} :
//         (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
//     %3 = tensor_load %arg0 : memref<2x2xf32>
//     %4 = "xla_hlo.mul"(%2, %3) {name = "multiply"} :
//         (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
//     tensor_store %4, %arg3 : memref<2x2xf32>
//     "xla_lhlo.terminator"() : () -> ()
//   }) {name = "fusion"} : () -> ()
//   return
// }
//
// Transformed fusion with LHLO ops.
// func @fusion(%arg0: memref<2x2xf32>,
//              %arg1: memref<2x2xf32>,
//              %arg2: memref<2x2xf32>,
//              %arg3: memref<2x2xf32>) {
//   "xla_lhlo.fusion"() ( {
//     %0 = alloc() {temp = true} : memref<2x2xf32>
//     "xla_lhlo.add"(%arg1, %arg2, %0) :
//         (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
//     "xla_lhlo.mul"(%0, %arg0, %arg3) :
//         (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
//     dealloc %0 : memref<2x2xf32>
//     "xla_lhlo.terminator"() : () -> ()
//   }) {name = "fusion"} : () -> ()
//   return
//  }
// }
struct HloLegalizeToLhlo : public FunctionPass<HloLegalizeToLhlo> {
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    ConversionTarget target(getContext());
    target.addLegalDialect<xla_lhlo::XlaLhloDialect>();

    auto func = getFunction();
    populateHLOToLHLOConversionPattern(func.getContext(), &patterns);
    if (failed(applyPartialConversion(func, target, patterns, nullptr))) {
      signalPassFailure();
    }
  }
};

}  // namespace

void populateHLOToLHLOConversionPattern(MLIRContext* context,
                                        OwningRewritePatternList* patterns) {
  // clang-format off
  patterns->insert<
      HloToLhloOpConverter<xla_hlo::AbsOp, xla_lhlo::AbsOp>,
      HloToLhloOpConverter<xla_hlo::AddOp, xla_lhlo::AddOp>,
      HloToLhloOpConverter<xla_hlo::AndOp, xla_lhlo::AndOp>,
      HloToLhloOpConverter<xla_hlo::BroadcastInDimOp,
                           xla_lhlo::BroadcastInDimOp>,
      HloToLhloOpConverter<xla_hlo::CeilOp, xla_lhlo::CeilOp>,
      HloToLhloOpConverter<xla_hlo::CompareOp, xla_lhlo::CompareOp>,
      HloToLhloOpConverter<xla_hlo::ConstOp, xla_lhlo::ConstOp>,
      HloToLhloOpConverter<xla_hlo::ConvertOp, xla_lhlo::ConvertOp>,
      HloToLhloOpConverter<xla_hlo::CosOp, xla_lhlo::CosOp>,
      HloToLhloOpConverter<xla_hlo::DivOp, xla_lhlo::DivOp>,
      HloToLhloOpConverter<xla_hlo::ExpOp, xla_lhlo::ExpOp>,
      HloToLhloOpConverter<xla_hlo::IotaOp, xla_lhlo::IotaOp>,
      HloToLhloOpConverter<xla_hlo::MaxOp, xla_lhlo::MaxOp>,
      HloToLhloOpConverter<xla_hlo::MinOp, xla_lhlo::MinOp>,
      HloToLhloOpConverter<xla_hlo::MulOp, xla_lhlo::MulOp>,
      HloToLhloOpConverter<xla_hlo::NegOp, xla_lhlo::NegOp>,
      HloToLhloOpConverter<xla_hlo::RemOp, xla_lhlo::RemOp>,
      HloToLhloOpConverter<xla_hlo::SelectOp, xla_lhlo::SelectOp>,
      HloToLhloOpConverter<xla_hlo::SignOp, xla_lhlo::SignOp>,
      HloToLhloOpConverter<xla_hlo::SubOp, xla_lhlo::SubOp>,
      HloToLhloOpConverter<xla_hlo::TanhOp, xla_lhlo::TanhOp>,
      HloToLHloReduceConverter, HloToLhloReturnConverter,
      HloToLhloTensorLoadConverter, HloToLhloTensorStoreConverter
  >(context);
  // clang-format on
}

std::unique_ptr<OpPassBase<FuncOp>> createLegalizeToLhloPass() {
  return absl::make_unique<HloLegalizeToLhlo>();
}

static PassRegistration<HloLegalizeToLhlo> legalize_pass(
    "hlo-legalize-to-lhlo", "Legalize from HLO dialect to LHLO dialect");

}  // namespace xla_hlo
}  // namespace mlir
