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
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // TF:llvm-project
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
#include "tensorflow/compiler/mlir/xla/transforms/hlo_shape_derivation.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"
#include "tensorflow/compiler/mlir/xla/transforms/rewriters.h"

namespace mlir {
namespace xla_hlo {
namespace {

constexpr StringRef kTempBufferAttr = "temp";

/// Returns DeallocOp to ensure that CopyOp is not inserted after dealloc.
Operation* FindInsertionPointForCopy(Value value) {
  for (const auto& user : value.getUsers()) {
    if (auto dealloc = dyn_cast<DeallocOp>(user)) {
      return user;
    }
  }
  return nullptr;
}

Value InsertDynamicAllocAndDealloc(Location loc, Value result,
                                   Value shape_operand,
                                   ConversionPatternRewriter* rewriter) {
  auto result_type = result.getType().dyn_cast<ShapedType>();
  if (!result_type) {
    result.getDefiningOp()->emitOpError()
        << "tensor to buffer conversion expects ranked results";
  }
  auto memref_type =
      MemRefType::get(result_type.getShape(), result_type.getElementType());

  Operation* op = result.getDefiningOp();
  auto block = op->getBlock();

  // Extract the required element out of the vector.
  SmallVector<Value, 4> dynamic_operands;
  for (auto shape_element : llvm::enumerate(result_type.getShape())) {
    if (shape_element.value() != ShapedType::kDynamicSize) continue;
    Value index = rewriter->create<ConstantOp>(
        loc, rewriter->getIntegerAttr(rewriter->getIndexType(),
                                      shape_element.index()));
    Value alloc_operand = rewriter->create<ExtractElementOp>(loc, shape_operand,
                                                             ValueRange{index});
    if (!alloc_operand.getType().isIndex()) {
      alloc_operand = rewriter->create<IndexCastOp>(loc, alloc_operand,
                                                    rewriter->getIndexType());
    }
    dynamic_operands.push_back(alloc_operand);
  }

  // Insert in front of op to ensure sizes are available.
  OpBuilder allocBuilder(op);
  auto alloc = allocBuilder.create<AllocOp>(loc, memref_type, dynamic_operands);

  alloc.setAttr(kTempBufferAttr, rewriter->getBoolAttr(true));

  allocBuilder.setInsertionPoint(block, std::prev(block->end()));
  allocBuilder.create<DeallocOp>(loc, alloc);

  return alloc;
}

Value InsertAllocAndDealloc(Location loc, Value result,
                            ConversionPatternRewriter* rewriter) {
  auto result_type = result.getType().dyn_cast<ShapedType>();
  if (!result_type || !result_type.hasStaticShape()) {
    result.getDefiningOp()->emitOpError()
        << "tensor to buffer conversion expects statically shaped results";
  }
  auto memref_type =
      MemRefType::get(result_type.getShape(), result_type.getElementType());

  Operation* op = result.getDefiningOp();
  auto block = op->getBlock();

  OpBuilder allocBuilder(op);
  allocBuilder.setInsertionPointToStart(block);  // Inserting at the beginning
  auto alloc = allocBuilder.create<AllocOp>(loc, memref_type);

  alloc.setAttr(kTempBufferAttr, rewriter->getBoolAttr(true));

  allocBuilder.setInsertionPoint(block, std::prev(block->end()));
  allocBuilder.create<DeallocOp>(loc, alloc);

  return alloc;
}

template <typename HloOpTy, typename LhloOpTy>
class HloToLhloOpConverter : public ConversionPattern {
 public:
  explicit HloToLhloOpConverter(MLIRContext* context)
      : ConversionPattern(HloOpTy::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(
      Operation* op, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const final {
    const auto& original_results = op->getResults();
    SmallVector<Value, 4> buffer_args(operands.begin(), operands.end());
    for (auto result : llvm::enumerate(original_results)) {
      RankedTensorType resultType =
          result.value().getType().dyn_cast<RankedTensorType>();
      if (!resultType) {
        return matchFailure();
      }
      if (resultType.hasStaticShape()) {
        buffer_args.push_back(
            InsertAllocAndDealloc(op->getLoc(), result.value(), &rewriter));
      } else {
        Value shape_value = ShapeDerivation<HloOpTy>::impl::deriveShapeFromOp(
            op, result.index(), &rewriter);
        if (!shape_value) {
          return matchFailure();
        }
        buffer_args.push_back(InsertDynamicAllocAndDealloc(
            op->getLoc(), result.value(), shape_value, &rewriter));
      }
    }
    rewriter.create<LhloOpTy>(op->getLoc(), llvm::None, buffer_args,
                              op->getAttrs());
    rewriter.replaceOp(op, ArrayRef<Value>(buffer_args).slice(operands.size()));
    return matchSuccess();
  }
};

struct HloToLHloDynamicBroadcastInDimOpConverter
    : public OpConversionPattern<xla_hlo::DynamicBroadcastInDimOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  PatternMatchResult matchAndRewrite(
      xla_hlo::DynamicBroadcastInDimOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const final {
    auto loc = op.getLoc();
    auto broadcast_dimensions = op.broadcast_dimensions();
    if (!broadcast_dimensions.hasValue()) {
      return matchFailure();
    }
    Value resultBuffer = InsertDynamicAllocAndDealloc(
        loc, op.getResult(), op.output_dimensions(), &rewriter);
    rewriter.create<xla_lhlo::BroadcastInDimOp>(
        loc, operands[0], resultBuffer, broadcast_dimensions.getValue());

    rewriter.replaceOp(op, {resultBuffer});

    return matchSuccess();
  }
};

struct HloToLHloReduceOpConverter
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
      op.emitOpError() << "tensor to buffer conversion expects a single block "
                          "in the region containing the operation";
      return matchFailure();
    }
    const auto& original_results = op.getResults();
    SmallVector<Value, 4> buffer_args(operands.begin(), operands.end());
    for (auto result : original_results) {
      buffer_args.push_back(InsertAllocAndDealloc(loc, result, &rewriter));
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
      auto old_type = old_arg.getType().cast<TensorType>();
      auto new_type =
          MemRefType::get(old_type.getShape(), old_type.getElementType());
      auto new_arg = entry_block.addArgument(new_type);
      rewriter.replaceUsesOfBlockArgument(old_arg, new_arg);
    }
    // Add an argument for the result.
    entry_block.addArgument(
        entry_block.getArgument(original_arg_count).getType());
    // Remove the old arguments.
    for (int i = original_arg_count - 1; i >= 0; --i) {
      entry_block.eraseArgument(i);
    }
    // Insert terminator at the end.
    rewriter.setInsertionPointToEnd(&entry_block);
    rewriter.create<xla_lhlo::TerminatorOp>(loc);

    rewriter.replaceOp(op, ArrayRef<Value>(buffer_args).slice(operands.size()));

    return matchSuccess();
  }
};

class HloToLhloTensorLoadOpConverter : public ConversionPattern {
 public:
  explicit HloToLhloTensorLoadOpConverter(MLIRContext* context)
      : ConversionPattern(TensorLoadOp::getOperationName(), 1, context) {}
  PatternMatchResult matchAndRewrite(
      Operation* op, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const final {
    rewriter.replaceOp(op, operands);
    return matchSuccess();
  }
};

// TODO(b/137624192): Rewrite into a copy and elide copy if possible.
class HloToLhloTensorStoreOpConverter : public ConversionPattern {
 public:
  explicit HloToLhloTensorStoreOpConverter(MLIRContext* context)
      : ConversionPattern(TensorStoreOp::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(
      Operation* op, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const final {
    rewriter.replaceOpWithNewOp<xla_lhlo::CopyOp>(
        op, llvm::None, operands.front(), operands.back());
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
//
// FuncOp signature conversion example:
//
// func @func_op(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
//   %0 = xla_hlo.max %arg0, %arg1 {name = "maximum.47"} : tensor<4xf32>
//   %1 = xla_hlo.add %arg0, %0 {name = "maximum.47"} : tensor<4xf32>
//   return %1 : tensor<4xf32>
// }
//
// Transformed function with an extra argument for the result. The types have
// been converted from tensor to memref.
//
// func @func_op(%arg0: memref<4xf32>,
//               %arg1: memref<4xf32>,
//               %arg2: memref<4xf32>) {
//   %0 = alloc() {temp = true} : memref<4xf32>
//   "xla_lhlo.max"(%arg0, %arg1, %0) {name = "maximum.47"} :
//         (memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
//   "xla_lhlo.add"(%arg0, %0, %arg2) {name = "maximum.47"} :
//         (memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
//   dealloc %1 : memref<4xf32>
//   "xla_lhlo.terminator"() : () -> ()
// }

struct HloLegalizeToLhlo : public ModulePass<HloLegalizeToLhlo> {
  void runOnModule() override {
    OwningRewritePatternList patterns;
    auto& context = getContext();
    ConversionTarget target(context);
    target.addLegalDialect<xla_lhlo::XlaLhloDialect>();
    target.addLegalDialect<StandardOpsDialect>();
    target.addLegalOp<ModuleOp>();
    target.addIllegalOp<mlir::ReturnOp>();
    target.addIllegalOp<mlir::TensorLoadOp>();
    target.addIllegalOp<mlir::TensorStoreOp>();
    target.addLegalOp<ModuleTerminatorOp>();
    target.addLegalOp<ScalarsToDimensionTensorOp>();
    target.addIllegalDialect<xla_hlo::XlaHloDialect>();
    target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
      auto inputs = op.getType().getInputs();
      return std::all_of(inputs.begin(), inputs.end(),
                         [](Type input) { return input.isa<MemRefType>(); });
    });

    auto module = getModule();
    populateHLOToLHLOConversionPattern(module.getContext(), &patterns);

    // Do partial conversion so we can have unknown ops in tests.
    if (failed(applyPartialConversion(module, target, patterns, nullptr))) {
      signalPassFailure();
    }
  }
};

Type ConvertType(Type t) {
  if (auto tensorType = t.dyn_cast<RankedTensorType>()) {
    return MemRefType::get(tensorType.getShape(), tensorType.getElementType());
  }
  return t;
}

}  // namespace

/// Transforms FuncOp arguments and results from tensors to buffers. Tensor
/// results are converted to memrefs and appended to the argument list.
class HloToLhloFuncOpConverter : public OpConversionPattern<FuncOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  PatternMatchResult matchAndRewrite(
      FuncOp funcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const final {
    if (funcOp.getBody().getBlocks().size() > 1) {
      funcOp.emitOpError() << "tensor to buffer conversion expects a single "
                              "block in the region containing the operation";
      return matchFailure();
    }

    auto funcType = funcOp.getType();

    TypeConverter::SignatureConversion conversion(funcType.getNumInputs());
    for (auto argType : llvm::enumerate(funcType.getInputs())) {
      conversion.addInputs(argType.index(), ConvertType(argType.value()));
    }
    for (auto resType : funcType.getResults()) {
      conversion.addInputs(ConvertType(resType));
    }
    rewriter.updateRootInPlace(funcOp, [&] {
      funcOp.setType(
          rewriter.getFunctionType(conversion.getConvertedTypes(), llvm::None));
      rewriter.applySignatureConversion(&funcOp.getBody(), conversion);
    });
    return matchSuccess();
  }
};

/// Transforms ReturnOp to LhloTerminator. CopyOp is inserted to copy each
/// result to the corresponding buffer argument.
class StdToLhloReturnOpConverter : public OpConversionPattern<mlir::ReturnOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  PatternMatchResult matchAndRewrite(
      mlir::ReturnOp returnOp, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const final {
    auto numReturnValues = returnOp.getNumOperands();
    auto funcOp = returnOp.getParentOfType<FuncOp>();
    auto numFuncArgs = funcOp.getNumArguments();
    auto loc = returnOp.getLoc();

    for (auto operand : llvm::enumerate(operands)) {
      auto returnArgNumber = numFuncArgs - numReturnValues + operand.index();
      auto dstBuffer = funcOp.getArgument(returnArgNumber);
      if (dstBuffer == operand.value()) {
        continue;
      }

      auto dealloc = FindInsertionPointForCopy(operand.value());

      if (dealloc == nullptr) {
        returnOp.emitOpError()
            << "Missing dealloc for operand " << operand.index();
        return matchFailure();
      }
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(dealloc);
      rewriter.create<xla_lhlo::CopyOp>(loc, llvm::None, operand.value(),
                                        funcOp.getArgument(returnArgNumber));
    }
    rewriter.replaceOpWithNewOp<xla_lhlo::TerminatorOp>(returnOp);
    return matchSuccess();
  }
};

void populateHLOToLHLOConversionPattern(MLIRContext* context,
                                        OwningRewritePatternList* patterns) {
  // clang-format off
  patterns->insert<
      HloToLHloDynamicBroadcastInDimOpConverter,
      HloToLhloFuncOpConverter,
      HloToLhloOpConverter<xla_hlo::AbsOp, xla_lhlo::AbsOp>,
      HloToLhloOpConverter<xla_hlo::AddOp, xla_lhlo::AddOp>,
      HloToLhloOpConverter<xla_hlo::AndOp, xla_lhlo::AndOp>,
      HloToLhloOpConverter<xla_hlo::BroadcastInDimOp,
                           xla_lhlo::BroadcastInDimOp>,
      HloToLhloOpConverter<xla_hlo::CeilOp, xla_lhlo::CeilOp>,
      HloToLhloOpConverter<xla_hlo::CompareOp, xla_lhlo::CompareOp>,
      HloToLhloOpConverter<xla_hlo::ConstOp, xla_lhlo::ConstOp>,
      HloToLhloOpConverter<xla_hlo::ConvertOp, xla_lhlo::ConvertOp>,
      HloToLhloOpConverter<xla_hlo::CopyOp, xla_lhlo::CopyOp>,
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
      HloToLHloReduceOpConverter,
      StdToLhloReturnOpConverter,
      HloToLhloTensorLoadOpConverter,
      HloToLhloTensorStoreOpConverter
  >(context);
  // clang-format on
}

/// Removes Lhlo.CopyOp that copies from an allocated buffer to the block
/// argument. All uses of the buffer are replaced with the block argument.
struct RedundantCopiesRemoval : mlir::FunctionPass<RedundantCopiesRemoval> {
  void runOnFunction() override {
    llvm::SmallVector<mlir::Operation*, 2> eraseList;
    getFunction().walk([&](mlir::xla_lhlo::CopyOp copyOp) {
      auto arguments = copyOp.getOperation()->getBlock()->getArguments();
      if (std::any_of(arguments.begin(), arguments.end(),
                      [&](mlir::BlockArgument arg) {
                        return copyOp.output() == arg;
                      }) &&
          std::none_of(arguments.begin(), arguments.end(),
                       [&](mlir::BlockArgument arg) {
                         return copyOp.operand() == arg;
                       })) {
        mlir::Value operand = copyOp.operand();
        mlir::Value output = copyOp.output();
        copyOp.erase();
        for (auto op : operand.getUsers()) {
          if (!mlir::isa<mlir::DeallocOp>(op)) {
            op->replaceUsesOfWith(operand, output);
          }
        }
        auto allocOp = operand.getDefiningOp();
        if (auto deallocOp =
                mlir::dyn_cast<mlir::DeallocOp>(*allocOp->getUsers().begin())) {
          eraseList.push_back(deallocOp);
          eraseList.push_back(allocOp);
        }
      }
    });
    for (auto op : eraseList) {
      op->erase();
    }
  };
};

std::unique_ptr<OpPassBase<ModuleOp>> createLegalizeToLhloPass() {
  return absl::make_unique<HloLegalizeToLhlo>();
}

std::unique_ptr<OpPassBase<FuncOp>> createLhloCopyRemovalPass() {
  return absl::make_unique<RedundantCopiesRemoval>();
}

static PassRegistration<HloLegalizeToLhlo> legalize_pass(
    "hlo-legalize-to-lhlo", "Legalize from HLO dialect to LHLO dialect");

static PassRegistration<RedundantCopiesRemoval> copies_removal_pass(
    "lhlo-redundant-copies-removal",
    "Legalize from HLO dialect to LHLO dialect");

}  // namespace xla_hlo
}  // namespace mlir
