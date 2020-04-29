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
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BlockAndValueMapping.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Function.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/ir/lhlo_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/map_hlo_to_lhlo_op.h"
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

template <typename HloOpTy>
class HloToLhloOpConverter : public ConversionPattern {
 public:
  explicit HloToLhloOpConverter(MLIRContext* context)
      : ConversionPattern(HloOpTy::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(
      Operation* op, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const final {
    const auto& original_results = op->getResults();
    SmallVector<Value, 4> buffer_args(operands.begin(), operands.end());
    for (auto result : llvm::enumerate(original_results)) {
      RankedTensorType resultType =
          result.value().getType().dyn_cast<RankedTensorType>();
      if (!resultType) {
        return failure();
      }
      if (resultType.hasStaticShape()) {
        buffer_args.push_back(
            InsertAllocAndDealloc(op->getLoc(), result.value(), &rewriter));
      } else {
        SmallVector<Value, 1> results_shape;
        auto shape_type_op = dyn_cast<InferShapedTypeOpInterface>(op);
        if (!shape_type_op) return failure();
        if (failed(
                shape_type_op.reifyReturnTypeShapes(rewriter, results_shape)))
          return failure();
        buffer_args.push_back(InsertDynamicAllocAndDealloc(
            op->getLoc(), result.value(), results_shape.front(), &rewriter));
      }
    }
    rewriter.create<xla_hlo::HloToLhloOp<HloOpTy>>(op->getLoc(), llvm::None,
                                                   buffer_args, op->getAttrs());
    rewriter.replaceOp(op, ArrayRef<Value>(buffer_args).slice(operands.size()));
    return success();
  }
};

struct HloToLhloDynamicBroadcastInDimOpConverter
    : public OpConversionPattern<xla_hlo::DynamicBroadcastInDimOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      xla_hlo::DynamicBroadcastInDimOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const final {
    auto loc = op.getLoc();
    Value resultBuffer = InsertDynamicAllocAndDealloc(
        loc, op.getResult(), op.output_dimensions(), &rewriter);
    rewriter.create<xla_lhlo::BroadcastInDimOp>(loc, operands[0], resultBuffer,
                                                op.broadcast_dimensions());

    rewriter.replaceOp(op, {resultBuffer});

    return success();
  }
};

struct HloToLhloReduceOpConverter
    : public OpConversionPattern<xla_hlo::ReduceOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      xla_hlo::ReduceOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const final {
    auto loc = op.getLoc();
    // TODO(b/137624192) Implement variadic reduce.
    if (op.getNumResults() != 1) return failure();
    if (op.getParentRegion()->getBlocks().size() != 1) {
      op.emitOpError() << "tensor to buffer conversion expects a single block "
                          "in the region containing the operation";
      return failure();
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

    return success();
  }
};

class HloToLhloTensorLoadOpConverter : public ConversionPattern {
 public:
  explicit HloToLhloTensorLoadOpConverter(MLIRContext* context)
      : ConversionPattern(TensorLoadOp::getOperationName(), 1, context) {}
  LogicalResult matchAndRewrite(
      Operation* op, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const final {
    rewriter.replaceOp(op, operands);
    return success();
  }
};

// TODO(b/137624192): Rewrite into a copy and elide copy if possible.
class HloToLhloTensorStoreOpConverter : public ConversionPattern {
 public:
  explicit HloToLhloTensorStoreOpConverter(MLIRContext* context)
      : ConversionPattern(TensorStoreOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(
      Operation* op, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const final {
    rewriter.replaceOpWithNewOp<xla_lhlo::CopyOp>(
        op, llvm::None, operands.front(), operands.back());
    return success();
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
//     %2 = "xla_hlo.add"(%0, %1) :
//         (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
//     %3 = tensor_load %arg0 : memref<2x2xf32>
//     %4 = "xla_hlo.multiply"(%2, %3) :
//         (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
//     tensor_store %4, %arg3 : memref<2x2xf32>
//     "xla_lhlo.terminator"() : () -> ()
//   }) : () -> ()
//   return
// }
//
// Transformed fusion with LHLO ops.
// func @fusion(%arg0: memref<2x2xf32>,
//              %arg1: memref<2x2xf32>,
//              %arg2: memref<2x2xf32>,
//              %arg3: memref<2x2xf32>) {
//   "xla_lhlo.fusion"() ( {
//     %0 = alloc() : memref<2x2xf32>
//     "xla_lhlo.add"(%arg1, %arg2, %0) :
//         (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
//     "xla_lhlo.multiply"(%0, %arg0, %arg3) :
//         (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
//     dealloc %0 : memref<2x2xf32>
//     "xla_lhlo.terminator"() : () -> ()
//   }) : () -> ()
//   return
//  }
// }
//
// FuncOp signature conversion example:
//
// func @func_op(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
//   %0 = "xla_hlo.maximum"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) ->
//   tensor<4xf32> %1 = "xla_hlo.add"(%arg0, %0)  : (tensor<4xf32>,
//   tensor<4xf32>) -> tensor<4xf32> return %1 : tensor<4xf32>
// }
//
// Transformed function with an extra argument for the result. The types have
// been converted from tensor to memref.
//
// func @func_op(%arg0: memref<4xf32>,
//               %arg1: memref<4xf32>,
//               %arg2: memref<4xf32>) {
//   %0 = alloc() : memref<4xf32>
//   %1 = alloc() : memref<4xf32>
//   "xla_lhlo.maximum"(%arg0, %arg1, %0) :
//         (memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
//   "xla_lhlo.add"(%arg0, %0, %1) :
//         (memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
//   "xla_lhlo.copy"(%1, %arg2) : (memref<4xf32>, memref<4xf32>) -> ()
//   dealloc %0 : memref<4xf32>
//   dealloc %1 : memref<4xf32>
//   "xla_lhlo.terminator"() : () -> ()
// }

struct HloLegalizeToLhlo
    : public PassWrapper<HloLegalizeToLhlo, OperationPass<ModuleOp>> {
  void runOnOperation() override {
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

    auto module = getOperation();
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

  LogicalResult matchAndRewrite(
      FuncOp funcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const final {
    if (funcOp.getBody().getBlocks().size() > 1) {
      funcOp.emitOpError() << "tensor to buffer conversion expects a single "
                              "block in the region containing the operation";
      return failure();
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
    return success();
  }
};

/// Transforms ReturnOp to LhloTerminator. CopyOp is inserted to copy each
/// result to the corresponding buffer argument.
class StdToLhloReturnOpConverter : public OpConversionPattern<mlir::ReturnOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
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
        return failure();
      }
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(dealloc);
      rewriter.create<xla_lhlo::CopyOp>(loc, llvm::None, operand.value(),
                                        funcOp.getArgument(returnArgNumber));
    }
    rewriter.replaceOpWithNewOp<xla_lhlo::TerminatorOp>(returnOp);
    return success();
  }
};

void populateHLOToLHLOConversionPattern(MLIRContext* context,
                                        OwningRewritePatternList* patterns) {
  // clang-format off
  patterns->insert<
      HloToLhloDynamicBroadcastInDimOpConverter,
      HloToLhloFuncOpConverter,
      HloToLhloOpConverter<xla_hlo::AbsOp>,
      HloToLhloOpConverter<xla_hlo::AddOp>,
      HloToLhloOpConverter<xla_hlo::AndOp>,
      HloToLhloOpConverter<xla_hlo::BroadcastInDimOp>,
      HloToLhloOpConverter<xla_hlo::CeilOp>,
      HloToLhloOpConverter<xla_hlo::CompareOp>,
      HloToLhloOpConverter<xla_hlo::ComplexOp>,
      HloToLhloOpConverter<xla_hlo::ConstOp>,
      HloToLhloOpConverter<xla_hlo::ConvertOp>,
      HloToLhloOpConverter<xla_hlo::CopyOp>,
      HloToLhloOpConverter<xla_hlo::CosOp>,
      HloToLhloOpConverter<xla_hlo::DivOp>,
      HloToLhloOpConverter<xla_hlo::ExpOp>,
      HloToLhloOpConverter<xla_hlo::ImagOp>,
      HloToLhloOpConverter<xla_hlo::IotaOp>,
      HloToLhloOpConverter<xla_hlo::LogOp>,
      HloToLhloOpConverter<xla_hlo::MaxOp>,
      HloToLhloOpConverter<xla_hlo::MinOp>,
      HloToLhloOpConverter<xla_hlo::MulOp>,
      HloToLhloOpConverter<xla_hlo::NegOp>,
      HloToLhloOpConverter<xla_hlo::RealOp>,
      HloToLhloOpConverter<xla_hlo::RemOp>,
      HloToLhloOpConverter<xla_hlo::RsqrtOp>,
      HloToLhloOpConverter<xla_hlo::SelectOp>,
      HloToLhloOpConverter<xla_hlo::SignOp>,
      HloToLhloOpConverter<xla_hlo::SqrtOp>,
      HloToLhloOpConverter<xla_hlo::SubOp>,
      HloToLhloOpConverter<xla_hlo::TanhOp>,
      HloToLhloReduceOpConverter,
      HloToLhloTensorLoadOpConverter,
      HloToLhloTensorStoreOpConverter,
      StdToLhloReturnOpConverter
  >(context);
  // clang-format on
}

std::unique_ptr<OperationPass<ModuleOp>> createLegalizeToLhloPass() {
  return absl::make_unique<HloLegalizeToLhlo>();
}

static PassRegistration<HloLegalizeToLhlo> legalize_pass(
    "hlo-legalize-to-lhlo", "Legalize from HLO dialect to LHLO dialect");

}  // namespace xla_hlo
}  // namespace mlir
