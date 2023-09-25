/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/tfrt/transforms/mlrt/tpu_conversion_patterns.h"

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tfrt/ir/mlrt/mlrt_dialect.h"
#include "tensorflow/compiler/mlir/tfrt/ir/mlrt/tf_mlrt_ops.h"
#include "tensorflow/compiler/mlir/tfrt/ir/mlrt/tf_mlrt_tpu_ops.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/mlrt/execute_op_registry.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/utils.h"

namespace tensorflow {
namespace mlrt_compiler {
namespace {

class TPUCompileMlirAndExecuteOpPreParallelizationConversion
    : public mlir::OpConversionPattern<mlir::TF::TPUCompileMlirAndExecuteOp> {
 public:
  TPUCompileMlirAndExecuteOpPreParallelizationConversion(
      mlir::MLIRContext* context, bool use_tpu_host_allocator_for_inputs)
      : OpConversionPattern(context),
        use_tpu_host_allocator_for_inputs_(use_tpu_host_allocator_for_inputs) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::TF::TPUCompileMlirAndExecuteOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    llvm::SmallVector<int> constant_operand_indices;
    llvm::SmallVector<int> non_constant_operand_indices;

    for (int i = 0; i < adaptor.getArgs().size(); ++i) {
      auto operand = adaptor.getOperands()[i];
      auto original_operand = op.getOperand(i);
      if (IsResultVariable(original_operand, operand)) {
        // NOTE: It's important to populate constant_operand_indices in
        // ascending order.
        constant_operand_indices.push_back(i);
      } else {
        non_constant_operand_indices.push_back(i);
      }
    }

    llvm::SmallVector<mlir::Value> operands = adaptor.getArgs();

    size_t tensor_operands_size = operands.size();
    operands.append(adaptor.getStaticShapes().begin(),
                    adaptor.getStaticShapes().end());

    auto producer_name = op->getAttrOfType<mlir::StringAttr>("producer_name");

    llvm::SmallVector<int32_t> operands_with_static_shapes;
    if (adaptor.getOperandsWithStaticShape().has_value()) {
      for (auto attr : adaptor.getOperandsWithStaticShapeAttr()
                           .getAsRange<mlir::IntegerAttr>()) {
        operands_with_static_shapes.push_back(
            static_cast<int32_t>(attr.getInt()));
      }
    }

    if (use_tpu_host_allocator_for_inputs_) {
      llvm::DenseMap<mlir::Operation*, mlir::Operation*> replaced_ops;

      for (int i : non_constant_operand_indices) {
        DCHECK_LT(i, op.getNumOperands());
        auto old_value = operands[i];
        mlir::Operation* def = old_value.getDefiningOp();

        if (def && llvm::isa<mlir::TF::TensorFlowDialect>(def->getDialect())) {
          auto*& op_with_device = replaced_ops[def];
          if (!op_with_device) {
            mlir::ConversionPatternRewriter::InsertionGuard guard(rewriter);
            rewriter.setInsertionPoint(def);

            op_with_device = rewriter.clone(*def);
            op_with_device->setAttr(kTfMlrtCustomDevice,
                                    rewriter.getStringAttr(kTpuHostDevice));
            rewriter.replaceOp(def, op_with_device->getResults());
          }
        }
      }
    }

    auto compile_and_execute_op =
        rewriter.create<tf_mlrt::TFTPUCompileAndExecuteOp>(
            op.getLoc(), op.getResultTypes(), operands,
            rewriter.getDenseI32ArrayAttr(constant_operand_indices),
            op.getMetadataAttr(), op.getMlirModuleAttr(),
            rewriter.getUI32IntegerAttr(tensor_operands_size),
            rewriter.getDenseI32ArrayAttr(operands_with_static_shapes),
            producer_name);

    rewriter.replaceOp(op, compile_and_execute_op->getResults());

    return mlir::success();
  }

 private:
  bool use_tpu_host_allocator_for_inputs_ = false;
};

class TPUCompileMlirAndExecuteOpConversion
    : public mlir::OpConversionPattern<tf_mlrt::TFTPUCompileAndExecuteOp> {
 public:
  TPUCompileMlirAndExecuteOpConversion(mlir::TypeConverter* type_converter,
                                       mlir::MLIRContext* context,
                                       ExecuteOpRegistry* execute_op_registry)
      : OpConversionPattern(*type_converter, context) {}

  mlir::LogicalResult matchAndRewrite(
      tf_mlrt::TFTPUCompileAndExecuteOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    llvm::SmallVector<mlir::Value> operands =
        adaptor.getOperandsAndStaticShapes();
    llvm::SmallVector<mlir::Type> result_types;
    result_types.push_back(rewriter.getType<tf_mlrt::TFTensorType>());
    result_types.append(op.getResults().size(),
                        rewriter.getType<mlrt::compiler::FutureType>());

    auto compile_and_execute_op =
        rewriter.create<tf_mlrt_tpu::CompileAndExecuteOp>(
            op.getLoc(), result_types, operands, op.getConstantOperandIndices(),
            op.getMetadataAttr(), op.getMlirModuleAttr(), op.getNumOperands(),
            op.getOperandsWithStaticShape(), op.getProducerName());

    rewriter.replaceOp(op, compile_and_execute_op->getResults());

    return mlir::success();
  }
};

}  // namespace

void PopulateTpuPreParallelizationConversionPatterns(
    mlir::ConversionTarget& target, mlir::RewritePatternSet& patterns,
    const TfrtPipelineOptions& options) {
  target.addIllegalOp<mlir::TF::TPUCompileMlirAndExecuteOp>();
  patterns.add<TPUCompileMlirAndExecuteOpPreParallelizationConversion>(
      patterns.getContext(), options.use_tpu_host_allocator_for_inputs);
}

void PopulateTpuConversionPatterns(mlir::ConversionTarget& target,
                                   mlir::RewritePatternSet& patterns,
                                   mlir::TypeConverter& type_converter,
                                   ExecuteOpRegistry& execute_op_registry,
                                   const TfrtPipelineOptions& options) {
  target.addIllegalOp<tf_mlrt::TFTPUCompileAndExecuteOp>();
  target.addLegalDialect<tf_mlrt_tpu::TensorflowMlrtTpuDialect>();

  patterns.add<TPUCompileMlirAndExecuteOpConversion>(
      &type_converter, patterns.getContext(), &execute_op_registry);
}

void RegisterTpuDialect(mlir::DialectRegistry& registry) {
  registry.insert<tf_mlrt_tpu::TensorflowMlrtTpuDialect>();
}

}  // namespace mlrt_compiler
}  // namespace tensorflow
