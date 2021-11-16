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

// This file implements lowering of TF dialect to TFRT data kernels.
#include "tensorflow/compiler/mlir/tfrt/transforms/tf_to_tfrt_data.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/protobuf/graph_debug_info.pb.h"
#include "tfrt/basic_kernels/opdefs/basic_kernels.h"  // from @tf_runtime
#include "tfrt/bef_converter/mlir_to_bef.h"  // from @tf_runtime
#include "tfrt/data/opdefs/data_ops.h"  // from @tf_runtime
#include "tfrt/data/opdefs/types.h"  // from @tf_runtime

#define DEBUG_TYPE "tf-to-tfrt-data"

namespace tensorflow {
namespace {

bool isIntScalar(Type t, size_t width) {
  if (auto ttype = t.dyn_cast<RankedTensorType>()) {
    if (ttype.hasStaticShape() && ttype.getNumElements() == 1 &&
        ttype.getRank() == 0 && ttype.getElementType().isSignlessInteger(width))
      return true;
  }
  return false;
}

// Converts `value_attr` from a TF Const node to the required type attr type `U`
template <typename T>
T ConstAttrToTypeAttr(ElementsAttr value_attr) {
  if (T type_attr = value_attr.dyn_cast<T>()) {
    return type_attr;
  } else if (auto v = value_attr.dyn_cast<SplatElementsAttr>()) {
    return v.getSplatValue<Attribute>().dyn_cast<T>();
  }
  return T(nullptr);
}

template <typename T>
LogicalResult ReplaceConst(TF::ConstOp &op, ConversionPatternRewriter &rewriter,
                           Type type) {
  IntegerAttr newAttr = ConstAttrToTypeAttr<IntegerAttr>(op.value());

  if (!newAttr) {
    return failure();
  }

  auto tfrtConst = rewriter.create<T>(op.getLoc(), type, newAttr);
  rewriter.replaceOp(op.getOperation(), tfrtConst.getResult());
  return success();
}

mlir::Type CreateDatasetType(mlir::Builder *builder) {
  return builder->getType<tfrt::data::DatasetType>();
}

// A helper class for converting data-specific types and attributes
class DataConverter : public mlir::TypeConverter {
 public:
  explicit DataConverter(mlir::MLIRContext *context) {
    addConversion([](Type type) { return type; });
    addConversion([context](TensorType type) {
      mlir::Builder builder(context);
      // tf.data datasets are represented by DT_VARIANT tensors in TF.
      // TODO(rachelim): Identify datasets more accurately.
      if (type.getElementType().dyn_cast<TF::VariantType>()) {
        return CreateDatasetType(&builder);
      }
      return type.dyn_cast<Type>();
    });
  }
};  // namespace

struct ConstOpConversion : public mlir::OpConversionPattern<TF::ConstOp> {
  explicit ConstOpConversion(MLIRContext *context)
      : OpConversionPattern<TF::ConstOp>(context) {}

  LogicalResult matchAndRewrite(
      TF::ConstOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (isIntScalar(op.getType(), 64)) {
      return ReplaceConst<tfrt::compiler::ConstantI64Op>(op, rewriter,
                                                         rewriter.getI64Type());
    }
    if (isIntScalar(op.getType(), 1)) {
      return ReplaceConst<tfrt::compiler::ConstantI1Op>(op, rewriter,
                                                        rewriter.getI1Type());
    }
    // TODO(rachelim): Support converting other const types.
    return failure();
  }
};

struct ReturnOpConversion : public mlir::OpConversionPattern<mlir::ReturnOp> {
  explicit ReturnOpConversion(MLIRContext *context)
      : OpConversionPattern<mlir::ReturnOp>(context) {}

  LogicalResult matchAndRewrite(
      mlir::ReturnOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<tfrt::compiler::ReturnOp>(
        op, adaptor.getOperands());
    return success();
  }
};

class RangeDatasetOpConversion
    : public OpConversionPattern<TF::RangeDatasetOp> {
 public:
  explicit RangeDatasetOpConversion(MLIRContext *context)
      : OpConversionPattern<TF::RangeDatasetOp>(context),
        builder_(context),
        dataset_type_(CreateDatasetType(&builder_)) {}

  LogicalResult matchAndRewrite(
      TF::RangeDatasetOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (op.output_types().size() != 1) {
      // Range dataset should only have one output type.
      return failure();
    }
    if (auto output_type = op.output_types().begin()->cast<TypeAttr>()) {
      rewriter.replaceOpWithNewOp<tfrt::data::RangeDatasetOp>(
          op, dataset_type_, adaptor.start(), adaptor.stop(), adaptor.step(),
          output_type);
      return success();
    }
    return failure();
  }

 private:
  mlir::Builder builder_;
  mlir::Type dataset_type_;
};

class BatchDatasetV2OpConversion
    : public OpConversionPattern<TF::BatchDatasetV2Op> {
 public:
  explicit BatchDatasetV2OpConversion(MLIRContext *context)
      : OpConversionPattern<TF::BatchDatasetV2Op>(context),
        builder_(context),
        dataset_type_(CreateDatasetType(&builder_)) {}

  LogicalResult matchAndRewrite(
      TF::BatchDatasetV2Op op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Since TFRT's BatchDataset doesn't have a drop_remainder=True option,
    // we only convert this op if its drop_remainder input is statically known
    // to be false.
    auto drop_remainder_op = op.drop_remainder().getDefiningOp<TF::ConstOp>();
    if (!drop_remainder_op) return failure();
    BoolAttr drop_remainder_val =
        ConstAttrToTypeAttr<BoolAttr>(drop_remainder_op.value());
    if (!drop_remainder_val || drop_remainder_val.getValue()) {
      return failure();
    }

    // TODO(b/155892156): Support converting non-unary BatchDataset
    if (op.output_types().size() != 1) return failure();

    // TODO(b/155892156): Support converting BatchDataset with unknown rank
    auto output_shape = op.output_shapes()[0].cast<TF::ShapeAttr>();
    if (!output_shape.hasRank()) {
      return failure();
    }

    if (output_shape.getRank() >= 2) {  // Input is a tensor
      rewriter.replaceOpWithNewOp<tfrt::data::BatchDatasetTensorOp>(
          op, dataset_type_, adaptor.input_dataset(), adaptor.batch_size(),
          /*same_input_metadata=*/rewriter.getBoolAttr(false));
      return success();
    }

    auto output_type = op.output_types()[0].cast<TypeAttr>().getValue();

    if (output_type.isInteger(32)) {
      rewriter.replaceOpWithNewOp<tfrt::data::BatchDatasetI32Op>(
          op, dataset_type_, adaptor.input_dataset(), adaptor.batch_size(),
          /*same_input_metadata=*/rewriter.getBoolAttr(false));
      return success();
    }
    if (output_type.isInteger(64)) {
      rewriter.replaceOpWithNewOp<tfrt::data::BatchDatasetI64Op>(
          op, dataset_type_, adaptor.input_dataset(), adaptor.batch_size(),
          /*same_input_metadata=*/rewriter.getBoolAttr(false));
      return success();
    }
    return failure();
  }

 private:
  mlir::Builder builder_;
  mlir::Type dataset_type_;
};

// This rewrite converts a tf.data function that returns a tf.data dataset (in
// the TF dialect) to the equivalent function in the TFRT and Data dialects that
// returns a `!tfrt.dataset`.
//
// For now, this can only lower a RangeDataset op and its inputs. As we add more
// native TFRT datasets, we add the corresponding lowering pattern here.
class TFToTFRTDataRewritePass
    : public mlir::PassWrapper<TFToTFRTDataRewritePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  llvm::StringRef getArgument() const final { return "tf-to-tfrt-data"; }
  llvm::StringRef getDescription() const final {
    return "Convert Tensorflow dialect to TFRT's data dialect.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tfrt::compiler::TFRTDialect, tfrt::data::DataDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto *context = &getContext();
    mlir::ConversionTarget target(*context);
    DataConverter data_converter(context);
    target.addIllegalDialect<TF::TensorFlowDialect>();
    target.addLegalDialect<tfrt::data::DataDialect>();
    target.addLegalDialect<tfrt::compiler::TFRTDialect>();
    target.addDynamicallyLegalOp<mlir::FuncOp>([&data_converter](FuncOp op) {
      return data_converter.isSignatureLegal(op.getType());
    });
    mlir::OwningRewritePatternList patterns(&getContext());
    patterns.insert<RangeDatasetOpConversion, BatchDatasetV2OpConversion,
                    ConstOpConversion, ReturnOpConversion>(context);
    mlir::populateFuncOpTypeConversionPattern(patterns, data_converter);

    auto result =
        mlir::applyPartialConversion(module, target, std::move(patterns));
    if (failed(result)) {
      signalPassFailure();
    }
  }
};

// Creates a pipeline of passes that converts MLIR TF Executor dialect to
// Hex and Data dialect.
void CreateTFExecutorToTFRTDataPipeline(mlir::OpPassManager &pm) {
  // Prune unused operations.
  pm.addPass(mlir::tf_executor::CreateTFExecutorGraphPruningPass());

  // Run the TF standard pipeline
  mlir::TF::StandardPipelineOptions tf_options;
  tf_options.enable_inliner = true;
  mlir::TF::CreateTFStandardPipeline(pm, tf_options);

  // After all the standard passes, lower to TFRT Data.
  pm.addPass(CreateTFToTFRTDataConversionPass());
}

Status TFDataGraphDefToTFDataMLIR(const GraphDef &graph_def,
                                  mlir::MLIRContext *mlir_ctx,
                                  mlir::OwningModuleRef *module_ref) {
  // Import to TF dialect
  string output_node;
  for (const auto &node : graph_def.node()) {
    if (node.op() == "_Retval") {
      output_node = node.input(0);
      VLOG(2) << "Output node: " << output_node;
      break;
    }
  }
  auto import_config = tensorflow::GraphImportConfig();
  import_config.outputs.push_back(std::move(output_node));
  import_config.prune_unused_nodes = true;
  TF_ASSIGN_OR_RETURN(*module_ref, ConvertGraphdefToMlir(
                                       graph_def, tensorflow::GraphDebugInfo(),
                                       std::move(import_config), mlir_ctx));

  return Status::OK();
}

Status CompileTFDataMLIRToBEF(mlir::ModuleOp module,
                              tfrt::BefBuffer *bef_buffer) {
  VLOG(1) << "TF Dialect: " << MlirModuleToString(module);

  mlir::PassManager pm(module.getContext());
  CreateTFExecutorToTFRTDataPipeline(pm);

  mlir::StatusScopedDiagnosticHandler diag_handler(module.getContext());
  if (mlir::failed(pm.run(module)))
    return diag_handler.Combine(
        errors::Internal("failed to lower TF Dialect to TFRT Data dialect."));

  VLOG(1) << "TFRT Dialect: " << MlirModuleToString(module);

  *bef_buffer =
      tfrt::ConvertMLIRToBEF(module, /*disable_optional_sections=*/false);
  if (bef_buffer->empty())
    return diag_handler.Combine(
        errors::Internal("failed to convert MLIR to BEF."));

  return Status::OK();
}

}  // namespace

std::unique_ptr<mlir::Pass> CreateTFToTFRTDataConversionPass() {
  return std::make_unique<TFToTFRTDataRewritePass>();
}

Status TFDataGraphDefToHostBEF(const GraphDef &graph_def,
                               tfrt::BefBuffer *bef) {
  mlir::MLIRContext mlir_ctx;
  mlir::OwningModuleRef module_ref;
  TF_RETURN_IF_ERROR(
      TFDataGraphDefToTFDataMLIR(graph_def, &mlir_ctx, &module_ref));
  TF_RETURN_IF_ERROR(CompileTFDataMLIRToBEF(module_ref.get(), bef));

  return Status::OK();
}

static mlir::PassRegistration<TFToTFRTDataRewritePass> pass;

}  // namespace tensorflow
