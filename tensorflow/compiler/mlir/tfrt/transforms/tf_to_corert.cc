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

// This file implements lowering of TF dialect to TFRT CoreRuntime ExecuteOp.
// This lowering pass is heavily experimental and incomplete. External code
// should not depend on the code here. And please do not take example on it as
// "the path forward" for this.

#include <vector>

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Pass/PassOptions.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/passes.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/tstring.h"
#include "tfrt/basic_kernels/opdefs/basic_kernels.h"
#include "tfrt/core_runtime/opdefs/attributes.h"
#include "tfrt/core_runtime/opdefs/core_runtime.h"

namespace tensorflow {
namespace {

// TODO(chky): define these dialect types instead of using opaque types.
mlir::Type CreateDeviceType(mlir::Builder *builder) {
  return mlir::OpaqueType::get(builder->getIdentifier("corert"), "device",
                               builder->getContext());
}

mlir::Type CreateTensorHandleType(mlir::Builder *builder) {
  return mlir::OpaqueType::get(builder->getIdentifier("corert"), "tensorhandle",
                               builder->getContext());
}

mlir::Type CreateStringType(mlir::Builder *builder) {
  return mlir::OpaqueType::get(builder->getIdentifier("hex"), "string",
                               builder->getContext());
}

// A helper class for converting CoreRT types and attributes.
class CoreRTConverter : public mlir::TypeConverter {
 public:
  explicit CoreRTConverter(mlir::MLIRContext *context)
      : builder_(context),
        device_type_(CreateDeviceType(&builder_)),
        tensor_handle_type_(CreateTensorHandleType(&builder_)) {
    addConversion([](Type type) { return type; });
    addConversion([=](TensorType type) { return tensor_handle_type_; });
  }

  // Create a single attribute that contains the named attribute lists. It is an
  // array of pairs. The key must be a string attribute, and the value can be
  // any attribute that is supported by CoreRuntime.
  mlir::ArrayAttr CreateOpAttrs(ArrayRef<NamedAttribute> attrs) {
    llvm::SmallVector<mlir::Attribute, 4> attr_array;
    for (auto key_and_value : attrs) {
      if (!IsUnusedAttribute(key_and_value.first)) {
        auto converted = ConvertAttribute(key_and_value.second);
        if (!converted) return {};

        mlir::StringAttr key = builder_.getStringAttr(key_and_value.first);
        attr_array.push_back(builder_.getArrayAttr({key, converted}));
      }
    }
    return builder_.getArrayAttr(attr_array);
  }

  // Convert the device attribute in `op` to a device value produced by the
  // corresponding GetDeviceOp in the current block. If there does not exist
  // one, insert a GetDeviceOp to the beginning of the block and return the
  // device value.
  Value ConvertDevice(mlir::Operation *op,
                      ConversionPatternRewriter *rewriter) const {
    auto device_attr = op->getAttr("device");
    if (!device_attr) {
      op->emitOpError("device attribute not found.");
      return {};
    }

    auto device_name = device_attr.cast<mlir::StringAttr>().getValue();
    if (device_name.empty()) {
      op->emitOpError("device has not been assigned.");
      return {};
    }

    op->removeAttr(rewriter->getIdentifier("device"));

    auto *block = op->getBlock();

    if (auto get_device_op = GetDeviceOrNull(device_name, block))
      return get_device_op.device();

    ConversionPatternRewriter::InsertionGuard insertion_guard(*rewriter);
    rewriter->setInsertionPointToStart(block);
    return rewriter
        ->create<tfrt::corert::GetDeviceOp>(block->getParent()->getLoc(),
                                            device_type(), device_name)
        .device();
  }

  mlir::Type device_type() const { return device_type_; }
  mlir::Type tensor_handle_type() const { return tensor_handle_type_; }

 private:
  // TODO(chky): attributes "_output_shapes" should be removed by any tool that
  // generates TF MLIR dialect, as they are not used by CoreRuntime. Remove this
  // filtering logic once unused attributes are cleaned up in the upper layer.
  bool IsUnusedAttribute(llvm::StringRef name) const {
    return name == "_output_shapes";
  }

  // Returns the converted attribute in TFRT dialect. If the conversion fails,
  // returns a null attribute instead.
  mlir::Attribute ConvertAttribute(mlir::Attribute attr) {
    // The supported attributes here should be kept consistent with
    // //third_party/tf_runtime/include/tfrt/core_runtime/op_attr_type.h
    //
    // Currently, not all tensorflow data types are supported. Unranked shape
    // attributes are not supported yet.

    // Return directly if the attribute is already supported.
    if (attr.isa<mlir::IntegerAttr>() || attr.isa<mlir::FloatAttr>() ||
        attr.isa<mlir::BoolAttr>() || attr.isa<mlir::TypeAttr>() ||
        attr.isa<mlir::StringAttr>() ||
        attr.isa<mlir::DenseIntOrFPElementsAttr>())
      return attr;

    // Convert the attribute to the corresponding format in TFRT dialect if
    // needed.
    if (auto shape_attr = attr.dyn_cast<mlir::TF::ShapeAttr>()) {
      if (!shape_attr.hasRank()) return {};
      return tfrt::corert::ShapeAttr::get(builder_.getContext(),
                                          shape_attr.getShape());
    }

    // For arrays, we recursively convert the elements.
    if (auto array_attr = attr.dyn_cast<mlir::ArrayAttr>()) {
      llvm::SmallVector<mlir::Attribute, 8> attrs;
      attrs.reserve(array_attr.size());
      for (auto attr : array_attr) {
        auto converted = ConvertAttribute(attr);
        if (!converted) return {};
        attrs.push_back(converted);
      }
      return builder_.getArrayAttr(attrs);
    }

    return {};
  }

  // Find a GetDeviceOp that matches the device_name at the beginning of the
  // block. Return nullptr if it does not find one.
  tfrt::corert::GetDeviceOp GetDeviceOrNull(StringRef device_name,
                                            Block *block) const {
    for (auto &op : *block) {
      auto get_device_op = llvm::dyn_cast<tfrt::corert::GetDeviceOp>(&op);
      if (!get_device_op) break;
      if (get_device_op.device_name() == device_name) return get_device_op;
    }
    return nullptr;
  }

  mlir::Builder builder_;
  mlir::Type device_type_;
  mlir::Type tensor_handle_type_;
};

// Lower a tf.Const op that creates a string tensor to a native
// corert.create_string_tensor op.
class CoreRTConstStringTensorOpConversion
    : public mlir::OpConversionPattern<mlir::TF::ConstOp> {
 public:
  CoreRTConstStringTensorOpConversion(mlir::MLIRContext *context,
                                      CoreRTConverter *corert_converter)
      : mlir::OpConversionPattern<mlir::TF::ConstOp>(context),
        corert_converter_(*corert_converter) {}

  LogicalResult matchAndRewrite(
      mlir::TF::ConstOp op, ArrayRef<mlir::Value> operands,
      ConversionPatternRewriter &rewriter) const override {  // NOLINT
    if (!op.dtype().isa<mlir::TF::StringType>()) return failure();

    DenseStringElementsAttr attr = op.value().cast<DenseStringElementsAttr>();

    llvm::SmallVector<Attribute, 4> values;
    values.reserve(attr.getNumElements());
    for (const auto &element : attr.getRawStringData())
      values.push_back(rewriter.getStringAttr(
          llvm::StringRef(element.data(), element.size())));

    // Create the shape attribute from the tensor shape.
    ArrayRef<int64_t> shape = op.value().getType().getShape();
    llvm::SmallVector<mlir::Attribute, 4> dims;
    dims.reserve(shape.size());
    auto i64_type = rewriter.getIntegerType(64);
    for (auto dim : shape)
      dims.push_back(rewriter.getIntegerAttr(i64_type, dim));

    auto new_op = rewriter.create<tfrt::corert::ConstStringTensorOp>(
        op.getLoc(), corert_converter_.tensor_handle_type(),
        rewriter.getArrayAttr(dims), rewriter.getArrayAttr(values));

    rewriter.replaceOp(op, new_op.result());

    return success();
  }

 private:
  CoreRTConverter &corert_converter_;
};

// Convert TF dialect operations with no side effects to CoreRT ExecuteOp. For
// example,
//
// %0 = "tf.MatMul"(%arg0, %arg1) {transpose_a = false, transpose_b = false} :
//    (tensor<3x1xf32>, tensor<1x3xf32>) -> tensor<3x3xf32>
//
// is converted to
//
// %result = corert.executeop(%device)
//    "tf.MatMul"(%arg0, %arg1) {transpose_a = false, transpose_b = false} :
//    (!corert.tensorhandle, !corert.tensorhandle) -> !corert.tensorhandle
//
// Note that it will fail to match if some attributes are not supported.
template <typename TF_Op>
class CoreRTExecuteOpConversion : public mlir::OpConversionPattern<TF_Op> {
 public:
  CoreRTExecuteOpConversion(mlir::MLIRContext *context,
                            CoreRTConverter *corert_converter)
      : mlir::OpConversionPattern<TF_Op>(context),
        corert_converter_(*corert_converter) {}

  LogicalResult matchAndRewrite(
      TF_Op op, ArrayRef<mlir::Value> operands,
      ConversionPatternRewriter &rewriter) const override {  // NOLINT
    mlir::StringAttr op_name = rewriter.getStringAttr(op.getOperationName());

    llvm::SmallVector<Type, 4> result_types;
    for (auto type : op.getOperation()->getResultTypes())
      result_types.push_back(corert_converter_.convertType(type));

    // Get the device, or create one if there does not exist one.
    auto device = corert_converter_.ConvertDevice(op, &rewriter);
    if (!device) return failure();

    auto derived_attrs = op.materializeDerivedAttributes();
    for (auto named_attr : derived_attrs) {
      op.setAttr(named_attr.first, named_attr.second);
    }

    ArrayAttr op_attrs = corert_converter_.CreateOpAttrs(op.getAttrs());
    if (!op_attrs) return failure();

    auto new_op = rewriter.create<tfrt::corert::ExecuteOp>(
        op.getLoc(), result_types, device, operands, op_attrs, op_name);

    rewriter.replaceOp(op, new_op.results());
    return success();
  }

 private:
  CoreRTConverter &corert_converter_;
};

// Deletes the op and forwards the arguments.
template <typename TF_Op>
class PassThroughConversion : public mlir::OpConversionPattern<TF_Op> {
 public:
  explicit PassThroughConversion(MLIRContext *context)
      : mlir::OpConversionPattern<TF_Op>(context) {}

  LogicalResult matchAndRewrite(
      TF_Op op, ArrayRef<mlir::Value> operands,
      ConversionPatternRewriter &rewriter) const override {  // NOLINT
    // Just forward the arguments to results.
    rewriter.replaceOp(op, operands);
    return success();
  }
};

// Convert standard ReturnOp to hex.return.
//
// TODO(chky): conversion to hex kernels should come from a common tf_to_hex
// library.
class ReturnOpConversion : public mlir::OpConversionPattern<mlir::ReturnOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::ReturnOp op, ArrayRef<mlir::Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<tfrt::hex::ReturnOp>(op, operands);
    return success();
  }
};

// Convert TF dialect to CoreRT dialect.
class TFToCoreRTConversionPass
    : public mlir::PassWrapper<TFToCoreRTConversionPass,
                               OperationPass<ModuleOp>> {
  void runOnOperation() override {
    auto module = getOperation();
    mlir::ConversionTarget target(getContext());
    mlir::OwningRewritePatternList patterns;
    if (failed(TFToCoreRTConversionPassRun(&getContext(), &module, &target,
                                           &patterns)))
      signalPassFailure();
  }
};

}  // namespace

LogicalResult TFToCoreRTConversionPassRun(
    mlir::MLIRContext *context, mlir::ModuleOp *module,
    mlir::ConversionTarget *target, mlir::OwningRewritePatternList *patterns) {
  module->removeAttr("tf_saved_model.semantics");

  mlir::Builder builder(context);
  auto bound_id = builder.getIdentifier("tf_saved_model.bound_input");
  auto path_id = builder.getIdentifier("tf_saved_model.index_path");

  module->walk([bound_id, path_id, module](mlir::Operation *op) mutable {
    if (auto func_op = dyn_cast<mlir::FuncOp>(op)) {
      // Remove tf_saved_model specific function arg attributes.
      for (unsigned i = 0, e = func_op.getNumArguments(); i != e; ++i) {
        func_op.removeArgAttr(i, bound_id);
        func_op.removeArgAttr(i, path_id);
      }
      for (unsigned i = 0, e = func_op.getNumResults(); i != e; ++i) {
        func_op.removeResultAttr(i, bound_id);
        func_op.removeResultAttr(i, path_id);
      }
      if (auto exported_names = func_op.getAttrOfType<mlir::ArrayAttr>(
              "tf_saved_model.exported_names")) {
        // Create a function for each exported name.
        //
        // TODO(b/148477882): TFRT dialect should have similar concepts of
        // exported names so that a function can be referenced by multiple
        // exported names.
        func_op.removeAttr("tf_saved_model.exported_names");
        for (auto exported_name : exported_names) {
          auto exported_func_op = func_op.clone();
          exported_func_op.setName(
              exported_name.cast<mlir::StringAttr>().getValue());
          module->insert(module->begin(), exported_func_op);
        }
        func_op.erase();
      }
    } else if (isa<mlir::tf_saved_model::GlobalTensorOp>(op)) {
      // Remove all global_tensor_ops.
      op->erase();
    }
  });

  CoreRTConverter corert_converter(context);

  target->addLegalDialect<tfrt::corert::CoreRTDialect>();
  target->addLegalDialect<tfrt::hex::HexDialect>();
  target->addIllegalDialect<TF::TensorFlowDialect>();
  target->addDynamicallyLegalOp<mlir::FuncOp>([&corert_converter](FuncOp op) {
    return corert_converter.isSignatureLegal(op.getType());
  });

  patterns->insert<PassThroughConversion<TF::ReadVariableOp>,
                   PassThroughConversion<TF::IdentityOp>, ReturnOpConversion>(
      context);

  // Here we use one specialized pattern for tf.Const with string tensors as
  // it will incorrect to use ExecuteOp pattern to convert string tensor
  // attribute.
  patterns->insert<CoreRTConstStringTensorOpConversion>(context,
                                                        &corert_converter);

  // TODO(b/148823030): Pattern registration for TF operations is not
  // sustainable currently. We need to figure out a plan
  patterns->insert<CoreRTExecuteOpConversion<TF::AddV2Op>,
                   // TODO(chky): Move the ReadVariableOp + Identity pattern
                   // to optimizer.
                   // CoreRTExecuteOpConversion<TF::IdentityOp>,
                   CoreRTExecuteOpConversion<TF::MulOp>,
                   CoreRTExecuteOpConversion<TF::BiasAddOp>,
                   CoreRTExecuteOpConversion<TF::Conv2DOp>,
                   CoreRTExecuteOpConversion<TF::ConcatV2Op>,
                   CoreRTExecuteOpConversion<TF::ConstOp>,
                   CoreRTExecuteOpConversion<TF::CastOp>,
                   CoreRTExecuteOpConversion<TF::ExpandDimsOp>,
                   CoreRTExecuteOpConversion<TF::TransposeOp>,
                   CoreRTExecuteOpConversion<TF::FusedBatchNormV3Op>,
                   CoreRTExecuteOpConversion<TF::FusedBatchNormExOp>,
                   CoreRTExecuteOpConversion<TF::MatMulOp>,
                   CoreRTExecuteOpConversion<TF::MaxPoolOp>,
                   CoreRTExecuteOpConversion<TF::MeanOp>,
                   CoreRTExecuteOpConversion<TF::PadOp>,
                   CoreRTExecuteOpConversion<TF::ParseExampleV2Op>,
                   CoreRTExecuteOpConversion<TF::ReluOp>,
                   CoreRTExecuteOpConversion<TF::SoftmaxOp>,
                   CoreRTExecuteOpConversion<TF::ShapeOp>,
                   CoreRTExecuteOpConversion<TF::TanhOp>>(context,
                                                          &corert_converter);

  mlir::populateFuncOpTypeConversionPattern(*patterns, context,
                                            corert_converter);
  return mlir::applyPartialConversion(*module, *target, *patterns);
}

std::unique_ptr<mlir::Pass> CreateTFToCoreRTConversionPass() {
  return std::make_unique<TFToCoreRTConversionPass>();
}

void CreateTFExecutorToTFPipeline(mlir::OpPassManager &pm,
                                  const CoreRTPipelineOptions &options) {
  // First, we prune unused operations in MLIR in TF Executor dialect.
  pm.addPass(mlir::tf_executor::CreateTFExecutorGraphPruningPass());

  // Then we pass the MLIR module through the TF standard pipeline, which for
  // instances does shape inference, canonicalization, inlining, etc.
  mlir::TF::StandardPipelineOptions tf_options;
  tf_options.enable_inliner = true;
  mlir::TF::CreateTFStandardPipeline(pm, tf_options);

  // After all standard passes run layout optimization to assign optimal data
  // format for all layout sensitive operations.
  mlir::TF::LayoutOptimizationPipelineOptions layout_optimization_options;
  layout_optimization_options.force_data_format =
      options.force_data_format.getValue();
  mlir::TF::CreateLayoutOptimizationPipeline(pm, layout_optimization_options);

  // Run canonicalization pipeline to remove unused constants and bypassed
  // transpose operations left in the IR after layout optimization.
  pm.addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());

  if (options.default_device == "gpu")
    pm.addNestedPass<mlir::FuncOp>(mlir::TF::CreateGpuOpFusionPass());

  // Then we assign default devices.
  pm.addNestedPass<mlir::FuncOp>(
      mlir::TF::CreateSimpleTFDeviceAssignmentPass(options.default_device));
}

void CreateTFExecutorToCoreRTPipeline(mlir::OpPassManager &pm,
                                      const CoreRTPipelineOptions &options) {
  CreateTFExecutorToTFPipeline(pm, options);

  // Convert it to MLIR in CoreRT dialect.
  pm.addPass(CreateTFToCoreRTConversionPass());

  // Run optimizer on the MLIR module in CoreRT dialect.
  if (options.enable_optimizer)
    pm.addNestedPass<mlir::FuncOp>(CreateCoreRTOptimizePass());
}

static mlir::PassRegistration<TFToCoreRTConversionPass> pass(
    "tf-to-corert",
    "Convert Tensorflow dialect to TFRT's CoreRuntime dialect.");

static mlir::PassPipelineRegistration<CoreRTPipelineOptions> pipeline(
    "tf-executor-to-corert-pipeline",
    "Convert Tensorflow Executor dialect to TFRT's CoreRuntime dialect, and "
    "also apply necessary optimization passes.",
    CreateTFExecutorToCoreRTPipeline);

}  // namespace tensorflow
