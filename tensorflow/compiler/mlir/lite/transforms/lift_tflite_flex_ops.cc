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

#include "tensorflow/compiler/mlir/lite/transforms/lift_tflite_flex_ops.h"

#include <string>
#include <utility>

#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_attributes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_attr.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/protobuf.h"

// The pass lifts TFLite Flex custom ops into TF dialect operations.
// Note: this pass is experimental, so not guaranteed to work with all Flex ops.

namespace mlir {
namespace TFL {
namespace {
#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/lite/transforms/passes.h.inc"

using ::tensorflow::StatusOr;

constexpr StringRef kFlexOpNamePrefix = "Flex";

// Pattern that converts TFL::CustomOp that encodes a Flex op into a TF dialect
// operation.
class LiftFlexCustomOp : public OpRewritePattern<TFL::CustomOp> {
 public:
  using OpRewritePattern<TFL::CustomOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::CustomOp op,
                                PatternRewriter& rewriter) const override {
    if (!op.custom_code().startswith(kFlexOpNamePrefix)) {
      return failure();
    }

    llvm::StringRef tf_op_name =
        op.custom_code().substr(kFlexOpNamePrefix.size());
    const std::string tf_op_full_name = llvm::Twine("tf.", tf_op_name).str();

    // Create the TF op
    OperationState op_state(op.getLoc(), tf_op_full_name);
    op_state.addOperands(op.getOperands());
    op_state.addTypes(op.getResultTypes());

    SmallVector<NamedAttribute, 2> attrs;
    std::string parsed_op_name;
    tensorflow::NodeDef node_def;
    if (failed(ParseCustomOption(op.custom_option().getValue(), op.getLoc(),
                                 parsed_op_name, attrs, node_def))) {
      return failure();
    }
    if (parsed_op_name != tf_op_name) {
      return op.emitOpError(
          "TF op names in 'custom_code' and 'custom_option' don't match");
    }
    const tensorflow::OpDef* op_def;

    // This will fail only if the op is not a registered TensorFlow op.
    if (!tensorflow::OpRegistry::Global()
             ->LookUpOpDef(parsed_op_name, &op_def)
             .ok()) {
      op->emitError() << "can't find registered TF op for " << parsed_op_name
                      << ". Please make sure the op library for "
                      << parsed_op_name << " is linked properly";
      return failure();
    }
    op_state.addAttributes(attrs);

    Operation* tf_op = rewriter.create(op_state);
    rewriter.replaceOp(op, tf_op->getResults());

    // Special type fixes for TF Resource Tensors that are casted to
    // Int32 tensor during MLIR->TFLite flatbuffer conversion.
    // TODO(b/146131919): correct handling of resource type
    if (auto tensor_array_v3_op = dyn_cast<TF::TensorArrayV3Op>(tf_op)) {
      Value handle = tensor_array_v3_op.handle();
      auto handle_type = handle.getType().cast<TensorType>();
      if (handle_type.getElementType().isInteger(/*width=*/32)) {
        Type resource_tensor_type =
            handle_type.clone(TF::ResourceType::get(rewriter.getContext()));
        handle.setType(resource_tensor_type);
      }
    }

    // Special type fixes for scalar tensor types.
    // TFLite flatbuffer schema doesn't distinguish scalar tensor shapes
    // and unranked tensor shapes (i.e. they are both represented as an empty
    // INT32 list), see b/138865275. MLIR importer conservatively treats them as
    // unranked tensor types. Here we set them to scalar tensor types when it is
    // safe.
    if (auto tensor_array_v3_op = dyn_cast<TF::TensorArrayV3Op>(tf_op)) {
      // The "flow" in TensorArrayV3 is always a scalar float tensor.
      // https://www.tensorflow.org/api_docs/python/tf/raw_ops/TensorArrayWriteV3
      Value flow = tensor_array_v3_op.flow();
      Type scalar_f32_tensor_type =
          RankedTensorType::get(/*shape=*/{}, rewriter.getF32Type());
      flow.setType(scalar_f32_tensor_type);
    }

    // Sets operand_segment_sizes or result_segment_sizes attribute to the op.
    // Referenced from tensorflow::ImporterBase::CreateOperation

    const auto set_segment_sizes_attr =
        [&](const tensorflow::NameRangeMap& arg_ranges,
            const tensorflow::protobuf::RepeatedPtrField<
                tensorflow::OpDef::ArgDef>& args,
            llvm::StringRef attr_name) {
          std::vector<mlir::Attribute> values;
          values.reserve(args.size());
          for (const auto& arg : args) {
            auto range = arg_ranges.at(arg.name());
            values.push_back(
                rewriter.getI32IntegerAttr(range.second - range.first));
          }
          auto attr_type =
              mlir::VectorType::get(args.size(), rewriter.getIntegerType(32));
          auto attr_value = mlir::DenseElementsAttr::get(attr_type, values);
          tf_op->setAttr(attr_name, attr_value);
        };
    if (tf_op->hasTrait<mlir::OpTrait::AttrSizedOperandSegments>() ||
        tf_op->hasTrait<mlir::OpTrait::AttrSizedResultSegments>()) {
      // The op has multiple variadic operands or results.
      // Calculate operand and result segment sizes using the OpDef.
      tensorflow::NameRangeMap input_ranges, output_ranges;
      // This will fail only if the OpDef is syntactically invalid.
      if (!NameRangesForNode(node_def, *op_def, &input_ranges, &output_ranges)
               .ok()) {
        tf_op->emitError("malformed opdef");
        return failure();
      }
      if (tf_op->hasTrait<mlir::OpTrait::AttrSizedOperandSegments>()) {
        // Add derived "operand_segment_sizes" attr to the created operation.
        // TODO(b/146937733): Don't use <void> here.
        set_segment_sizes_attr(input_ranges, op_def->input_arg(),
                               mlir::OpTrait::AttrSizedOperandSegments<
                                   void>::getOperandSegmentSizeAttr());
      }

      if (tf_op->hasTrait<mlir::OpTrait::AttrSizedResultSegments>()) {
        // Add derived "result_segment_sizes" attr to the created operation.
        // TODO(b/146937733): Don't use <void> here.
        set_segment_sizes_attr(output_ranges, op_def->output_arg(),
                               mlir::OpTrait::AttrSizedResultSegments<
                                   void>::getResultSegmentSizeAttr());
      }
    }

    return success();
  }

 private:
  // Parses TFLite Flex op's `custom_options` and returns the TF
  // `op_name` and TF dialect MLIR op attributes.
  static LogicalResult ParseCustomOption(
      StringRef custom_options, Location loc, std::string& op_name,
      SmallVectorImpl<NamedAttribute>& attributes,
      tensorflow::NodeDef& node_def) {
    // The flexbuffer contains a vector where the first elements is the
    // op name and the second is a serialized NodeDef.
    const flexbuffers::Vector& v =
        flexbuffers::GetRoot(
            reinterpret_cast<const uint8_t*>(custom_options.data()),
            custom_options.size())
            .AsVector();

    op_name = v[0].AsString().str();

    if (!node_def.ParseFromString(v[1].AsString().str())) {
      return emitError(
          loc, "failed to parse 'custom_options' data into a valid NodeDef");
    }

    OpBuilder builder(loc.getContext());
    for (const auto& name_and_value : node_def.attr()) {
      const std::string& attr_name = name_and_value.first;
      const tensorflow::AttrValue& attr_value = name_and_value.second;
      StatusOr<Attribute> mlir_attr =
          tensorflow::ConvertAttributeValue(attr_value, &builder);
      if (!mlir_attr.ok()) {
        return emitError(loc, mlir_attr.status().error_message());
      }
      attributes.push_back(builder.getNamedAttr(attr_name, *mlir_attr));
    }
    return success();
  }
};

class LiftTfliteFlexOpsPass
    : public LiftTfliteFlexOpsPassBase<LiftTfliteFlexOpsPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<TF::TensorFlowDialect>();
  }

 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LiftTfliteFlexOpsPass)

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    func::FuncOp func = getOperation();

    mlir::RewritePatternSet patterns(context);
    patterns.add<LiftFlexCustomOp>(context);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreateLiftTfliteFlexOpsPass() {
  return std::make_unique<LiftTfliteFlexOpsPass>();
}
}  // namespace TFL
}  // namespace mlir
