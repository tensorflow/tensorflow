/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Rewrite/FrozenRewritePatternSet.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "re2/re2.h"
#include "tensorflow/compiler/mlir/quantization/common/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/quantization/common/lift_as_function_call.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/quantization/common/tf_attrs_and_constraints.h"
#include "tensorflow/compiler/mlir/quantization/common/tf_quantization_lib/tf_quantization_utils.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/cc/quantization_unit_loc.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/ops/temp_tf_op_quant_spec.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace tf_quant {
namespace {

using QuantizationUnit =
    ::tensorflow::quantization::UnitWiseQuantizationSpec::QuantizationUnit;
using quant::AppendToVector;
using quant::FunctionCallOpType;
using quant::IsEinsumSupportedByXlaDotV2;
using quant::IsInLiftedFunc;
using ::tensorflow::quantization::OpSet;
using ::tensorflow::quantization::QuantizationComponentSpec;
using ::tensorflow::quantization::QuantizationMethod;
using ::tensorflow::quantization::QuantizationOptions;
using ::tensorflow::quantization::UnitWiseQuantizationSpec;
using tf_quant::HasStaticShapeAtDims;
using tf_quant::kAttrMapAttribute;
using tf_quant::kQuantTraitAttrName;

class TFLiftQuantizableSpotsAsFunctionsPass
    : public PassWrapper<TFLiftQuantizableSpotsAsFunctionsPass,
                         OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TFLiftQuantizableSpotsAsFunctionsPass)

  TFLiftQuantizableSpotsAsFunctionsPass() : test_mode_(true) {
    initializeForTest();
  }

  explicit TFLiftQuantizableSpotsAsFunctionsPass(
      const QuantizationOptions& quant_options)
      : quant_options_(quant_options), test_mode_(false) {}

  TFLiftQuantizableSpotsAsFunctionsPass(
      const TFLiftQuantizableSpotsAsFunctionsPass& other) {
    quant_options_ = other.quant_options_;
    test_mode_ = other.test_mode_;
    op_set_ = other.op_set_;
    initializeForTest();
  }

  StringRef getArgument() const final {
    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "tf-quant-lift-quantizable-spots-as-functions";
  }

  StringRef getDescription() const final {
    // This is a brief description of the pass.
    return "Replace quantization candidates with composite functions into the "
           "module";
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<TF::TensorFlowDialect>();
  }

  void runOnOperation() override;

 private:
  QuantizationOptions quant_options_;
  bool test_mode_;
  Option<OpSet> op_set_{
      *this, "target-opset", llvm::cl::init(OpSet::TF),
      llvm::cl::desc("Choose target opset."),
      llvm::cl::values(
          clEnumValN(OpSet::TF, "TF",
                     "Uses TF ops that mimic quantization behavior"),
          clEnumValN(OpSet::XLA, "XLA", "Uses TF XLA ops"),
          clEnumValN(OpSet::UNIFORM_QUANTIZED, "UNIFORM_QUANTIZED",
                     "Uses TF Uniform Quantized ops"))};

  // Initialize for tests.
  void initializeForTest() {
    if (!test_mode_) return;

    op_set_.setCallback([this](const OpSet& new_op_set) {
      quant_options_.set_op_set(new_op_set);
    });

    // Set the test quantization method to static-range.
    if (quant_options_.quantization_method().preset_method() ==
        QuantizationMethod::METHOD_UNSPECIFIED) {
      quant_options_.mutable_quantization_method()->set_preset_method(
          QuantizationMethod::METHOD_STATIC_RANGE_INT8);
    }

    if (quant_options_.quantization_method()
            .quantization_component_specs()
            .empty()) {
      auto add_new_spec =
          [this](QuantizationComponentSpec::QuantizationComponent component,
                 QuantizationComponentSpec::TensorType type) {
            QuantizationComponentSpec* new_spec =
                quant_options_.mutable_quantization_method()
                    ->add_quantization_component_specs();
            new_spec->set_quantization_component(component);
            new_spec->set_tensor_type(type);
          };

      add_new_spec(QuantizationComponentSpec::COMPONENT_ACTIVATION,
                   QuantizationComponentSpec::TENSORTYPE_INT_8);
      add_new_spec(QuantizationComponentSpec::COMPONENT_WEIGHT,
                   QuantizationComponentSpec::TENSORTYPE_INT_8);
      add_new_spec(QuantizationComponentSpec::COMPONENT_BIAS,
                   QuantizationComponentSpec::TENSORTYPE_INT_32);
    }

    if (quant_options_.unit_wise_quantization_specs().empty()) {
      // Opt-out a node named `test_opt_out`.
      UnitWiseQuantizationSpec* new_spec =
          quant_options_.add_unit_wise_quantization_specs();
      QuantizationUnit* new_unit = new_spec->add_unit();
      new_unit->set_node_name("test_opt_out");
      new_spec->mutable_quantization_method()->set_preset_method(
          QuantizationMethod::METHOD_NO_QUANTIZE);
    }
  }
};

class CheckQuantizableOps
    : public mlir::OpRewritePattern<TF::PartitionedCallOp> {
 public:
  explicit CheckQuantizableOps(MLIRContext* context,
                               const QuantizationOptions& quant_options)
      : OpRewritePattern<TF::PartitionedCallOp>(context),
        quant_options_(quant_options) {}

 private:
  LogicalResult matchAndRewrite(TF::PartitionedCallOp call_op,
                                PatternRewriter& rewriter) const override {
    StringRef function_name =
        mlir::cast<FlatSymbolRefAttr>(call_op.getFAttr()).getValue();
    if (!function_name.starts_with("composite_") ||
        !call_op->hasAttr(kQuantTraitAttrName)) {
      return failure();
    }

    absl::Status check_status;
    // TODO(b/270906404): Support weight-only gather for uniform quantized opset
    // in PTQ mode
    if (quant_options_.op_set() == OpSet::UNIFORM_QUANTIZED &&
        function_name.contains("gather")) {
      check_status.Update(absl::InternalError("Weight-only op is skipped."));
    }

    if (quant_options_.op_set() == OpSet::XLA) {
      check_status.Update(checkQuantizableOpsForXla(call_op, function_name));
    }

    // Only the composite functions with f32 inputs are quantizable.
    if (call_op.getResults().size() == 1 &&
        !mlir::cast<ShapedType>(call_op->getResult(0).getType())
             .getElementType()
             .isF32()) {
      check_status.Update(absl::InternalError(
          "Composite functions for quantization should be f32 type."));
    }

    // The OK status means this op is quantizable. Return failure since the
    // pattern doesn't rewrite anything yet.
    if (check_status.ok()) return failure();
    call_op->removeAttr(kQuantTraitAttrName);
    removeAttrMapAttribute(call_op, function_name, check_status.message());
    return success();
  }

  // Get the quantization method to apply to this composite function. If set,
  // the unit-wise quantization method overrides the default one.
  std::optional<QuantizationMethod> getUnitWiseQuantizationMethod(
      TF::PartitionedCallOp call_op) const {
    // If unit-wise quantization config is found, overwrite the default config.
    auto quantization_unit =
        quant::FindQuantizationUnitFromLoc(call_op.getLoc());
    if (!quantization_unit.has_value()) return std::nullopt;

    for (const auto& unit_config :
         quant_options_.unit_wise_quantization_specs()) {
      for (const auto& unit : unit_config.unit()) {
        if (!unit.op_type().empty() &&
            quantization_unit.value().op_type() != unit.op_type()) {
          continue;
        }

        if (!unit.node_name().empty()) {
          const RE2 node_name_regex(unit.node_name());
          if (!RE2::FullMatch(quantization_unit.value().node_name(),
                              node_name_regex)) {
            continue;
          }
        }

        if (!unit.func_name().empty()) {
          const RE2 func_name_regex(unit.func_name());
          if (!RE2::FullMatch(quantization_unit.value().func_name(),
                              func_name_regex)) {
            continue;
          }
        }

        // Overrides the default quantization method.
        return unit_config.quantization_method();
      }
    }
    return std::nullopt;
  }

  absl::Status checkQuantizableOpsForXla(TF::PartitionedCallOp call_op,
                                         StringRef function_name) const {
    // Disable quantization for the DepthwiseConv since it has no benefits in
    // the XLA opset.
    if (function_name.contains("depthwise_conv2d")) {
      return absl::InternalError(
          "DepthwiseConv2D doesn't get any benefit of quantization in XLA.");
    } else if (function_name.contains("conv2d")) {
      // For Conv2D, the channel dimension must be static to calculate the
      // feature group count.
      if (!HasStaticShapeAtDims(call_op->getOperand(0), /*dims=*/3)) {
        return absl::InternalError(
            "The channel dimension of Conv2D is required to be static.");
      }
    } else if (function_name.contains("conv3d")) {
      // For Conv3D, the channel dimension must be static to calculate the
      // feature group count.
      if (!HasStaticShapeAtDims(call_op->getOperand(0), /*dims=*/4)) {
        return absl::InternalError(
            "The channel dimension of Conv3D is required to be static.");
      }
    } else if (function_name.contains("batch_matmul")) {
      // For BatchMatMul, the input must be ranked to determine the batch
      // dimensions.
      ShapedType shaped_type =
          mlir::dyn_cast<ShapedType>(call_op->getOperand(0).getType());
      if (!shaped_type || !shaped_type.hasRank()) {
        return absl::InternalError("The input of BatchMatMul must have rank.");
      }
    } else if (function_name.contains("gather")) {
      // This op is guaranteed to be a constant as ODS checks IsConstTensor.
      // Check if the number of elements meets the requirement.
      int64_t num_elements =
          mlir::cast<ShapedType>(call_op.getOperand(0).getType())
              .getNumElements();
      if (num_elements < quant_options_.min_num_elements_for_weights()) {
        return absl::InternalError(
            "The params of Gather have fewer number of elements than "
            "the `min_num_elements_for_weights`.");
      }
    }

    // Disable quantization if the quantization method is NO_QUANTIZE.
    QuantizationMethod quantization_method =
        quant_options_.quantization_method();
    if (quantization_method.quantization_component_specs().empty()) {
      return absl::InternalError(
          "The quantization method has been set to METHOD_NO_QUANTIZE.");
    }

    // The unit-wise quantization config should override the loser-grained
    // quantization config, such as `enable_two_input_tensors`.
    bool is_unitwise_quantization_enabled = false;
    std::optional<QuantizationMethod> unit_wise_quantization_method =
        getUnitWiseQuantizationMethod(call_op);
    if (unit_wise_quantization_method.has_value()) {
      if (unit_wise_quantization_method.value()
              .quantization_component_specs()
              .empty()) {
        return absl::InternalError(
            "The unit-wise quantization method has been set to "
            "METHOD_NO_QUANTIZE.");
      }
      is_unitwise_quantization_enabled = true;
    }

    std::unique_ptr<tf_quant::OpQuantSpec> spec =
        tf_quant::GetTFOpQuantSpec(call_op);
    for (auto iter : spec->coeff_op_quant_dim) {
      Operation* preceding_op = call_op.getOperand(iter.first).getDefiningOp();
      // The XLA opset only supports constant filter/weight at the moment.
      bool is_weight_constant =
          preceding_op && preceding_op->hasTrait<OpTrait::ConstantLike>();

      // There might be q/dq ops after the filter/weight.
      if (auto dq_op =
              llvm::dyn_cast_or_null<mlir::quant::ir::DequantizeCastOp>(
                  preceding_op)) {
        if (auto q_op = llvm::dyn_cast_or_null<mlir::quant::ir::QuantizeCastOp>(
                dq_op.getArg().getDefiningOp())) {
          Operation* q_op_input = q_op.getArg().getDefiningOp();
          is_weight_constant =
              q_op_input && q_op_input->hasTrait<OpTrait::ConstantLike>();
        }
      }

      if (!is_weight_constant) {
        if (!function_name.contains("matmul") &&
            !function_name.contains("einsum")) {
          return absl::InternalError(
              "Non-constant weights are not supported at the moment,"
              " except matmul and einsum.");
        } else if (!quant_options_.enable_two_input_tensors() &&
                   !is_unitwise_quantization_enabled) {
          return absl::InternalError(
              "Quantization is disabled for this op due to the non-constant "
              "weight. You can enable it by setting `enable_two_input_tensors` "
              "to true or using unit-wise quantization config.");
        }
      }
    }

    return absl::OkStatus();
  }

  void removeAttrMapAttribute(TF::PartitionedCallOp call_op,
                              StringRef function_name,
                              StringRef error_message) const {
    ModuleOp module = call_op->getParentOfType<ModuleOp>();
    SymbolTable symbol_table(module);
    mlir::func::FuncOp composite_func =
        dyn_cast<func::FuncOp>(symbol_table.lookup(function_name));
    if (!composite_func) return;

    composite_func.walk([&](Operation* op) {
      if (op->hasAttr(kAttrMapAttribute)) {
        op->removeAttr(kAttrMapAttribute);

        std::string log_message;
        llvm::raw_string_ostream log_stream(log_message);
        op->getLoc().print(log_stream);
        log_stream << ": Quantization disabled on this op: ";
        log_stream << error_message << "\n";
        log_stream << "See the current operation:\n";
        op->print(log_stream);
        VLOG(2) << log_message;
      }
    });
  }

  const QuantizationOptions& quant_options_;
};

static PassRegistration<TFLiftQuantizableSpotsAsFunctionsPass> pass;

#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/tf_lift_quantizable_spots_as_functions.inc"

void TFLiftQuantizableSpotsAsFunctionsPass::runOnOperation() {
  MLIRContext* ctx = &getContext();
  RewritePatternSet patterns(ctx);
  ModuleOp module = getOperation();

  populateWithGenerated(patterns);
  patterns.add<CheckQuantizableOps>(ctx, quant_options_);
  FrozenRewritePatternSet frozen_patterns(std::move(patterns));

  // Iterate over the sorted list of functions to keep the order deterministic.
  for (func::FuncOp func : quant::GetSortedFunctions(module)) {
    if (failed(applyPatternsGreedily(func, frozen_patterns))) {
      func.emitError() << "quant-lift-quantizable-spots-as-functions failed.";
      signalPassFailure();
    }
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateTFLiftQuantizableSpotsAsFunctionsPass(
    const QuantizationOptions& quant_options) {
  return std::make_unique<TFLiftQuantizableSpotsAsFunctionsPass>(quant_options);
}

}  // namespace tf_quant
}  // namespace mlir
