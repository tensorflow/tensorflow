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
#include <memory>
#include <utility>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/tf_quant_ops.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/utils/lift_as_function_call_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace quant {
namespace {

using ::tensorflow::quantization::CalibrationOptions;

constexpr StringRef kQuantTraitAttrName = "_tfl_quant_trait";

class InsertCustomAggregationOpsPass
    : public PassWrapper<InsertCustomAggregationOpsPass,
                         OperationPass<func::FuncOp>> {
 public:
  explicit InsertCustomAggregationOpsPass() : test_mode_(true) {
    initializeForTest();
  }

  explicit InsertCustomAggregationOpsPass(const CalibrationOptions &calib_opts)
      : test_mode_(false), calib_opts_(calib_opts) {}

  InsertCustomAggregationOpsPass(const InsertCustomAggregationOpsPass &other) {
    test_mode_ = other.test_mode_;
    test_case_ = other.test_case_;
    calib_opts_ = other.calib_opts_;
    initializeForTest();
  }

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertCustomAggregationOpsPass)

  StringRef getArgument() const final {
    // This is the argument used to refer to the pass in the textual format (on
    // the commandline for example).
    return "quant-insert-custom-aggregation-ops";
  }

  StringRef getDescription() const final {
    // This is a brief description of the pass.
    return "Insert custom aggregation ops for the calibration procedure";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TF::TensorFlowDialect>();
  }

  void runOnOperation() override;

 private:
  enum TestCase {
    TEST_CASE_MIN_MAX,
    TEST_CASE_AVERAGE_MIN_MAX,
    TEST_CASE_HISTOGRAM_PERCENTILE,
    TEST_CASE_HISTOGRAM_MSE_BRUTEFORCE,
    TEST_CASE_HISTOGRAM_MSE_MAX_FREQUENCY,
    TEST_CASE_HISTOGRAM_MSE_SYMMETRIC,
  };

  bool test_mode_;
  CalibrationOptions calib_opts_;
  Option<TestCase> test_case_{
      *this, "test-case",
      llvm::cl::desc(
          "Select a the test case for testing various calibration methods. It "
          "sets the value of calib_opts_ when test_mode_ is true."),
      llvm::cl::init(TEST_CASE_MIN_MAX),
      llvm::cl::values(
          clEnumValN(TEST_CASE_MIN_MAX, "MIN_MAX",
                     "Uses MIN_MAX calibration method"),
          clEnumValN(TEST_CASE_AVERAGE_MIN_MAX, "AVERAGE_MIN_MAX",
                     "Uses AVERAGE_MIN_MAX calibration method"),
          clEnumValN(TEST_CASE_HISTOGRAM_PERCENTILE, "HISTOGRAM_PERCENTILE",
                     "Uses HISTOGRAM_PERCENTILE calibration method"),
          clEnumValN(TEST_CASE_HISTOGRAM_MSE_BRUTEFORCE,
                     "HISTOGRAM_MSE_BRUTEFORCE",
                     "Uses HISTOGRAM_MSE_BRUTEFORCE calibration method"),
          clEnumValN(TEST_CASE_HISTOGRAM_MSE_MAX_FREQUENCY,
                     "HISTOGRAM_MSE_MAX_FREQUENCY",
                     "Uses HISTOGRAM_MSE_MAX_FREQUENCY calibration "
                     "method"),
          clEnumValN(TEST_CASE_HISTOGRAM_MSE_SYMMETRIC,
                     "HISTOGRAM_MSE_SYMMETRIC",
                     "Uses HISTOGRAM_MSE_SYMMETRIC calibration "
                     "method"))};

  // Initialize for tests.
  void initializeForTest() {
    if (!test_mode_) return;

    switch (test_case_.getValue()) {
      case TEST_CASE_MIN_MAX:
        calib_opts_.set_calibration_method(
            CalibrationOptions::CALIBRATION_METHOD_MIN_MAX);
        break;
      case TEST_CASE_AVERAGE_MIN_MAX:
        calib_opts_.set_calibration_method(
            CalibrationOptions::CALIBRATION_METHOD_AVERAGE_MIN_MAX);
        break;
      case TEST_CASE_HISTOGRAM_PERCENTILE: {
        calib_opts_.set_calibration_method(
            CalibrationOptions::CALIBRATION_METHOD_HISTOGRAM_PERCENTILE);
        auto calibration_parameters =
            CalibrationOptions::CalibrationParameters();
        calibration_parameters.set_initial_num_bins(256);
        calibration_parameters.set_min_percentile(0.001);
        calibration_parameters.set_max_percentile(99.999);
        calib_opts_.mutable_calibration_parameters()->CopyFrom(
            calibration_parameters);
        break;
      }
      case TEST_CASE_HISTOGRAM_MSE_BRUTEFORCE: {
        calib_opts_.set_calibration_method(
            CalibrationOptions::CALIBRATION_METHOD_HISTOGRAM_MSE_BRUTEFORCE);
        auto calibration_parameters =
            CalibrationOptions::CalibrationParameters();
        calibration_parameters.set_initial_num_bins(256);
        calib_opts_.mutable_calibration_parameters()->CopyFrom(
            calibration_parameters);
        break;
      }
      case TEST_CASE_HISTOGRAM_MSE_MAX_FREQUENCY: {
        calib_opts_.set_calibration_method(
            CalibrationOptions::CALIBRATION_METHOD_HISTOGRAM_MSE_MAX_FREQUENCY);
        auto calibration_parameters =
            CalibrationOptions::CalibrationParameters();
        calibration_parameters.set_initial_num_bins(256);
        calib_opts_.mutable_calibration_parameters()->CopyFrom(
            calibration_parameters);
        break;
      }
      case TEST_CASE_HISTOGRAM_MSE_SYMMETRIC: {
        calib_opts_.set_calibration_method(
            CalibrationOptions::CALIBRATION_METHOD_HISTOGRAM_MSE_SYMMETRIC);
        auto calibration_parameters =
            CalibrationOptions::CalibrationParameters();
        calibration_parameters.set_initial_num_bins(256);
        calib_opts_.mutable_calibration_parameters()->CopyFrom(
            calibration_parameters);
        break;
      }
    }
  }
};

static PassRegistration<InsertCustomAggregationOpsPass> pass;

class AddCustomAggregationOp : public RewritePattern {
 public:
  // Does not take ownership of context, which must refer to a valid value that
  // outlives this object.
  explicit AddCustomAggregationOp(MLIRContext *context,
                                  const CalibrationOptions &calib_opts)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context),
        calib_opts_(calib_opts) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Return early if the given operator is the custom aggregator op.
    if (dyn_cast_or_null<TF::CustomAggregatorOp>(op)) return failure();

    // Return early if the given op is a non-quantizable op.
    auto call_op = dyn_cast_or_null<TF::PartitionedCallOp>(op);
    if (call_op && !op->hasAttr(kQuantTraitAttrName)) {
      return failure();
    }

    bool mutated = false;
    for (Value input : op->getOperands()) {
      Type element_type = getElementTypeOrSelf(input.getType());
      // Non-float cases won't be calibrated.
      if (!element_type.isF32()) {
        continue;
      }

      // Skip when the given operator is under the quantizable spot.
      if (IsInLiftedFunc(op)) {
        continue;
      }

      // Skip when there is any already existing CustomAggregatorOp found.
      Operation *defining_op = input.getDefiningOp();
      if (dyn_cast_or_null<TF::CustomAggregatorOp>(defining_op)) {
        continue;
      }

      // Skip calibration when the given operand comes from a constant.
      if (defining_op != nullptr &&
          defining_op->hasTrait<OpTrait::ConstantLike>()) {
        continue;
      }

      // ID attribute will have empty value for now.
      SmallVector<NamedAttribute, 5> attributes{
          rewriter.getNamedAttr("id", rewriter.getStringAttr("")),
          rewriter.getNamedAttr(
              "calibration_method",
              rewriter.getI32IntegerAttr(calib_opts_.calibration_method())),
          rewriter.getNamedAttr(
              "initial_num_bins",
              rewriter.getI32IntegerAttr(
                  calib_opts_.calibration_parameters().initial_num_bins())),
          rewriter.getNamedAttr(
              "min_percentile",
              rewriter.getF32FloatAttr(
                  calib_opts_.calibration_parameters().min_percentile())),
          rewriter.getNamedAttr(
              "max_percentile",
              rewriter.getF32FloatAttr(
                  calib_opts_.calibration_parameters().max_percentile())),
      };

      // Insert custom aggregation op between operand and operator.
      rewriter.setInsertionPointAfterValue(input);
      Operation *aggregator_op = rewriter.create<TF::CustomAggregatorOp>(
          op->getLoc(), input.getType(), input, attributes);

      Value aggregator_op_result = aggregator_op->getOpResult(0);
      input.replaceAllUsesWith(aggregator_op_result);
      aggregator_op->replaceUsesOfWith(aggregator_op_result, input);

      // Mark mutated.
      mutated = true;
    }

    // Return failure when there is no matching operand.
    return mutated ? success() : failure();
  }

 private:
  CalibrationOptions calib_opts_;
};

void InsertCustomAggregationOpsPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  func::FuncOp func = getOperation();

  patterns.add<AddCustomAggregationOp>(ctx, calib_opts_);
  if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
    func.emitError() << "quant-insert-custom-aggregation-ops failed.";
    signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
CreateInsertCustomAggregationOpsPass(const CalibrationOptions &calib_opts) {
  return std::make_unique<InsertCustomAggregationOpsPass>(calib_opts);
}

}  // namespace quant
}  // namespace mlir
