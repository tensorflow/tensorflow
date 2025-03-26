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
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/pre_calibration.h"

#include <type_traits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/common/test_base.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/config.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/tf_quant_ops.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "xla/tsl/platform/status_matchers.h"

namespace mlir::quant::stablehlo {
namespace {

using ::stablehlo::quantization::ExpandPresets;
using ::stablehlo::quantization::PopulateDefaults;
using ::stablehlo::quantization::QuantizationConfig;
using ::testing::Contains;
using ::testing::SizeIs;
using ::testing::StartsWith;
using ::testing::StrEq;
using ::tsl::testing::IsOk;

// Matches an operation whose `getSymName` equals `name`.
MATCHER_P(HasSymName, name, "") {
  auto non_const_arg = const_cast<std::remove_const_t<decltype(arg)>>(arg);
  *result_listener << "where the name is " << non_const_arg.getSymName().str();
  return non_const_arg.getSymName() == name;
}

// Matches an operation that has a StringAttr whose name is `name` and value
// matches `value_matcher`.
MATCHER_P2(HasStringAttr, name, value_matcher,
           absl::StrCat(negation ? "doesn't have" : "has",
                        "string attribute: ", name, ", with desirable value")) {
  auto non_const_arg = const_cast<std::remove_const_t<decltype(arg)>>(arg);
  return non_const_arg->template hasAttrOfType<StringAttr>(name) &&
         ExplainMatchResult(
             value_matcher,
             non_const_arg->template getAttrOfType<StringAttr>(name).str(),
             result_listener);
}

// Matches an operation that has a FlatSymbolRefAttr whose name is `name` and
// value matches `value_matcher`.
MATCHER_P2(HasSymNameAttr, name, value_matcher,
           absl::StrCat(negation ? "doesn't have" : "has",
                        "string attribute: ", name, ", with desirable value")) {
  auto non_const_arg = const_cast<std::remove_const_t<decltype(arg)>>(arg);
  return non_const_arg->template hasAttrOfType<FlatSymbolRefAttr>(name) &&
         ExplainMatchResult(
             value_matcher,
             non_const_arg->template getAttrOfType<FlatSymbolRefAttr>(name)
                 .getValue()
                 .str(),
             result_listener);
}

using PreCalibrationComponentTest = ::mlir::quant::QuantizationTestBase;

TEST_F(PreCalibrationComponentTest,
       HasCustomAggregatorOpAndQuantizableFuncForSimpleDotGeneral) {
  PreCalibrationComponent component(ctx_.get());
  OwningOpRef<ModuleOp> module_op = ParseModuleOpString(R"mlir(
    module attributes {} {
      func.func @main(%arg0: tensor<1x4xf32>) -> tensor<1x3xf32> attributes {} {
        %0 = stablehlo.constant dense<1.0> : tensor<4x3xf32>
        %1 = stablehlo.dot_general %arg0, %0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4xf32>, tensor<4x3xf32>) -> tensor<1x3xf32>
        return %1 : tensor<1x3xf32>
      }
    }
  )mlir");
  ASSERT_TRUE(module_op);

  QuantizationConfig quantization_config{};
  quantization_config.mutable_static_range_ptq_preset();
  quantization_config = ExpandPresets(PopulateDefaults(quantization_config));
  absl::StatusOr<ModuleOp> pre_calibration_result =
      component.Run(*module_op, quantization_config);

  EXPECT_THAT(pre_calibration_result, IsOk());

  SmallVector<func::FuncOp> func_ops;
  for (auto func_op : pre_calibration_result->getOps<func::FuncOp>()) {
    func_ops.push_back(func_op);
  }
  ASSERT_THAT(func_ops, SizeIs(2));
  EXPECT_THAT(func_ops, Contains(HasSymName("main")));
  EXPECT_THAT(func_ops, Contains(HasSymName("composite_dot_general_fn_1")));

  // Tests that there is a XlaCallModuleOp that calls the composite quantizable
  // function.
  SmallVector<TF::XlaCallModuleOp> xla_call_module_ops;
  for (auto xla_call_module_op : func_ops[0].getOps<TF::XlaCallModuleOp>()) {
    xla_call_module_ops.push_back(xla_call_module_op);
  }
  ASSERT_THAT(xla_call_module_ops, SizeIs(1));
  auto xla_call_module_op = xla_call_module_ops[0];
  EXPECT_THAT(xla_call_module_op,
              HasStringAttr("_tfl_quant_trait", StrEq("fully_quantizable")));
  EXPECT_THAT(xla_call_module_op,
              HasSymNameAttr("_entry_function",
                             StartsWith("composite_dot_general_fn")));
  EXPECT_THAT(xla_call_module_op,
              HasStringAttr("_original_entry_function",
                            StartsWith("composite_dot_general_fn")));

  // Tests that there are CustomAggregatorOps inserted.
  SmallVector<TF::CustomAggregatorOp> custom_aggregator_ops;
  for (auto custom_aggregator_op :
       func_ops[0].getOps<TF::CustomAggregatorOp>()) {
    custom_aggregator_ops.push_back(custom_aggregator_op);
  }
  EXPECT_THAT(custom_aggregator_ops, SizeIs(2));
}

}  // namespace
}  // namespace mlir::quant::stablehlo
