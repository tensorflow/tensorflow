/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/quantization/stablehlo/instrumentations/save_report.h"

#include <memory>
#include <optional>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/common/test_base.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/io.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/passes.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tsl/platform/protobuf.h"  // IWYU pragma: keep
#include "tsl/platform/status_matchers.h"

namespace mlir::quant::stablehlo {
namespace {

using ::stablehlo::quantization::QuantizationResults;
using ::stablehlo::quantization::io::ReadFileToString;
using ::testing::SizeIs;
using ::testing::StrEq;
using ::tsl::protobuf::TextFormat;
using ::tsl::testing::IsOk;
using ::tsl::testing::StatusIs;

using SaveQuantizationReportInstrumentationTest = QuantizationTestBase;

TEST_F(SaveQuantizationReportInstrumentationTest, SaveReport) {
  constexpr absl::string_view kModuleWithCompositeDotGeneral = R"mlir(
    func.func @main(%arg0: tensor<1x2xf32>) -> tensor<1x3xf32> {
      %cst = "tf.Const"() {value = dense<3.00000000e-1> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
      %0 = "quantfork.stats"(%arg0) {layerStats = dense<[6.00000000e-6, 9.00000000e-1]> : tensor<2xf32>} : (tensor<1x2xf32>) -> tensor<1x2xf32>
      %1 = "tf.XlaCallModule"(%0, %cst) {Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_fn, _original_entry_function = "composite_dot_general_fn", _quantization_method = "static_range_ptq { }", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable", device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
      %2 = "quantfork.stats"(%1) {layerStats = dense<[5.00000000e-6, 7.00000000e-1]> : tensor<2xf32>} : (tensor<1x3xf32>) -> tensor<1x3xf32>
      return %2 : tensor<1x3xf32>
    }

    func.func private @composite_dot_general_fn(%arg0: tensor<1x2xf32>, %arg1: tensor<2x3xf32>) -> tensor<1x3xf32> attributes {_from_xla_call_module} {
      %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
      return %0 : tensor<1x3xf32>
    }
  )mlir";

  const OwningOpRef<ModuleOp> module_op =
      ParseModuleOpString(kModuleWithCompositeDotGeneral);
  ASSERT_TRUE(module_op);

  // Create a pass manager with `SaveQuantizationReportInstrumentation` and
  // `QuantizeCompositeFunctionsPass`. Run the passes against `module_op`.
  PassManager pm(ctx_.get());

  QuantizeCompositeFunctionsPassOptions options;
  pm.addPass(createQuantizeCompositeFunctionsPass(options));

  const std::string report_file_path =
      absl::StrCat(testing::TempDir(), "/save_report.txtpb");
  pm.addInstrumentation(std::make_unique<SaveQuantizationReportInstrumentation>(
      report_file_path));

  const LogicalResult run_result = pm.run(*module_op);
  ASSERT_TRUE(succeeded(run_result));

  // Check that the report file contains `QuantizationResults` textproto,
  // reflecting the quantization results, in this case the
  // `composite_dot_general_fn` with quantized with `static_range_ptq` method.
  const absl::StatusOr<std::string> file_data =
      ReadFileToString(report_file_path);
  ASSERT_THAT(file_data, IsOk());

  /*
  results {
    quantizable_unit {
      name: "composite_dot_general_fn"
    }
    method { static_range_ptq { } }
  }
  */
  QuantizationResults results{};
  ASSERT_TRUE(TextFormat::ParseFromString(*file_data, &results));
  ASSERT_THAT(results.results(), SizeIs(1));
  EXPECT_THAT(results.results(0).quantizable_unit().name(),
              StrEq("composite_dot_general_fn"));
  EXPECT_TRUE(results.results(0).method().has_static_range_ptq());
}

TEST_F(SaveQuantizationReportInstrumentationTest,
       ReportNotSavedWhenNoQuantizeCompositeFunctionsPass) {
  constexpr absl::string_view kModuleWithCompositeDotGeneral = R"mlir(
    func.func @main(%arg0: tensor<1x2xf32>) -> tensor<1x3xf32> {
      %cst = "stablehlo.constant"() {value = dense<3.00000000e-1> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
      %0 = "quantfork.stats"(%arg0) {layerStats = dense<[6.00000000e-6, 9.00000000e-1]> : tensor<2xf32>} : (tensor<1x2xf32>) -> tensor<1x2xf32>
      %1 = "tf.XlaCallModule"(%0, %cst) {Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_fn, _original_entry_function = "composite_dot_general_fn", _quantization_method = "static_range_ptq { }", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable", device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
      %2 = "quantfork.stats"(%1) {layerStats = dense<[5.00000000e-6, 7.00000000e-1]> : tensor<2xf32>} : (tensor<1x3xf32>) -> tensor<1x3xf32>
      return %2 : tensor<1x3xf32>
    }

    func.func private @composite_dot_general_fn(%arg0: tensor<1x2xf32>, %arg1: tensor<2x3xf32>) -> tensor<1x3xf32> attributes {_from_xla_call_module} {
      %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
      return %0 : tensor<1x3xf32>
    }
  )mlir";

  const OwningOpRef<ModuleOp> module_op =
      ParseModuleOpString(kModuleWithCompositeDotGeneral);
  ASSERT_TRUE(module_op);

  // Create a pass manager with `SaveQuantizationReportInstrumentation` a pass
  // that is not `QuantizeCompositeFunctionsPass`. Run the passes against
  // `module_op`.
  PassManager pm(ctx_.get());

  pm.addPass(createPrepareQuantizePass());

  const std::string report_file_path = absl::StrCat(
      testing::TempDir(),
      "/report_not_saved_no_quantize_composite_functions_pass.txtpb");
  pm.addInstrumentation(std::make_unique<SaveQuantizationReportInstrumentation>(
      report_file_path));

  const LogicalResult run_result = pm.run(*module_op);
  ASSERT_TRUE(succeeded(run_result));

  // The report file is not created because `QuantizeCompositeFunctionsPass` was
  // not run.
  EXPECT_THAT(ReadFileToString(report_file_path),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(SaveQuantizationReportInstrumentationTest,
       ReportNotSavedWhenReportFilePathIsNullopt) {
  constexpr absl::string_view kModuleWithCompositeDotGeneral = R"mlir(
    func.func @main(%arg0: tensor<1x2xf32>) -> tensor<1x3xf32> {
      %cst = "stablehlo.constant"() {value = dense<3.00000000e-1> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
      %0 = "quantfork.stats"(%arg0) {layerStats = dense<[6.00000000e-6, 9.00000000e-1]> : tensor<2xf32>} : (tensor<1x2xf32>) -> tensor<1x2xf32>
      %1 = "tf.XlaCallModule"(%0, %cst) {Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_fn, _original_entry_function = "composite_dot_general_fn", _quantization_method = "static_range_ptq { }", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable", device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
      %2 = "quantfork.stats"(%1) {layerStats = dense<[5.00000000e-6, 7.00000000e-1]> : tensor<2xf32>} : (tensor<1x3xf32>) -> tensor<1x3xf32>
      return %2 : tensor<1x3xf32>
    }

    func.func private @composite_dot_general_fn(%arg0: tensor<1x2xf32>, %arg1: tensor<2x3xf32>) -> tensor<1x3xf32> attributes {_from_xla_call_module} {
      %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
      return %0 : tensor<1x3xf32>
    }
  )mlir";

  const OwningOpRef<ModuleOp> module_op =
      ParseModuleOpString(kModuleWithCompositeDotGeneral);
  ASSERT_TRUE(module_op);

  PassManager pm(ctx_.get());

  QuantizeCompositeFunctionsPassOptions options;
  pm.addPass(createQuantizeCompositeFunctionsPass(options));
  pm.addInstrumentation(std::make_unique<SaveQuantizationReportInstrumentation>(
      /*file_path=*/std::nullopt));

  // The report file is not created and `SaveQuantizationReportInstrumentation`
  // is not run, but the passes still run without errors.
  const LogicalResult run_result = pm.run(*module_op);
  ASSERT_TRUE(succeeded(run_result));
}

}  // namespace
}  // namespace mlir::quant::stablehlo
