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
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/report.h"

#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/common/test_base.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/io.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tsl/platform/protobuf.h"  // IWYU pragma: keep
#include "tsl/platform/status_matchers.h"

namespace mlir::quant::stablehlo {
namespace {

using ::stablehlo::quantization::Method;
using ::stablehlo::quantization::QuantizableUnit;
using ::stablehlo::quantization::QuantizationResult;
using ::stablehlo::quantization::QuantizationResults;
using ::stablehlo::quantization::io::ReadFileToString;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::SizeIs;
using ::testing::StrEq;
using ::testing::TempDir;
using ::tsl::protobuf::TextFormat;
using ::tsl::testing::IsOk;

using QuantizationReportTest = ::mlir::quant::QuantizationTestBase;

TEST_F(QuantizationReportTest, GetQuantizationResultsReturnsEmptyResults) {
  QuantizationReport report{};

  const QuantizationResults& results = report.GetQuantizationResults();
  ASSERT_THAT(results.results(), IsEmpty());
}

TEST_F(QuantizationReportTest, AddQuantizationResult) {
  // Construct a `QuantizationResult` to add, representing a unit named
  // `quantized_my_function` that is not quantized.
  QuantizationResult result{};
  QuantizableUnit& quantizable_unit = *result.mutable_quantizable_unit();
  quantizable_unit.set_name("quantized_my_function");

  Method& method = *result.mutable_method();
  method.mutable_no_quantization();

  QuantizationReport report{};
  report.AddQuantizationResult(std::move(result));

  const QuantizationResults& results = report.GetQuantizationResults();
  ASSERT_THAT(results.results(), SizeIs(1));

  const QuantizationResult& first_result = results.results(0);
  EXPECT_THAT(first_result.quantizable_unit().name(),
              StrEq("quantized_my_function"));
  EXPECT_TRUE(first_result.method().has_no_quantization());
}

TEST_F(QuantizationReportTest, InitializeWithModuleOp) {
  constexpr absl::string_view kQuantizedDotGeneral = R"mlir(
    func.func @main(%arg0: tensor<1x2xf32>) -> tensor<1x3xf32> {
      %0 = stablehlo.constant() {value = dense<127> : tensor<2x3xi8>} : () -> tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {1.000000e+0,2.000000e+0,3.000000e+0}>>
      %1 = stablehlo.uniform_quantize %arg0 : (tensor<1x2xf32>) -> tensor<1x2x!quant.uniform<i8:f32, 4.000000e+0>>
      %2 = call @quantized_dot_general_fn(%1, %0) {_quantization_method = "static_range_ptq { }"} : (tensor<1x2x!quant.uniform<i8:f32, 4.000000e+0>>, tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {1.000000e+0,2.000000e+0,3.000000e+0}>>) -> tensor<1x3x!quant.uniform<i8:f32, 5.000000e+0>>
      %3 = stablehlo.uniform_dequantize %2 : (tensor<1x3x!quant.uniform<i8:f32, 5.000000e+0>>) -> tensor<1x3xf32>
      return %3 : tensor<1x3xf32>
    }

    func.func private @quantized_dot_general_fn(%arg0: tensor<1x2x!quant.uniform<i8:f32, 4.000000e+0>>, %arg1: tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {1.000000e+0,2.000000e+0,3.000000e+0}>>) -> tensor<1x3x!quant.uniform<i8:f32, 5.000000e+0>> {
      %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<1x2x!quant.uniform<i8:f32, 4.000000e+0>>, tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {1.000000e+0,2.000000e+0,3.000000e+0}>>) -> tensor<1x3x!quant.uniform<i32:f32:1, {6.000000e+0,7.000000e+0,8.000000e+0}>>
      %1 = stablehlo.uniform_quantize %0 : (tensor<1x3x!quant.uniform<i32:f32:1, {6.000000e+0,7.000000e+0,8.000000e+0}>>) -> tensor<1x3x!quant.uniform<i8:f32, 5.000000e+0>>
      return %1 : tensor<1x3x!quant.uniform<i8:f32, 5.000000e+0>>
    }
  )mlir";

  const OwningOpRef<ModuleOp> module_op =
      ParseModuleOpString(kQuantizedDotGeneral);
  ASSERT_TRUE(module_op);

  const QuantizationReport report(*module_op);
  const QuantizationResults& results = report.GetQuantizationResults();
  ASSERT_THAT(results.results(), SizeIs(1));

  // Test that the quantized `QuantizableUnit` corresponding to
  // `composite_dot_general_fn` is captured.
  const QuantizationResult& result = results.results(0);
  EXPECT_THAT(result.quantizable_unit().name(),
              StrEq("composite_dot_general_fn"));
  EXPECT_TRUE(result.method().has_static_range_ptq());
}

TEST_F(QuantizationReportTest,
       InitializeWithModuleOpWithoutQuantizationMethodAttribute) {
  // A quantized dot_general op but the `CallOp` is missing the
  // `_quantization_method` attribute.
  constexpr absl::string_view
      kQuantizedDotGeneralMissingQuantizationMethodAttr = R"mlir(
    func.func @main(%arg0: tensor<1x2xf32>) -> tensor<1x3xf32> {
      %0 = stablehlo.constant() {value = dense<127> : tensor<2x3xi8>} : () -> tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {1.000000e+0,2.000000e+0,3.000000e+0}>>
      %1 = stablehlo.uniform_quantize %arg0 : (tensor<1x2xf32>) -> tensor<1x2x!quant.uniform<i8:f32, 4.000000e+0>>
      %2 = call @quantized_dot_general_fn(%1, %0) : (tensor<1x2x!quant.uniform<i8:f32, 4.000000e+0>>, tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {1.000000e+0,2.000000e+0,3.000000e+0}>>) -> tensor<1x3x!quant.uniform<i8:f32, 5.000000e+0>>
      %3 = stablehlo.uniform_dequantize %2 : (tensor<1x3x!quant.uniform<i8:f32, 5.000000e+0>>) -> tensor<1x3xf32>
      return %3 : tensor<1x3xf32>
    }

    func.func private @quantized_dot_general_fn(%arg0: tensor<1x2x!quant.uniform<i8:f32, 4.000000e+0>>, %arg1: tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {1.000000e+0,2.000000e+0,3.000000e+0}>>) -> tensor<1x3x!quant.uniform<i8:f32, 5.000000e+0>> {
      %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<1x2x!quant.uniform<i8:f32, 4.000000e+0>>, tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {1.000000e+0,2.000000e+0,3.000000e+0}>>) -> tensor<1x3x!quant.uniform<i32:f32:1, {6.000000e+0,7.000000e+0,8.000000e+0}>>
      %1 = stablehlo.uniform_quantize %0 : (tensor<1x3x!quant.uniform<i32:f32:1, {6.000000e+0,7.000000e+0,8.000000e+0}>>) -> tensor<1x3x!quant.uniform<i8:f32, 5.000000e+0>>
      return %1 : tensor<1x3x!quant.uniform<i8:f32, 5.000000e+0>>
    }
  )mlir";

  const OwningOpRef<ModuleOp> module_op =
      ParseModuleOpString(kQuantizedDotGeneralMissingQuantizationMethodAttr);
  ASSERT_TRUE(module_op);

  const QuantizationReport report(*module_op);
  const QuantizationResults& results = report.GetQuantizationResults();
  // The quantized call op without the _quantization_method attribute is not
  // captured as a `QuantizationResult`.
  ASSERT_THAT(results.results(), IsEmpty());
}

TEST_F(QuantizationReportTest, InitializeWithModuleOpWithInvalidCalleeName) {
  // A quantized dot_general op but the callee function has an invalid name. It
  // is expected to start with `quantized_`.
  constexpr absl::string_view kQuantizedDotGeneralWithInvalidCalleeName =
      R"mlir(
    func.func @main(%arg0: tensor<1x2xf32>) -> tensor<1x3xf32> {
      %0 = stablehlo.constant() {value = dense<127> : tensor<2x3xi8>} : () -> tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {1.000000e+0,2.000000e+0,3.000000e+0}>>
      %1 = stablehlo.uniform_quantize %arg0 : (tensor<1x2xf32>) -> tensor<1x2x!quant.uniform<i8:f32, 4.000000e+0>>
      %2 = call @invalid_quantized_dot_general_fn(%1, %0) {_quantization_method = "static_range_ptq { }"} : (tensor<1x2x!quant.uniform<i8:f32, 4.000000e+0>>, tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {1.000000e+0,2.000000e+0,3.000000e+0}>>) -> tensor<1x3x!quant.uniform<i8:f32, 5.000000e+0>>
      %3 = stablehlo.uniform_dequantize %2 : (tensor<1x3x!quant.uniform<i8:f32, 5.000000e+0>>) -> tensor<1x3xf32>
      return %3 : tensor<1x3xf32>
    }

    func.func private @invalid_quantized_dot_general_fn(%arg0: tensor<1x2x!quant.uniform<i8:f32, 4.000000e+0>>, %arg1: tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {1.000000e+0,2.000000e+0,3.000000e+0}>>) -> tensor<1x3x!quant.uniform<i8:f32, 5.000000e+0>> {
      %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<1x2x!quant.uniform<i8:f32, 4.000000e+0>>, tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {1.000000e+0,2.000000e+0,3.000000e+0}>>) -> tensor<1x3x!quant.uniform<i32:f32:1, {6.000000e+0,7.000000e+0,8.000000e+0}>>
      %1 = stablehlo.uniform_quantize %0 : (tensor<1x3x!quant.uniform<i32:f32:1, {6.000000e+0,7.000000e+0,8.000000e+0}>>) -> tensor<1x3x!quant.uniform<i8:f32, 5.000000e+0>>
      return %1 : tensor<1x3x!quant.uniform<i8:f32, 5.000000e+0>>
    }
  )mlir";

  const OwningOpRef<ModuleOp> module_op =
      ParseModuleOpString(kQuantizedDotGeneralWithInvalidCalleeName);
  ASSERT_TRUE(module_op);

  const QuantizationReport report(*module_op);
  const QuantizationResults& results = report.GetQuantizationResults();
  // The quantized call op whose callee doesn't start with `quantized_` is not
  // captured as a `QuantizationResult`.
  ASSERT_THAT(results.results(), IsEmpty());
}

TEST_F(QuantizationReportTest, InitializeWithModuleOpWithNonQuantizedOp) {
  constexpr absl::string_view kNonQuantizedDotGeneral = R"mlir(
    func.func @main(%arg0: tensor<1x2xf32>) -> tensor<1x3xf32> {
      %0 = stablehlo.constant dense<3.000000e+0> : tensor<2x3xf32>
      %1 = "tf.XlaCallModule"(%arg0, %0) {Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_fn, _stablehlo_version = "1.0.0", _original_entry_function = "composite_dot_general_fn", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable",   device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
      return %1 : tensor<1x3xf32>
    }

    func.func private @composite_dot_general_fn(%arg0: tensor<1x2xf32>, %arg1: tensor<2x3xf32>) -> tensor<1x3xf32> {
      %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
      return %0 : tensor<1x3xf32>
    }
  )mlir";

  const OwningOpRef<ModuleOp> module_op =
      ParseModuleOpString(kNonQuantizedDotGeneral);
  ASSERT_TRUE(module_op);

  const QuantizationReport report(*module_op);
  const QuantizationResults& results = report.GetQuantizationResults();
  ASSERT_THAT(results.results(), SizeIs(1));

  // Test that the unquantized `QuantizableUnit` corresponding to
  // `composite_dot_general_fn` is captured. The `Method` contains
  // `NoQuantization`.
  const QuantizationResult& result = results.results(0);
  EXPECT_THAT(result.quantizable_unit().name(),
              StrEq("composite_dot_general_fn"));
  EXPECT_TRUE(result.method().has_no_quantization());
}

TEST_F(QuantizationReportTest,
       InitializeWithModuleOpWithQuantizedAndNonQuantizedOps) {
  constexpr absl::string_view kQuantizedDotGeneralAndNonQuantizedDotGeneral =
      R"mlir(
    func.func @main(%arg0: tensor<1x2xf32>, %arg1: tensor<1x2xf32>) -> tensor<1x3xf32> {
      // Non-quantized dot_general.
      %0 = stablehlo.constant dense<3.000000e+0> : tensor<2x3xf32>
      %1 = "tf.XlaCallModule"(%arg0, %0) {Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_fn_1, _stablehlo_verison = "1.0.0", _original_entry_function = "composite_dot_general_fn_1", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable", device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
      // Quantized dot_general.
      %2 = stablehlo.constant() {value = dense<127> : tensor<2x3xi8>} : () -> tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {1.000000e+0,2.000000e+0,3.000000e+0}>>
      %3 = stablehlo.uniform_quantize %arg1 : (tensor<1x2xf32>) -> tensor<1x2x!quant.uniform<i8:f32, 4.000000e+0>>
      %4 = call @quantized_dot_general_fn_2(%3, %2) {_quantization_method = "static_range_ptq { }"} : (tensor<1x2x!quant.uniform<i8:f32, 4.000000e+0>>, tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {1.000000e+0,2.000000e+0,3.000000e+0}>>) -> tensor<1x3x!quant.uniform<i8:f32, 5.000000e+0>>
      %5 = stablehlo.uniform_dequantize %4 : (tensor<1x3x!quant.uniform<i8:f32, 5.000000e+0>>) -> tensor<1x3xf32>
      // Add is there to prevent from dot_generals from being DCEed.
      %6 = stablehlo.add %1, %5 : tensor<1x3xf32>
      return %6 : tensor<1x3xf32>
    }

    // Callee of non-quantized op.
    func.func private @composite_dot_general_fn_1(%arg0: tensor<1x2xf32>, %arg1: tensor<2x3xf32>) -> tensor<1x3xf32> {
      %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
      return %0 : tensor<1x3xf32>
    }

    // Callee of quantized op.
    func.func private @quantized_dot_general_fn_2(%arg0: tensor<1x2x!quant.uniform<i8:f32, 4.000000e+0>>, %arg1: tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {1.000000e+0,2.000000e+0,3.000000e+0}>>) -> tensor<1x3x!quant.uniform<i8:f32, 5.000000e+0>> {
      %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<1x2x!quant.uniform<i8:f32, 4.000000e+0>>, tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {1.000000e+0,2.000000e+0,3.000000e+0}>>) -> tensor<1x3x!quant.uniform<i32:f32:1, {6.000000e+0,7.000000e+0,8.000000e+0}>>
      %1 = stablehlo.uniform_quantize %0 : (tensor<1x3x!quant.uniform<i32:f32:1, {6.000000e+0,7.000000e+0,8.000000e+0}>>) -> tensor<1x3x!quant.uniform<i8:f32, 5.000000e+0>>
      return %1 : tensor<1x3x!quant.uniform<i8:f32, 5.000000e+0>>
    }
  )mlir";

  const OwningOpRef<ModuleOp> module_op =
      ParseModuleOpString(kQuantizedDotGeneralAndNonQuantizedDotGeneral);
  ASSERT_TRUE(module_op);

  const QuantizationReport report(*module_op);
  const QuantizationResults& results = report.GetQuantizationResults();
  ASSERT_THAT(results.results(), SizeIs(2));

  // Test that the quantized op is captured in `results`.
  const QuantizationResult& quantized_result = results.results(0);
  EXPECT_THAT(quantized_result.quantizable_unit().name(),
              StrEq("composite_dot_general_fn_2"));
  EXPECT_TRUE(quantized_result.method().has_static_range_ptq());

  // Test that the non-quantized op is captured in `results`.
  const QuantizationResult& non_quantized_result = results.results(1);
  EXPECT_THAT(non_quantized_result.quantizable_unit().name(),
              StrEq("composite_dot_general_fn_1"));
  EXPECT_TRUE(non_quantized_result.method().has_no_quantization());
}

TEST_F(QuantizationReportTest, ToString) {
  QuantizationResult result{};
  QuantizableUnit& quantizable_unit = *result.mutable_quantizable_unit();
  quantizable_unit.set_name("quantized_my_function");

  Method& method = *result.mutable_method();
  method.mutable_no_quantization();

  QuantizationReport report{};
  report.AddQuantizationResult(std::move(result));

  // Check that the report string is equivalent to the textproto representation
  // of the `QuantizationResults`.
  std::string result_str{};
  TextFormat::PrintToString(report.GetQuantizationResults(), &result_str);

  EXPECT_THAT(report.ToString(), HasSubstr("Quantization Report"));
  EXPECT_THAT(report.ToString(), HasSubstr(result_str));
  EXPECT_THAT(report.ToString(), HasSubstr("Quantization Report End"));
}

TEST_F(QuantizationReportTest, Save) {
  constexpr absl::string_view kQuantizedDotGeneral = R"mlir(
    func.func @main(%arg0: tensor<1x2xf32>) -> tensor<1x3xf32> {
      %0 = stablehlo.constant() {value = dense<127> : tensor<2x3xi8>} : () -> tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {1.000000e+0,2.000000e+0,3.000000e+0}>>
      %1 = stablehlo.uniform_quantize %arg0 : (tensor<1x2xf32>) -> tensor<1x2x!quant.uniform<i8:f32, 4.000000e+0>>
      %2 = call @quantized_dot_general_fn(%1, %0) {_quantization_method = "static_range_ptq { }"} : (tensor<1x2x!quant.uniform<i8:f32, 4.000000e+0>>, tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {1.000000e+0,2.000000e+0,3.000000e+0}>>) -> tensor<1x3x!quant.uniform<i8:f32, 5.000000e+0>>
      %3 = stablehlo.uniform_dequantize %2 : (tensor<1x3x!quant.uniform<i8:f32, 5.000000e+0>>) -> tensor<1x3xf32>
      return %3 : tensor<1x3xf32>
    }

    func.func private @quantized_dot_general_fn(%arg0: tensor<1x2x!quant.uniform<i8:f32, 4.000000e+0>>, %arg1: tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {1.000000e+0,2.000000e+0,3.000000e+0}>>) -> tensor<1x3x!quant.uniform<i8:f32, 5.000000e+0>> {
      %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<1x2x!quant.uniform<i8:f32, 4.000000e+0>>, tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {1.000000e+0,2.000000e+0,3.000000e+0}>>) -> tensor<1x3x!quant.uniform<i32:f32:1, {6.000000e+0,7.000000e+0,8.000000e+0}>>
      %1 = stablehlo.uniform_quantize %0 : (tensor<1x3x!quant.uniform<i32:f32:1, {6.000000e+0,7.000000e+0,8.000000e+0}>>) -> tensor<1x3x!quant.uniform<i8:f32, 5.000000e+0>>
      return %1 : tensor<1x3x!quant.uniform<i8:f32, 5.000000e+0>>
    }
  )mlir";

  const OwningOpRef<ModuleOp> module_op =
      ParseModuleOpString(kQuantizedDotGeneral);
  ASSERT_TRUE(module_op);

  const QuantizationReport report(*module_op);

  const std::string dst_file_path =
      absl::StrCat(TempDir(), "/quantization_report.txtpb");
  const absl::Status save_status = report.Save(dst_file_path);
  ASSERT_THAT(save_status, IsOk());

  const absl::StatusOr<std::string> file_data = ReadFileToString(dst_file_path);
  ASSERT_THAT(file_data, IsOk());

  // Test that the file data can be parsed as `QuantizationResults`.
  QuantizationResults results{};
  ASSERT_TRUE(TextFormat::ParseFromString(*file_data, &results));

  // Check that `results` reflects the information of the quantized units
  // properly.
  ASSERT_THAT(results.results(), SizeIs(1));
  EXPECT_THAT(results.results(0).quantizable_unit().name(),
              StrEq("composite_dot_general_fn"));
  EXPECT_TRUE(results.results(0).method().has_static_range_ptq());
}

}  // namespace
}  // namespace mlir::quant::stablehlo
