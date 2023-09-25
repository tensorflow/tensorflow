/* Copyright 2023 The StableHLO Authors. All Rights Reserved.

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
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "stablehlo/dialect/ChloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/bridge/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "xla/error_spec.h"
#include "xla/literal_util.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/tfrt_cpu_pjrt_client.h"
#include "xla/statusor.h"
#include "xla/tests/literal_test_util.h"

namespace mlir::quant::stablehlo {
namespace {

class ConvertTfQuantToMhloIntTest : public ::testing::Test {
 protected:
  void SetUp() override {
    DialectRegistry dialects;
    dialects.insert<TF::TensorFlowDialect, func::FuncDialect, chlo::ChloDialect,
                    mhlo::MhloDialect, quant::QuantizationDialect>();
    ctx_ = std::make_unique<MLIRContext>(dialects);

    // Create a CPU client with 1 device.
    TF_ASSERT_OK_AND_ASSIGN(
        pjrt_client_,
        xla::GetTfrtCpuClient(/*asynchronous=*/false, /*cpu_device_count=*/1));
    device_ = pjrt_client_->addressable_devices().front();
    CHECK(device_);
  }

  absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>> CompileProgram(
      absl::string_view program) {
    // Parse the program.
    auto module_op = parseSourceString<ModuleOp>(program, ctx_.get());
    CHECK(module_op);
    // Run the Convert TF Quant Types, TF Quant -> MHLO Quant and MHLO Quant ->
    // MHLO int passes.
    PassManager pm(module_op->getContext());
    pm.addNestedPass<func::FuncOp>(CreateConvertTFQuantTypesPass());
    pm.addNestedPass<func::FuncOp>(CreateConvertTFQuantOpsToMHLOPass());
    pm.addNestedPass<func::FuncOp>(
        stablehlo::createConvertMHLOQuantToIntPass(false));
    CHECK(succeeded(pm.run(module_op.get())));
    // Compile the program.
    return pjrt_client_->Compile(*module_op, xla::CompileOptions{});
  }

  absl::StatusOr<std::shared_ptr<xla::Literal>>
  ExecutePromgramAndReturnSingleResult(
      xla::PjRtLoadedExecutable* executable,
      absl::Span<const xla::Literal* const> arguments) {
    // Process and buffer arguments.
    std::vector<std::unique_ptr<xla::PjRtBuffer>> buffers;
    std::vector<xla::PjRtBuffer*> buffer_ptrs;
    buffers.reserve(arguments.size());
    for (const xla::Literal* argument : arguments) {
      TF_ASSIGN_OR_RETURN(
          auto buffer, pjrt_client_->BufferFromHostLiteral(*argument, device_));
      buffer_ptrs.push_back(buffer.get());
      buffers.push_back(std::move(buffer));
    }
    // Run the executable.
    TF_ASSIGN_OR_RETURN(auto result,
                        executable->Execute({buffer_ptrs}, /*options=*/{}));
    CHECK(result.size() == 1 && result[0].size() == 1);
    return result[0][0]->ToLiteralSync();
  }

  std::unique_ptr<MLIRContext> ctx_;
  std::unique_ptr<xla::PjRtClient> pjrt_client_;
  xla::PjRtDevice* device_;
};

TEST_F(ConvertTfQuantToMhloIntTest, UniformQuantizeAndDequantize) {
  constexpr absl::string_view kProgram = R"mlir(
func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %scale = "tf.Const"() { value = dense<10.0> : tensor<f32> } : ()
    -> tensor<f32>
  %zp = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>
  %0 = "tf.UniformQuantize"(%arg0, %scale, %zp) {
    quantization_axis = -1 : i64,
    quantization_min_val = -128 : i64,
    quantization_max_val = 127 : i64
  } : (tensor<4xf32>, tensor<f32>, tensor<i32>) -> tensor<4x!tf_type.qint8>
  %1 = "tf.UniformDequantize"(%0, %scale, %zp) {
    quantization_axis = -1 : i64,
    quantization_min_val = -128 : i64,
    quantization_max_val = 127 : i64
  } : (tensor<4x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
})mlir";
  TF_ASSERT_OK_AND_ASSIGN(auto executable, this->CompileProgram(kProgram));

  auto arg0 =
      xla::LiteralUtil::CreateR1<float>({100.0f, 20000.0f, -2409.0f, -25.1f});
  TF_ASSERT_OK_AND_ASSIGN(
      auto result_literal,
      this->ExecutePromgramAndReturnSingleResult(executable.get(), {&arg0}));
  xla::LiteralTestUtil::ExpectR1Near<float>({100.0f, 1240.0f, -1310.0f, -30.0f},
                                            *result_literal,
                                            xla::ErrorSpec(0.001f));
}

TEST_F(ConvertTfQuantToMhloIntTest, UniformQuantizeConvolution) {
  constexpr absl::string_view kProgram = R"mlir(
func.func @main(%input: tensor<1x2x2x1xf32>, %filter: tensor<2x1x1x1xf32>) -> tensor<1x2x2x1xf32> {
    %input_scale = "tf.Const"() { value = dense<7.3> : tensor<f32> } : ()
    -> tensor<f32>
    %input_zp = "tf.Const"() { value = dense<-45> : tensor<i32> } : () -> tensor<i32>
    %filter_scale = "tf.Const"() { value = dense<0.047> : tensor<f32> } : ()
    -> tensor<f32>
    %filter_zp = "tf.Const"() { value = dense<-86> : tensor<i32> } : () -> tensor<i32>
    %accum_scale = "tf.Const"() { value = dense<0.343> : tensor<f32> } : ()
    -> tensor<f32>
    %accum_zp = "tf.Const"() { value = dense<0> : tensor<i32> } : () -> tensor<i32>
    %quant_input = "tf.UniformQuantize"(%input, %input_scale, %input_zp) {Tin = "tfdtype$DT_FLOAT", Tout = "tfdtype$DT_QINT8", attr_map = "", quantization_axis = -1 : i64, quantization_max_val = 127 : i64, quantization_min_val = -128 : i64} : (tensor<1x2x2x1xf32>, tensor<f32>, tensor<i32>) -> tensor<1x2x2x1x!tf_type.qint8>
    %quant_filter = "tf.UniformQuantize"(%filter, %filter_scale, %filter_zp) {Tin = "tfdtype$DT_FLOAT", Tout = "tfdtype$DT_QINT8", attr_map = "", quantization_axis = -1 : i64, quantization_max_val = 127 : i64, quantization_min_val = -128 : i64} : (tensor<2x1x1x1xf32>, tensor<f32>, tensor<i32>) -> tensor<2x1x1x1x!tf_type.qint8>
    %0 = "tf.UniformQuantizedConvolution"(%quant_input, %quant_filter, %input_scale, %input_zp, %filter_scale, %filter_zp, %accum_scale, %accum_zp) {Tin = "tfdtype$DT_QINT8", Tout = "tfdtype$DT_QINT32", attr_map = "", batch_group_count = 1 : i64, dimension_numbers = "\10\03\1A\02\01\02 \02(\032\02\00\01@\03J\02\01\02", explicit_padding = [], feature_group_count = 1 : i64, lhs_dilation = [1, 1], lhs_quantization_axis = -1 : i64, lhs_quantization_max_val = 127 : i64, lhs_quantization_min_val = -128 : i64, output_quantization_axis = -1 : i64, output_quantization_max_val = 2147483647 : i64, output_quantization_min_val = -2147483648 : i64, padding = "SAME", rhs_dilation = [1, 1], rhs_quantization_axis = -1 : i64, rhs_quantization_max_val = 127 : i64, rhs_quantization_min_val = -128 : i64, window_strides = [1, 1]} : (tensor<1x2x2x1x!tf_type.qint8>, tensor<2x1x1x1x!tf_type.qint8>, tensor<f32>, tensor<i32>, tensor<f32>, tensor<i32>, tensor<f32>, tensor<i32>) -> tensor<1x2x2x1x!tf_type.qint32>
    %output = "tf.UniformDequantize"(%0, %accum_scale, %accum_zp) {quantization_axis = -1 : i64, quantization_min_val = -128 : i64, quantization_max_val = 127 : i64} : (tensor<1x2x2x1x!tf_type.qint32>, tensor<f32>, tensor<i32>) -> tensor<1x2x2x1xf32>
    return %output : tensor<1x2x2x1xf32>
})mlir";

  TF_ASSERT_OK_AND_ASSIGN(auto executable, this->CompileProgram(kProgram));

  auto input = xla::LiteralUtil::CreateR4<float>(
      {{{{14.f}, {-100.f}}, {{-600.f}, {1250.f}}}});
  auto filter = xla::LiteralUtil::CreateR4<float>({{{{10.f}}}, {{{-2.f}}}});

  TF_ASSERT_OK_AND_ASSIGN(auto result_literal,
                          this->ExecutePromgramAndReturnSingleResult(
                              executable.get(), {&input, &filter}));
  xla::LiteralTestUtil::ExpectR4Near<float>(
      {{{{1340.f}, {-3500.f}}, {{-6000.f}, {12500.f}}}}, *result_literal,
      xla::ErrorSpec(20.f));
}

TEST_F(ConvertTfQuantToMhloIntTest, UniformQuantizeDot) {
  constexpr absl::string_view kProgram = R"mlir(
func.func @main(%input: tensor<1x2xf32>, %filter: tensor<2x3xf32>) -> tensor<1x3xf32> {
    %input_scale = "tf.Const"() { value = dense<0.588> : tensor<f32> } : ()
    -> tensor<f32>
    %input_zp = "tf.Const"() { value = dense<42> : tensor<i32> } : () -> tensor<i32>
    %filter_scale = "tf.Const"() { value = dense<0.0235> : tensor<f32> } : ()
    -> tensor<f32>
    %filter_zp = "tf.Const"() { value = dense<0> : tensor<i32> } : () -> tensor<i32>
    %accum_scale = "tf.Const"() { value = dense<0.0138> : tensor<f32> } : ()
    -> tensor<f32>
    %accum_zp = "tf.Const"() { value = dense<0> : tensor<i32> } : () -> tensor<i32>
    %quant_input = "tf.UniformQuantize"(%input, %input_scale, %input_zp) {Tin = "tfdtype$DT_FLOAT", Tout = "tfdtype$DT_QINT8", attr_map = "", quantization_axis = -1 : i64, quantization_max_val = 127 : i64, quantization_min_val = -128 : i64} : (tensor<1x2xf32>, tensor<f32>, tensor<i32>) -> tensor<1x2x!tf_type.qint8>
    %quant_filter = "tf.UniformQuantize"(%filter, %filter_scale, %filter_zp) {Tin = "tfdtype$DT_FLOAT", Tout = "tfdtype$DT_QINT8", attr_map = "", quantization_axis = -1 : i64, quantization_max_val = 127 : i64, quantization_min_val = -128 : i64} : (tensor<2x3xf32>, tensor<f32>, tensor<i32>) -> tensor<2x3x!tf_type.qint8>
    %0 = "tf.UniformQuantizedDot"(%quant_input, %quant_filter, %input_scale, %input_zp, %filter_scale, %filter_zp, %accum_scale, %accum_zp) {Tin = "tfdtype$DT_QINT8", Tout = "tfdtype$DT_QINT32", attr_map = "", device = "", lhs_quantization_axis = -1 : i64, lhs_quantization_max_val = 127 : i64, lhs_quantization_min_val = -128 : i64, output_quantization_axis = -1 : i64, output_quantization_max_val = 2147483647 : i64, output_quantization_min_val = -2147483648 : i64, rhs_quantization_axis = -1 : i64, rhs_quantization_max_val = 127 : i64, rhs_quantization_min_val = -128 : i64} : (tensor<1x2x!tf_type.qint8>, tensor<2x3x!tf_type.qint8>, tensor<f32>, tensor<i32>, tensor<f32>, tensor<i32>, tensor<f32>, tensor<i32>) -> tensor<1x3x!tf_type.qint32>
    %output = "tf.UniformDequantize"(%0, %accum_scale, %accum_zp) {quantization_axis = -1 : i64, quantization_min_val = -128 : i64, quantization_max_val = 127 : i64} : (tensor<1x3x!tf_type.qint32>, tensor<f32>, tensor<i32>) -> tensor<1x3xf32>
    return %output : tensor<1x3xf32>
})mlir";

  TF_ASSERT_OK_AND_ASSIGN(auto executable, this->CompileProgram(kProgram));

  auto input = xla::LiteralUtil::CreateR2<float>({{50.f, -100.f}});
  auto filter =
      xla::LiteralUtil::CreateR2<float>({{1.f, 2.f, 3.f}, {-1.f, -3.f, 1.f}});

  TF_ASSERT_OK_AND_ASSIGN(auto result_literal,
                          this->ExecutePromgramAndReturnSingleResult(
                              executable.get(), {&input, &filter}));
  xla::LiteralTestUtil::ExpectR2Near<float>(
      {{150.f, 400.f, 50.f}}, *result_literal, xla::ErrorSpec(2.f));
}

}  // namespace
}  // namespace mlir::quant::stablehlo
