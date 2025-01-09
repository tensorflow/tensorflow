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

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/Quant.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "stablehlo/dialect/ChloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/bridge/passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/cc/constant_fold.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "xla/error_spec.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_client_options.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/literal_test_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace mlir::quant::stablehlo {
namespace {

using ::testing::Test;

class ConvertTfQuantToMhloIntTest : public Test {
 protected:
  void SetUp() override {
    DialectRegistry dialects;
    dialects.insert<TF::TensorFlowDialect, func::FuncDialect, chlo::ChloDialect,
                    mhlo::MhloDialect, quant::QuantDialect>();
    ctx_ = std::make_unique<MLIRContext>(dialects);
    ctx_->loadAllAvailableDialects();

    // Create a CPU client with 1 device.
    xla::CpuClientOptions options;
    options.asynchronous = false;
    options.cpu_device_count = 1;
    TF_ASSERT_OK_AND_ASSIGN(pjrt_client_, xla::GetXlaPjrtCpuClient(options));
    device_ = pjrt_client_->addressable_devices().front();
    CHECK(device_);
  }

  absl::StatusOr<OwningOpRef<ModuleOp>> ReplaceFuncArgsByConstant(
      absl::string_view program,
      absl::Span<const xla::Literal* const> arguments,
      bool use_mhlo_const = false) {
    auto module_op = parseSourceString<ModuleOp>(program, ctx_.get());
    CHECK(module_op);
    auto func_op = llvm::dyn_cast<func::FuncOp>(
        *module_op->getBodyRegion().getOps().begin());
    if (!func_op) {
      return absl::InternalError("Input MLIR must have only 1 func");
    }
    if (arguments.size() != func_op.getNumArguments()) {
      return absl::InternalError("Input argument has wrong size");
    }

    // Convert input xla::Literal arguments to constants, this allows using
    // constant folding to evaluate function return value.
    mlir::OpBuilder builder(ctx_.get());
    for (int i = 0; i < arguments.size(); ++i) {
      const xla::Literal* const xla_literal = arguments[i];
      tensorflow::TensorShape shape;
      TF_ASSIGN_OR_RETURN(auto data_type,
                          tensorflow::EncodePrimitiveTypeAsDataType(
                              xla_literal->shape().element_type()));
      TF_RETURN_IF_ERROR(
          tensorflow::XLAShapeToTensorShape(xla_literal->shape(), &shape));
      tensorflow::Tensor tensor(data_type, shape);
      std::memcpy(static_cast<char*>(tensor.data()),
                  xla_literal->untyped_data(),
                  xla::ShapeUtil::ByteSizeOfPrimitiveType(
                      xla_literal->shape().element_type()) *
                      xla_literal->element_count());
      TF_ASSIGN_OR_RETURN(auto attrs,
                          tensorflow::ConvertTensor(tensor, &builder));
      builder.setInsertionPoint(
          &func_op.getFunctionBody().getBlocks().front().front());
      // Use mhlo.Constant when it is consumed by the lowering passes since they
      // can't lower tf.Const.
      Value cst;
      if (use_mhlo_const) {
        cst = builder.create<mhlo::ConstantOp>(func_op->getLoc(), attrs);
      } else {
        cst = builder.create<TF::ConstOp>(func_op->getLoc(), attrs);
      }
      func_op.getArgument(i).replaceAllUsesWith(cst);
    }
    return module_op;
  }

  // Evaluate return value of a function using TF kernel.
  // This assumes that the module op has only 1 function and it has TF ops only.
  absl::StatusOr<std::shared_ptr<xla::Literal>> EvaluateTfFunction(
      absl::string_view program,
      absl::Span<const xla::Literal* const> arguments) {
    TF_ASSIGN_OR_RETURN(auto module_op,
                        ReplaceFuncArgsByConstant(program, arguments));
    // Constant fold the func.Return op's producer op to evaluate the return
    // value. The evaluation will use TF kernels.
    // This assumes that func.Return is the last op in the function and it
    // returns only 1 value.
    auto& return_op = llvm::dyn_cast<func::FuncOp>(
                          *module_op->getBodyRegion().getOps().begin())
                          .getFunctionBody()
                          .getBlocks()
                          .back()
                          .back();
    if (!llvm::isa<func::ReturnOp>(return_op) ||
        return_op.getNumOperands() != 1) {
      return absl::InternalError(
          "Func must have ReturnOp as last op and must return 1 value");
    }
    auto def_op = return_op.getOperand(0).getDefiningOp();
    auto fold_results = ConstantFoldOpIfPossible(def_op);
    if (fold_results.size() != 1 ||
        !llvm::isa<TF::ConstOp>(fold_results[0].getDefiningOp())) {
      return absl::InternalError("Failed to evaluate TF ops");
    }

    // Convert output tensor back to xla::Literal.
    tensorflow::Tensor tensor;
    TF_RETURN_IF_ERROR(tensorflow::ConvertToTensor(
        llvm::dyn_cast<TF::ConstOp>(fold_results[0].getDefiningOp()).getValue(),
        &tensor));
    xla::Shape xla_shape;
    TF_RETURN_IF_ERROR(tensorflow::TensorShapeToXLAShape(
        tensor.dtype(), tensor.shape(), &xla_shape));
    xla::PjRtClient::HostBufferSemantics host_buffer_semantics =
        xla::PjRtClient::HostBufferSemantics::kImmutableUntilTransferCompletes;
    TF_ASSIGN_OR_RETURN(
        auto buffer,
        pjrt_client_->BufferFromHostBuffer(
            tensor.data(), xla_shape.element_type(), xla_shape.dimensions(),
            /*byte_strides=*/std::nullopt, host_buffer_semantics,
            /*on_done_with_host_buffer=*/nullptr, device_));
    return buffer->ToLiteralSync();
  }

  absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>> CompileProgram(
      absl::string_view program,
      absl::Span<const xla::Literal* const> arguments) {
    // Replace args by mhlo.constant since the lowering passes can't lower
    // tf.Const.
    TF_ASSIGN_OR_RETURN(
        auto module_op,
        ReplaceFuncArgsByConstant(program, arguments, /*use_mhlo_const=*/true));

    // Run the Convert TF Quant Types, TF Quant -> MHLO Quant and MHLO Quant ->
    // MHLO int passes.
    PassManager pm(module_op->getContext());
    pm.addNestedPass<func::FuncOp>(CreateConvertTFQuantTypesPass());
    AddQuantizationLoweringPasses(pm);
    CHECK(succeeded(pm.run(module_op.get())));
    // Compile the program.
    return pjrt_client_->Compile(*module_op, xla::CompileOptions{});
  }

  absl::StatusOr<std::shared_ptr<xla::Literal>>
  ExecuteProgramAndReturnSingleResult(
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

  void ExecuteAndCompareResultsWithTfKernel(
      absl::string_view program,
      absl::Span<const xla::Literal* const> arguments,
      std::optional<absl::string_view> tf_program = std::nullopt,
      double error_tolerance = 0.1) {
    // Expected result is calculated by evaluating using TF kernels. In some
    // cases, TF kernel behaves differently from lowered graph (e.g. Hybrid
    // ops). So we optionally use a different graph to calculate the expected
    // result.
    TF_ASSERT_OK_AND_ASSIGN(
        auto expected,
        this->EvaluateTfFunction(
            (tf_program.has_value() ? *tf_program : program), arguments));

    TF_ASSERT_OK_AND_ASSIGN(auto executable,
                            this->CompileProgram(program, arguments));
    TF_ASSERT_OK_AND_ASSIGN(
        auto result,
        this->ExecuteProgramAndReturnSingleResult(executable.get(), arguments));

    // Convert to double for comparison. This is needed for comparing integers
    // since it LiteralTestUtil asserts different integers even if it is within
    // error_spec.
    TF_ASSERT_OK_AND_ASSIGN(auto expected_double, expected->Convert(xla::F64))
    TF_ASSERT_OK_AND_ASSIGN(auto result_double, result->Convert(xla::F64))
    EXPECT_TRUE(xla::LiteralTestUtil::Near(expected_double, result_double,
                                           xla::ErrorSpec(error_tolerance)));
  }

  absl::StatusOr<xla::Literal> CreateRandomF32Literal(
      absl::Span<const int64_t> dims, float min = -100, float max = 100) {
    TF_ASSIGN_OR_RETURN(auto shape,
                        xla::ShapeUtil::MakeValidatedShape(xla::F32, dims));
    return xla::LiteralUtil::CreateLiteralWithGenerator<xla::F32, float>(
        shape, [this, min, max](absl::Span<const int64_t> dims) -> float {
          return absl::Uniform(bitgen_, min, max);
        });
  }

  absl::StatusOr<xla::Literal> CreateRandomI8Literal(
      absl::Span<const int64_t> dims, int8_t min = -128, int8_t max = 127) {
    TF_ASSIGN_OR_RETURN(auto shape,
                        xla::ShapeUtil::MakeValidatedShape(xla::S8, dims));
    return xla::LiteralUtil::CreateLiteralWithGenerator<xla::S8, int8_t>(
        shape, [this, min, max](absl::Span<const int64_t> dims) -> int8_t {
          return absl::Uniform(bitgen_, min, max);
        });
  }

  absl::StatusOr<xla::Literal> CreateRandomI32Literal(
      absl::Span<const int64_t> dims, int32_t min = -128, int32_t max = 127) {
    TF_ASSIGN_OR_RETURN(auto shape,
                        xla::ShapeUtil::MakeValidatedShape(xla::S32, dims));
    return xla::LiteralUtil::CreateLiteralWithGenerator<xla::S32, int32_t>(
        shape, [this, min, max](absl::Span<const int64_t> dims) -> int32_t {
          return absl::Uniform(bitgen_, min, max);
        });
  }

  std::unique_ptr<MLIRContext> ctx_;
  std::unique_ptr<xla::PjRtClient> pjrt_client_;
  xla::PjRtDevice* device_;
  absl::BitGen bitgen_;
};

TEST_F(ConvertTfQuantToMhloIntTest, UniformQuantizeAndDequantizeToValidGraph) {
  constexpr absl::string_view kProgram = R"mlir(
func.func @main(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %scale = "tf.Const"() { value = dense<0.347> : tensor<f32> } : () -> tensor<f32>
  %zp = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>
  %0 = "tf.UniformQuantize"(%arg0, %scale, %zp) {
    quantization_axis = -1 : i64,
    quantization_min_val = -128 : i64,
    quantization_max_val = 127 : i64
  } : (tensor<10xf32>, tensor<f32>, tensor<i32>) -> tensor<10x!tf_type.qint8>
  %1 = "tf.UniformDequantize"(%0, %scale, %zp) {
    quantization_axis = -1 : i64,
    quantization_min_val = -128 : i64,
    quantization_max_val = 127 : i64
  } : (tensor<10x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<10xf32>
  return %1 : tensor<10xf32>
})mlir";
  TF_ASSERT_OK_AND_ASSIGN(auto arg0, CreateRandomF32Literal({10}));
  // error_tolerance is set to be slightly > scale because different rounding
  // implementations for UniformQuantize in TF kernel and the lowering passes
  // may cause +/-1 differences.
  ExecuteAndCompareResultsWithTfKernel(
      kProgram, {&arg0}, /*tf_program=*/std::nullopt, /*error_tolerance=*/0.35);
}

TEST_F(ConvertTfQuantToMhloIntTest, UniformQuantizePerChannelToValidGraph) {
  constexpr absl::string_view kProgram = R"mlir(
func.func @main(
    %arg0: tensor<10x10xf32>, %scale: tensor<10xf32>, %zp: tensor<10xi32>
  ) -> tensor<10x10xi8> {
  %0 = "tf.UniformQuantize"(%arg0, %scale, %zp) {
    quantization_axis = 1 : i64,
    quantization_min_val = -128 : i64,
    quantization_max_val = 127 : i64
  } : (tensor<10x10xf32>, tensor<10xf32>, tensor<10xi32>) -> tensor<10x10x!tf_type.qint8>
  %1 = "tf.Cast"(%0) {} : (tensor<10x10x!tf_type.qint8>) -> tensor<10x10xi8>
  return %1 : tensor<10x10xi8>
})mlir";
  TF_ASSERT_OK_AND_ASSIGN(auto arg0, CreateRandomF32Literal({10, 10}));
  TF_ASSERT_OK_AND_ASSIGN(
      auto scale, CreateRandomF32Literal({10}, /*min=*/0.0001, /*max=*/2));
  TF_ASSERT_OK_AND_ASSIGN(auto zp, CreateRandomI32Literal({10}));
  // Different rounding implementations for UniformQuantize in TF kernel and the
  // lowering passes may cause +/-1 differences.
  ExecuteAndCompareResultsWithTfKernel(kProgram, {&arg0, &scale, &zp},
                                       /*tf_program=*/std::nullopt,
                                       /*error_tolerance=*/1.0);
}

TEST_F(ConvertTfQuantToMhloIntTest, UniformDequantizePerChannelToValidGraph) {
  constexpr absl::string_view kProgram = R"mlir(
func.func @main(
    %arg0: tensor<10x10xi8>, %scale: tensor<10xf32>, %zp: tensor<10xi32>
  ) -> tensor<10x10xf32> {
  %0 = "tf.Cast"(%arg0) {} : (tensor<10x10xi8>) -> tensor<10x10x!tf_type.qint8>
  %1 = "tf.UniformDequantize"(%0, %scale, %zp) {
    quantization_axis = 1 : i64,
    quantization_min_val = -128 : i64,
    quantization_max_val = 127 : i64
  } : (tensor<10x10x!tf_type.qint8>, tensor<10xf32>, tensor<10xi32>) -> tensor<10x10xf32>
  return %1 : tensor<10x10xf32>
})mlir";
  TF_ASSERT_OK_AND_ASSIGN(auto arg0, CreateRandomI8Literal({10, 10}));
  TF_ASSERT_OK_AND_ASSIGN(
      auto scale, CreateRandomF32Literal({10}, /*min=*/0.0001, /*max=*/2));
  TF_ASSERT_OK_AND_ASSIGN(auto zp, CreateRandomI32Literal({10}));
  ExecuteAndCompareResultsWithTfKernel(kProgram, {&arg0, &scale, &zp});
}

TEST_F(ConvertTfQuantToMhloIntTest, UniformQuantizeConvolutionToValidGraph) {
  constexpr absl::string_view kProgram = R"mlir(
func.func @main(%input: tensor<1x9x9x9xi8>, %filter: tensor<3x3x9x10xi8>) -> tensor<1x9x9x10xi32> {
  %input_scale = "tf.Const"() { value = dense<2.0> : tensor<f32> } : () -> tensor<f32>
  %input_zp = "tf.Const"() { value = dense<-10> : tensor<i32> } : () -> tensor<i32>
  %filter_scale = "tf.Const"() { value = dense<0.5> : tensor<f32> } : () -> tensor<f32>
  %filter_zp = "tf.Const"() { value = dense<0> : tensor<i32> } : () -> tensor<i32>
  %accum_scale = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %accum_zp = "tf.Const"() { value = dense<0> : tensor<i32> } : () -> tensor<i32>
  %quant_input = "tf.Cast"(%input) {} : (tensor<1x9x9x9xi8>) ->
    tensor<1x9x9x9x!tf_type.qint8>
  %quant_filter = "tf.Cast"(%filter) {} : (tensor<3x3x9x10xi8>) ->
    tensor<3x3x9x10x!tf_type.qint8>
  %0 = "tf.UniformQuantizedConvolution"(
    %quant_input, %quant_filter, %input_scale, %input_zp,
    %filter_scale, %filter_zp, %accum_scale, %accum_zp
  ) {
    Tin = "tfdtype$DT_QINT8", Tout = "tfdtype$DT_QINT32",
    attr_map = "", batch_group_count = 1 : i64,
    dimension_numbers = "\10\03\1A\02\01\02 \02(\032\02\00\01@\03J\02\01\02",
    explicit_padding = [], feature_group_count = 1 : i64, lhs_dilation = [1, 1],
    lhs_quantization_axis = -1 : i64, lhs_quantization_max_val = 127 : i64,
    lhs_quantization_min_val = -128 : i64, output_quantization_axis = -1 : i64,
    output_quantization_max_val = 2147483647 : i64,
    output_quantization_min_val = -2147483648 : i64, padding = "SAME",
    rhs_dilation = [1, 1], rhs_quantization_axis = -1 : i64,
    rhs_quantization_max_val = 127 : i64, rhs_quantization_min_val = -128 : i64,
    window_strides = [1, 1]
  } : (tensor<1x9x9x9x!tf_type.qint8>, tensor<3x3x9x10x!tf_type.qint8>,
    tensor<f32>, tensor<i32>, tensor<f32>, tensor<i32>, tensor<f32>, tensor<i32>
  ) -> tensor<1x9x9x10x!tf_type.qint32>
  %output = "tf.Cast"(%0) {} : (tensor<1x9x9x10x!tf_type.qint32>) -> tensor<1x9x9x10xi32>
  return %output : tensor<1x9x9x10xi32>
})mlir";
  TF_ASSERT_OK_AND_ASSIGN(auto input, CreateRandomI8Literal({1, 9, 9, 9}));
  TF_ASSERT_OK_AND_ASSIGN(auto filter, CreateRandomI8Literal({3, 3, 9, 10}));
  ExecuteAndCompareResultsWithTfKernel(kProgram, {&input, &filter});
}

TEST_F(ConvertTfQuantToMhloIntTest,
       UniformQuantizeConvolutionPerChannelToValidGraph) {
  constexpr absl::string_view kProgram = R"mlir(
func.func @main(
    %input: tensor<1x9x9x9xi8>, %filter: tensor<3x3x9x10xi8>, %scale: tensor<10xf32>
  ) -> tensor<1x9x9x10xi32> {
  %input_scale = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %input_zp = "tf.Const"() { value = dense<-10> : tensor<i32> } : () -> tensor<i32>
  %zp = "tf.Const"() { value = dense<0> : tensor<10xi32> } : () -> tensor<10xi32>
  %quant_input = "tf.Cast"(%input) {} : (tensor<1x9x9x9xi8>) ->
    tensor<1x9x9x9x!tf_type.qint8>
  %quant_filter = "tf.Cast"(%filter) {} : (tensor<3x3x9x10xi8>) ->
    tensor<3x3x9x10x!tf_type.qint8>
  %0 = "tf.UniformQuantizedConvolution"(
    %quant_input, %quant_filter, %input_scale, %input_zp, %scale, %zp, %scale, %zp
  ) {
    Tin = "tfdtype$DT_QINT8", Tout = "tfdtype$DT_QINT32",
    attr_map = "", batch_group_count = 1 : i64,
    dimension_numbers = "\10\03\1A\02\01\02 \02(\032\02\00\01@\03J\02\01\02",
    explicit_padding = [], feature_group_count = 1 : i64, lhs_dilation = [1, 1],
    lhs_quantization_axis = -1 : i64, lhs_quantization_max_val = 127 : i64,
    lhs_quantization_min_val = -128 : i64, output_quantization_axis = 3 : i64,
    output_quantization_max_val = 2147483647 : i64,
    output_quantization_min_val = -2147483648 : i64, padding = "SAME",
    rhs_dilation = [1, 1], rhs_quantization_axis = 3 : i64,
    rhs_quantization_max_val = 127 : i64, rhs_quantization_min_val = -128 : i64,
    window_strides = [1, 1]
  } : (tensor<1x9x9x9x!tf_type.qint8>, tensor<3x3x9x10x!tf_type.qint8>,
    tensor<f32>, tensor<i32>, tensor<10xf32>, tensor<10xi32>, tensor<10xf32>, tensor<10xi32>
  ) -> tensor<1x9x9x10x!tf_type.qint32>
  %output = "tf.Cast"(%0) {} : (tensor<1x9x9x10x!tf_type.qint32>) -> tensor<1x9x9x10xi32>
  return %output : tensor<1x9x9x10xi32>
})mlir";
  TF_ASSERT_OK_AND_ASSIGN(auto input, CreateRandomI8Literal({1, 9, 9, 9}));
  TF_ASSERT_OK_AND_ASSIGN(auto filter, CreateRandomI8Literal({3, 3, 9, 10}));
  TF_ASSERT_OK_AND_ASSIGN(
      auto scale, CreateRandomF32Literal({10}, /*min=*/0.0001, /*max=*/2));
  ExecuteAndCompareResultsWithTfKernel(kProgram, {&input, &filter, &scale});
}

TEST_F(ConvertTfQuantToMhloIntTest,
       UniformQuantizeConvolutionHybridToValidGraph) {
  constexpr absl::string_view kTfProgram = R"mlir(
func.func @main(%input: tensor<2x10x10x10xf32>, %filter: tensor<3x3x10x20xi8>) -> tensor<2x10x10x20xf32> {
  %filter_scale = "tf.Const"() { value = dense<0.047> : tensor<f32> } : () -> tensor<f32>
  %filter_zp = "tf.Const"() { value = dense<0> : tensor<i32> } : () -> tensor<i32>
  %quant_filter = "tf.Cast"(%filter) {} : (tensor<3x3x10x20xi8>) ->
    tensor<3x3x10x20x!tf_type.qint8>
  %filter_new = "tf.UniformDequantize"(%quant_filter, %filter_scale, %filter_zp) {
    quantization_axis = -1 : i64, quantization_min_val = -128 : i64,
    quantization_max_val = 127 : i64
  } : (
    tensor<3x3x10x20x!tf_type.qint8>, tensor<f32>, tensor<i32>
  ) -> tensor<3x3x10x20xf32>
  %0 = "tf.Conv2D"(%input, %filter_new) {
    Tin = "tfdtype$DT_FLOAT", Tout = "tfdtype$DT_FLOAT",
    attr_map = "", batch_group_count = 1 : i64,
    explicit_padding = [], feature_group_count = 1 : i64, lhs_dilation = [1, 1],
    padding = "SAME", rhs_dilation = [1, 1], strides = [1, 1, 1, 1]
  } : (tensor<2x10x10x10xf32>, tensor<3x3x10x20xf32>) -> tensor<2x10x10x20xf32>
  return %0 : tensor<2x10x10x20xf32>
})mlir";
  constexpr absl::string_view kProgram = R"mlir(
func.func @main(%input: tensor<2x10x10x10xf32>, %filter: tensor<3x3x10x20xi8>) -> tensor<2x10x10x20xf32> {
  %filter_scale = "tf.Const"() { value = dense<0.047> : tensor<f32> } : () -> tensor<f32>
  %filter_zp = "tf.Const"() { value = dense<0> : tensor<i32> } : () -> tensor<i32>
  %quant_filter = "tf.Cast"(%filter) {} : (tensor<3x3x10x20xi8>) -> tensor<3x3x10x20x!tf_type.qint8>
  %0 = "tf.UniformQuantizedConvolutionHybrid"(
    %input, %quant_filter, %filter_scale, %filter_zp
  ) {
    Tin = "tfdtype$DT_QINT8", Tout = "tfdtype$DT_FLOAT",
    attr_map = "", batch_group_count = 1 : i64,
    dimension_numbers = "\10\03\1A\02\01\02 \02(\032\02\00\01@\03J\02\01\02",
    explicit_padding = [], feature_group_count = 1 : i64, lhs_dilation = [1, 1],
    padding = "SAME", rhs_dilation = [1, 1], rhs_quantization_axis = -1 : i64,
    rhs_quantization_max_val = 127 : i64, rhs_quantization_min_val = -128 : i64,
    window_strides = [1, 1]
  } : (tensor<2x10x10x10xf32>, tensor<3x3x10x20x!tf_type.qint8>,
    tensor<f32>, tensor<i32>) -> tensor<2x10x10x20xf32>
  return %0 : tensor<2x10x10x20xf32>
})mlir";
  TF_ASSERT_OK_AND_ASSIGN(auto input, CreateRandomF32Literal({2, 10, 10, 10}));
  TF_ASSERT_OK_AND_ASSIGN(auto filter, CreateRandomI8Literal({3, 3, 10, 20}));
  // TF kernels for UniformQuantizedConvolutionHybrid does DRQ. But StableHLO
  // hybrid ops does weight-only. So we use a different TF graph for evaluating
  // expected weight-only quantized results.
  ExecuteAndCompareResultsWithTfKernel(kProgram, {&input, &filter}, kTfProgram);
}

TEST_F(ConvertTfQuantToMhloIntTest, UniformQuantizeDotToValidGraph) {
  constexpr absl::string_view kProgram = R"mlir(
func.func @main(%input: tensor<8x9xi8>, %filter: tensor<9x10xi8>) -> tensor<8x10xi32> {
  %input_scale = "tf.Const"() { value = dense<0.588> : tensor<f32> } : () -> tensor<f32>
  %input_zp = "tf.Const"() { value = dense<42> : tensor<i32> } : () -> tensor<i32>
  %filter_scale = "tf.Const"() { value = dense<0.0235> : tensor<f32> } : () -> tensor<f32>
  %filter_zp = "tf.Const"() { value = dense<0> : tensor<i32> } : () -> tensor<i32>
  %accum_scale = "tf.Const"() { value = dense<0.013818> : tensor<f32> } : () -> tensor<f32>
  %accum_zp = "tf.Const"() { value = dense<0> : tensor<i32> } : () -> tensor<i32>
  %quant_input = "tf.Cast"(%input) {} : (tensor<8x9xi8>) -> tensor<8x9x!tf_type.qint8>
  %quant_filter = "tf.Cast"(%filter) {} : (tensor<9x10xi8>) -> tensor<9x10x!tf_type.qint8>
  %0 = "tf.UniformQuantizedDot"(
    %quant_input, %quant_filter, %input_scale, %input_zp, %filter_scale,
    %filter_zp, %accum_scale, %accum_zp
  ) {
    Tin = "tfdtype$DT_QINT8", Tout = "tfdtype$DT_QINT32", attr_map = "",
    device = "", lhs_quantization_axis = -1 : i64,
    lhs_quantization_max_val = 127 : i64,
    lhs_quantization_min_val = -128 : i64,
    output_quantization_axis = -1 : i64,
    output_quantization_max_val = 2147483647 : i64,
    output_quantization_min_val = -2147483648 : i64,
    rhs_quantization_axis = -1 : i64,
    rhs_quantization_max_val = 127 : i64,
    rhs_quantization_min_val = -128 : i64
  } : (
    tensor<8x9x!tf_type.qint8>, tensor<9x10x!tf_type.qint8>, tensor<f32>,
    tensor<i32>, tensor<f32>, tensor<i32>, tensor<f32>, tensor<i32>
  ) -> tensor<8x10x!tf_type.qint32>
  %output = "tf.Cast"(%0) {} : (tensor<8x10x!tf_type.qint32>) -> tensor<8x10xi32>
  return %output : tensor<8x10xi32>
})mlir";
  TF_ASSERT_OK_AND_ASSIGN(auto input, CreateRandomI8Literal({8, 9}));
  TF_ASSERT_OK_AND_ASSIGN(auto filter, CreateRandomI8Literal({9, 10}));
  ExecuteAndCompareResultsWithTfKernel(kProgram, {&input, &filter});
}

TEST_F(ConvertTfQuantToMhloIntTest, UniformQuantizeDotHybridToValidGraph) {
  constexpr absl::string_view kTfProgram = R"mlir(
func.func @main(%input: tensor<8x9xf32>, %filter: tensor<9x10xi8>) -> tensor<8x10xf32> {
  %filter_scale = "tf.Const"() { value = dense<0.0235> : tensor<f32> } : () -> tensor<f32>
  %filter_zp = "tf.Const"() { value = dense<0> : tensor<i32> } : () -> tensor<i32>
  %quant_filter = "tf.Cast"(%filter) {} : (tensor<9x10xi8>) -> tensor<9x10x!tf_type.qint8>
  %filter_new = "tf.UniformDequantize"(%quant_filter, %filter_scale, %filter_zp) {
    quantization_axis = -1 : i64, quantization_min_val = -128 : i64,
    quantization_max_val = 127 : i64
  } : (tensor<9x10x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<9x10xf32>
  %0 = "tf.MatMul"(%input, %filter_new) {
  } : (tensor<8x9xf32>, tensor<9x10xf32>) -> tensor<8x10xf32>
  return %0 : tensor<8x10xf32>
})mlir";
  constexpr absl::string_view kProgram = R"mlir(
func.func @main(%input: tensor<8x9xf32>, %filter: tensor<9x10xi8>) -> tensor<8x10xf32> {
  %filter_scale = "tf.Const"() { value = dense<0.0235> : tensor<f32> } : ()
  -> tensor<f32>
  %filter_zp = "tf.Const"() { value = dense<0> : tensor<i32> } : () -> tensor<i32>
  %quant_filter = "tf.Cast"(%filter) {} : (tensor<9x10xi8>) -> tensor<9x10x!tf_type.qint8>
  %0 = "tf.UniformQuantizedDotHybrid"(
    %input, %quant_filter, %filter_scale, %filter_zp
  ) {
    Tin = "tfdtype$DT_QINT8", Tout = "tfdtype$DT_FLOAT", attr_map = "",
    device = "", rhs_quantization_axis = -1 : i64,
    rhs_quantization_max_val = 127 : i64, rhs_quantization_min_val = -128 : i64
  } : (tensor<8x9xf32>, tensor<9x10x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<8x10xf32>
  return %0 : tensor<8x10xf32>
})mlir";
  TF_ASSERT_OK_AND_ASSIGN(auto input, CreateRandomF32Literal({8, 9}));
  TF_ASSERT_OK_AND_ASSIGN(auto filter, CreateRandomI8Literal({9, 10}));
  // TF kernels for UniformQuantizedDotHybrid does DRQ. But StableHLO hybrid ops
  // does weight-only. So we use a different TF graph for evaluating expected
  // weight-only quantized results.
  ExecuteAndCompareResultsWithTfKernel(kProgram, {&input, &filter}, kTfProgram);
}

TEST_F(ConvertTfQuantToMhloIntTest, UniformRequantizeToValidGraph) {
  constexpr absl::string_view kProgram = R"mlir(
func.func @main(%input: tensor<10xi8>) -> tensor<10xi8> {
  %input_scale = "tf.Const"() { value = dense<0.2235> : tensor<f32> } : () -> tensor<f32>
  %input_zp = "tf.Const"() { value = dense<-2> : tensor<i32> } : () -> tensor<i32>
  %output_scale = "tf.Const"() { value = dense<0.11> : tensor<f32> } : () -> tensor<f32>
  %output_zp = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>
  %0 = "tf.Cast"(%input) {} : (tensor<10xi8>) -> tensor<10x!tf_type.qint8>
  %1 = "tf.UniformRequantize"(
    %0, %input_scale, %input_zp, %output_scale, %output_zp
  ) {
    Tin = "tfdtype$DT_QINT8", Tout = "tfdtype$DT_QINT8", attr_map = "",
    device = "", input_quantization_axis = -1,
    input_quantization_max_val = 127 : i64,
    input_quantization_min_val = -128 : i64,
    output_quantization_axis = -1 : i64,
    output_quantization_max_val = 127 : i64,
    output_quantization_min_val = -128 : i64
  } : (
    tensor<10x!tf_type.qint8>, tensor<f32>, tensor<i32>, tensor<f32>,
    tensor<i32>
  ) -> tensor<10x!tf_type.qint8>
  %2 = "tf.Cast"(%1) {} : (tensor<10x!tf_type.qint8>) -> tensor<10xi8>
  return %2 : tensor<10xi8>
})mlir";
  TF_ASSERT_OK_AND_ASSIGN(auto input, CreateRandomI8Literal({10}));
  ExecuteAndCompareResultsWithTfKernel(kProgram, {&input});
}

TEST_F(ConvertTfQuantToMhloIntTest, UniformRequantizePerChannelToValidGraph) {
  constexpr absl::string_view kProgram = R"mlir(
func.func @main(
    %input: tensor<10x10xi8>, %input_scale: tensor<10xf32>,
    %input_zp: tensor<10xi32>, %output_scale: tensor<10xf32>,
    %output_zp: tensor<10xi32>
  ) -> tensor<10x10xi8> {
  %0 = "tf.Cast"(%input) {} : (tensor<10x10xi8>) -> tensor<10x10x!tf_type.qint8>
  %1 = "tf.UniformRequantize"(
    %0, %input_scale, %input_zp, %output_scale, %output_zp
  ) {
    Tin = "tfdtype$DT_QINT8", Tout = "tfdtype$DT_QINT8", attr_map = "",
    device = "", input_quantization_axis = 1,
    input_quantization_max_val = 127 : i64,
    input_quantization_min_val = -128 : i64,
    output_quantization_axis = 1 : i64,
    output_quantization_max_val = 127 : i64,
    output_quantization_min_val = -128 : i64
  } : (
    tensor<10x10x!tf_type.qint8>, tensor<10xf32>, tensor<10xi32>,
    tensor<10xf32>, tensor<10xi32>
  ) -> tensor<10x10x!tf_type.qint8>
  %2 = "tf.Cast"(%1) {} : (tensor<10x10x!tf_type.qint8>) -> tensor<10x10xi8>
  return %2 : tensor<10x10xi8>
})mlir";
  TF_ASSERT_OK_AND_ASSIGN(auto input, CreateRandomI8Literal({10, 10}));
  TF_ASSERT_OK_AND_ASSIGN(
      auto input_scale,
      CreateRandomF32Literal({10}, /*min=*/0.0001, /*max=*/2));
  TF_ASSERT_OK_AND_ASSIGN(auto input_zp, CreateRandomI32Literal({10}));
  TF_ASSERT_OK_AND_ASSIGN(
      auto output_scale,
      CreateRandomF32Literal({10}, /*min=*/0.0001, /*max=*/2));
  TF_ASSERT_OK_AND_ASSIGN(auto output_zp, CreateRandomI32Literal({10}));
  // error_tolerance is set to be 1 because different rounding implementations
  // in TF kernel and the lowering passes may cause +/-1 differences.
  ExecuteAndCompareResultsWithTfKernel(
      kProgram, {&input, &input_scale, &input_zp, &output_scale, &output_zp},
      /*tf_program=*/std::nullopt,
      /*error_tolerance=*/1.0);
}

TEST_F(ConvertTfQuantToMhloIntTest,
       UniformRequantizePerTensorToPerChannelToValidGraph) {
  constexpr absl::string_view kProgram = R"mlir(
func.func @main(
    %input: tensor<10x10xi8>, %input_scale: tensor<f32>, %input_zp: tensor<i32>,
    %output_scale: tensor<10xf32>, %output_zp: tensor<10xi32>
  ) -> tensor<10x10xi8> {
  %0 = "tf.Cast"(%input) {} : (tensor<10x10xi8>) -> tensor<10x10x!tf_type.qint8>
  %1 = "tf.UniformRequantize"(
    %0, %input_scale, %input_zp, %output_scale, %output_zp
  ) {
    Tin = "tfdtype$DT_QINT8", Tout = "tfdtype$DT_QINT8", attr_map = "",
    device = "", input_quantization_axis = -1,
    input_quantization_max_val = 127 : i64,
    input_quantization_min_val = -128 : i64,
    output_quantization_axis = 1 : i64,
    output_quantization_max_val = 127 : i64,
    output_quantization_min_val = -128 : i64
  } : (
    tensor<10x10x!tf_type.qint8>, tensor<f32>, tensor<i32>,
    tensor<10xf32>, tensor<10xi32>
  ) -> tensor<10x10x!tf_type.qint8>
  %2 = "tf.Cast"(%1) {} : (tensor<10x10x!tf_type.qint8>) -> tensor<10x10xi8>
  return %2 : tensor<10x10xi8>
})mlir";
  TF_ASSERT_OK_AND_ASSIGN(auto input, CreateRandomI8Literal({10, 10}));
  TF_ASSERT_OK_AND_ASSIGN(
      auto input_scale, CreateRandomF32Literal({}, /*min=*/0.0001, /*max=*/2));
  TF_ASSERT_OK_AND_ASSIGN(auto input_zp, CreateRandomI32Literal({}));
  TF_ASSERT_OK_AND_ASSIGN(
      auto output_scale,
      CreateRandomF32Literal({10}, /*min=*/0.0001, /*max=*/2));
  TF_ASSERT_OK_AND_ASSIGN(auto output_zp, CreateRandomI32Literal({10}));
  // error_tolerance is set to be 1 because different rounding implementations
  // in TF kernel and the lowering passes may cause +/-1 differences.
  ExecuteAndCompareResultsWithTfKernel(
      kProgram, {&input, &input_scale, &input_zp, &output_scale, &output_zp},
      /*tf_program=*/std::nullopt,
      /*error_tolerance=*/1.0);
}

TEST_F(ConvertTfQuantToMhloIntTest,
       UniformRequantizePerChannelToPerTensorToValidGraph) {
  constexpr absl::string_view kProgram = R"mlir(
func.func @main(
    %input: tensor<10x10xi8>, %input_scale: tensor<10xf32>,
    %input_zp: tensor<10xi32>, %output_scale: tensor<f32>, %output_zp: tensor<i32>
  ) -> tensor<10x10xi8> {
  %0 = "tf.Cast"(%input) {} : (tensor<10x10xi8>) -> tensor<10x10x!tf_type.qint8>
  %1 = "tf.UniformRequantize"(
    %0, %input_scale, %input_zp, %output_scale, %output_zp
  ) {
    Tin = "tfdtype$DT_QINT8", Tout = "tfdtype$DT_QINT8", attr_map = "",
    device = "", input_quantization_axis = 1,
    input_quantization_max_val = 127 : i64,
    input_quantization_min_val = -128 : i64,
    output_quantization_axis = -1 : i64,
    output_quantization_max_val = 127 : i64,
    output_quantization_min_val = -128 : i64
  } : (
    tensor<10x10x!tf_type.qint8>, tensor<10xf32>, tensor<10xi32>,
    tensor<f32>, tensor<i32>
  ) -> tensor<10x10x!tf_type.qint8>
  %2 = "tf.Cast"(%1) {} : (tensor<10x10x!tf_type.qint8>) -> tensor<10x10xi8>
  return %2 : tensor<10x10xi8>
})mlir";
  TF_ASSERT_OK_AND_ASSIGN(auto input, CreateRandomI8Literal({10, 10}));
  TF_ASSERT_OK_AND_ASSIGN(
      auto input_scale,
      CreateRandomF32Literal({10}, /*min=*/0.0001, /*max=*/2));
  TF_ASSERT_OK_AND_ASSIGN(auto input_zp, CreateRandomI32Literal({10}));
  TF_ASSERT_OK_AND_ASSIGN(
      auto output_scale, CreateRandomF32Literal({}, /*min=*/0.0001, /*max=*/2));
  TF_ASSERT_OK_AND_ASSIGN(auto output_zp, CreateRandomI32Literal({}));
  // error_tolerance is set to be 1 because different rounding implementations
  // in TF kernel and the lowering passes may cause +/-1 differences.
  ExecuteAndCompareResultsWithTfKernel(
      kProgram, {&input, &input_scale, &input_zp, &output_scale, &output_zp},
      /*tf_program=*/std::nullopt,
      /*error_tolerance=*/1.0);
}

TEST_F(ConvertTfQuantToMhloIntTest, UniformQuantizeAddToValidGraph) {
  constexpr absl::string_view kProgram = R"mlir(
func.func @main(%lhs: tensor<10x10xi32>, %rhs: tensor<10x10xi32>) -> tensor<10x10xi32> {
  %lhs_scale = "tf.Const"() { value = dense<0.518> : tensor<f32> } : () -> tensor<f32>
  %lhs_zp = "tf.Const"() { value = dense<42> : tensor<i32> } : () -> tensor<i32>
  %rhs_scale = "tf.Const"() { value = dense<0.0239> : tensor<f32> } : () -> tensor<f32>
  %rhs_zp = "tf.Const"() { value = dense<0> : tensor<i32> } : () -> tensor<i32>
  %accum_scale = "tf.Const"() { value = dense<0.013> : tensor<f32> } : () -> tensor<f32>
  %accum_zp = "tf.Const"() { value = dense<0> : tensor<i32> } : () -> tensor<i32>
  %quant_lhs = "tf.Cast"(%lhs) {} : (tensor<10x10xi32>) -> tensor<10x10x!tf_type.qint32>
  %quant_rhs = "tf.Cast"(%rhs) {} : (tensor<10x10xi32>) -> tensor<10x10x!tf_type.qint32>
  %0 = "tf.UniformQuantizedAdd"(
    %quant_lhs, %quant_rhs, %lhs_scale, %lhs_zp, %rhs_scale,
    %rhs_zp, %accum_scale, %accum_zp
  ) {
    Tin = "tfdtype$DT_QINT32", Tout = "tfdtype$DT_QINT32", attr_map = "",
    device = "", lhs_quantization_axis = -1 : i64,
    lhs_quantization_max_val = 2147483647 : i64,
    lhs_quantization_min_val = -2147483648 : i64,
    output_quantization_axis = -1 : i64,
    output_quantization_max_val = 2147483647 : i64,
    output_quantization_min_val = -2147483648 : i64,
    rhs_quantization_axis = -1 : i64,
    rhs_quantization_max_val = 2147483647 : i64,
    rhs_quantization_min_val = -2147483648 : i64
  } : (
    tensor<10x10x!tf_type.qint32>, tensor<10x10x!tf_type.qint32>, tensor<f32>,
    tensor<i32>, tensor<f32>, tensor<i32>, tensor<f32>, tensor<i32>
  ) -> tensor<10x10x!tf_type.qint32>
  %1 = "tf.Cast"(%0) {} : (tensor<10x10x!tf_type.qint32>) ->  tensor<10x10xi32>
  return %1 : tensor<10x10xi32>
})mlir";
  TF_ASSERT_OK_AND_ASSIGN(auto lhs, CreateRandomI32Literal({10, 10}));
  TF_ASSERT_OK_AND_ASSIGN(auto rhs, CreateRandomI32Literal({10, 10}));
  // error_tolerance is set to be 1 because different rounding implementations
  // in TF kernel and the lowering passes may cause +/-1 differences.
  ExecuteAndCompareResultsWithTfKernel(kProgram, {&lhs, &rhs},
                                       /*tf_program=*/std::nullopt,
                                       /*error_tolerance=*/1.0);
}

}  // namespace
}  // namespace mlir::quant::stablehlo
