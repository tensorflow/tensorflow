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

#include "tensorflow/compiler/mlir/lite/quantization/common/quantization_lib/quantization_driver.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/Quant.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/quantization/common/quantization_lib/quantization_utils.h"
#include "tensorflow/compiler/mlir/lite/quantization/common/quantization_lib/tfl_quantization_driver.h"
#include "tensorflow/compiler/mlir/lite/quantization/common/test_base.h"
#include "tensorflow/compiler/mlir/lite/quantization/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/quantization/common/attrs_and_constraints.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/quantization/common/func.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir::TFL {
namespace {

using ApplyQuantizationParamsPropagationTest =
    mlir::quant::QuantizationTestBase;
using ::testing::IsEmpty;
using ::testing::Not;

constexpr absl::string_view kModuleTFLite = R"mlir(
  module {
    func.func @main(%arg0: tensor<1x4x4x3xf32>) -> tensor<1x4x4x3xf32> attributes {_from_xla_call_module} {
      %cst_0 = arith.constant dense<1.0> : tensor<3x1x1x3xf32>
      %cst_1 = arith.constant dense<2.0> : tensor<3xf32>
      %0 = "tf.XlaCallModule"(%arg0, %cst_0, %cst_1) <{Sout = [#tf_type.shape<1x4x4x3>], module = "", version = 9 : i64}> {_entry_function = @composite_fn_1, _stablehlo_version = "1.0.0", _original_entry_function = "composite_fn_1", _tfl_quant_trait = "fully_quantizable"} : (tensor<1x4x4x3xf32>, tensor<3x1x1x3xf32>, tensor<3xf32>) -> tensor<1x4x4x3xf32>
      %1 = "tf.XlaCallModule"(%0, %cst_0, %cst_1) <{Sout = [#tf_type.shape<1x4x4x3>], module = "", version = 9 : i64}> {_entry_function = @composite_fn_2, _stablehlo_version = "1.0.0", _original_entry_function = "composite_fn_2", _tfl_quant_trait = "fully_quantizable"} : (tensor<1x4x4x3xf32>, tensor<3x1x1x3xf32>, tensor<3xf32>) -> tensor<1x4x4x3xf32>
      return %1 : tensor<1x4x4x3xf32>
    }
    func.func private @composite_fn_1(%arg0: tensor<1x4x4x3xf32>, %arg1: tensor<3x1x1x3xf32>, %arg2: tensor<3xf32>) -> tensor<1x4x4x3xf32> attributes {tf_quant.composite_function} {
      %0 = "tfl.conv_2d"(%arg0, %arg1, %arg2) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "RELU", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x4x4x3xf32>, tensor<3x1x1x3xf32>, tensor<3xf32>) -> tensor<1x4x4x3xf32>
      return %0 : tensor<1x4x4x3xf32>
    }
    func.func private @composite_fn_2(%arg0: tensor<1x4x4x3xf32>, %arg1: tensor<3x1x1x3xf32>, %arg2: tensor<3xf32>) -> tensor<1x4x4x3xf32> attributes {tf_quant.composite_function} {
      %0 = "tfl.conv_2d"(%arg0, %arg1, %arg2) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "RELU", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x4x4x3xf32>, tensor<3x1x1x3xf32>, tensor<3xf32>) -> tensor<1x4x4x3xf32>
      return %0 : tensor<1x4x4x3xf32>
    }
  }
)mlir";

// TODO: b/323478683 - Directly use types rather than creating a `unique_ptr`.
std::unique_ptr<OpQuantSpec> GetOpQuantSpec(
    const mlir::Operation* op,
    bool disable_per_channel_for_dense_layers = false) {
  auto spec = std::make_unique<OpQuantSpec>();
  spec->coeff_op_quant_dim[1] = 3;
  spec->biases_params[2] = {{0, 1}, GetUniformQuantizedTypeForBias};
  for (const auto& [key, value] : spec->coeff_op_quant_dim) {
    spec->quantizable_operands.insert(key);
  }
  return spec;
}

TEST_F(ApplyQuantizationParamsPropagationTest,
       ConstsUsedMultipleTimesAreDuplicated) {
  const OwningOpRef<ModuleOp> module_op_ref =
      mlir::quant::QuantizationTestBase::ParseModuleOpString(kModuleTFLite);
  func::FuncOp main_fn = mlir::quant::FindMainFuncOp(*module_op_ref);

  auto op_quant_spec_getter = [&](mlir::Operation* op) {
    return GetOpQuantSpec(op, /*disable_per_channel_for_dense_layers=*/false);
  };
  QuantizationDriver quantization_driver(
      main_fn, /*is_signed=*/true, /*bit_width=*/8,
      /*disable_per_channel=*/false, op_quant_spec_getter,
      GetDefaultQuantScaleSpec,
      /*infer_tensor_range=*/true, /*legacy_float_scale=*/false,
      /*is_qdq_conversion=*/false);

  quantization_driver.Initialize();

  int64_t num_constant_op = 0;
  main_fn.walk([&](arith::ConstantOp cst) { ++num_constant_op; });
  EXPECT_EQ(num_constant_op, 4);
}

TEST_F(ApplyQuantizationParamsPropagationTest,
       PropagateParamsCreatesQuantState) {
  const OwningOpRef<ModuleOp> module_op_ref =
      ParseModuleOpString(kModuleTFLite);
  func::FuncOp main_fn = mlir::quant::FindMainFuncOp(*module_op_ref);

  auto op_quant_spec_getter = [&](mlir::Operation* op) {
    return GetOpQuantSpec(op, /*disable_per_channel_for_dense_layers=*/false);
  };
  QuantizationDriver quantization_driver(
      main_fn, /*is_signed=*/true, /*bit_width=*/8,
      /*disable_per_channel=*/false, op_quant_spec_getter,
      GetDefaultQuantScaleSpec,
      /*infer_tensor_range=*/true, /*legacy_float_scale=*/false,
      /*is_qdq_conversion=*/false);

  quantization_driver.Initialize();
  ASSERT_TRUE(quantization_driver.PropagateParamsAndReturnIfChanged());
  EXPECT_THAT(quantization_driver.GetArgs(), Not(IsEmpty()));

  for (const auto& arg : quantization_driver.GetArgs()) {
    const QuantState& state = quantization_driver.GetArgQuantState(arg);
    EXPECT_TRUE(isa<quant::QuantizedType>(state.params));
  }
  for (const auto& result : quantization_driver.GetResultStates()) {
    mlir::Operation* op = result.first.first;
    const int res_index = result.first.second;
    const QuantState state =
        quantization_driver.GetResultQuantState(op, res_index);
    EXPECT_TRUE(isa<quant::QuantizedType>(state.params));
  }
}

TEST_F(ApplyQuantizationParamsPropagationTest, FinalizeInsertsQDQOps) {
  const OwningOpRef<ModuleOp> module_op_ref =
      ParseModuleOpString(kModuleTFLite);
  func::FuncOp main_fn = mlir::quant::FindMainFuncOp(*module_op_ref);

  auto op_quant_spec_getter = [&](mlir::Operation* op) {
    return GetOpQuantSpec(op, /*disable_per_channel_for_dense_layers=*/false);
  };
  ApplyQuantizationParamsPropagation(
      main_fn, /*is_signed=*/true, /*bit_width=*/8,
      /*disable_per_channel=*/false, op_quant_spec_getter,
      /*infer_tensor_ranges=*/true, /*legacy_float_scale=*/false,
      /*is_qdq_conversion=*/false);
  mlir::Operation* xla_call_module_op =
      quant::FindOperationOfType<TF::XlaCallModuleOp>(main_fn);
  mlir::Operation* filter_dcast_op =
      xla_call_module_op->getOperand(1).getDefiningOp();
  mlir::Operation* filter_qcast_op =
      filter_dcast_op->getOperand(0).getDefiningOp();
  ASSERT_NE(filter_qcast_op, nullptr);
  EXPECT_TRUE(isa<quantfork::QuantizeCastOp>(filter_qcast_op));
  EXPECT_TRUE(isa<quantfork::DequantizeCastOp>(filter_dcast_op));
  EXPECT_TRUE(isa<quant::UniformQuantizedPerAxisType>(
      mlir::cast<TensorType>(filter_qcast_op->getResult(0).getType())
          .getElementType()));
}

TEST_F(ApplyQuantizationParamsPropagationTest,
       ConcatWithMismatchedUint8OperandsRoundsHalfIntegerZeroPoint) {
  constexpr absl::string_view kConcatModule = R"mlir(
    module {
      func.func @main(%arg0: tensor<1x4xf32>, %arg1: tensor<1x4xf32>) -> tensor<1x8xf32> {
        %q0 = "quantfork.qcast"(%arg0) : (tensor<1x4xf32>) -> tensor<1x4x!quant.uniform<u8:f32, 0.0039215686274509803:255>>
        %dq0 = "quantfork.dcast"(%q0) : (tensor<1x4x!quant.uniform<u8:f32, 0.0039215686274509803:255>>) -> tensor<1x4xf32>
        %q1 = "quantfork.qcast"(%arg1) : (tensor<1x4xf32>) -> tensor<1x4x!quant.uniform<u8:f32, 0.0039215686274509803:0>>
        %dq1 = "quantfork.dcast"(%q1) : (tensor<1x4x!quant.uniform<u8:f32, 0.0039215686274509803:0>>) -> tensor<1x4xf32>
        %concat = "tfl.concatenation"(%dq0, %dq1) {axis = 1 : i32, fused_activation_function = "NONE"} : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x8xf32>
        return %concat : tensor<1x8xf32>
      }
    }
  )mlir";
  const OwningOpRef<ModuleOp> module_op_ref =
      ParseModuleOpString(kConcatModule);
  func::FuncOp main_fn = mlir::quant::FindMainFuncOp(*module_op_ref);

  auto op_quant_spec_getter = [&](mlir::Operation* op) {
    return std::make_unique<OpQuantSpec>();
  };
  temp::QuantizationDriver quantization_driver(
      main_fn, /*is_signed=*/false, /*bit_width=*/8,
      /*disable_per_channel=*/false, op_quant_spec_getter,
      GetDefaultQuantScaleSpec,
      /*infer_tensor_range=*/true, QDQConversionMode::kQDQNone,
      /*legacy_float_scale=*/false);

  quantization_driver.Initialize();
  ASSERT_TRUE(quantization_driver.PropagateParamsAndReturnIfChanged());

  mlir::Operation* concat_op = nullptr;
  main_fn.walk([&](mlir::Operation* op) {
    if (op->getName().getStringRef() == "tfl.concatenation") {
      concat_op = op;
    }
  });
  ASSERT_NE(concat_op, nullptr);
  const temp::QuantState& result_state =
      quantization_driver.GetResultQuantState(concat_op, 0);
  auto merged_qtype =
      mlir::dyn_cast_or_null<quant::UniformQuantizedType>(result_state.params);
  ASSERT_NE(merged_qtype, nullptr);
  // Operands with ranges [-1.0, 0.0] and [0.0, 1.0] merge to [-1.0, 1.0] with
  // scale = 2.0 / 255.0 and q_min - real_min / scale = 127.5, which rounds to
  // 128 rather than truncating to 127.
  EXPECT_NEAR(merged_qtype.getScale(), 2.0 / 255.0, 1e-12);
  EXPECT_EQ(merged_qtype.getZeroPoint(), 128);
}

TEST(DownCastScaleTest,
     DownCastScalePreservesUniformZeroPointForSymmetricRange) {
  MLIRContext context;
  context.loadDialect<quant::QuantDialect>();
  Builder builder(&context);

  // An artificial non-zero zero-point (zp = 5) on a [-127, 127] symmetric
  // storage range is used here to verify that DownCastScale preserves the
  // existing zero-point from the source type rather than resetting it to 0.
  auto uq_type = quant::UniformQuantizedType::get(
      quant::QuantizationFlags::Signed, builder.getIntegerType(8),
      builder.getF32Type(), /*scale=*/0.1, /*zeroPoint=*/5,
      /*storageTypeMin=*/-127, /*storageTypeMax=*/127);
  auto downcast =
      mlir::dyn_cast<quant::UniformQuantizedType>(mlir::TFL::DownCastScale(
          uq_type, /*min=*/-1.0, /*max=*/1.0, builder.getUnknownLoc()));
  ASSERT_NE(downcast, nullptr);
  EXPECT_EQ(downcast.getZeroPoint(), 5);
}

TEST(CalculateNewQuantizedTypeTest, RoundsHalfIntegerZeroPoint) {
  MLIRContext context;
  context.loadDialect<quant::QuantDialect>();
  Builder builder(&context);

  // Two signed int8 types with ranges [-1.0, 0.0] and [0.0, 1.0].
  // Mismatched operand qparams merge to [-1.0, 1.0], scale = 2.0 / 255.0.
  // q_min - (real_min / scale) = -128 - (-1.0 / (2.0 / 255.0)) = -0.5.
  // Truncation previously truncated to 0; std::llround rounds -0.5 to -1.
  const double scale = 1.0 / 255.0;
  auto qtype1 = quant::UniformQuantizedType::get(
      quant::QuantizationFlags::Signed, builder.getIntegerType(8),
      builder.getF32Type(), scale, /*zeroPoint=*/127,
      /*storageTypeMin=*/-128, /*storageTypeMax=*/127);
  auto qtype2 = quant::UniformQuantizedType::get(
      quant::QuantizationFlags::Signed, builder.getIntegerType(8),
      builder.getF32Type(), scale, /*zeroPoint=*/-128,
      /*storageTypeMin=*/-128, /*storageTypeMax=*/127);

  auto merged = mlir::dyn_cast<quant::UniformQuantizedType>(
      temp::internal::CalculateNewQuantizedType({qtype1, qtype2}));
  ASSERT_NE(merged, nullptr);
  EXPECT_EQ(merged.getZeroPoint(), -1);
}

TEST(CalculateNewQuantizedTypeTest, ClampsZeroPointToStorageRange) {
  MLIRContext context;
  context.loadDialect<quant::QuantDialect>();
  Builder builder(&context);

  // Using zeroPoint = 200 on [-128, 127] gives an all-negative range
  // [-3.28, -0.73] (exercising real_max = lowest() with real_max < 0) and an
  // unclamped zero-point of 200 that clamps to storage_max (127).
  auto qtype_high1 = quant::UniformQuantizedType::get(
      quant::QuantizationFlags::Signed, builder.getIntegerType(8),
      builder.getF32Type(), /*scale=*/0.01, /*zeroPoint=*/200,
      /*storageTypeMin=*/-128, /*storageTypeMax=*/127);
  auto qtype_high2 = quant::UniformQuantizedType::get(
      quant::QuantizationFlags::Signed, builder.getIntegerType(8),
      builder.getF32Type(), /*scale=*/0.01, /*zeroPoint=*/200,
      /*storageTypeMin=*/-128, /*storageTypeMax=*/127);

  auto merged_high = mlir::dyn_cast<quant::UniformQuantizedType>(
      temp::internal::CalculateNewQuantizedType({qtype_high1, qtype_high2}));
  ASSERT_NE(merged_high, nullptr);
  EXPECT_EQ(merged_high.getZeroPoint(), 127);

  // Symmetrically, zeroPoint = -200 gives an all-positive range [0.72, 3.27]
  // and an unclamped zero-point of -200 that clamps to storage_min (-128).
  auto qtype_low1 = quant::UniformQuantizedType::get(
      quant::QuantizationFlags::Signed, builder.getIntegerType(8),
      builder.getF32Type(), /*scale=*/0.01, /*zeroPoint=*/-200,
      /*storageTypeMin=*/-128, /*storageTypeMax=*/127);
  auto qtype_low2 = quant::UniformQuantizedType::get(
      quant::QuantizationFlags::Signed, builder.getIntegerType(8),
      builder.getF32Type(), /*scale=*/0.01, /*zeroPoint=*/-200,
      /*storageTypeMin=*/-128, /*storageTypeMax=*/127);

  auto merged_low = mlir::dyn_cast<quant::UniformQuantizedType>(
      temp::internal::CalculateNewQuantizedType({qtype_low1, qtype_low2}));
  ASSERT_NE(merged_low, nullptr);
  EXPECT_EQ(merged_low.getZeroPoint(), -128);

  EXPECT_EQ(temp::internal::CalculateNewQuantizedType({}), nullptr);
}

TEST(CalculateNewQuantizedTypeTest, GuardsAgainstZeroScaleDivisionByZero) {
  MLIRContext context;
  context.loadDialect<quant::QuantDialect>();
  Builder builder(&context);

  // Using a valid storage range [-128, 127] with zeroPoint = 1LL << 62 passes
  // QuantizedType::verifyInvariants in debug builds while
  // static_cast<double>(storage_min - zp) and
  // static_cast<double>(storage_max - zp) round to the same double, collapsing
  // real_max - real_min to 0.0 (scale == 0.0) and returning qtype1 early.
  constexpr int64_t kLargeZeroPoint = 1LL << 62;
  auto qtype1 = quant::UniformQuantizedType::get(
      quant::QuantizationFlags::Signed, builder.getIntegerType(8),
      builder.getF32Type(), /*scale=*/1.0, /*zeroPoint=*/kLargeZeroPoint,
      /*storageTypeMin=*/-128, /*storageTypeMax=*/127);
  auto qtype2 = quant::UniformQuantizedType::get(
      quant::QuantizationFlags::Signed, builder.getIntegerType(8),
      builder.getF32Type(), /*scale=*/1.0, /*zeroPoint=*/kLargeZeroPoint,
      /*storageTypeMin=*/-128, /*storageTypeMax=*/127);

  auto merged = mlir::dyn_cast<quant::UniformQuantizedType>(
      temp::internal::CalculateNewQuantizedType({qtype1, qtype2}));
  ASSERT_NE(merged, nullptr);
  EXPECT_EQ(merged, qtype1);
}

}  // namespace
}  // namespace mlir::TFL
