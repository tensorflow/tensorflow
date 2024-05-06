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

#include "tensorflow/compiler/mlir/quantization/stablehlo/ops/stablehlo_op_quant_spec.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/quantization/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/quantization/common/attrs_and_constraints.h"
#include "tensorflow/compiler/mlir/quantization/common/func.h"
#include "tensorflow/compiler/mlir/quantization/common/quantization_lib/quantization_utils.h"
#include "tensorflow/compiler/mlir/quantization/common/test_base.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/platform/test.h"

namespace mlir::quant::stablehlo {
namespace {

using ::mlir::stablehlo::GatherOp;
using ::testing::IsEmpty;
using ::testing::IsTrue;
using ::testing::NotNull;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

using IsOpQuantizableStableHloTest = ::mlir::quant::QuantizationTestBase;

// Quantizable ops: constants
// Non-quantizable ops: normal StableHLO ops and terminators
constexpr absl::string_view kModuleConstantAdd = R"mlir(
  module {
    func.func @constant_add() -> (tensor<3x2xf32>) {
      %cst1 = stablehlo.constant dense<2.4> : tensor<3x2xf32>
      %cst2 = stablehlo.constant dense<5.7> : tensor<3x2xf32>
      %add = stablehlo.add %cst1, %cst2 : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x2xf32>
      func.return %add : tensor<3x2xf32>
    }
  }
)mlir";

// Quantizable ops: XlaCallModule op with "fully_quantizable" attribute and
// same-scale StableHLO ops
// Non-quantizable ops: quantize/dequantize ops
constexpr absl::string_view kModuleCompositeSameScale = R"mlir(
  module {
    func.func @same_scale_after_composite() -> tensor<3x1xf32> {
      %0 = "tf.XlaCallModule"() {Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable",   device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : () -> tensor<1x3xf32>
      %1 = "quantfork.qcast"(%0) {volatile} : (tensor<1x3xf32>) -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
      %2 = "quantfork.dcast"(%1) : (tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<1x3xf32>
      %3 = stablehlo.reshape %2 : (tensor<1x3xf32>) -> tensor<3x1xf32>
      %4 = "quantfork.qcast"(%3) {volatile} : (tensor<3x1xf32>) -> tensor<3x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
      %5 = "quantfork.dcast"(%4) : (tensor<3x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<3x1xf32>
      return %5 : tensor<3x1xf32>
    }
  }
)mlir";

// Non-quantizable ops: XlaCallModule op without "fully_quantizable" attribute
constexpr absl::string_view kModuleCompositeNoAttr = R"mlir(
  module {
    func.func @composite_without_attr() -> tensor<1x3xf32> {
      %0 = "tf.XlaCallModule"() {Sout = [#tf_type.shape<1x3>], _entry_function = @non_quantizable_composite, _original_entry_function = "non_quantizable_composite", _stablehlo_module_attrs = {}, device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : () -> tensor<1x3xf32>
      return %0 : tensor<1x3xf32>
    }
  }
)mlir";

TEST_F(IsOpQuantizableStableHloTest, ConstantOpQuantizable) {
  OwningOpRef<ModuleOp> module_op = ParseModuleOpString(kModuleConstantAdd);
  ASSERT_TRUE(module_op);

  auto test_func = module_op->lookupSymbol<func::FuncOp>("constant_add");
  ASSERT_THAT(test_func, NotNull());

  auto constant_op =
      FindOperationOfType<mlir::stablehlo::ConstantOp>(test_func);
  EXPECT_TRUE(IsOpQuantizableStableHlo(constant_op));
}

TEST_F(IsOpQuantizableStableHloTest, TerminatorOpNotQuantizable) {
  OwningOpRef<ModuleOp> module_op = ParseModuleOpString(kModuleConstantAdd);
  ASSERT_TRUE(module_op);

  auto test_func = module_op->lookupSymbol<func::FuncOp>("constant_add");
  ASSERT_THAT(test_func, NotNull());

  auto return_op = FindOperationOfType<func::ReturnOp>(test_func);
  EXPECT_FALSE(IsOpQuantizableStableHlo(return_op));
}

TEST_F(IsOpQuantizableStableHloTest, SameScaleOpQuantizable) {
  OwningOpRef<ModuleOp> module_op =
      ParseModuleOpString(kModuleCompositeSameScale);
  ASSERT_TRUE(module_op);

  auto test_func =
      module_op->lookupSymbol<func::FuncOp>("same_scale_after_composite");
  ASSERT_THAT(test_func, NotNull());

  auto reshape_op = FindOperationOfType<mlir::stablehlo::ReshapeOp>(test_func);
  EXPECT_TRUE(IsOpQuantizableStableHlo(reshape_op));
}

TEST_F(IsOpQuantizableStableHloTest, NonSameScaleOpNotQuantizable) {
  OwningOpRef<ModuleOp> module_op = ParseModuleOpString(kModuleConstantAdd);
  ASSERT_TRUE(module_op);

  auto test_func = module_op->lookupSymbol<func::FuncOp>("constant_add");
  ASSERT_THAT(test_func, NotNull());

  auto add_op = FindOperationOfType<mlir::stablehlo::AddOp>(test_func);
  EXPECT_FALSE(IsOpQuantizableStableHlo(add_op));
}

TEST_F(IsOpQuantizableStableHloTest, ValidXlaCallModuleOpQuantizable) {
  OwningOpRef<ModuleOp> module_op =
      ParseModuleOpString(kModuleCompositeSameScale);
  ASSERT_TRUE(module_op);

  auto test_func =
      module_op->lookupSymbol<func::FuncOp>("same_scale_after_composite");
  ASSERT_THAT(test_func, NotNull());

  auto xla_call_module_op = FindOperationOfType<TF::XlaCallModuleOp>(test_func);
  EXPECT_TRUE(IsOpQuantizableStableHlo(xla_call_module_op));
}

TEST_F(IsOpQuantizableStableHloTest, InvalidXlaCallModuleOpNotQuantizable) {
  OwningOpRef<ModuleOp> module_op = ParseModuleOpString(kModuleCompositeNoAttr);
  ASSERT_TRUE(module_op);

  auto test_func =
      module_op->lookupSymbol<func::FuncOp>("composite_without_attr");
  ASSERT_THAT(test_func, NotNull());

  auto xla_call_module_op = FindOperationOfType<TF::XlaCallModuleOp>(test_func);
  EXPECT_FALSE(IsOpQuantizableStableHlo(xla_call_module_op));
}

TEST_F(IsOpQuantizableStableHloTest, QuantizeDequantizeOpNotQuantizable) {
  OwningOpRef<ModuleOp> module_op =
      ParseModuleOpString(kModuleCompositeSameScale);
  ASSERT_TRUE(module_op);

  auto test_func =
      module_op->lookupSymbol<func::FuncOp>("same_scale_after_composite");
  ASSERT_THAT(test_func, NotNull());

  auto quantize_op = FindOperationOfType<quantfork::QuantizeCastOp>(test_func);
  EXPECT_FALSE(IsOpQuantizableStableHlo(quantize_op));

  auto dequantize_op =
      FindOperationOfType<quantfork::DequantizeCastOp>(test_func);
  EXPECT_FALSE(IsOpQuantizableStableHlo(dequantize_op));
}

TEST_F(IsOpQuantizableStableHloTest,
       XlaCallModuleOpQuantizableWhenNotDenylisted) {
  // A `TF::XlaCallModuleOp` with `_quantization_method = ""`.
  constexpr absl::string_view
      kModuleXlaCallModuleOpWithDefaultQuantizationMethod = R"mlir(
    func.func @xla_call_module_default_quantization_method(%arg0: tensor<1x1x3xf32>, %arg1: tensor<3x4xf32>) -> tensor<1x1x4xf32> {
      %0 = "tf.XlaCallModule"(%arg0, %arg1) <{Sout = [#tf_type.shape<1x1x4>], dim_args_spec = [], disabled_checks = [], function_list = [], has_token_input_output = false, module = "", platforms = ["CPU"], version = 9 : i64}> {_entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _quantization_method = "", _stablehlo_module_attrs = {jax.uses_shape_polymorphism = true}, _tfl_quant_trait = "fully_quantizable"} : (tensor<1x1x3xf32>, tensor<3x4xf32>) -> tensor<1x1x4xf32>
      return %0 : tensor<1x1x4xf32>
    }
  )mlir";

  OwningOpRef<ModuleOp> module_op =
      ParseModuleOpString(kModuleXlaCallModuleOpWithDefaultQuantizationMethod);
  ASSERT_TRUE(module_op);

  auto test_func = module_op->lookupSymbol<func::FuncOp>(
      "xla_call_module_default_quantization_method");
  ASSERT_THAT(test_func, NotNull());

  auto xla_call_module_op = FindOperationOfType<TF::XlaCallModuleOp>(test_func);
  EXPECT_TRUE(IsOpQuantizableStableHlo(xla_call_module_op));
}

TEST_F(IsOpQuantizableStableHloTest, DenylistedXlaCallModuleOpNotQuantizable) {
  // A `TF::XlaCallModuleOp` with `_quantization_method = "no_quantization {}"`,
  // indicating it has been explicitly denylisted by the user.
  constexpr absl::string_view kModuleDenylistedXlaCallModuleOp = R"mlir(
    func.func @xla_call_module_denylisted(%arg0: tensor<1x1x3xf32>, %arg1: tensor<3x4xf32>) -> tensor<1x1x4xf32> {
      %0 = "tf.XlaCallModule"(%arg0, %arg1) <{Sout = [#tf_type.shape<1x1x4>], dim_args_spec = [], disabled_checks = [], function_list = [], has_token_input_output = false, module = "", platforms = ["CPU"], version = 9 : i64}> {_entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _quantization_method = "no_quantization {}", _stablehlo_module_attrs = {jax.uses_shape_polymorphism = true}, _tfl_quant_trait = "fully_quantizable"} : (tensor<1x1x3xf32>, tensor<3x4xf32>) -> tensor<1x1x4xf32>
      return %0 : tensor<1x1x4xf32>
    }
  )mlir";

  OwningOpRef<ModuleOp> module_op =
      ParseModuleOpString(kModuleDenylistedXlaCallModuleOp);
  ASSERT_TRUE(module_op);

  auto test_func =
      module_op->lookupSymbol<func::FuncOp>("xla_call_module_denylisted");
  ASSERT_THAT(test_func, NotNull());

  auto xla_call_module_op = FindOperationOfType<TF::XlaCallModuleOp>(test_func);
  EXPECT_FALSE(IsOpQuantizableStableHlo(xla_call_module_op));
}

using GetStableHloOpQuantSpecTest = ::mlir::quant::QuantizationTestBase;

TEST_F(GetStableHloOpQuantSpecTest,
       EmptyCoeffOpQuantDimForPerTensorQuantizedConvolution) {
  // A `TF::XlaCallModuleOp` with `_quantization_method = "static_range_ptq
  // {}"`, representing a per-tensor static-range PTQ quantization.
  constexpr absl::string_view
      kXlaCallModuleOpWithPerTensorQuantizedConvolution = R"mlir(
    func.func @main(%arg0: tensor<1x1x3xf32>, %arg1: tensor<3x4xf32>) -> tensor<1x1x4xf32> {
      %0 = "tf.XlaCallModule"(%arg0, %arg1) <{Sout = [#tf_type.shape<1x1x4>], dim_args_spec = [], disabled_checks = [], function_list = [], has_token_input_output = false, module = "", platforms = ["CPU"], version = 9 : i64}>
          {
            _entry_function = @composite_conv_fn_1,
            _original_entry_function = "composite_conv_fn_1",
            _quantization_method = "static_range_ptq {}",
            _stablehlo_module_attrs = {jax.uses_shape_polymorphism = true},
            _tfl_quant_trait = "fully_quantizable"
          } : (tensor<1x1x3xf32>, tensor<3x4xf32>) -> tensor<1x1x4xf32>
      return %0 : tensor<1x1x4xf32>
    }
  )mlir";

  const OwningOpRef<ModuleOp> module_op =
      ParseModuleOpString(kXlaCallModuleOpWithPerTensorQuantizedConvolution);
  ASSERT_TRUE(module_op);

  const FailureOr<TF::XlaCallModuleOp> xla_call_module_op =
      FindFirstOpFromMainFunc<TF::XlaCallModuleOp>(*module_op);
  ASSERT_TRUE(succeeded(xla_call_module_op));

  const std::unique_ptr<OpQuantSpec> op_quant_spec =
      GetStableHloOpQuantSpec(*xla_call_module_op);
  ASSERT_THAT(op_quant_spec, NotNull());

  EXPECT_THAT(op_quant_spec->coeff_op_quant_dim, IsEmpty());
}

TEST_F(GetStableHloOpQuantSpecTest,
       EmptyCoeffOpQuantDimForPerChannelQuantizedConvolution) {
  constexpr absl::string_view
      kXlaCallModuleOpWithPerChannelQuantizedConvolution = R"mlir(
    func.func @main(%arg0: tensor<1x1x3xf32>, %arg1: tensor<3x4xf32>) -> tensor<1x1x4xf32> {
      %0 = "tf.XlaCallModule"(%arg0, %arg1) <{Sout = [#tf_type.shape<1x1x4>], dim_args_spec = [], disabled_checks = [], function_list = [], has_token_input_output = false, module = "", platforms = ["CPU"], version = 9 : i64}>
          {
            _entry_function = @composite_conv_fn_1,
            _original_entry_function = "composite_conv_fn_1",
            _quantization_method = "static_range_ptq {input_quantized_types {key: 1, value {dimension_specs {dimension: 3}}}}",
            _stablehlo_module_attrs = {jax.uses_shape_polymorphism = true},
            _tfl_quant_trait = "fully_quantizable"
          } : (tensor<1x1x3xf32>, tensor<3x4xf32>) -> tensor<1x1x4xf32>
      return %0 : tensor<1x1x4xf32>
    }
  )mlir";

  const OwningOpRef<ModuleOp> module_op =
      ParseModuleOpString(kXlaCallModuleOpWithPerChannelQuantizedConvolution);
  ASSERT_TRUE(module_op);

  const FailureOr<TF::XlaCallModuleOp> xla_call_module_op =
      FindFirstOpFromMainFunc<TF::XlaCallModuleOp>(*module_op);
  ASSERT_TRUE(succeeded(xla_call_module_op));

  const std::unique_ptr<OpQuantSpec> op_quant_spec =
      GetStableHloOpQuantSpec(*xla_call_module_op);
  ASSERT_THAT(op_quant_spec, NotNull());

  EXPECT_THAT(op_quant_spec->coeff_op_quant_dim,
              UnorderedElementsAre(Pair(1, 3)));
}

using GetStableHloQuantConstraintsTest = ::mlir::quant::QuantizationTestBase;

TEST_F(GetStableHloQuantConstraintsTest,
       HasSameOperandAndResultTypeRequirementSucceeds) {
  // Quantizable ops: constants
  // Non-quantizable ops: normal StableHLO ops and terminators
  constexpr absl::string_view kModuleGather = R"mlir(
    module {
      func.func @main() -> (tensor<2x3x2x2xf32>) {
        %0 = stablehlo.constant dense<1.0> : tensor<3x4x2xf32>
        %1 = stablehlo.constant dense<2> : tensor<2x3x2xi64>
        %2 = "stablehlo.gather"(%0, %1) {
          dimension_numbers = #stablehlo.gather<
            offset_dims = [2, 3],
            collapsed_slice_dims = [0],
            start_index_map = [1, 0],
            index_vector_dim = 2>,
          slice_sizes = array<i64: 1, 2, 2>,
          indices_are_sorted = false
        } : (tensor<3x4x2xf32>, tensor<2x3x2xi64>) -> tensor<2x3x2x2xf32>
        func.return %2 : tensor<2x3x2x2xf32>
      }
    }
  )mlir";
  OwningOpRef<ModuleOp> module_op = ParseModuleOpString(kModuleGather);
  ASSERT_TRUE(module_op);

  func::FuncOp main_fn = FindMainFuncOp(*module_op);
  ASSERT_THAT(main_fn, NotNull());

  Operation* gather_op = FindOperationOfType<GatherOp>(main_fn);
  const auto spec = GetStableHloQuantConstraints(gather_op);

  EXPECT_THAT(spec, NotNull());
  EXPECT_THAT(spec->has_same_operand_and_result_type_requirement, IsTrue());
}

}  // namespace
}  // namespace mlir::quant::stablehlo
