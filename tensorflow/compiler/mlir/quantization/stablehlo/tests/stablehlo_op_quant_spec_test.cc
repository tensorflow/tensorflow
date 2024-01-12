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

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/quantization/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/quantization/common/test_base.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/platform/test.h"

namespace mlir::quant::stablehlo {
namespace {

using ::mlir::quant::QuantizationTestBase;

class IsOpQuantizableStableHloTest : public QuantizationTestBase {};

// Quantizable ops: constants
// Non-quantizable ops: normal StableHLO ops and terminators
constexpr absl::string_view module_constant_add = R"mlir(
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
constexpr absl::string_view module_composite_same_scale = R"mlir(
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
constexpr absl::string_view module_composite_no_attr = R"mlir(
  module {
    func.func @composite_without_attr() -> tensor<1x3xf32> {
      %0 = "tf.XlaCallModule"() {Sout = [#tf_type.shape<1x3>], _entry_function = @non_quantizable_composite, _original_entry_function = "non_quantizable_composite", _stablehlo_module_attrs = {}, device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : () -> tensor<1x3xf32>
      return %0 : tensor<1x3xf32>
    }
  }
)mlir";

TEST_F(IsOpQuantizableStableHloTest, ConstantOpQuantizable) {
  OwningOpRef<ModuleOp> module_op_ref =
      ParseModuleOpString(module_constant_add);
  func::FuncOp test_func =
      GetFunctionFromModule(*module_op_ref, "constant_add");
  Operation* constant_op =
      FindOperationOfType<mlir::stablehlo::ConstantOp>(test_func);
  bool is_constant_quantizable =
      mlir::quant::stablehlo::IsOpQuantizableStableHlo(constant_op);

  EXPECT_TRUE(is_constant_quantizable);
}

TEST_F(IsOpQuantizableStableHloTest, TerminatorOpNotQuantizable) {
  OwningOpRef<ModuleOp> module_op_ref =
      ParseModuleOpString(module_constant_add);
  func::FuncOp test_func =
      GetFunctionFromModule(*module_op_ref, "constant_add");
  Operation* return_op = FindOperationOfType<func::ReturnOp>(test_func);
  bool is_return_quantizable =
      mlir::quant::stablehlo::IsOpQuantizableStableHlo(return_op);

  EXPECT_FALSE(is_return_quantizable);
}

TEST_F(IsOpQuantizableStableHloTest, SameScaleOpQuantizable) {
  OwningOpRef<ModuleOp> module_op_ref =
      ParseModuleOpString(module_composite_same_scale);
  func::FuncOp test_func =
      GetFunctionFromModule(*module_op_ref, "same_scale_after_composite");
  Operation* reshape_op =
      FindOperationOfType<mlir::stablehlo::ReshapeOp>(test_func);
  bool is_reshape_quantizable =
      mlir::quant::stablehlo::IsOpQuantizableStableHlo(reshape_op);

  EXPECT_TRUE(is_reshape_quantizable);
}

TEST_F(IsOpQuantizableStableHloTest, NonSameScaleOpNotQuantizable) {
  OwningOpRef<ModuleOp> module_op_ref =
      ParseModuleOpString(module_constant_add);
  func::FuncOp test_func =
      GetFunctionFromModule(*module_op_ref, "constant_add");
  Operation* add_op = FindOperationOfType<mlir::stablehlo::AddOp>(test_func);
  bool is_add_quantizable =
      mlir::quant::stablehlo::IsOpQuantizableStableHlo(add_op);

  EXPECT_FALSE(is_add_quantizable);
}

TEST_F(IsOpQuantizableStableHloTest, ValidXlaCallModuleOpQuantizable) {
  OwningOpRef<ModuleOp> module_op_ref =
      ParseModuleOpString(module_composite_same_scale);
  func::FuncOp test_func =
      GetFunctionFromModule(*module_op_ref, "same_scale_after_composite");
  Operation* xla_call_module_op =
      FindOperationOfType<TF::XlaCallModuleOp>(test_func);
  bool is_xla_call_module_quantizable =
      mlir::quant::stablehlo::IsOpQuantizableStableHlo(xla_call_module_op);

  EXPECT_TRUE(is_xla_call_module_quantizable);
}

TEST_F(IsOpQuantizableStableHloTest, InvalidXlaCallModuleOpNotQuantizable) {
  OwningOpRef<ModuleOp> module_op_ref =
      ParseModuleOpString(module_composite_no_attr);
  func::FuncOp test_func =
      GetFunctionFromModule(*module_op_ref, "composite_without_attr");
  Operation* xla_call_module_op =
      FindOperationOfType<TF::XlaCallModuleOp>(test_func);
  bool is_xla_call_module_quantizable =
      mlir::quant::stablehlo::IsOpQuantizableStableHlo(xla_call_module_op);

  EXPECT_FALSE(is_xla_call_module_quantizable);
}

TEST_F(IsOpQuantizableStableHloTest, QuantizeDequantizeOpNotQuantizable) {
  OwningOpRef<ModuleOp> module_op_ref =
      ParseModuleOpString(module_composite_same_scale);
  func::FuncOp test_func =
      GetFunctionFromModule(*module_op_ref, "same_scale_after_composite");
  Operation* quantize_op =
      FindOperationOfType<quantfork::QuantizeCastOp>(test_func);
  Operation* dequantize_op =
      FindOperationOfType<quantfork::DequantizeCastOp>(test_func);
  bool is_quantize_quantizable =
      mlir::quant::stablehlo::IsOpQuantizableStableHlo(quantize_op);
  bool is_dequantize_quantizable =
      mlir::quant::stablehlo::IsOpQuantizableStableHlo(dequantize_op);

  EXPECT_FALSE(is_quantize_quantizable);
  EXPECT_FALSE(is_dequantize_quantizable);
}

}  // namespace
}  // namespace mlir::quant::stablehlo
