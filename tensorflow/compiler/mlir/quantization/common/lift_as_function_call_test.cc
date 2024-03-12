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

#include "tensorflow/compiler/mlir/quantization/common/lift_as_function_call.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/quantization/common/func.h"
#include "tensorflow/compiler/mlir/quantization/common/test_base.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tsl/platform/status_matchers.h"

namespace mlir::quant {
namespace {

using ::stablehlo::quantization::Method;
using ::testing::HasSubstr;
using ::testing::NotNull;
using ::tsl::testing::IsOk;
using ::tsl::testing::StatusIs;

using LiftAsFunctionCallTest = ::mlir::quant::QuantizationTestBase;

constexpr absl::string_view kModuleLifted = R"mlir(
  module {
    func.func private @composite_dot_general_fn_1(%arg0: tensor<1x1024xf32>, %arg1: tensor<1024x3xf32>) -> tensor<1x3xf32> attributes {_from_xla_call_module, tf_quant.composite_function} {
      %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1024xf32>, tensor<1024x3xf32>) -> tensor<1x3xf32>
      return %0 : tensor<1x3xf32>
    }
  }
)mlir";

TEST_F(LiftAsFunctionCallTest, LiftedFunctionSucceeds) {
  OwningOpRef<ModuleOp> module_op = ParseModuleOpString(kModuleLifted);
  ASSERT_TRUE(module_op);

  auto composite_dot_general_fn =
      module_op->lookupSymbol<func::FuncOp>("composite_dot_general_fn_1");
  ASSERT_THAT(composite_dot_general_fn, NotNull());

  Operation* dot_general_op =
      FindOperationOfType<mlir::stablehlo::DotGeneralOp>(
          composite_dot_general_fn);
  EXPECT_TRUE(IsInLiftedFunc(*dot_general_op));
}

constexpr absl::string_view kModuleStableHlo = R"mlir(
  module {
    func.func @main(%arg0: tensor<1x1024xf32>, %arg1: tensor<1024x3xf32>) -> tensor<1x3xf32> attributes {_from_xla_call_module} {
      %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [] : (tensor<1x1024xf32>, tensor<1024x3xf32>) -> tensor<1x3xf32>
      return %0 : tensor<1x3xf32>
    }
  }
)mlir";

TEST_F(LiftAsFunctionCallTest, FunctionLiftedAsXlaCallModuleOp) {
  OwningOpRef<ModuleOp> module_op = ParseModuleOpString(kModuleStableHlo);
  ASSERT_TRUE(module_op);

  func::FuncOp main_fn = FindMainFuncOp(*module_op);
  ASSERT_THAT(main_fn, NotNull());

  Operation* dot_general_op =
      FindOperationOfType<mlir::stablehlo::DotGeneralOp>(main_fn);

  const SmallVector<NamedAttribute>& attributes = {
      builder_.getNamedAttr(
          "precision_config",
          builder_.getArrayAttr(SmallVector<Attribute>(
              1, mlir::stablehlo::PrecisionAttr::get(
                     ctx_.get(), mlir::stablehlo::Precision::DEFAULT)))),
  };
  Operation* lifted_op =
      LiftAsFunctionCall(builder_, dot_general_op->getLoc(),
                         FunctionCallOpType::TFXlaCallModuleOp,
                         "composite_dot_general_fn",
                         dot_general_op->getOperands(),
                         dot_general_op->getResults(), attributes)[0]
          .getDefiningOp();
  const auto entry_function_symbol_ref =
      lifted_op->getAttrOfType<FlatSymbolRefAttr>("_entry_function");
  SymbolTable symbol_table(*module_op);
  auto entry_func = dyn_cast_or_null<func::FuncOp>(
      symbol_table.lookup(entry_function_symbol_ref.getValue()));
  Operation* lifted_dot_general_op =
      FindOperationOfType<mlir::stablehlo::DotGeneralOp>(entry_func);

  EXPECT_TRUE(isa<TF::XlaCallModuleOp>(lifted_op));
  EXPECT_EQ(lifted_op->getAttr("_original_entry_function").cast<StringAttr>(),
            "composite_dot_general_fn_1");
  EXPECT_EQ(
      lifted_dot_general_op->getAttr("precision_config").cast<ArrayAttr>(),
      builder_.getArrayAttr(SmallVector<Attribute>(
          1, mlir::stablehlo::PrecisionAttr::get(
                 ctx_.get(), mlir::stablehlo::Precision::DEFAULT))));
}

TEST_F(LiftAsFunctionCallTest, FunctionNoAttrLiftedAsXlaCallModuleOp) {
  OwningOpRef<ModuleOp> module_op = ParseModuleOpString(kModuleStableHlo);
  ASSERT_TRUE(module_op);

  func::FuncOp main_fn = FindMainFuncOp(*module_op);
  ASSERT_THAT(main_fn, NotNull());

  Operation* dot_general_op =
      FindOperationOfType<mlir::stablehlo::DotGeneralOp>(main_fn);
  Operation* lifted_op =
      LiftAsFunctionCall(
          builder_, dot_general_op->getLoc(),
          FunctionCallOpType::TFXlaCallModuleOp, "composite_dot_general_fn",
          dot_general_op->getOperands(), dot_general_op->getResults())[0]
          .getDefiningOp();
  EXPECT_TRUE(isa<TF::XlaCallModuleOp>(lifted_op));
  EXPECT_EQ(lifted_op->getAttr("_original_entry_function").cast<StringAttr>(),
            "composite_dot_general_fn_1");
}

TEST_F(LiftAsFunctionCallTest, EinsumSupportedForXlaDotV2Succeeds) {
  StringAttr einsum_supported_by_xla_dot_v2_attr =
      builder_.getStringAttr("ijk,ikm->ijm");
  StringAttr einsum_one_operand = builder_.getStringAttr("ijk->ikj");
  StringAttr einsum_ellipsis = builder_.getStringAttr("...gse->...gs");
  EXPECT_TRUE(IsEinsumSupportedByXlaDotV2(einsum_supported_by_xla_dot_v2_attr));
  EXPECT_FALSE(IsEinsumSupportedByXlaDotV2(einsum_one_operand));
  EXPECT_FALSE(IsEinsumSupportedByXlaDotV2(einsum_ellipsis));
}

TEST_F(LiftAsFunctionCallTest, GetQuantizationMethodSucceeds) {
  // Function containing a simple `TF::XlaCallModuleOp` with a valid string
  // attribute `_quantization_method` set to `"no_quantization {}"`.
  constexpr absl::string_view kXlaCallModuleOpWithQuantizationMethodAttr =
      R"mlir(
    func.func @main(%arg0: tensor<1x1x3xf32>, %arg1: tensor<3x4xf32>) -> tensor<1x1x4xf32> {
      %0 = "tf.XlaCallModule"(%arg0, %arg1) <{Sout = [#tf_type.shape<1x1x4>], dim_args_spec = [], disabled_checks = [], function_list = [], has_token_input_output = false, module = "", platforms = ["CPU"], version = 9 : i64}> {_entry_function = @composite_dot_general_fn_1, _quantization_method = "no_quantization {}", _stablehlo_module_attrs = {jax.uses_shape_polymorphism = true}} : (tensor<1x1x3xf32>, tensor<3x4xf32>) -> tensor<1x1x4xf32>
      return %0 : tensor<1x1x4xf32>
    }
  )mlir";

  const OwningOpRef<ModuleOp> module_op =
      ParseModuleOpString(kXlaCallModuleOpWithQuantizationMethodAttr);
  ASSERT_TRUE(module_op);

  func::FuncOp main_fn = FindMainFuncOp(*module_op);
  ASSERT_THAT(main_fn, NotNull());

  auto xla_call_module_ops = main_fn.getOps<TF::XlaCallModuleOp>();
  ASSERT_FALSE(xla_call_module_ops.empty());

  // Test that `GetQuantizationMethod` returns a valid `Method` corresponding to
  // `"no_quantization {}"`.
  const absl::StatusOr<Method> method =
      GetQuantizationMethod(*xla_call_module_ops.begin());
  ASSERT_THAT(method, IsOk());
  EXPECT_TRUE(method->has_no_quantization());
}

TEST_F(LiftAsFunctionCallTest,
       GetQuantizationMethodFailsWhenNoQuantizationMethodAttr) {
  // Function containing a simple `TF::XlaCallModuleOp` that doesn't have the
  // attribute "_quantization_method".
  constexpr absl::string_view kXlaCallModuleOpWithNoQuantizationMethodAttr =
      R"mlir(
    func.func @main(%arg0: tensor<1x1x3xf32>, %arg1: tensor<3x4xf32>) -> tensor<1x1x4xf32> {
      %0 = "tf.XlaCallModule"(%arg0, %arg1) <{Sout = [#tf_type.shape<1x1x4>], dim_args_spec = [], disabled_checks = [], function_list = [], has_token_input_output = false, module = "", platforms = ["CPU"], version = 9 : i64}> {_entry_function = @composite_dot_general_fn_1, _stablehlo_module_attrs = {jax.uses_shape_polymorphism = true}} : (tensor<1x1x3xf32>, tensor<3x4xf32>) -> tensor<1x1x4xf32>
      return %0 : tensor<1x1x4xf32>
    }
  )mlir";

  const OwningOpRef<ModuleOp> module_op =
      ParseModuleOpString(kXlaCallModuleOpWithNoQuantizationMethodAttr);
  ASSERT_TRUE(module_op);

  func::FuncOp main_fn = FindMainFuncOp(*module_op);
  ASSERT_THAT(main_fn, NotNull());

  auto xla_call_module_ops = main_fn.getOps<TF::XlaCallModuleOp>();
  ASSERT_FALSE(xla_call_module_ops.empty());

  // Test that `GetQuantizationMethod` returns a `absl::InvalidArgumentError`
  // because there is no `_quantization_method` attribute.
  const absl::StatusOr<Method> method =
      GetQuantizationMethod(*xla_call_module_ops.begin());
  EXPECT_THAT(
      method,
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Attribute _quantization_method is not found")));
}

TEST_F(LiftAsFunctionCallTest,
       GetQuantizationMethodFailsWhenMalformedQuantizationMethodAttr) {
  // Function containing a simple `TF::XlaCallModuleOp` with an invalid
  // `_quantization_method` attribute.
  constexpr absl::string_view kXlaCallModuleOpWithNoQuantizationMethodAttr =
      R"mlir(
    func.func @main(%arg0: tensor<1x1x3xf32>, %arg1: tensor<3x4xf32>) -> tensor<1x1x4xf32> {
      %0 = "tf.XlaCallModule"(%arg0, %arg1) <{Sout = [#tf_type.shape<1x1x4>], dim_args_spec = [], disabled_checks = [], function_list = [], has_token_input_output = false, module = "", platforms = ["CPU"], version = 9 : i64}> {_entry_function = @composite_dot_general_fn_1, _quantization_method = "invalid_field: 123", _stablehlo_module_attrs = {jax.uses_shape_polymorphism = true}} : (tensor<1x1x3xf32>, tensor<3x4xf32>) -> tensor<1x1x4xf32>
      return %0 : tensor<1x1x4xf32>
    }
  )mlir";

  const OwningOpRef<ModuleOp> module_op =
      ParseModuleOpString(kXlaCallModuleOpWithNoQuantizationMethodAttr);
  ASSERT_TRUE(module_op);

  func::FuncOp main_fn = FindMainFuncOp(*module_op);
  ASSERT_THAT(main_fn, NotNull());

  auto xla_call_module_ops = main_fn.getOps<TF::XlaCallModuleOp>();
  ASSERT_FALSE(xla_call_module_ops.empty());

  const absl::StatusOr<Method> method =
      GetQuantizationMethod(*xla_call_module_ops.begin());
  EXPECT_THAT(method,
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Failed to parse Method from textproto")));
}

}  // namespace
}  // namespace mlir::quant
