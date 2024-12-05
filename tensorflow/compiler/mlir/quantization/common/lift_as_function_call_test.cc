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
#include "absl/algorithm/container.h"
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
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/quantization/common/attrs_and_constraints.h"
#include "tensorflow/compiler/mlir/quantization/common/func.h"
#include "tensorflow/compiler/mlir/quantization/common/test_base.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tsl/platform/protobuf.h"  // IWYU pragma: keep
#include "tsl/platform/status_matchers.h"

namespace mlir::quant {
namespace {

using ::stablehlo::quantization::Method;
using ::testing::HasSubstr;
using ::testing::NotNull;
using ::testing::SizeIs;
using ::testing::StrEq;
using ::tsl::protobuf::util::MessageDifferencer;
using ::tsl::testing::IsOk;
using ::tsl::testing::StatusIs;

using LiftAsFunctionCallTest = QuantizationTestBase;

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

  auto dot_general_op = FindOperationOfType<mlir::stablehlo::DotGeneralOp>(
      composite_dot_general_fn);
  EXPECT_TRUE(IsInLiftedFunc(dot_general_op));
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

  auto dot_general_op =
      FindOperationOfType<mlir::stablehlo::DotGeneralOp>(main_fn);

  const SmallVector<NamedAttribute>& attributes = {
      builder_.getNamedAttr(
          "precision_config",
          builder_.getArrayAttr(SmallVector<Attribute>(
              1, mlir::stablehlo::PrecisionAttr::get(
                     ctx_.get(), mlir::stablehlo::Precision::DEFAULT)))),
  };
  const SmallVector<Value> operands(dot_general_op->getOperands());
  const SmallVector<Value> results(dot_general_op->getResults());
  Operation* lifted_op =
      LiftAsFunctionCall(builder_, dot_general_op->getLoc(),
                         FunctionCallOpType::TFXlaCallModuleOp,
                         "composite_dot_general_fn", operands, results,
                         attributes)[0]
          .getDefiningOp();
  const auto entry_function_symbol_ref =
      lifted_op->getAttrOfType<FlatSymbolRefAttr>("_entry_function");
  SymbolTable symbol_table(*module_op);
  auto entry_func = dyn_cast_or_null<func::FuncOp>(
      symbol_table.lookup(entry_function_symbol_ref.getValue()));
  auto lifted_dot_general_op =
      FindOperationOfType<mlir::stablehlo::DotGeneralOp>(entry_func);

  EXPECT_TRUE(isa<TF::XlaCallModuleOp>(lifted_op));
  EXPECT_EQ(
      mlir::cast<StringAttr>(lifted_op->getAttr("_original_entry_function")),
      "composite_dot_general_fn_1");
  EXPECT_EQ(
      mlir::cast<ArrayAttr>(lifted_dot_general_op->getAttr("precision_config")),
      builder_.getArrayAttr(SmallVector<Attribute>(
          1, mlir::stablehlo::PrecisionAttr::get(
                 ctx_.get(), mlir::stablehlo::Precision::DEFAULT))));
}

TEST_F(LiftAsFunctionCallTest, FunctionNoAttrLiftedAsXlaCallModuleOp) {
  OwningOpRef<ModuleOp> module_op = ParseModuleOpString(kModuleStableHlo);
  ASSERT_TRUE(module_op);

  func::FuncOp main_fn = FindMainFuncOp(*module_op);
  ASSERT_THAT(main_fn, NotNull());

  auto dot_general_op =
      FindOperationOfType<mlir::stablehlo::DotGeneralOp>(main_fn);
  const SmallVector<Value> operands(dot_general_op->getOperands());
  const SmallVector<Value> results(dot_general_op->getResults());
  Operation* lifted_op =
      LiftAsFunctionCall(builder_, dot_general_op->getLoc(),
                         FunctionCallOpType::TFXlaCallModuleOp,
                         "composite_dot_general_fn", operands, results)[0]
          .getDefiningOp();
  EXPECT_TRUE(isa<TF::XlaCallModuleOp>(lifted_op));
  EXPECT_EQ(
      mlir::cast<StringAttr>(lifted_op->getAttr("_original_entry_function")),
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

constexpr absl::string_view kFunctionWithRegion =
    R"mlir(
  func.func @main(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
    %if = "stablehlo.if"(%arg0) ({
      %0 = stablehlo.add %arg1, %arg1 : tensor<f32>
      stablehlo.return %0 : tensor<f32>
    }, {
      %1 = stablehlo.add %arg2, %arg2 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<i1>) -> (tensor<f32>)
    %subtract = stablehlo.subtract %if, %if : tensor<f32>
    return %subtract : tensor<f32>
  }
)mlir";

TEST_F(LiftAsFunctionCallTest, IsInRegionSucceedsWhenOpInsideRegion) {
  const OwningOpRef<ModuleOp> module_op =
      ParseModuleOpString(kFunctionWithRegion);
  ASSERT_TRUE(module_op);

  func::FuncOp main_fn = FindMainFuncOp(*module_op);
  ASSERT_THAT(main_fn, NotNull());

  auto if_op = FindOperationOfType<mlir::stablehlo::IfOp>(main_fn);
  Block& block = if_op->getRegion(0).front();
  Operation& add_op = *absl::c_find_if(block, [](Operation& entry) {
    return dyn_cast_or_null<::mlir::stablehlo::AddOp>(&entry);
  });
  EXPECT_TRUE(IsInStableHloOpRegion(&add_op));
}

TEST_F(LiftAsFunctionCallTest, IsInRegionFailsWhenOpNotInsideRegion) {
  const OwningOpRef<ModuleOp> module_op =
      ParseModuleOpString(kFunctionWithRegion);
  ASSERT_TRUE(module_op);

  func::FuncOp main_fn = FindMainFuncOp(*module_op);
  ASSERT_THAT(main_fn, NotNull());

  auto subtract_op = FindOperationOfType<mlir::stablehlo::SubtractOp>(main_fn);
  EXPECT_FALSE(IsInStableHloOpRegion(subtract_op));
}

TEST_F(LiftAsFunctionCallTest,
       GetQuantizationMethodOrDefaultReturnsCorrectMethod) {
  // Function containing a simple `TF::XlaCallModuleOp` with a valid string
  // attribute `_quantization_method` set to `"no_quantization { }"`.
  constexpr absl::string_view kXlaCallModuleOpWithQuantizationMethodAttr =
      R"mlir(
    func.func @main(%arg0: tensor<1x1x3xf32>, %arg1: tensor<3x4xf32>) -> tensor<1x1x4xf32> {
      %0 = "tf.XlaCallModule"(%arg0, %arg1) <{Sout = [#tf_type.shape<1x1x4>], dim_args_spec = [], disabled_checks = [], function_list = [], has_token_input_output = false, module = "", platforms = ["CPU"], version = 9 : i64}>
          {
            _entry_function = @composite_dot_general_fn_1,
            _quantization_method = "no_quantization { }",
            _stablehlo_module_attrs = {jax.uses_shape_polymorphism = true}
          } : (tensor<1x1x3xf32>, tensor<3x4xf32>) -> tensor<1x1x4xf32>
      return %0 : tensor<1x1x4xf32>
    }
  )mlir";

  const OwningOpRef<ModuleOp> module_op =
      ParseModuleOpString(kXlaCallModuleOpWithQuantizationMethodAttr);
  ASSERT_TRUE(module_op);

  FailureOr<TF::XlaCallModuleOp> xla_call_module_op =
      FindFirstOpFromMainFunc<TF::XlaCallModuleOp>(*module_op);
  ASSERT_TRUE(succeeded(xla_call_module_op));

  // Test that `GetQuantizationMethodOrDefault` returns a valid `Method`
  // corresponding to `"no_quantization {}"`.
  const Method method = GetQuantizationMethodOrDefault(*xla_call_module_op);
  EXPECT_TRUE(method.has_no_quantization());
}

TEST_F(
    LiftAsFunctionCallTest,
    GetQuantizationMethodOrDefaultReturnsDefaultWhenNoQuantizationMethodAttr) {
  // Function containing a simple `TF::XlaCallModuleOp` that is missing the
  // "_quantization_method" attribute.
  constexpr absl::string_view kXlaCallModuleOpWithoutQuantizationMethodAttr =
      R"mlir(
    func.func @main(%arg0: tensor<1x1x3xf32>, %arg1: tensor<3x4xf32>) -> tensor<1x1x4xf32> {
      %0 = "tf.XlaCallModule"(%arg0, %arg1) <{Sout = [#tf_type.shape<1x1x4>], dim_args_spec = [], disabled_checks = [], function_list = [], has_token_input_output = false, module = "", platforms = ["CPU"], version = 9 : i64}>
          {
            _entry_function = @composite_dot_general_fn_1,
            _stablehlo_module_attrs = {jax.uses_shape_polymorphism = true}
          } : (tensor<1x1x3xf32>, tensor<3x4xf32>) -> tensor<1x1x4xf32>
      return %0 : tensor<1x1x4xf32>
    }
  )mlir";

  const OwningOpRef<ModuleOp> module_op =
      ParseModuleOpString(kXlaCallModuleOpWithoutQuantizationMethodAttr);
  ASSERT_TRUE(module_op);

  FailureOr<TF::XlaCallModuleOp> xla_call_module_op =
      FindFirstOpFromMainFunc<TF::XlaCallModuleOp>(*module_op);
  ASSERT_TRUE(succeeded(xla_call_module_op));

  // Test that `GetQuantizationMethodOrDefault` returns the default instance.
  const Method method = GetQuantizationMethodOrDefault(*xla_call_module_op);
  EXPECT_TRUE(MessageDifferencer::Equals(method, Method::default_instance()));
}
constexpr absl::string_view kModuleDotWeightOnlyPtq = R"mlir(
  module {
    func.func @main(%arg0: tensor<?x2xf32> {tf_saved_model.index_path = ["input_tensor"]}) -> (tensor<?x2xf32>) {
      %0 = stablehlo.constant dense<[-0.211145893, -0.708605706]> : tensor<2xf32>
      %1 = stablehlo.constant dense<[[-0.630731344, 0.54962182], [0.180364341, -0.764542698]]> : tensor<2x2xf32>
      %2 = "tf.XlaCallModule"(%arg0, %1, %0) <{Sout = [#tf_type.shape<?x2>], module = "", version = 9 : i64}> {_entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _tfl_quant_trait = "fully_quantizable", _quantization_method = "weight_only_ptq { }"} : (tensor<?x2xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<?x2xf32>
      return %2 : tensor<?x2xf32>
    }
    func.func private @composite_dot_general_fn_1(%arg0: tensor<?x2xf32>, %arg1: tensor<2x2xf32>, %arg2: tensor<2xf32>) -> tensor<?x2xf32> attributes {_from_xla_call_module, tf_quant.composite_function} {
      %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<?x2xf32>, tensor<2x2xf32>) -> tensor<?x2xf32>
      return %0 : tensor<?x2xf32>
    }
  }
)mlir";

TEST_F(LiftAsFunctionCallTest, HasWeightOnlyPtqMethodExists) {
  OwningOpRef<ModuleOp> module_op =
      ParseModuleOpString(kModuleDotWeightOnlyPtq);
  ASSERT_TRUE(module_op);

  func::FuncOp main_fn = FindMainFuncOp(*module_op);
  ASSERT_THAT(main_fn, NotNull());

  auto call_op = *main_fn.getOps<TF::XlaCallModuleOp>().begin();
  EXPECT_TRUE(HasWeightOnlyPtqMethod(call_op));
}

TEST_F(LiftAsFunctionCallTest, HasWeightOnlyPtqMethodDifferentMethod) {
  const absl::string_view kModuleDotNoQuantization = R"mlir(
    module {
      func.func @main(%arg0: tensor<?x2xf32> {tf_saved_model.index_path = ["input_tensor"]}) -> (tensor<?x2xf32>) {
        %0 = stablehlo.constant dense<[-0.211145893, -0.708605706]> : tensor<2xf32>
        %1 = stablehlo.constant dense<[[-0.630731344, 0.54962182], [0.180364341, -0.764542698]]> : tensor<2x2xf32>
        %2 = "tf.XlaCallModule"(%arg0, %1, %0) <{Sout = [#tf_type.shape<?x2>], module = "", version = 9 : i64}> {_entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _tfl_quant_trait = "fully_quantizable", _quantization_method = "no_quantization { }"} : (tensor<?x2xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<?x2xf32>
        return %2 : tensor<?x2xf32>
      }
      func.func private @composite_dot_general_fn_1(%arg0: tensor<?x2xf32>, %arg1: tensor<2x2xf32>, %arg2: tensor<2xf32>) -> tensor<?x2xf32> attributes {_from_xla_call_module, tf_quant.composite_function} {
        %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<?x2xf32>, tensor<2x2xf32>) -> tensor<?x2xf32>
        return %0 : tensor<?x2xf32>
      }
    }
  )mlir";
  OwningOpRef<ModuleOp> module_op =
      ParseModuleOpString(kModuleDotNoQuantization);
  ASSERT_TRUE(module_op);

  func::FuncOp main_fn = FindMainFuncOp(*module_op);
  ASSERT_THAT(main_fn, NotNull());

  auto call_op = *main_fn.getOps<TF::XlaCallModuleOp>().begin();
  EXPECT_FALSE(HasWeightOnlyPtqMethod(call_op));
}

TEST_F(LiftAsFunctionCallTest, HasWeightOnlyPtqMethodNoMethod) {
  const absl::string_view kModuleXlaCallModule = R"mlir(
    module {
      func.func @main(%arg0: tensor<?x2xf32> {tf_saved_model.index_path = ["input_tensor"]}) -> (tensor<?x2xf32>) {
        %0 = stablehlo.constant dense<[-0.211145893, -0.708605706]> : tensor<2xf32>
        %1 = stablehlo.constant dense<[[-0.630731344, 0.54962182], [0.180364341, -0.764542698]]> : tensor<2x2xf32>
        %2 = "tf.XlaCallModule"(%arg0, %1, %0) <{Sout = [#tf_type.shape<?x2>], module = "", version = 9 : i64}> {_entry_function = @composite_fn_1, _stablehlo_version = "1.0.0", _original_entry_function = "composite_fn_1", _tfl_quant_trait = "fully_quantizable"} : (tensor<?x2xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<?x2xf32>
        return %2 : tensor<?x2xf32>
      }
      func.func private @composite_fn_1(%arg0: tensor<?x2xf32>, %arg1: tensor<2x2xf32>, %arg2: tensor<2xf32>) -> tensor<?x2xf32> attributes {_from_xla_call_module, tf_quant.composite_function} {
        return %arg0 : tensor<?x2xf32>
      }
    }
  )mlir";
  OwningOpRef<ModuleOp> module_op = ParseModuleOpString(kModuleXlaCallModule);
  ASSERT_TRUE(module_op);

  func::FuncOp main_fn = FindMainFuncOp(*module_op);
  ASSERT_THAT(main_fn, NotNull());

  auto call_op = *main_fn.getOps<TF::XlaCallModuleOp>().begin();
  EXPECT_FALSE(HasWeightOnlyPtqMethod(call_op));
}

TEST_F(LiftAsFunctionCallTest, IsWeightOnlyQuantizableOpDot) {
  OwningOpRef<ModuleOp> module_op =
      ParseModuleOpString(kModuleDotWeightOnlyPtq);
  ASSERT_TRUE(module_op);

  func::FuncOp main_fn = FindMainFuncOp(*module_op);
  ASSERT_THAT(main_fn, NotNull());

  auto call_op = *main_fn.getOps<TF::XlaCallModuleOp>().begin();
  EXPECT_TRUE(IsWeightOnlyQuantizableOp(*call_op));
}

TEST_F(LiftAsFunctionCallTest, IsWeightOnlyQuantizableOpNotTfXlaCallModuleOp) {
  const absl::string_view kModulePartitionedCallDot = R"mlir(
    module {
      func.func @main(%arg0: tensor<?x2xf32> {tf_saved_model.index_path = ["input_tensor"]}) -> (tensor<?x2xf32>) {
        %0 = stablehlo.constant dense<[-0.211145893, -0.708605706]> : tensor<2xf32>
        %1 = stablehlo.constant dense<[[-0.630731344, 0.54962182], [0.180364341, -0.764542698]]> : tensor<2x2xf32>
        %2 = "tf.PartitionedCall"(%arg0, %1, %0)  {_tfl_quant_trait = "fully_quantizable", config = "", config_proto = "", executor_type = "", f = @composite_dot_general_fn_1, _quantization_method = "weight_only_ptq { }"} : (tensor<?x2xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<?x2xf32>
        return %2 : tensor<?x2xf32>
      }
      func.func private @composite_dot_general_fn_1(%arg0: tensor<?x2xf32>, %arg1: tensor<2x2xf32>, %arg2: tensor<2xf32>) -> tensor<?x2xf32> attributes {_from_xla_call_module, tf_quant.composite_function} {
        %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<?x2xf32>, tensor<2x2xf32>) -> tensor<?x2xf32>
        return %0 : tensor<?x2xf32>
      }
    }
  )mlir";
  OwningOpRef<ModuleOp> module_op =
      ParseModuleOpString(kModulePartitionedCallDot);
  ASSERT_TRUE(module_op);

  func::FuncOp main_fn = FindMainFuncOp(*module_op);
  ASSERT_THAT(main_fn, NotNull());

  auto call_op = *main_fn.getOps<TF::PartitionedCallOp>().begin();
  EXPECT_FALSE(IsWeightOnlyQuantizableOp(*call_op));
}

TEST_F(LiftAsFunctionCallTest, IsWeightOnlyQuantizableOpNoConvNoDot) {
  constexpr absl::string_view kModuleXlaCallModule = R"mlir(
    module {
      func.func @main(%arg0: tensor<?x2xf32> {tf_saved_model.index_path = ["input_tensor"]}) -> (tensor<?x2xf32>) {
        %0 = stablehlo.constant dense<[-0.211145893, -0.708605706]> : tensor<2xf32>
        %1 = stablehlo.constant dense<[[-0.630731344, 0.54962182], [0.180364341, -0.764542698]]> : tensor<2x2xf32>
        %2 = "tf.XlaCallModule"(%arg0, %1, %0) <{Sout = [#tf_type.shape<?x2>], module = "", version = 9 : i64}> {_entry_function = @composite_fn_1, _stablehlo_version = "1.0.0", _original_entry_function = "composite_fn_1", _tfl_quant_trait = "fully_quantizable", _quantization_method = "weight_only_ptq { }"} : (tensor<?x2xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<?x2xf32>
        return %2 : tensor<?x2xf32>
      }
      func.func private @composite_fn_1(%arg0: tensor<?x2xf32>, %arg1: tensor<2x2xf32>, %arg2: tensor<2xf32>) -> tensor<?x2xf32> attributes {_from_xla_call_module, tf_quant.composite_function} {
        return %arg0 : tensor<?x2xf32>
      }
    }
  )mlir";
  OwningOpRef<ModuleOp> module_op = ParseModuleOpString(kModuleXlaCallModule);
  ASSERT_TRUE(module_op);

  func::FuncOp main_fn = FindMainFuncOp(*module_op);
  ASSERT_THAT(main_fn, NotNull());

  auto call_op = *main_fn.getOps<TF::XlaCallModuleOp>().begin();
  EXPECT_FALSE(IsWeightOnlyQuantizableOp(*call_op));
}

TEST_F(LiftAsFunctionCallTest, GetSortedFunctions) {
  constexpr absl::string_view kModuleXlaCallModule = R"mlir(
    module {
      func.func @conv_3_fn(%arg0: tensor<1x3x3x4xf32>) -> tensor<1x3x3x4xf32> {
        %0 = stablehlo.constant dense<2.000000e+00> : tensor<3x3x4x4xf32>
        %1 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x3x4xf32>, tensor<3x3x4x4xf32>) -> tensor<1x3x3x4xf32>
        %2 = stablehlo.convolution(%1, %0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x3x4xf32>, tensor<3x3x4x4xf32>) -> tensor<1x3x3x4xf32>
        func.return %2: tensor<1x3x3x4xf32>
      }

      func.func @conv_1_fn(%arg0: tensor<1x3x3x4xf32>) -> tensor<1x3x3x4xf32> {
        %0 = stablehlo.constant dense<2.000000e+00> : tensor<3x3x4x4xf32>
        %1 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x3x4xf32>, tensor<3x3x4x4xf32>) -> tensor<1x3x3x4xf32>
        %2 = stablehlo.convolution(%1, %0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x3x4xf32>, tensor<3x3x4x4xf32>) -> tensor<1x3x3x4xf32>
        func.return %2: tensor<1x3x3x4xf32>
      }

      func.func @conv_2_fn(%arg0: tensor<1x3x3x4xf32>) -> tensor<1x3x3x4xf32> {
        %0 = stablehlo.constant dense<2.000000e+00> : tensor<3x3x4x4xf32>
        %1 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x3x4xf32>, tensor<3x3x4x4xf32>) -> tensor<1x3x3x4xf32>
        %2 = stablehlo.convolution(%1, %0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x3x4xf32>, tensor<3x3x4x4xf32>) -> tensor<1x3x3x4xf32>
        func.return %2: tensor<1x3x3x4xf32>
      }
    }
  )mlir";
  OwningOpRef<ModuleOp> module_op = ParseModuleOpString(kModuleXlaCallModule);
  ASSERT_TRUE(module_op);

  SmallVector<func::FuncOp> funcs = GetSortedFunctions(*module_op);
  ASSERT_THAT(funcs, SizeIs(3));
  EXPECT_THAT(funcs[0].getSymName(), StrEq("conv_1_fn"));
  EXPECT_THAT(funcs[1].getSymName(), StrEq("conv_2_fn"));
  EXPECT_THAT(funcs[2].getSymName(), StrEq("conv_3_fn"));
}

}  // namespace
}  // namespace mlir::quant
