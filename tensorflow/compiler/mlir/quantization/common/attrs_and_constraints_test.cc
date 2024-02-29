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

#include "tensorflow/compiler/mlir/quantization/common/attrs_and_constraints.h"

#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/quantization/common/func.h"
#include "tensorflow/compiler/mlir/quantization/common/test_base.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir::quant {
namespace {

using ::mlir::quant::QuantizationTestBase;
using ::mlir::stablehlo::AddOp;
using ::mlir::stablehlo::ConvolutionOp;
using ::mlir::stablehlo::DotGeneralOp;
using ::mlir::stablehlo::SubtractOp;
using ::testing::ElementsAreArray;
using ::testing::Eq;
using ::testing::NotNull;

class AttrsAndConstraintsTest : public QuantizationTestBase {};

constexpr absl::string_view kModuleStatic = R"mlir(
  module {
    func.func @main(%arg0: tensor<1x1024xf32>, %arg1: tensor<1024x3xf32>) -> tensor<1x3xf32> attributes {_from_xla_call_module} {
      %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [] : (tensor<1x1024xf32>, tensor<1024x3xf32>) -> tensor<1x3xf32>
      return %0 : tensor<1x3xf32>
    }
  }
)mlir";

constexpr absl::string_view kModuleDynamic = R"mlir(
  module {
    func.func @main(%arg0: tensor<?x1024xf32>, %arg1: tensor<1024x3xf32>) -> tensor<?x3xf32> attributes {_from_xla_call_module} {
      %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [] : (tensor<?x1024xf32>, tensor<1024x3xf32>) -> tensor<?x3xf32>
      return %0 : tensor<?x3xf32>
    }
  }
)mlir";

constexpr absl::string_view kModuleMultipleUses = R"mlir(
  module {
    func.func @main(%arg0: tensor<1x1024xf32>, %arg1: tensor<1024x3xf32>, %arg2: tensor<1x3xf32>) -> tensor<1x3xf32> attributes {_from_xla_call_module} {
      %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [] : (tensor<1x1024xf32>, tensor<1024x3xf32>) -> tensor<1x3xf32>
      %1 = stablehlo.subtract %0, %arg2 : tensor<1x3xf32>
      %2 = stablehlo.add %0, %arg2 : tensor<1x3xf32>
      return %2 : tensor<1x3xf32>
    }
  }
)mlir";

constexpr absl::string_view kModuleXlaCallModule = R"mlir(
  module {
    func.func @main(%arg0: tensor<?x2xf32> {tf_saved_model.index_path = ["input_tensor"]}) -> (tensor<?x2xf32>) {
      %0 = stablehlo.constant dense<[-0.211145893, -0.708605706]> : tensor<2xf32>
      %1 = stablehlo.constant dense<[[-0.630731344, 0.54962182], [0.180364341, -0.764542698]]> : tensor<2x2xf32>
      %2 = "tf.XlaCallModule"(%arg0, %1, %0) <{Sout = [#tf_type.shape<?x2>], module = "", version = 9 : i64}> {_entry_function = @composite_fn_1, _original_entry_function = "composite_fn_1", _tfl_quant_trait = "fully_quantizable"} : (tensor<?x2xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<?x2xf32>
      return %2 : tensor<?x2xf32>
    }
    func.func private @composite_fn_1(%arg0: tensor<?x2xf32>, %arg1: tensor<2x2xf32>, %arg2: tensor<2xf32>) -> tensor<?x2xf32> attributes {_from_xla_call_module, tf_quant.composite_function} {
      return %arg0 : tensor<?x2xf32>
    }
  }
)mlir";

constexpr absl::string_view kModuleXlaCallModuleNoEntryNoQuantTrait = R"mlir(
  module {
    func.func @main(%arg0: tensor<?x2xf32> {tf_saved_model.index_path = ["input_tensor"]}) -> (tensor<?x2xf32>) {
      %0 = stablehlo.constant dense<[-0.211145893, -0.708605706]> : tensor<2xf32>
      %1 = stablehlo.constant dense<[[-0.630731344, 0.54962182], [0.180364341, -0.764542698]]> : tensor<2x2xf32>
      %2 = "tf.XlaCallModule"(%arg0, %1, %0) <{Sout = [#tf_type.shape<?x2>], module = "", version = 9 : i64}> {_original_entry_function = "composite_fn_1"} : (tensor<?x2xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<?x2xf32>
      return %2 : tensor<?x2xf32>
    }
    func.func private @composite_fn_1(%arg0: tensor<?x2xf32>, %arg1: tensor<2x2xf32>, %arg2: tensor<2xf32>) -> tensor<?x2xf32> attributes {_from_xla_call_module, tf_quant.composite_function} {
      return %arg0 : tensor<?x2xf32>
    }
  }
)mlir";

constexpr absl::string_view kModulePartitionedCall = R"mlir(
  module {
    func.func @main(%arg0: tensor<2x2xf32> {tf_saved_model.index_path = ["input_tensor"]}) -> (tensor<2x2xf32>) {
      %cst = "tf.Const"() {device = "", value = dense<[[-0.630731344, 0.54962182], [0.180364341, -0.764542698]]> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
      %0 = "tf.PartitionedCall"(%arg0, %cst) {_tfl_quant_trait = "fully_quantizable", config = "", config_proto = "", executor_type = "", f = @composite_fn_1} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32> loc(callsite("test@main"("MatMul") at "QuantizationUnit(\12\06MatMul\1a\07main)"))
      return %0 : tensor<2x2xf32>
    }
    func.func private @composite_fn_1(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> attributes {tf_quant.composite_function} {
      %0 = "tf.MatMul"(%arg0, %arg1) {attr_map = "0:transpose_a,1:transpose_b", device = "", transpose_a = false, transpose_b = false} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
      return %0 : tensor<2x2xf32>
    }
  }
)mlir";

TEST_F(AttrsAndConstraintsTest, HasStaticShapeSucceedsWithStaticShapes) {
  OwningOpRef<ModuleOp> module_op_ref = ParseModuleOpString(kModuleStatic);
  func::FuncOp main_fn = FindMainFuncOp(*module_op_ref);
  ASSERT_THAT(main_fn, NotNull());

  Value dot_general_result =
      FindOperationOfType<DotGeneralOp>(main_fn)->getResult(0);
  EXPECT_TRUE(HasStaticShape(dot_general_result));
  EXPECT_TRUE(HasStaticShapeAtDims(dot_general_result, /*dims=*/{0}));
  EXPECT_TRUE(HasStaticShapeAtDims(dot_general_result, /*dims=*/{1}));
}

TEST_F(AttrsAndConstraintsTest, HasStaticShapeFailsWithDynamicShapes) {
  OwningOpRef<ModuleOp> module_op_ref = ParseModuleOpString(kModuleDynamic);
  func::FuncOp main_fn = FindMainFuncOp(*module_op_ref);
  ASSERT_THAT(main_fn, NotNull());

  Value dot_general_result =
      FindOperationOfType<DotGeneralOp>(main_fn)->getResult(0);
  EXPECT_FALSE(HasStaticShape(dot_general_result));
  EXPECT_FALSE(HasStaticShapeAtDims(dot_general_result, /*dims=*/{0}));
  EXPECT_TRUE(HasStaticShapeAtDims(dot_general_result, /*dims=*/{1}));
}

TEST_F(AttrsAndConstraintsTest, HasRankOfReturnsTrueForMatchingRank) {
  constexpr absl::string_view kConstantOpWithRankFour =
      R"mlir(%0 = stablehlo.constant dense<0> : tensor<1x1x1x1xi8>)mlir";
  OwningOpRef<ModuleOp> module_op_ref =
      ParseModuleOpString(kConstantOpWithRankFour);

  ASSERT_FALSE(module_op_ref->getBodyRegion().empty());
  ASSERT_FALSE(module_op_ref->getBodyRegion().front().empty());

  auto constant_op = dyn_cast_or_null<mlir::stablehlo::ConstantOp>(
      module_op_ref->getBodyRegion().front().front());
  ASSERT_THAT(constant_op, NotNull());

  EXPECT_TRUE(HasRankOf(constant_op, /*rank=*/4));
}

TEST_F(AttrsAndConstraintsTest, HasRankOfReturnsFalseForNonMatchingRank) {
  constexpr absl::string_view kConstantOpWithRankFour =
      R"mlir(%0 = stablehlo.constant dense<0> : tensor<1x1x1x1xi8>)mlir";
  OwningOpRef<ModuleOp> module_op_ref =
      ParseModuleOpString(kConstantOpWithRankFour);

  ASSERT_FALSE(module_op_ref->getBodyRegion().empty());
  ASSERT_FALSE(module_op_ref->getBodyRegion().front().empty());

  auto constant_op = dyn_cast_or_null<mlir::stablehlo::ConstantOp>(
      module_op_ref->getBodyRegion().front().front());
  ASSERT_THAT(constant_op, NotNull());

  EXPECT_FALSE(HasRankOf(constant_op, /*rank=*/3));
}

TEST_F(AttrsAndConstraintsTest,
       HasRankOfReturnsTrueForMatchingRankWithUnknownDimensions) {
  // Argument has rank 2, but its dimensions are unknown.
  constexpr absl::string_view kArgumentWithUnknownDims = R"mlir(
    func.func @unknown_dims_arg(%arg: tensor<?x?xi8>) -> tensor<?x?xi8> {
      return %arg : tensor<?x?xi8>
    }
  )mlir";

  OwningOpRef<ModuleOp> module_op_ref =
      ParseModuleOpString(kArgumentWithUnknownDims);

  auto func_op = module_op_ref->lookupSymbol<func::FuncOp>("unknown_dims_arg");
  ASSERT_THAT(func_op, NotNull());
  ASSERT_THAT(func_op.getNumArguments(), Eq(1));

  EXPECT_TRUE(HasRankOf(func_op.getArgument(0), /*rank=*/2));
}

TEST_F(AttrsAndConstraintsTest, HasRankOfReturnsFalseForUnknownRank) {
  constexpr absl::string_view kArgumentWithUnknownRank = R"mlir(
    func.func @unknown_rank_arg(%arg: tensor<*xi8>) -> tensor<*xi8> {
      return %arg : tensor<*xi8>
    }
  )mlir";

  OwningOpRef<ModuleOp> module_op_ref =
      ParseModuleOpString(kArgumentWithUnknownRank);

  auto func_op = module_op_ref->lookupSymbol<func::FuncOp>("unknown_rank_arg");
  ASSERT_THAT(func_op, NotNull());
  ASSERT_THAT(func_op.getNumArguments(), Eq(1));

  EXPECT_FALSE(HasRankOf(func_op.getArgument(0), /*rank=*/1));
}

TEST_F(AttrsAndConstraintsTest, TryCastSucceeds) {
  OwningOpRef<ModuleOp> module_op_ref = ParseModuleOpString(kModuleStatic);
  func::FuncOp main_fn = FindMainFuncOp(*module_op_ref);
  ASSERT_THAT(main_fn, NotNull());

  Operation* dot_general_op = FindOperationOfType<DotGeneralOp>(main_fn);
  EXPECT_TRUE(succeeded(
      TryCast<DotGeneralOp>(dot_general_op, /*name=*/"dot_general_op")));
}

TEST_F(AttrsAndConstraintsTest, TryCastFailsOnWrongType) {
  OwningOpRef<ModuleOp> module_op_ref = ParseModuleOpString(kModuleStatic);
  func::FuncOp main_fn = FindMainFuncOp(*module_op_ref);
  ASSERT_THAT(main_fn, NotNull());

  Operation* dot_general_op = FindOperationOfType<DotGeneralOp>(main_fn);
  EXPECT_TRUE(
      failed(TryCast<AddOp>(dot_general_op, /*name=*/"dot_general_op")));
}

TEST_F(AttrsAndConstraintsTest, TryCastFailsOnNullPtr) {
  OwningOpRef<ModuleOp> module_op_ref = ParseModuleOpString(kModuleStatic);
  func::FuncOp main_fn = FindMainFuncOp(*module_op_ref);
  ASSERT_THAT(main_fn, NotNull());

  Operation* op_nullptr =
      FindOperationOfType<DotGeneralOp>(main_fn)->getNextNode()->getNextNode();
  // getNextNode() returns a nullptr if at the very last node.
  EXPECT_EQ(op_nullptr, nullptr);
  EXPECT_TRUE(failed(TryCast<DotGeneralOp>(op_nullptr, /*name=*/"op_nullptr")));
  EXPECT_TRUE(failed(TryCast<DotGeneralOp>(nullptr, /*name=*/"nullptr")));
}

TEST_F(AttrsAndConstraintsTest, I64ValueInI32RangeAreCastedCorrectly) {
  EXPECT_TRUE(succeeded(CastI64ToI32(llvm::minIntN(32))));
  EXPECT_TRUE(succeeded(CastI64ToI32(llvm::maxIntN(32))));
}

TEST_F(AttrsAndConstraintsTest, CastingFailsForI64ValueOutOfI32Range) {
  EXPECT_TRUE(failed(CastI64ToI32(llvm::minIntN(32) - 10)));
  EXPECT_TRUE(failed(CastI64ToI32(llvm::maxIntN(32) + 10)));
}

TEST_F(AttrsAndConstraintsTest, I64ArrayInI32RangeAreCastedCorrectly) {
  const SmallVector<int64_t> array_i64 = {llvm::minIntN(32), -2, -1, 0, 1, 2,
                                          llvm::maxIntN(32)};

  FailureOr<SmallVector<int32_t>> array_i32 = CastI64ArrayToI32(array_i64);
  EXPECT_TRUE(succeeded(array_i32));
  EXPECT_THAT(
      *array_i32,
      ElementsAreArray({static_cast<int32_t>(llvm::minIntN(32)), -2, -1, 0, 1,
                        2, static_cast<int32_t>(llvm::maxIntN(32))}));
}

TEST_F(AttrsAndConstraintsTest, CastingFailsForI64ArrayUnderI32Range) {
  const int64_t under_min_i32 = -2147483658;
  ArrayRef<int64_t> array_i64{under_min_i32};
  EXPECT_EQ(under_min_i32, llvm::minIntN(32) - 10);
  EXPECT_TRUE(failed(CastI64ArrayToI32(array_i64)));
}

TEST_F(AttrsAndConstraintsTest, CastingFailsForI64ArrayAboveI32Range) {
  const int64_t below_max_i32 = 2147483657;
  ArrayRef<int64_t> array_i64{below_max_i32};
  EXPECT_EQ(below_max_i32, llvm::maxIntN(32) + 10);
  EXPECT_TRUE(failed(CastI64ArrayToI32(array_i64)));
}

TEST_F(AttrsAndConstraintsTest, FindUserOfDifferentTypes) {
  OwningOpRef<ModuleOp> module_op_ref =
      ParseModuleOpString(kModuleMultipleUses);
  func::FuncOp main_fn = FindMainFuncOp(*module_op_ref);
  ASSERT_THAT(main_fn, NotNull());

  Operation* dot_general_op = FindOperationOfType<DotGeneralOp>(main_fn);
  ASSERT_NE(FindUserOfType<AddOp>(dot_general_op), nullptr);
  ASSERT_NE(FindUserOfType<SubtractOp>(dot_general_op), nullptr);
  ASSERT_NE(FindUserOfType<>(dot_general_op), nullptr);
  ASSERT_EQ(FindUserOfType<ConvolutionOp>(dot_general_op), nullptr);
}

TEST_F(AttrsAndConstraintsTest, CallGetFuncAttr) {
  OwningOpRef<ModuleOp> xla_module_op_ref =
      ParseModuleOpString(kModuleXlaCallModule);
  func::FuncOp xml_main_fn = FindMainFuncOp(*xla_module_op_ref);
  Operation* xla_op = FindOperationOfType<TF::XlaCallModuleOp>(xml_main_fn);
  auto xla_call_op = dyn_cast_or_null<TF::XlaCallModuleOp>(*xla_op);
  FlatSymbolRefAttr xla_call_op_attr = GetFuncAttr(xla_call_op);
  EXPECT_EQ(xla_call_op_attr.getValue(), "composite_fn_1");

  OwningOpRef<ModuleOp> partitioned_module_op_ref =
      ParseModuleOpString(kModulePartitionedCall);
  func::FuncOp partitioned_main_fn = FindMainFuncOp(*partitioned_module_op_ref);
  Operation* partitioned_op =
      FindOperationOfType<TF::PartitionedCallOp>(partitioned_main_fn);
  auto partitioned_call_op =
      dyn_cast_or_null<TF::PartitionedCallOp>(*partitioned_op);
  FlatSymbolRefAttr partitioned_call_op_attr = GetFuncAttr(partitioned_call_op);
  EXPECT_EQ(partitioned_call_op_attr.getValue(), "composite_fn_1");
}

TEST_F(AttrsAndConstraintsTest, GetEntryFunctionNameCorrectly) {
  OwningOpRef<ModuleOp> xla_module_op_ref =
      ParseModuleOpString(kModuleXlaCallModule);
  func::FuncOp xml_main_fn = FindMainFuncOp(*xla_module_op_ref);
  Operation* xla_op = FindOperationOfType<TF::XlaCallModuleOp>(xml_main_fn);
  auto xla_call_op = dyn_cast_or_null<TF::XlaCallModuleOp>(*xla_op);

  EXPECT_EQ(GetEntryFunctionName(xla_call_op), StringRef("composite_fn_1"));
}

TEST_F(AttrsAndConstraintsTest, GetEntryFunctionNameWhenNotSet) {
  OwningOpRef<ModuleOp> xla_module_op_ref =
      ParseModuleOpString(kModuleXlaCallModuleNoEntryNoQuantTrait);
  func::FuncOp xml_main_fn = FindMainFuncOp(*xla_module_op_ref);
  Operation* xla_op = FindOperationOfType<TF::XlaCallModuleOp>(xml_main_fn);
  auto xla_call_op = dyn_cast_or_null<TF::XlaCallModuleOp>(*xla_op);

  EXPECT_EQ(GetEntryFunctionName(xla_call_op), StringRef());
}

TEST_F(AttrsAndConstraintsTest, HasQuantizableTraitTrue) {
  OwningOpRef<ModuleOp> xla_module_op_ref =
      ParseModuleOpString(kModuleXlaCallModule);
  func::FuncOp xml_main_fn = FindMainFuncOp(*xla_module_op_ref);
  Operation* xla_op = FindOperationOfType<TF::XlaCallModuleOp>(xml_main_fn);
  auto xla_call_op = dyn_cast_or_null<TF::XlaCallModuleOp>(*xla_op);

  EXPECT_TRUE(HasQuantizableTrait(xla_call_op));
}

TEST_F(AttrsAndConstraintsTest, HasQuantizableTraitFalse) {
  OwningOpRef<ModuleOp> xla_module_op_ref =
      ParseModuleOpString(kModuleXlaCallModuleNoEntryNoQuantTrait);
  func::FuncOp xml_main_fn = FindMainFuncOp(*xla_module_op_ref);
  Operation* xla_op = FindOperationOfType<TF::XlaCallModuleOp>(xml_main_fn);
  auto xla_call_op = dyn_cast_or_null<TF::XlaCallModuleOp>(*xla_op);

  EXPECT_FALSE(HasQuantizableTrait(xla_call_op));
}

}  // namespace
}  // namespace mlir::quant
