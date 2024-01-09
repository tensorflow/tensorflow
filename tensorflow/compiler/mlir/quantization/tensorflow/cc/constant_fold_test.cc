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
#include "tensorflow/compiler/mlir/quantization/tensorflow/cc/constant_fold.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/platform/test.h"

namespace mlir {
namespace quant {
namespace {

using ::testing::NotNull;
using ::testing::SizeIs;

class ConstantFoldingTest : public ::testing::Test {
 protected:
  ConstantFoldingTest() {
    ctx_.loadDialect<func::FuncDialect, TF::TensorFlowDialect>();
  }

  // Parses `module_op_str` to create a `ModuleOp`. Checks whether the created
  // module op is valid.
  OwningOpRef<ModuleOp> ParseModuleOpString(absl::string_view module_op_str) {
    auto module_op_ref = parseSourceString<ModuleOp>(module_op_str, &ctx_);
    EXPECT_TRUE(module_op_ref);
    return module_op_ref;
  }

  // Gets the function with the given name from the module.
  func::FuncOp GetFunctionFromModule(ModuleOp module,
                                     absl::string_view function_name) {
    SymbolTable symbol_table(module);
    return symbol_table.lookup<func::FuncOp>(function_name);
  }

  // Returns the first operation with the given type in the function.
  template <typename OpType>
  OpType FindOperationOfType(func::FuncOp function) {
    for (auto op : function.getBody().getOps<OpType>()) {
      return op;
    }
    return nullptr;
  }

  MLIRContext ctx_{};
};

TEST_F(ConstantFoldingTest, FoldLargeConstant) {
  constexpr absl::string_view kModuleCode = R"mlir(
    module {
      func.func @test_fold_constant() -> (tensor<1024x24x24x3xf32>) {
        %zp = "tf.Const"() {value = dense<2> : tensor<i32>} : () -> tensor<i32>
        %scale = "tf.Const"() {value = dense<2.0> : tensor<f32>} : () -> tensor<f32>
        %weight = "tf.Const"() {value = dense<1> : tensor<1024x24x24x3xi8>} : () -> tensor<1024x24x24x3xi8>
        %input_i32 = "tf.Cast"(%weight) : (tensor<1024x24x24x3xi8>) -> tensor<1024x24x24x3xi32>
        %output = "tf.Sub"(%input_i32, %zp) : (tensor<1024x24x24x3xi32>, tensor<i32>) -> tensor<1024x24x24x3xi32>
        %cast = "tf.Cast"(%output) : (tensor<1024x24x24x3xi32>) -> tensor<1024x24x24x3xf32>
        %mul = "tf.Mul"(%cast, %scale) : (tensor<1024x24x24x3xf32>, tensor<f32>) -> tensor<1024x24x24x3xf32>
        func.return %mul : tensor<1024x24x24x3xf32>
      }
    }
  )mlir";

  OwningOpRef<ModuleOp> module_op_ref = ParseModuleOpString(kModuleCode);
  func::FuncOp test_func =
      GetFunctionFromModule(*module_op_ref, "test_fold_constant");
  Operation* mul_op = FindOperationOfType<TF::MulOp>(test_func);
  SmallVector<Value> results = ConstantFoldOpIfPossible(mul_op);
  EXPECT_THAT(results, SizeIs(1));
  EXPECT_TRUE(isa<TF::ConstOp>(results[0].getDefiningOp()));
}

TEST_F(ConstantFoldingTest, NotFoldingIdentity) {
  constexpr absl::string_view kModuleCode = R"mlir(
    module {
      func.func @test_fold_constant() -> (tensor<1024x24x24x3xf32>) {
        %zp = "tf.Const"() {value = dense<2> : tensor<i32>} : () -> tensor<i32>
        %scale = "tf.Const"() {value = dense<2.0> : tensor<f32>} : () -> tensor<f32>
        %weight = "tf.Const"() {value = dense<1> : tensor<1024x24x24x3xi8>} : () -> tensor<1024x24x24x3xi8>
        %input_i32 = "tf.Cast"(%weight) : (tensor<1024x24x24x3xi8>) -> tensor<1024x24x24x3xi32>
        %output = "tf.Sub"(%input_i32, %zp) : (tensor<1024x24x24x3xi32>, tensor<i32>) -> tensor<1024x24x24x3xi32>
        %cast = "tf.Cast"(%output) : (tensor<1024x24x24x3xi32>) -> tensor<1024x24x24x3xf32>
        %identity = "tf.Identity"(%scale) : (tensor<f32>) -> tensor<f32>
        %mul = "tf.Mul"(%cast, %identity) : (tensor<1024x24x24x3xf32>, tensor<f32>) -> tensor<1024x24x24x3xf32>
        func.return %mul : tensor<1024x24x24x3xf32>
      }
    }
  )mlir";

  OwningOpRef<ModuleOp> module_op_ref = ParseModuleOpString(kModuleCode);
  func::FuncOp test_func =
      GetFunctionFromModule(*module_op_ref, "test_fold_constant");
  Operation* op_to_fold = FindOperationOfType<TF::MulOp>(test_func);
  SmallVector<Value> results = ConstantFoldOpIfPossible(op_to_fold);
  EXPECT_THAT(results, SizeIs(1));
  // No constant-folding since the IdentityOp has `TF_NoConstantFold` trait.
  auto mul_op = dyn_cast_or_null<TF::MulOp>(results[0].getDefiningOp());
  EXPECT_THAT(mul_op, NotNull());
  // Even though the preceding CastOp is foldable, it shouldn't be folded since
  // we are calling from the MulOp.
  EXPECT_TRUE(isa<TF::CastOp>(mul_op.getX().getDefiningOp()));
}

TEST_F(ConstantFoldingTest, NotFoldingArgument) {
  constexpr absl::string_view kModuleCode = R"mlir(
    module {
      func.func @test_fold_constant(%arg0: tensor<f32>) -> (tensor<1024x24x24x3xf32>) {
        %zp = "tf.Const"() {value = dense<2> : tensor<i32>} : () -> tensor<i32>
        %weight = "tf.Const"() {value = dense<1> : tensor<1024x24x24x3xi8>} : () -> tensor<1024x24x24x3xi8>
        %input_i32 = "tf.Cast"(%weight) : (tensor<1024x24x24x3xi8>) -> tensor<1024x24x24x3xi32>
        %output = "tf.Sub"(%input_i32, %zp) : (tensor<1024x24x24x3xi32>, tensor<i32>) -> tensor<1024x24x24x3xi32>
        %cast = "tf.Cast"(%output) : (tensor<1024x24x24x3xi32>) -> tensor<1024x24x24x3xf32>
        %mul = "tf.Mul"(%cast, %arg0) : (tensor<1024x24x24x3xf32>, tensor<f32>) -> tensor<1024x24x24x3xf32>
        func.return %mul : tensor<1024x24x24x3xf32>
      }
    }
  )mlir";

  OwningOpRef<ModuleOp> module_op_ref = ParseModuleOpString(kModuleCode);
  func::FuncOp test_func =
      GetFunctionFromModule(*module_op_ref, "test_fold_constant");
  Operation* op_to_fold = FindOperationOfType<TF::MulOp>(test_func);
  SmallVector<Value> results = ConstantFoldOpIfPossible(op_to_fold);
  EXPECT_THAT(results, SizeIs(1));
  // No constant-folding since the second operand is an argument.
  TF::MulOp mul_op = dyn_cast_or_null<TF::MulOp>(results[0].getDefiningOp());
  EXPECT_THAT(mul_op, NotNull());
  // Even though the preceding CastOp is foldable, it shouldn't be folded since
  // we are calling from the MulOp.
  EXPECT_TRUE(isa<TF::CastOp>(mul_op.getX().getDefiningOp()));
}

TEST_F(ConstantFoldingTest, FoldDepthwiseConvWeight) {
  constexpr absl::string_view kModuleCode = R"mlir(
    module {
      func.func @test_fold_constant(%arg0: tensor<*xf32>) -> (tensor<?x?x?x3xf32>) {
        %cst = "tf.Const"() {value = dense<2.000000e+00> : tensor<2x3x3x1xf32>} : () -> tensor<2x3x3x1xf32>
        %cst_0 = "tf.Const"() {value = dense<0.400000e+00> : tensor<3xf32>} : () -> tensor<3xf32>
        %cst_1 = "tf.Const"() {value = dense<0.500000e+00> : tensor<3xf32>} : () -> tensor<3xf32>
        %cst_2 = "tf.Const"() {value = dense<3.0> : tensor<f32>} : () -> tensor<f32>
        %w = "tf.Mul"(%cst, %cst_2) : (tensor<2x3x3x1xf32>, tensor<f32>) -> tensor<2x3x3x1xf32>
        %0 = "tf.DepthwiseConv2dNative"(%arg0, %w) {data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<*xf32>, tensor<2x3x3x1xf32>) -> tensor<?x?x?x3xf32>
        %1 = "tf.BiasAdd"(%0, %cst_0) {data_format = "NHWC"} : (tensor<?x?x?x3xf32>, tensor<3xf32>) -> tensor<?x?x?x3xf32>
        %2 = "tf.Mul"(%1, %cst_1) : (tensor<?x?x?x3xf32>, tensor<3xf32>) -> tensor<?x?x?x3xf32>
        func.return %2 : tensor<?x?x?x3xf32>
      }
    }
  )mlir";

  OwningOpRef<ModuleOp> module_op_ref = ParseModuleOpString(kModuleCode);
  func::FuncOp test_func =
      GetFunctionFromModule(*module_op_ref, "test_fold_constant");

  RewritePatternSet patterns(&ctx_);
  patterns.add<ConstantFoldQuantizableOperands>(&ctx_);
  EXPECT_TRUE(
      succeeded(applyPatternsAndFoldGreedily(test_func, std::move(patterns))));

  auto depthwise_conv_op =
      FindOperationOfType<TF::DepthwiseConv2dNativeOp>(test_func);
  EXPECT_THAT(depthwise_conv_op, NotNull());
  // The filter of the DepthwiseConv2dNativeOp is expected to be a constant.
  EXPECT_TRUE(isa<TF::ConstOp>(depthwise_conv_op.getFilter().getDefiningOp()));
}

TEST_F(ConstantFoldingTest, DepthwiseConvWeightNotFoldable) {
  constexpr absl::string_view kModuleCode = R"mlir(
    module {
      func.func @test_fold_constant(%arg0: tensor<*xf32>, %arg1: tensor<f32>) -> (tensor<?x?x?x3xf32>) {
        %cst = "tf.Const"() {value = dense<2.000000e+00> : tensor<2x3x3x1xf32>} : () -> tensor<2x3x3x1xf32>
        %cst_0 = "tf.Const"() {value = dense<0.400000e+00> : tensor<3xf32>} : () -> tensor<3xf32>
        %cst_1 = "tf.Const"() {value = dense<0.500000e+00> : tensor<3xf32>} : () -> tensor<3xf32>
        %w = "tf.Mul"(%cst, %arg1) : (tensor<2x3x3x1xf32>, tensor<f32>) -> tensor<2x3x3x1xf32>
        %0 = "tf.DepthwiseConv2dNative"(%arg0, %w) {data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<*xf32>, tensor<2x3x3x1xf32>) -> tensor<?x?x?x3xf32>
        %1 = "tf.BiasAdd"(%0, %cst_0) {data_format = "NHWC"} : (tensor<?x?x?x3xf32>, tensor<3xf32>) -> tensor<?x?x?x3xf32>
        %2 = "tf.Mul"(%1, %cst_1) : (tensor<?x?x?x3xf32>, tensor<3xf32>) -> tensor<?x?x?x3xf32>
        func.return %2 : tensor<?x?x?x3xf32>
      }
    }
  )mlir";

  OwningOpRef<ModuleOp> module_op_ref = ParseModuleOpString(kModuleCode);
  func::FuncOp test_func =
      GetFunctionFromModule(*module_op_ref, "test_fold_constant");

  RewritePatternSet patterns(&ctx_);
  patterns.add<ConstantFoldQuantizableOperands>(&ctx_);
  EXPECT_TRUE(
      succeeded(applyPatternsAndFoldGreedily(test_func, std::move(patterns))));

  auto depthwise_conv_op =
      FindOperationOfType<TF::DepthwiseConv2dNativeOp>(test_func);
  EXPECT_THAT(depthwise_conv_op, NotNull());
  // The filter of the DepthwiseConv2dNativeOp is not constant-foldable.
  EXPECT_TRUE(isa<TF::MulOp>(depthwise_conv_op.getFilter().getDefiningOp()));
}

}  // namespace
}  // namespace quant
}  // namespace mlir
