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

#include "tensorflow/compiler/mlir/quantization/tensorflow/utils/tf_to_uniform_attribute_utils.h"

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/IR/AsmState.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir::quant {
namespace {

using QuantMethod =
    tensorflow::quantization::QuantizationMethod::ExperimentalMethod;

class EmptyPatternRewriter : public mlir::PatternRewriter {
 public:
  explicit EmptyPatternRewriter(const OpBuilder& op_builder)
      : mlir::PatternRewriter(op_builder) {}
  ~EmptyPatternRewriter() override = default;
};

class TfToUniformAttributeUtilsTestPeer {
 public:
  explicit TfToUniformAttributeUtilsTestPeer() = delete;
  explicit TfToUniformAttributeUtilsTestPeer(MLIRContext* ctx)
      : rewriter_(OpBuilder(ctx)) {}

  EmptyPatternRewriter rewriter_;
};

class TfToUniformAttributeUtilsTest : public ::testing::Test {
 protected:
  TfToUniformAttributeUtilsTest() : ctx_() {
    ctx_.loadDialect<TF::TensorFlowDialect>();
  }

  MLIRContext ctx_;
};

TF::UniformQuantizedConvolutionOp ParseUniformQuantizedConvolutionOp(
    const absl::string_view conv_op_str, Block& block, MLIRContext& ctx) {
  const LogicalResult parse_result =
      parseSourceString(conv_op_str, &block, ParserConfig(&ctx));
  EXPECT_TRUE(succeeded(parse_result));

  auto uq_conv_op =
      dyn_cast_or_null<TF::UniformQuantizedConvolutionOp>(block.back());
  EXPECT_TRUE(uq_conv_op);

  return uq_conv_op;
}

TEST_F(TfToUniformAttributeUtilsTest, UniformQuantizedConvolutionOpAttributes) {
  TfToUniformAttributeUtilsTestPeer test_peer(&ctx_);

  constexpr absl::string_view kConvOpExpr =
      R"mlir(
      %0 = "tf.Const"() {value = #tf_type<tensor_proto : "0x746674656"> : tensor<1x3x4x3x!tf_type.qint8>} : () -> tensor<1x3x4x3x!tf_type.qint8>
      %1 = "tf.Const"() {value = #tf_type<tensor_proto : "0x746674656"> : tensor<2x3x3x2x!tf_type.qint8>} : () -> tensor<2x3x3x2x!tf_type.qint8>
      %2 = "tf.Const"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
      %3 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
      %4 = "tf.UniformQuantizedConvolution"(%0, %1, %2, %3, %2, %3, %2, %3) {Tin = "tfdtype$DT_QINT8", Tout = "tfdtype$DT_QINT32", attr_map = "", batch_group_count = 1 : i64, dimension_numbers = "", explicit_padding = [], feature_group_count = 1 : i64, lhs_dilation = [], lhs_quantization_axis = -1 : i64, lhs_quantization_max_val = 1 : i64, lhs_quantization_min_val = -1 : i64, output_quantization_axis = -1 : i64, output_quantization_max_val = 1 : i64, output_quantization_min_val = -1 : i64, padding = "SAME", rhs_dilation = [], rhs_quantization_axis = -1 : i64, rhs_quantization_max_val = 1 : i64, rhs_quantization_min_val = -1 : i64, window_strides = [1, 1]} : (tensor<1x3x4x3x!tf_type.qint8>, tensor<2x3x3x2x!tf_type.qint8>, tensor<f32>, tensor<i32>, tensor<f32>, tensor<i32>, tensor<f32>, tensor<i32>) -> tensor<1x3x4x2x!tf_type.qint32>
      )mlir";

  Block block{};
  TF::UniformQuantizedConvolutionOp op =
      ParseUniformQuantizedConvolutionOp(kConvOpExpr, block, ctx_);

  llvm::StringMap<Attribute> identifier_to_attr;
  QuantMethod quantization_method =
      tensorflow::quantization::QuantizationMethod::STATIC_RANGE;
  auto res = FillAttributesForUniformQuantizedAddOp(
      test_peer.rewriter_, op, identifier_to_attr, quantization_method,
      /*enable_per_channel_quantization=*/false);
  ASSERT_TRUE(succeeded(res));
  ASSERT_EQ(127, op.getLhsQuantizationMaxValAttr().getInt());
  ASSERT_EQ(-128, op.getLhsQuantizationMinValAttr().getInt());
  ASSERT_EQ(127, op.getRhsQuantizationMaxValAttr().getInt());
  ASSERT_EQ(-128, op.getRhsQuantizationMinValAttr().getInt());
  ASSERT_EQ(2147483647, op.getOutputQuantizationMaxValAttr().getInt());
  ASSERT_EQ(-2147483648, op.getOutputQuantizationMinValAttr().getInt());
}

}  // namespace
}  // namespace mlir::quant
