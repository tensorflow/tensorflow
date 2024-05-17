/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/quantization/tensorflow/cc/const_op_size.h"

#include "absl/strings/string_view.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/AsmState.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/platform/test.h"

namespace mlir {
namespace quant {
namespace {

using ::testing::Eq;

class GetSizeInBytesTest : public ::testing::Test {
 protected:
  GetSizeInBytesTest() : ctx_() { ctx_.loadDialect<TF::TensorFlowDialect>(); }

  MLIRContext ctx_;
};

TF::ConstOp ParseConstOp(const absl::string_view const_op_str, Block& block,
                         MLIRContext& ctx) {
  const LogicalResult parse_result =
      parseSourceString(const_op_str, &block, ParserConfig(&ctx));
  EXPECT_TRUE(succeeded(parse_result));

  auto const_op = dyn_cast_or_null<TF::ConstOp>(block.front());
  EXPECT_TRUE(const_op);

  return const_op;
}

TEST_F(GetSizeInBytesTest, Int32ScalarConstOpSizeInBytes) {
  constexpr absl::string_view kConstOpExpr =
      R"mlir(%cst = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>)mlir";

  Block block{};
  TF::ConstOp int_tensor_const_op = ParseConstOp(kConstOpExpr, block, ctx_);

  const int64_t num_bytes = GetSizeInBytes(int_tensor_const_op);
  EXPECT_THAT(num_bytes, Eq(4));
}

TEST_F(GetSizeInBytesTest, Int32ConstOpSizeInBytes) {
  constexpr absl::string_view kConstOpExpr =
      R"mlir(%cst = "tf.Const"() {value = dense<1> : tensor<2xi32>} : () -> tensor<2xi32>)mlir";

  Block block{};
  TF::ConstOp int_tensor_const_op = ParseConstOp(kConstOpExpr, block, ctx_);

  const int64_t num_bytes = GetSizeInBytes(int_tensor_const_op);
  EXPECT_THAT(num_bytes, Eq(8));
}

TEST_F(GetSizeInBytesTest, Int8ConstOpSizeInBytes) {
  constexpr absl::string_view kConstOpExpr =
      R"mlir(%cst = "tf.Const"() {value = dense<2> : tensor<3xi8>} : () -> tensor<3xi8>)mlir";

  Block block{};
  TF::ConstOp int_tensor_const_op = ParseConstOp(kConstOpExpr, block, ctx_);

  const int64_t num_bytes = GetSizeInBytes(int_tensor_const_op);
  EXPECT_THAT(num_bytes, Eq(3));
}

TEST_F(GetSizeInBytesTest, Float32ConstOpSizeInBytes) {
  constexpr absl::string_view kConstOpExpr =
      R"mlir(%cst = "tf.Const"() {value = dense<3.0> : tensor<4xf32>} : () -> tensor<4xf32>)mlir";

  Block block{};
  TF::ConstOp int_tensor_const_op = ParseConstOp(kConstOpExpr, block, ctx_);

  const int64_t num_bytes = GetSizeInBytes(int_tensor_const_op);
  EXPECT_THAT(num_bytes, Eq(16));
}

TEST_F(GetSizeInBytesTest, Float64ConstOpSizeInBytes) {
  constexpr absl::string_view kConstOpExpr =
      R"mlir(%cst = "tf.Const"() {value = dense<3.0> : tensor<2xf64>} : () -> tensor<2xf64>)mlir";

  Block block{};
  TF::ConstOp int_tensor_const_op = ParseConstOp(kConstOpExpr, block, ctx_);

  const int64_t num_bytes = GetSizeInBytes(int_tensor_const_op);
  EXPECT_THAT(num_bytes, Eq(16));
}

TEST_F(GetSizeInBytesTest, Bfloat16ConstOpSizeInBytes) {
  constexpr absl::string_view kConstOpExpr = R"mlir(
    %cst = "tf.Const"() {value = dense<1.0> : tensor<7xbf16>} : () -> tensor<7xbf16>
  )mlir";

  Block block{};
  TF::ConstOp int_tensor_const_op = ParseConstOp(kConstOpExpr, block, ctx_);

  const int64_t num_bytes = GetSizeInBytes(int_tensor_const_op);
  EXPECT_THAT(num_bytes, Eq(14));
}

TEST_F(GetSizeInBytesTest, TfStringConstOpSizeInBytes) {
  constexpr absl::string_view kConstOpExpr = R"mlir(
    %cst = "tf.Const"() {value = dense<["Hello World", "Quantization"]> : tensor<2x!tf_type.string>} : () -> tensor<2x!tf_type.string>
  )mlir";

  Block block{};
  TF::ConstOp int_tensor_const_op = ParseConstOp(kConstOpExpr, block, ctx_);

  // Sum of the number of characters in "Hello World" and "Quantization".
  const int64_t num_bytes = GetSizeInBytes(int_tensor_const_op);
  EXPECT_THAT(num_bytes, Eq(23));
}

TEST_F(GetSizeInBytesTest, ConstOpWithUnknownSizeAssumes4BytesPerElement) {
  constexpr absl::string_view kConstOpExpr = R"mlir(
    %cst = "tf.Const"() {value = #tf_type<tensor_proto : "0xDEADBAAD"> : tensor<!tf_type.variant>} : () -> tensor<!tf_type.variant>
  )mlir";

  Block block{};
  TF::ConstOp int_tensor_const_op = ParseConstOp(kConstOpExpr, block, ctx_);

  // For non-fixed size like tf_type.variant, the size of each element is
  // assumed to be 4 bytes.
  const int64_t num_bytes = GetSizeInBytes(int_tensor_const_op);
  EXPECT_THAT(num_bytes, Eq(4));
}

}  // namespace
}  // namespace quant
}  // namespace mlir
