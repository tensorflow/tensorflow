/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/common_shape_fns.h"

#include <string>

#include <gtest/gtest.h>
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace shape_inference {

namespace {

PartialTensorShape S(std::initializer_list<int64_t> dims) {
  return PartialTensorShape(dims);
}

PartialTensorShape Unknown() { return PartialTensorShape(); }

OpDef MakeOpDef(int num_inputs, int num_outputs) {
  OpRegistrationData op_reg_data;
  OpDefBuilder b("dummy");
  for (int i = 0; i < num_inputs; ++i) {
    b.Input(absl::StrCat("i", i, ": float"));
  }
  for (int i = 0; i < num_outputs; ++i) {
    b.Output(absl::StrCat("o", i, ": float"));
  }
  CHECK(b.Attr("foo:string").Finalize(&op_reg_data).ok());
  return op_reg_data.op_def;
}

}  // namespace

TEST(CommonShapeFnsTest, NoOutputShapeTest) {
  OpRegistrationData op_reg_data;
  TF_CHECK_OK(OpDefBuilder("Assert")
                  .Input("condition: bool")
                  .Input("data: float")
                  .Finalize(&op_reg_data));
  OpDef op_def = op_reg_data.op_def;

  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("test", "Assert")
                  .Input("condition", 0, DT_BOOL)
                  .Input({{"data", 0, DT_FLOAT}})
                  .Finalize(&def));

  InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def, {S({}), S({10})}, {},
                     {}, {});
  TF_EXPECT_OK(NoOutputs(&c));
  EXPECT_EQ(0, c.num_outputs());
}

TEST(CommonShapeFnsTest, ScalarShapeTest) {
  OpRegistrationData op_reg_data;
  TF_CHECK_OK(OpDefBuilder("L2Loss")
                  .Input("t: float")
                  .Output("t: float")
                  .Finalize(&op_reg_data));
  OpDef op_def = op_reg_data.op_def;

  NodeDef def;
  TF_CHECK_OK(
      NodeDefBuilder("test", "L2Loss").Input("t", 0, DT_FLOAT).Finalize(&def));

  {
    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def, {S({})}, {}, {}, {});
    TF_EXPECT_OK(ScalarShape(&c));
    ShapeHandle output = c.output(0);
    EXPECT_EQ(0, c.Rank(output));
  }

  {
    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def, {S({1, 23, 4, 4, 2})},
                       {}, {}, {});
    TF_EXPECT_OK(ScalarShape(&c));
    ShapeHandle output = c.output(0);
    EXPECT_EQ(0, c.Rank(output));
  }
}

TEST(CommonShapeFnsTest, MatMulShapeTest) {
  OpRegistrationData op_reg_data;
  TF_CHECK_OK(OpDefBuilder("MatMul")
                  .Input("a: float")
                  .Input("b: float")
                  .Output("c: float")
                  .Attr("transpose_a:bool=false")
                  .Attr("transpose_b:bool=false")
                  .Finalize(&op_reg_data));
  OpDef op_def = op_reg_data.op_def;

  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("test", "MatMul")
                  .Input("a", 0, DT_FLOAT)
                  .Input("b", 0, DT_FLOAT)
                  .Attr("transpose_a", false)
                  .Attr("transpose_b", false)
                  .Finalize(&def));

  {
    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def,
                       {S({2, 3}), S({3, 4})}, {}, {}, {});
    TF_EXPECT_OK(MatMulShape(&c));
    ShapeHandle output = c.output(0);
    EXPECT_EQ(2, c.Value(c.Dim(output, 0)));
    EXPECT_EQ(4, c.Value(c.Dim(output, 1)));
  }

  {
    // Unknown inner dimension for one
    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def,
                       {S({2, -1}), S({3, 4})}, {}, {}, {});
    TF_EXPECT_OK(MatMulShape(&c));
    ShapeHandle output = c.output(0);
    EXPECT_EQ(2, c.Value(c.Dim(output, 0)));
    EXPECT_EQ(4, c.Value(c.Dim(output, 1)));
  }

  {
    // Invalid rank.
    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def, {S({2}), S({3, 4})},
                       {}, {}, {});
    auto s = MatMulShape(&c);
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code(), error::INVALID_ARGUMENT);
    EXPECT_TRUE(
        absl::StrContains(s.message(), "Shape must be rank 2 but is rank 1"));
  }

  {
    // Unknown outer dimension
    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def,
                       {S({2, 3}), S({3, -1})}, {}, {}, {});
    TF_EXPECT_OK(MatMulShape(&c));
    ShapeHandle output = c.output(0);
    EXPECT_EQ(2, c.Value(c.Dim(output, 0)));
    EXPECT_FALSE(c.ValueKnown(c.Dim(output, 1)));
  }

  {
    // Inner shapes not compatible
    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def,
                       {S({2, 5}), S({3, 4})}, {}, {}, {});
    auto s = MatMulShape(&c);
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code(), error::INVALID_ARGUMENT);
    EXPECT_TRUE(absl::StrContains(s.message(),
                                  "Dimensions must be equal, but are 5 and 3"));
  }

  {
    // Inner shapes not compatible
    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def,
                       {S({2, 5, 3}), S({3, 5, 4})}, {}, {}, {});
    auto s = MatMulShape(&c);
    EXPECT_EQ(s.code(), error::INVALID_ARGUMENT);
    EXPECT_TRUE(
        absl::StrContains(s.message(), "Shape must be rank 2 but is rank 3"));
  }

  {
    // transpose_a
    TF_CHECK_OK(NodeDefBuilder("test", "MatMul")
                    .Input("a", 0, DT_FLOAT)
                    .Input("b", 0, DT_FLOAT)
                    .Attr("transpose_a", true)
                    .Attr("transpose_b", false)
                    .Attr("type", DT_FLOAT)
                    .Finalize(&def));

    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def,
                       {S({3, 2}), S({3, 4})}, {}, {}, {});
    auto s = MatMulShape(&c);
    ShapeHandle output = c.output(0);
    EXPECT_EQ(2, c.Value(c.Dim(output, 0)));
    EXPECT_EQ(4, c.Value(c.Dim(output, 1)));
  }

  {
    // transpose_b
    TF_CHECK_OK(NodeDefBuilder("test", "MatMul")
                    .Input("a", 0, DT_FLOAT)
                    .Input("b", 0, DT_FLOAT)
                    .Attr("transpose_a", false)
                    .Attr("transpose_b", true)
                    .Attr("type", DT_FLOAT)
                    .Finalize(&def));

    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def,
                       {S({2, 3}), S({4, 3})}, {}, {}, {});
    auto s = MatMulShape(&c);
    ShapeHandle output = c.output(0);
    EXPECT_EQ(2, c.Value(c.Dim(output, 0)));
    EXPECT_EQ(4, c.Value(c.Dim(output, 1)));
  }
}

TEST(CommonShapeFnsTest, Einsum_ShapeFn) {
  ShapeInferenceTestOp op("Einsum");
  auto set_equation = [&op](int n, string equation) {
    std::vector<NodeDefBuilder::NodeOut> input_list;
    input_list.reserve(n);
    for (int i = 0; i < n; ++i) {
      input_list.emplace_back("a", 0, DT_FLOAT);
    }
    TF_ASSERT_OK(NodeDefBuilder("test", "Einsum")
                     .Input(input_list)
                     .Attr("equation", equation)
                     .Finalize(&op.node_def));
  };

  // Unary cases.
  set_equation(1, "abc->c");
  INFER_OK(op, "[?,?,?]", "[d0_2]");
  set_equation(1, "abc->aabbcc");
  INFER_OK(op, "[?,?,?]", "[d0_0,d0_0,d0_1,d0_1,d0_2,d0_2]");
  set_equation(1, "abc->");
  INFER_OK(op, "[?,?,?]", "[]");
  set_equation(1, "->");
  INFER_OK(op, "[]", "[]");

  // Binary cases.
  set_equation(2, "ij,jk->ik");
  INFER_OK(op, "[?,?];[?,?]", "[d0_0,d1_1]");
  set_equation(2, "ij,jk->ik");
  INFER_OK(op, "[?,?];[?,?]", "[d0_0,d1_1]");
  set_equation(2, "ab,ab->");
  INFER_OK(op, "[?,?];[?,?]", "[]");
  set_equation(2, "ab,->b");
  INFER_OK(op, "[?,?];[]", "[d0_1]");
  set_equation(2, ",->");
  INFER_OK(op, "[];[]", "[]");
  set_equation(2, "aaa,b->abbb");
  INFER_OK(op, "[?,?,?];[?]", "[d0_0,d1_0,d1_0,d1_0]");
  set_equation(2, ",abcd->badc");
  INFER_OK(op, "[];[?,?,?,?]", "[d1_1,d1_0,d1_3,d1_2]");

  // Ellipsis cases.
  set_equation(1, "a...bc->c...");
  INFER_OK(op, "[?,?,?,?,?]", "[d0_4,d0_1,d0_2]");
  set_equation(2, "...ij,...jk->...ik");
  INFER_OK(op, "[?,?,?,?,?];[1,?,?]", "[d0_0,d0_1,d0_2,d0_3,d1_2]");
  INFER_OK(op, "[1,?,?];[?,?,?,?,?]", "[d1_0,d1_1,d1_2,d0_1,d1_4]");

  // Unknown rank.
  set_equation(1, "abc->c");
  INFER_OK(op, "?", "[?]");
  set_equation(1, "a...bc->c");
  INFER_OK(op, "?", "[?]");
  set_equation(1, "a...bc->c...");
  INFER_OK(op, "?", "?");

  set_equation(2, "...ij,...jk->...ik");
  INFER_OK(op, "?;?", "?");
  INFER_OK(op, "[?,?,?];?", "?");
  INFER_OK(op, "?;[?,?,?]", "?");
  set_equation(2, "...ij,...jk->ik");
  INFER_OK(op, "?;?", "[?,?]");
  set_equation(2, "abd,b...c->...cad");
  INFER_OK(op, "[?,?,?];[?,?,?,?]", "[d1_1,d1_2,d1_3,d0_0,d0_2]");
  set_equation(2, "...ab,b...c->ac...");
  INFER_OK(op, "[?,1,?,?];[?,?,?]", "[d0_2,d1_2,d0_0,d1_1]");

  // Wrong number of inputs.
  set_equation(2, "ab->b");
  INFER_ERROR("got: 2", op, "[?,?];[?,?]");
  set_equation(1, "ab,a->b");
  INFER_ERROR("got: 1", op, "[?,?]");

  // Invalid format. Implicit form is not supported.
  set_equation(1, "a");
  INFER_ERROR("equation", op, "[2]");
  set_equation(2, "ab,bc");
  INFER_ERROR("equation", op, "[2,2];[2,2]");

  // Wrong number of ellipsis or periods outside of ellipsis.
  set_equation(1, "..a.->a...");
  INFER_ERROR("ellipsis", op, "[1,1,2,1]");
  set_equation(1, "...a->.a..");
  INFER_ERROR("ellipsis", op, "[1,1,1,2]");
  set_equation(1, "...a...->...a");
  INFER_ERROR("ellipsis", op, "[1,1,1,2]");
  set_equation(1, "..a..b..->...ab");
  INFER_ERROR("ellipsis", op, "[1,1,2,1]");
  set_equation(2, "...a...,ab->a");
  INFER_ERROR("ellipsis", op, "[1,2,1];[2,1]");
  set_equation(2, "a,...ab...->a");
  INFER_ERROR("ellipsis", op, "[2];[1,2,1,1]");
  set_equation(2, "a,ab->a......");
  INFER_ERROR("ellipsis", op, "[2];[2,1]");

  // Output label doesn't appear in input.
  set_equation(1, "abc->d");
  INFER_ERROR("'d'", op, "[?,?,?]");

  // Mismatch in input rank.
  set_equation(1, "abc->c");
  INFER_ERROR("4", op, "[?,?,?,?]");
  INFER_ERROR("2", op, "[?,?]");
  set_equation(1, "...abc->...c");
  INFER_ERROR("2", op, "[?,?]");

  // Input dimensions are not consistent.
  set_equation(2, "ab,ab->a");
  INFER_ERROR("are 1 and 2", op, "[1,2];[2,1]");
  set_equation(2, "aa,bb->a");
  INFER_ERROR("are 1 and 2", op, "[1,2];[2,2]");

  // Invalid broadcasting dimensions.
  set_equation(2, "...ij,...jk->...ik");
  INFER_ERROR("are 2 and 3", op, "[2,?,?];[3,?,?]");
  set_equation(2, "i...j,jk...->...ik");
  INFER_ERROR("are 2 and 3", op, "[?,2,?];[?,?,3]");
  set_equation(2, "...ij,...jk->ik");
  set_equation(2, "i...j,jk...->ik");
  INFER_ERROR("non-empty broadcasting", op, "[?,2,?];[?,?]");
  set_equation(2, "...ab,b...c->ac...");
  INFER_OK(op, "?;[4,5,3]", "?");
}

TEST(CommonShapeFnsTest, BatchMatMulV2_ShapeFn) {
  ShapeInferenceTestOp op("BatchMatMulV2");
  auto set_adj = [&op](bool adj_x, bool adj_y) {
    TF_ASSERT_OK(NodeDefBuilder("test", "BatchMatMulV2")
                     .Input({"a", 0, DT_FLOAT})
                     .Input({"b", 0, DT_FLOAT})
                     .Attr("adj_x", adj_x)
                     .Attr("adj_y", adj_y)
                     .Finalize(&op.node_def));
  };

  set_adj(false, false);

  // Rank checks.
  INFER_ERROR("at least rank 2", op, "[];?");
  INFER_ERROR("at least rank 2", op, "[1];?");
  INFER_ERROR("at least rank 2", op, "?;[]");
  INFER_ERROR("at least rank 2", op, "?;[2]");

  INFER_OK(op, "?;?", "?");

  // 0 batch dims.
  INFER_OK(op, "[?,?];[?,?]", "[d0_0,d1_1]");

  // 1 batch dims.
  INFER_OK(op, "[3,?,?];[3,?,?]", "[d0_0,d0_1,d1_2]");
  INFER_OK(op, "[?,?,?];[1,?,?]", "[d0_0,d0_1,d1_2]");
  INFER_OK(op, "[?,?,?];[2,?,?]", "[d1_0,d0_1,d1_2]");
  INFER_OK(op, "[1,?,?];[?,?,?]", "[d1_0,d0_1,d1_2]");
  INFER_OK(op, "[2,?,?];[?,?,?]", "[d0_0,d0_1,d1_2]");
  INFER_OK(op, "[?,?,?];[?,?,?]", "[?,d0_1,d1_2]");

  // Empty batch dim with broadcasting.
  INFER_OK(op, "[?,?];[?,?,?]", "[d1_0,d0_0,d1_2]");
  INFER_OK(op, "[?,?,?];[?,?]", "[d0_0,d0_1,d1_1]");
  INFER_OK(op, "[?,?];[?,?,?,?]", "[d1_0,d1_1,d0_0,d1_3]");
  INFER_OK(op, "[?,?,?,?];[?,?]", "[d0_0,d0_1,d0_2,d1_1]");

  // Unknown number of batch dims.
  INFER_OK(op, "[?,?];?", "?");
  INFER_OK(op, "?;[?,?]", "?");
  INFER_OK(op, "[?,?,?,?];?", "?");

  // Large number of batch dims.
  INFER_OK(op, "[?,?,?,?,?];[1,?,?]", "[d0_0,d0_1,d0_2,d0_3,d1_2]");
  INFER_OK(op, "[1,?,?];[?,?,?,?,?]", "[d1_0,d1_1,d1_2,d0_1,d1_4]");

  // Batch dim mismatch.
  INFER_ERROR("are 2 and 3", op, "[?,?,2,?,?];[3,?,?]");
  INFER_ERROR("are 2 and 3", op, "[2,?,?];[?,?,3,?,?]");

  // Test adj_a, testing output and that inner dims are compared.
  set_adj(false, false);
  INFER_OK(op, "[2,2,3,4];[2,2,?,?]", "[d0_0,d0_1,d0_2,d1_3]");
  INFER_ERROR("are 2 and 3", op, "[?,1,2];[?,3,1]");  // inner dim mismatch
  set_adj(true, false);
  INFER_OK(op, "[2,2,3,4];[2,2,?,?]", "[d0_0,d0_1,d0_3,d1_3]");
  INFER_ERROR("are 2 and 3", op, "[?,2,1];[?,3,1]");  // inner dim mismatch

  // Test adj_b=true.
  set_adj(false, true);
  INFER_OK(op, "[2,2,?,?];[2,2,3,4]", "[d0_0,d0_1,d0_2,d1_2]");
  INFER_ERROR("are 2 and 3", op, "[?,1,2];[?,1,3]");  // inner dim mismatch
  set_adj(true, true);
  INFER_OK(op, "[2,2,?,?];[2,2,3,4]", "[d0_0,d0_1,d0_3,d1_2]");
  INFER_ERROR("are 2 and 3", op, "[?,2,1];[?,1,3]");  // inner dim mismatch
}

TEST(CommonShapeFnsTest, BiasAddShapeTest) {
  OpRegistrationData op_reg_data;
  TF_CHECK_OK(OpDefBuilder("BiasAdd")
                  .Input("a: float")
                  .Input("b: float")
                  .Output("c: float")
                  .Finalize(&op_reg_data));

  OpDef op_def = op_reg_data.op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("test", "BiasAdd")
                  .Input("a", 0, DT_FLOAT)
                  .Input("b", 0, DT_FLOAT)
                  .Finalize(&def));

  {
    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def, {S({2, 10}), S({10})},
                       {}, {}, {});
    TF_EXPECT_OK(BiasAddShape(&c));
    ShapeHandle output = c.output(0);
    EXPECT_EQ(2, c.Value(c.Dim(output, 0)));
    EXPECT_EQ(10, c.Value(c.Dim(output, 1)));
  }

  {
    // Unknown ranks.
    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def,
                       {Unknown(), Unknown()}, {}, {}, {});
    TF_EXPECT_OK(BiasAddShape(&c));
    ShapeHandle output = c.output(0);
    EXPECT_FALSE(c.RankKnown(output));
  }

  {
    // Rank > 2
    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def,
                       {S({4, 3, 4, 2, 15}), S({15})}, {}, {}, {});
    TF_EXPECT_OK(BiasAddShape(&c));
    ShapeHandle output = c.output(0);
    EXPECT_EQ("[4,3,4,2,15]", c.DebugString(output));
  }

  {
    // NCHW format
    TF_CHECK_OK(NodeDefBuilder("test", "BiasAdd")
                    .Input("a", 0, DT_FLOAT)
                    .Input("b", 0, DT_FLOAT)
                    .Attr("data_format", "NCHW")
                    .Finalize(&def));
    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def,
                       {S({2, 3, 4, 5}), S({3})}, {}, {}, {});
    TF_EXPECT_OK(BiasAddShape(&c));
    ShapeHandle output = c.output(0);
    EXPECT_EQ("[2,3,4,5]", c.DebugString(output));
  }

  {
    // NCHW format with high input rank
    TF_CHECK_OK(NodeDefBuilder("test", "BiasAdd")
                    .Input("a", 0, DT_FLOAT)
                    .Input("b", 0, DT_FLOAT)
                    .Attr("data_format", "NCHW")
                    .Finalize(&def));
    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def,
                       {S({8, 6, 4, 2, 3, 4, 5}), S({3})}, {}, {}, {});
    EXPECT_FALSE(BiasAddShape(&c).ok());
  }

  {
    // NCHW format with input rank 3
    TF_CHECK_OK(NodeDefBuilder("test", "BiasAdd")
                    .Input("a", 0, DT_FLOAT)
                    .Input("b", 0, DT_FLOAT)
                    .Attr("data_format", "NCHW")
                    .Finalize(&def));
    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def,
                       {S({10, 11, 12}), S({11})}, {}, {}, {});
    TF_EXPECT_OK(BiasAddShape(&c));
    ShapeHandle output = c.output(0);
    EXPECT_EQ("[10,11,12]", c.DebugString(output));
  }

  {
    // Input rank not high enough
    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def, {S({3}), S({3})}, {},
                       {}, {});
    EXPECT_FALSE(BiasAddShape(&c).ok());
  }

  {
    // NCHW rank not high enough
    TF_CHECK_OK(NodeDefBuilder("test", "BiasAdd")
                    .Input("a", 0, DT_FLOAT)
                    .Input("b", 0, DT_FLOAT)
                    .Attr("data_format", "NCHW")
                    .Finalize(&def));
    // NCHW format
    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def, {S({2, 3}), S({3})},
                       {}, {}, {});
    EXPECT_FALSE(BiasAddShape(&c).ok());
  }
}

TEST(CommonShapeFnsTest, FusedBatchNormExTest) {
  ShapeInferenceTestOp op("_FusedBatchNormEx");

  std::vector<NodeDefBuilder::NodeOut> no_side_inputs;
  TF_CHECK_OK(NodeDefBuilder("test", "_FusedBatchNormEx")
                  .Input("x", 0, DT_HALF)
                  .Input("scale", 0, DT_FLOAT)
                  .Input("offset", 0, DT_FLOAT)
                  .Input("mean", 0, DT_FLOAT)
                  .Input("variance", 0, DT_FLOAT)
                  .Input(no_side_inputs)
                  .Attr("T", DT_HALF)
                  .Attr("U", DT_FLOAT)
                  .Attr("epsilon", 0.001)
                  .Attr("data_format", "NHWC")
                  .Attr("activation_mode", "Relu")
                  .Attr("num_side_inputs", 0)
                  .Attr("is_training", true)
                  .Finalize(&op.node_def));

  // Channels are not multiple of 4.
  INFER_ERROR("must be divisible by 4", op, "[2,2,2,2];[2];[2];[2];[2]");

  INFER_OK(op, "[2,2,2,4];[4];[4];[4];[4]",
           "[d0_0,d0_1,d0_2,d0_3];[d0_3];[d0_3];[d0_3];[d0_3];?");
}

TEST(CommonShapeFnsTest, BiasAddGradShapeTest) {
  OpRegistrationData op_reg_data;
  TF_CHECK_OK(OpDefBuilder("BiasAddGrad")
                  .Input("a: float")
                  .Output("b: float")
                  .Finalize(&op_reg_data));

  OpDef op_def = op_reg_data.op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("test", "BiasAddGrad")
                  .Input("a", 0, DT_FLOAT)
                  .Finalize(&def));

  {
    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def, {S({2, 10})}, {}, {},
                       {});
    TF_EXPECT_OK(BiasAddGradShape(&c));
    ShapeHandle output = c.output(0);
    EXPECT_EQ(10, c.Value(c.Dim(output, 0)));
  }

  {
    // Rank > 2
    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def, {S({5, 7, 2, 10})},
                       {}, {}, {});
    TF_EXPECT_OK(BiasAddGradShape(&c));
    ShapeHandle output = c.output(0);
    EXPECT_EQ(10, c.Value(c.Dim(output, 0)));
  }

  {
    // NCHW format
    TF_CHECK_OK(NodeDefBuilder("test", "BiasAddGrad")
                    .Input("a", 0, DT_FLOAT)
                    .Attr("data_format", "NCHW")
                    .Finalize(&def));
    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def, {S({2, 3, 4, 5})}, {},
                       {}, {});
    TF_EXPECT_OK(BiasAddGradShape(&c));
    ShapeHandle output = c.output(0);
    EXPECT_EQ(3, c.Value(c.Dim(output, 0)));
  }

  {
    // NCHW format with high input rank
    TF_CHECK_OK(NodeDefBuilder("test", "BiasAddGrad")
                    .Input("a", 0, DT_FLOAT)
                    .Attr("data_format", "NCHW")
                    .Finalize(&def));
    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def,
                       {S({8, 6, 4, 2, 3, 4, 5})}, {}, {}, {});
    TF_EXPECT_OK(BiasAddGradShape(&c));
    ShapeHandle output = c.output(0);
    EXPECT_EQ(6, c.Value(c.Dim(output, 0)));
  }

  {
    // NCHW format with input rank 3
    TF_CHECK_OK(NodeDefBuilder("test", "BiasAddGrad")
                    .Input("a", 0, DT_FLOAT)
                    .Attr("data_format", "NCHW")
                    .Finalize(&def));
    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def, {S({10, 11, 12})}, {},
                       {}, {});
    TF_EXPECT_OK(BiasAddGradShape(&c));
    ShapeHandle output = c.output(0);
    EXPECT_EQ(11, c.Value(c.Dim(output, 0)));
  }

  {
    // Input rank not high enough
    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def, {S({3})}, {}, {}, {});
    EXPECT_FALSE(BiasAddGradShape(&c).ok());
  }

  {
    // NCHW rank not high enough
    TF_CHECK_OK(NodeDefBuilder("test", "BiasAddGrad")
                    .Input("a", 0, DT_FLOAT)
                    .Attr("data_format", "NCHW")
                    .Finalize(&def));
    // NCHW format
    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def, {S({2, 3})}, {}, {},
                       {});
    EXPECT_FALSE(BiasAddGradShape(&c).ok());
  }
}

TEST(CommonShapeFnsTest, ConvTest) {
  ShapeInferenceTestOp op("Conv");
  auto set_op = [&op](const std::vector<int32>& strides, const string& padding,
                      string data_format, int batch_dims, int groups) {
    TF_CHECK_OK(NodeDefBuilder("test", op.name)
                    .Input("input", 0, DT_FLOAT)
                    .Input("filter", 0, DT_FLOAT)
                    .Attr("strides", strides)
                    .Attr("padding", padding)
                    .Attr("data_format", data_format)
                    .Attr("batch_dims", batch_dims)
                    .Attr("groups", groups)
                    .Finalize(&op.node_def));
  };

  // Different input and filter ranks.
  set_op(/*strides=*/{{1, 1, 1, 1}}, /*padding=*/"VALID",
         /*data_format=*/"CHANNELS_LAST", /*batch_dims=*/1, /*groups=*/1);
  INFER_ERROR("Input tensor rank must be the same as filter rank.", op,
              "[2,2,1,1,1];[2,1,1,1]");

  // Negative batch dimension.
  set_op(/*strides=*/{{1, 1, 1, 1}}, /*padding=*/"VALID",
         /*data_format=*/"CHANNELS_LAST", /*batch_dims=*/-1, /*groups=*/1);
  INFER_ERROR("must be non-negative", op, "[1,2,2,1];[1,1,1,1]");

  // Too large batch dimension.
  set_op(/*strides=*/{{1, 1, 1, 1}}, /*padding=*/"VALID",
         /*data_format=*/"CHANNELS_LAST", /*batch_dims=*/5, /*groups=*/1);
  INFER_ERROR(
      "Input tensor must be rank 4 or 5, excluding extra "
      "batch dimensions, but got: 0",
      op, "[1,2,2,1];[1,1,1,1]");

  // Invalid batch dimension (default 1 but should be 0).
  set_op(/*strides=*/{{1, 1, 1, 1}}, /*padding=*/"VALID",
         /*data_format=*/"CHANNELS_LAST", /*batch_dims=*/1, /*groups=*/1);
  INFER_ERROR("extra batch dimensions", op, "[1,2,3];[1,1,1,1]");

  // Invalid batch dimension (default 1 but should be 2).
  set_op(/*strides=*/{{1, 1, 1, 1, 1}}, /*padding=*/"VALID",
         /*data_format=*/"CHANNELS_LAST", /*batch_dims=*/1, /*groups=*/1);
  INFER_ERROR("extra batch dimensions", op, "[1,2,3,4,5,6];[1,1,1,1,1]");

  // Negative groups number.
  set_op(/*strides=*/{{1, 1, 1, 1, 1}}, /*padding=*/"VALID",
         /*data_format=*/"CHANNELS_LAST", /*batch_dims=*/1, /*groups=*/0);
  INFER_ERROR("should be a positive integer", op, "[1,2,3,4,5];[1,1,1,1,1]");
  set_op(/*strides=*/{{1, 1, 1, 1, 1}}, /*padding=*/"VALID",
         /*data_format=*/"CHANNELS_LAST", /*batch_dims=*/1, /*groups=*/-1);
  INFER_ERROR("should be a positive integer", op, "[1,2,3,4,5];[1,1,1,1,1]");

  // Groups number doesn't divide input depth.
  set_op(/*strides=*/{{1, 1, 1, 1, 1}}, /*padding=*/"VALID",
         /*data_format=*/"CHANNELS_LAST", /*batch_dims=*/1, /*groups=*/3);
  INFER_ERROR("should divide input depth", op, "[1,1,1,1,13];[3,3,3,13,3]");

  // Groups number doesn't divide output depth.
  set_op(/*strides=*/{{1, 1, 1, 1, 1}}, /*padding=*/"VALID",
         /*data_format=*/"CHANNELS_LAST", /*batch_dims=*/1, /*groups=*/3);
  INFER_ERROR("should divide output depth", op, "[3,3,3,3,3];[1,1,1,3,13]");

  set_op(/*strides=*/{{1, 1, 1, 1, 1}}, /*padding=*/"SAME",
         /*data_format=*/"CHANNELS_LAST", /*batch_dims=*/1, /*groups=*/2);
  // 4x4 input of depth 10, 2x2 filter with depth 5, 1x1 stride.
  INFER_OK(op, "[1,4,4,4,10];[2,2,2,5,2]", "[d0_0,d0_1,d0_2,d0_3,d1_4]");

  // Test output multiple of group size is ok:
  // 4x4 input of depth 10, 2x2 filter with depth 5, 1x1 stride.
  INFER_OK(op, "[1,4,4,4,10];[2,2,2,5,2]", "[d0_0,d0_1,d0_2,d0_3,d1_4]");

  // Input depth / filter input depth != groups.
  set_op(/*strides=*/{{1, 1, 1, 1, 1}}, /*padding=*/"SAME",
         /*data_format=*/"CHANNELS_LAST", /*batch_dims=*/1, /*groups=*/1);
  INFER_ERROR(
      "Input depth divided by filter input depth does not match with groups "
      "parameter (1)",
      op, "[1,4,4,4,10];[2,2,2,5,2]");

  set_op(/*strides=*/{{1, 1, 1, 1, 1}}, /*padding=*/"SAME",
         /*data_format=*/"CHANNELS_LAST", /*batch_dims=*/1, /*groups=*/10);
  // Depthwise convolution first step:
  // 4x4 input of depth 10, 2x2 filter with depth 1, 1x1 stride.
  INFER_OK(op, "[1,4,4,4,10];[2,2,2,1,10]", "[d0_0,d0_1,d0_2,d0_3,d1_4]");
}

TEST(CommonShapeFnsTest, Conv2DFormatsTest) {
  ShapeInferenceTestOp op("Conv2D");
  auto set_op = [&op](const std::vector<int32>& strides, const string& padding,
                      const string& data_format, const string& filter_format,
                      const std::vector<int32>& explicit_paddings = {}) {
    TF_CHECK_OK(NodeDefBuilder("test", op.name)
                    .Input("input", 0, DT_FLOAT)
                    .Input("filter", 0, DT_FLOAT)
                    .Attr("strides", strides)
                    .Attr("padding", padding)
                    .Attr("explicit_paddings", explicit_paddings)
                    .Attr("data_format", data_format)
                    .Attr("filter_format", filter_format)
                    .Finalize(&op.node_def));
  };

  // Tests for NCHW_VECT_C.
  // 1x1 filter.
  set_op(/*strides=*/{{1, 1, 1, 1}}, /*padding=*/"VALID",
         /*data_format=*/"NCHW_VECT_C", /*filter_format=*/"OIHW_VECT_I");
  INFER_OK(op, "[1,1,2,2,4];[4,1,1,1,4]", "[d0_0,1,2,2,d0_4]");

  // 2x2 filter.
  set_op(/*strides=*/{{1, 1, 1, 1}}, /*padding=*/"VALID",
         /*data_format=*/"NCHW_VECT_C", /*filter_format=*/"OIHW_VECT_I");
  INFER_OK(op, "[1,1,2,2,4];[4,1,2,2,4]", "[d0_0,1,1,1,d0_4]");

  // 3x3 input, 1x1 filter, 2x2 stride.
  set_op(/*strides=*/{{1, 1, 2, 2}}, /*padding=*/"VALID",
         /*data_format=*/"NCHW_VECT_C", /*filter_format=*/"OIHW_VECT_I");
  INFER_OK(op, "[1,1,3,3,4];[8,1,1,1,4]", "[d0_0,2,2,2,d0_4]");

  // 3x3 input, 1x1 filter, 2x1 stride.
  set_op(/*strides=*/{{1, 1, 2, 1}}, /*padding=*/"VALID",
         /*data_format=*/"NCHW_VECT_C", /*filter_format=*/"OIHW_VECT_I");
  INFER_OK(op, "[1,1,3,3,4];[4,1,1,1,4]", "[d0_0,1,2,3,d0_4]");

  // 4x4 input, 2x1 filter, 1x2 stride.
  set_op(/*strides=*/{{1, 1, 1, 2}}, /*padding=*/"VALID",
         /*data_format=*/"NCHW_VECT_C", /*filter_format=*/"OIHW_VECT_I");
  INFER_OK(op, "[1,1,4,4,4];[4,1,2,1,4]", "[d0_0,1,3,2,d0_4]");

  // int8x32 input.
  set_op(/*strides=*/{{1, 1, 1, 2}}, /*padding=*/"VALID",
         /*data_format=*/"NCHW_VECT_C", /*filter_format=*/"OIHW_VECT_I");
  INFER_OK(op, "[1,1,4,4,32];[32,1,2,1,32]", "[d0_0,1,3,2,d0_4]");
}

class Conv2DShapeTest : public ::testing::TestWithParam<string> {};

TEST_P(Conv2DShapeTest, Conv2DShapeTest) {
  const string op_name = GetParam();
  ShapeInferenceTestOp op(op_name);
  auto set_op = [&op](const std::vector<int32>& strides, const string& padding,
                      const string& data_format, const string& filter_format,
                      const std::vector<int32>& explicit_paddings = {}) {
    string format;
    if (op.name == "Conv")
      format = (data_format == "NHWC") ? "CHANNELS_LAST" : "CHANNELS_FIRST";
    else
      format = data_format;
    TF_CHECK_OK(NodeDefBuilder("test", op.name)
                    .Input("input", 0, DT_FLOAT)
                    .Input("filter", 0, DT_FLOAT)
                    .Attr("strides", strides)
                    .Attr("padding", padding)
                    .Attr("explicit_paddings", explicit_paddings)
                    .Attr("data_format", format)
                    .Attr("filter_format", filter_format)
                    .Finalize(&op.node_def));
  };

  set_op(/*strides=*/{{1, 1, 0, 1}}, /*padding=*/"VALID",
         /*data_format=*/"NHWC", /*filter_format=*/"HWIO");
  // Invalid rank for input
  INFER_ERROR("must be rank 4", op, "[4,4];[2,1,1,1]");
  // Invalid rank for filter
  INFER_ERROR("must be rank 4", op, "[1,4,4,1];[2,1,1]");

  // Invalid value for strides
  set_op(/*strides=*/{{1, 1, 0, 1}}, /*padding=*/"VALID",
         /*data_format=*/"NHWC", /*filter_format=*/"HWIO");
  INFER_ERROR("must be > 0", op, "[1,2,2,1];[1,1,1,1]");

  // 1x1 filter
  set_op(/*strides=*/{{1, 1, 1, 1}}, /*padding=*/"VALID",
         /*data_format=*/"NHWC", /*filter_format=*/"HWIO");
  INFER_OK(op, "[1,2,2,1];[1,1,1,1]", "[d0_0,2,2,d1_3]");

  // 2x2 filter
  set_op(/*strides=*/{{1, 1, 1, 1}}, /*padding=*/"VALID",
         /*data_format=*/"NHWC", /*filter_format=*/"HWIO");
  INFER_OK(op, "[1,2,2,1];[2,2,1,1]", "[d0_0,1,1,d1_3]");

  // 3x3 input, 1x1 filter, 2x2 stride
  set_op(/*strides=*/{{1, 2, 2, 1}}, /*padding=*/"VALID",
         /*data_format=*/"NHWC", /*filter_format=*/"HWIO");
  INFER_OK(op, "[1,3,3,1];[1,1,1,1]", "[d0_0,2,2,d1_3]");

  // 3x3 input, 1x1 filter, 2x1 stride
  set_op(/*strides=*/{{1, 2, 1, 1}}, /*padding=*/"VALID",
         /*data_format=*/"NHWC", /*filter_format=*/"HWIO");
  INFER_OK(op, "[1,3,3,1];[1,1,1,1]", "[d0_0,2,3,d1_3]");

  // 4x4 input, 2x1 filter, 1x2 stride
  set_op(/*strides=*/{{1, 1, 2, 1}}, /*padding=*/"VALID",
         /*data_format=*/"NHWC", /*filter_format=*/"HWIO");
  INFER_OK(op, "[1,4,4,1];[2,1,1,1]", "[d0_0,3,2,d1_3]");

  // Unknown dims in the critical fields lead to partial inference.
  INFER_OK(op, "[1,4,4,1];[2,1,1,1]", "[d0_0,3,2,d1_3]");
  INFER_OK(op, "[1,?,4,1];[2,1,1,1]", "[d0_0,?,2,d1_3]");
  INFER_OK(op, "[1,4,?,1];[2,1,1,1]", "[d0_0,3,?,d1_3]");
  INFER_OK(op, "[1,4,4,?];[2,1,1,1]", "[d0_0,3,2,d1_3]");
  INFER_OK(op, "[1,4,4,1];[?,1,1,1]", "[d0_0,?,2,d1_3]");
  INFER_OK(op, "[1,4,4,1];[2,?,1,1]", "[d0_0,3,?,d1_3]");

  // input depths must be multiple of filter.
  INFER_ERROR(
      "Depth of input (10) is not a multiple of input depth of filter (10000)",
      op, "[1,2,2,10];[1,1,10000,20]");

  // Tests for NCHW
  // 1x1 filter
  set_op(/*strides=*/{{1, 1, 1, 1}}, /*padding=*/"VALID",
         /*data_format=*/"NCHW", /*filter_format=*/"HWIO");
  INFER_OK(op, "[1,1,2,2];[1,1,1,1]", "[d0_0,d1_3,2,2]");

  // 2x2 filter
  set_op(/*strides=*/{{1, 1, 1, 1}}, /*padding=*/"VALID",
         /*data_format=*/"NCHW", /*filter_format=*/"HWIO");
  INFER_OK(op, "[1,1,2,2];[2,2,1,1]", "[d0_0,d1_3,1,1]");

  // 3x3 input, 1x1 filter, 2x2 stride
  set_op(/*strides=*/{{1, 1, 2, 2}}, /*padding=*/"VALID",
         /*data_format=*/"NCHW", /*filter_format=*/"HWIO");
  INFER_OK(op, "[1,1,3,3];[1,1,1,1]", "[d0_0,d1_3,2,2]");

  // 3x3 input, 1x1 filter, 2x1 stride
  set_op(/*strides=*/{{1, 1, 2, 1}}, /*padding=*/"VALID",
         /*data_format=*/"NCHW", /*filter_format=*/"HWIO");
  INFER_OK(op, "[1,1,3,3];[1,1,1,1]", "[d0_0,d1_3,2,3]");

  // 4x4 input, 2x1 filter, 1x2 stride
  set_op(/*strides=*/{{1, 1, 1, 2}}, /*padding=*/"VALID",
         /*data_format=*/"NCHW", /*filter_format=*/"HWIO");
  INFER_OK(op, "[1,1,4,4];[2,1,1,1]", "[d0_0,d1_3,3,2]");

  // Some tests for "SAME" padding

  // 4x4 input, 1x1 filter, 1x1 stride
  set_op(/*strides=*/{{1, 1, 1, 1}}, /*padding=*/"SAME", /*data_format=*/"NHWC",
         /*filter_format=*/"HWIO");
  INFER_OK(op, "[1,4,4,1];[1,1,1,1]", "[d0_0,d0_1,d0_2,d1_3]");

  // 3x3 input, 2x2 filter, 1x1 stride
  set_op(/*strides=*/{{1, 1, 1, 1}}, /*padding=*/"SAME", /*data_format=*/"NHWC",
         /*filter_format=*/"HWIO");
  INFER_OK(op, "[1,3,3,1];[2,2,1,1]", "[d0_0,d0_1,d0_2,d1_3]");

  // 4x4 input, 2x2 filter, 2x2 stride
  set_op(/*strides=*/{{1, 2, 2, 1}}, /*padding=*/"SAME", /*data_format=*/"NHWC",
         /*filter_format=*/"HWIO");
  INFER_OK(op, "[1,4,4,1];[2,2,1,1]", "[d0_0,2,2,d1_3]");

  // 4x4 input, 2x2 filter, 1x1 stride
  set_op(/*strides=*/{{1, 1, 1, 1}}, /*padding=*/"SAME", /*data_format=*/"NHWC",
         /*filter_format=*/"HWIO");
  INFER_OK(op, "[1,4,4,1];[2,2,1,1]", "[d0_0,d0_1,d0_2,d1_3]");

  // With stride 1x1 and SAME, unknown dims don't matter - filter dims except
  // for output channels are ignored for output, so all inputs are carried
  // through to output.
  set_op(/*strides=*/{{1, 1, 1, 1}}, /*padding=*/"SAME", /*data_format=*/"NHWC",
         /*filter_format=*/"HWIO");
  INFER_OK(op, "[1,4,4,1];[?,?,?,?]", "[d0_0,d0_1,d0_2,d1_3]");
  INFER_OK(op, "[1,?,4,1];[?,?,?,?]", "[d0_0,d0_1,d0_2,d1_3]");
  INFER_OK(op, "[1,4,?,1];[?,?,?,?]", "[d0_0,d0_1,d0_2,d1_3]");
  INFER_OK(op, "[1,4,4,?];[?,?,?,?]", "[d0_0,d0_1,d0_2,d1_3]");
  INFER_OK(op, "[?,4,4,1];[?,?,?,?]", "[d0_0,d0_1,d0_2,d1_3]");

  // With stride != 1, the input HW dims are divided to produce output dims.
  set_op(/*strides=*/{{1, 2, 2, 1}}, /*padding=*/"SAME", /*data_format=*/"NHWC",
         /*filter_format=*/"HWIO");
  INFER_OK(op, "[?,4,4,1];[?,?,?,?]", "[d0_0,2,2,d1_3]");
  INFER_OK(op, "[1,?,4,1];[?,?,?,?]", "[d0_0,?,2,d1_3]");
  INFER_OK(op, "[1,4,?,1];[?,?,?,?]", "[d0_0,2,?,d1_3]");
  INFER_OK(op, "[1,4,4,?];[?,?,?,?]", "[d0_0,2,2,d1_3]");

  // Some tests for "EXPLICIT" padding

  // 4x4 input, 1x1 filter, 1x1 stride, [0, 2, 1, 4] padding
  set_op(/*strides=*/{{1, 1, 1, 1}}, /*padding=*/"EXPLICIT",
         /*data_format=*/"NHWC", /*filter_format=*/"HWIO",
         /*explicit_paddings=*/{0, 0, 0, 2, 1, 4, 0, 0});
  INFER_OK(op, "[1,4,4,1];[1,1,1,1]", "[d0_0,6,9,d1_3]");

  // 3x3 input, 2x2 filter, 1x1 stride, [1, 0, 1, 2] padding
  set_op(/*strides=*/{{1, 1, 1, 1}}, /*padding=*/"EXPLICIT",
         /*data_format=*/"NHWC", /*filter_format=*/"HWIO",
         /*explicit_paddings=*/{0, 0, 1, 0, 1, 2, 0, 0});
  INFER_OK(op, "[1,3,3,1];[2,2,1,1]", "[d0_0,3,5,d1_3]");

  // 4x4 input, 2x2 filter, 2x2 stride, [3, 2, 1, 0] padding
  set_op(/*strides=*/{{1, 2, 2, 1}}, /*padding=*/"EXPLICIT",
         /*data_format=*/"NHWC", /*filter_format=*/"HWIO",
         /*explicit_paddings=*/{0, 0, 3, 2, 1, 0, 0, 0});
  INFER_OK(op, "[1,4,4,2];[2,2,2,3]", "[d0_0,4,2,d1_3]");

  // 2x2 input, 2x1 filter, 1x2 stride, [1, 1, 2, 2] padding
  set_op(/*strides=*/{{1, 1, 2, 1}}, /*padding=*/"EXPLICIT",
         /*data_format=*/"NHWC", /*filter_format=*/"HWIO",
         /*explicit_paddings=*/{0, 0, 1, 1, 2, 2, 0, 0});
  INFER_OK(op, "[1,2,2,1];[2,1,1,1]", "[d0_0,3,3,d1_3]");

  // Unknown dims in the critical fields lead to partial inference.
  INFER_OK(op, "[1,4,4,1];[2,1,1,1]", "[d0_0,5,4,d1_3]");
  INFER_OK(op, "[1,?,4,1];[2,1,1,1]", "[d0_0,?,4,d1_3]");
  INFER_OK(op, "[1,4,?,1];[2,1,1,1]", "[d0_0,5,?,d1_3]");
  INFER_OK(op, "[1,4,4,?];[2,1,1,1]", "[d0_0,5,4,d1_3]");
  INFER_OK(op, "[1,4,4,1];[?,1,1,1]", "[d0_0,?,4,d1_3]");
  INFER_OK(op, "[1,4,4,1];[2,?,1,1]", "[d0_0,5,?,d1_3]");

  // Explicit padding errors
  // Negative padding
  set_op(/*strides=*/{{1, 1, 1, 1}}, /*padding=*/"EXPLICIT",
         /*data_format=*/"NHWC", /*filter_format=*/"HWIO",
         /*explicit_paddings=*/{0, 0, 0, -1, 0, 0, 0, 0});
  INFER_ERROR("must be nonnegative", op, "[1,2,2,1];[1,1,1,1]");

  // Too little padding (7 explicit paddings instead of 8)
  set_op(/*strides=*/{{1, 1, 1, 1}}, /*padding=*/"EXPLICIT",
         /*data_format=*/"NHWC", /*filter_format=*/"HWIO",
         /*explicit_paddings=*/{0, 0, 0, 0, 0, 0, 0});
  INFER_ERROR("must contain 8 values", op, "[1,2,2,1];[1,1,1,1]");

  // Too much padding (9 explicit paddings instead of 8)
  set_op({{1, 1, 1, 1}}, "EXPLICIT", "NHWC", "HWIO",
         {0, 0, 0, 0, 0, 0, 0, 0, 0});
  INFER_ERROR("must contain 8 values", op, "[1,2,2,1];[1,1,1,1]");

  // Padding in batch dimension
  set_op(/*strides=*/{{1, 1, 1, 1}}, /*padding=*/"EXPLICIT",
         /*data_format=*/"NHWC", /*filter_format=*/"HWIO",
         /*explicit_paddings=*/{1, 0, 0, 0, 0, 0, 0, 0});
  INFER_ERROR("batch or depth dimensions", op, "[1,2,2,1];[1,1,1,1]");

  // Padding in depth dimension
  set_op(/*strides=*/{{1, 1, 1, 1}}, /*padding=*/"EXPLICIT",
         /*data_format=*/"NHWC", /*filter_format=*/"HWIO",
         /*explicit_paddings=*/{0, 0, 0, 0, 0, 0, 1, 0});
  INFER_ERROR("batch or depth dimensions", op, "[1,2,2,1];[1,1,1,1]");

  // Padding explicit_paddings when padding is not EXPLICIT
  set_op(/*strides=*/{{1, 1, 1, 1}}, /*padding=*/"VALID",
         /*data_format=*/"NHWC", /*filter_format=*/"HWIO",
         /*explicit_paddings=*/{0, 0, 0, 0, 0, 0, 0, 0});
  INFER_ERROR("must be empty", op, "[1,2,2,1];[1,1,1,1]");
}

TEST_P(Conv2DShapeTest, Conv2DDilatedShapeTest) {
  const string op_name = GetParam();
  ShapeInferenceTestOp op(op_name);
  auto set_op = [&op](const std::vector<int32>& dilations,
                      const std::vector<int32>& strides, const string& padding,
                      const string& data_format,
                      const std::vector<int32>& explicit_paddings = {}) {
    string format;
    if (op.name == "Conv")
      format = (data_format == "NHWC") ? "CHANNELS_LAST" : "CHANNELS_FIRST";
    else
      format = data_format;
    TF_CHECK_OK(NodeDefBuilder("test", "Conv2D")
                    .Input("input", 0, DT_FLOAT)
                    .Input("filter", 0, DT_FLOAT)
                    .Attr("dilations", dilations)
                    .Attr("strides", strides)
                    .Attr("padding", padding)
                    .Attr("explicit_paddings", explicit_paddings)
                    .Attr("data_format", format)
                    .Attr("batch_dims", 1)
                    .Attr("groups", 1)
                    .Finalize(&op.node_def));
  };

  // Invalid rank for dilation
  set_op(/*dilations=*/{{1, 2, 1}}, /*strides=*/{{1, 1, 1, 1}},
         /*padding=*/"VALID", /*data_format=*/"NHWC");
  INFER_ERROR("contain 4 values", op, "[1,2,2,1];[1,1,1,1]");

  // Invalid value for dilation
  set_op(/*dilations=*/{{1, 0, 1, 1}}, /*strides=*/{{1, 1, 1, 1}},
         /*padding=*/"VALID", /*data_format=*/"NHWC");
  INFER_ERROR("must be >= 1", op, "[1,2,2,1];[1,1,1,1]");

  // Tests for NHWC
  // 1x1 filter, 2x1 dilations, 1x1 strides
  set_op(/*dilations=*/{{1, 2, 1, 1}}, /*strides=*/{{1, 1, 1, 1}},
         /*padding=*/"VALID", /*data_format=*/"NHWC");
  INFER_OK(op, "[1,2,2,1];[1,1,1,1]", "[d0_0,2,2,d1_3]");

  // 1x1 filter, 2x1 dilations, 2x1 strides
  set_op(/*dilations=*/{{1, 2, 1, 1}}, /*strides=*/{{1, 2, 1, 1}},
         /*padding=*/"VALID", /*data_format=*/"NHWC");
  INFER_OK(op, "[1,4,4,1];[1,1,1,1]", "[d0_0,2,4,d1_3]");

  // 1x1 filter, 2x1 dilations, 2x2 strides
  set_op(/*dilations=*/{{1, 2, 1, 1}}, /*strides=*/{{1, 2, 2, 1}},
         /*padding=*/"VALID", /*data_format=*/"NHWC");
  INFER_OK(op, "[1,4,4,1];[1,1,1,1]", "[d0_0,2,2,d1_3]");

  // 3x3 filter, 2x1 dilations, 1x1 strides
  set_op(/*dilations=*/{{1, 2, 1, 1}}, /*strides=*/{{1, 1, 1, 1}},
         /*padding=*/"VALID", /*data_format=*/"NHWC");
  INFER_OK(op, "[1,5,5,1];[3,3,1,1]", "[d0_0,1,3,d1_3]");

  // 3x3 filter, 2x1 dilations, 2x1 strides
  set_op(/*dilations=*/{{1, 2, 1, 1}}, /*strides=*/{{1, 2, 1, 1}},
         /*padding=*/"VALID", /*data_format=*/"NHWC");
  INFER_OK(op, "[1,5,5,1];[3,3,1,1]", "[d0_0,1,3,d1_3]");

  // 3x3 filter, 1x2 dilations, 2x2 strides
  set_op(/*dilations=*/{{1, 1, 2, 1}}, /*strides=*/{{1, 2, 2, 1}},
         /*padding=*/"VALID", /*data_format=*/"NHWC");
  INFER_OK(op, "[1,5,5,1];[3,3,1,1]", "[d0_0,2,1,d1_3]");

  // Tests for NCHW
  // 1x1 filter, 2x1 dilations, 1x1 strides
  set_op(/*dilations=*/{{1, 1, 2, 1}}, /*strides=*/{{1, 1, 1, 1}},
         /*padding=*/"VALID", /*data_format=*/"NCHW");
  INFER_OK(op, "[1,1,2,2];[1,1,1,1]", "[d0_0,d1_3,2,2]");

  // 1x1 filter, 2x1 dilations, 2x1 strides
  set_op(/*dilations=*/{{1, 1, 2, 1}}, /*strides=*/{{1, 1, 2, 1}},
         /*padding=*/"VALID", /*data_format=*/"NCHW");
  INFER_OK(op, "[1,1,4,4];[1,1,1,1]", "[d0_0,d1_3,2,4]");

  // 1x1 filter, 2x1 dilations, 2x2 strides
  set_op(/*dilations=*/{{1, 1, 2, 1}}, /*strides=*/{{1, 1, 2, 2}},
         /*padding=*/"VALID", /*data_format=*/"NCHW");
  INFER_OK(op, "[1,1,4,4];[1,1,1,1]", "[d0_0,d1_3,2,2]");

  // 3x3 filter, 2x1 dilations, 1x1 strides
  set_op(/*dilations=*/{{1, 1, 2, 1}}, /*strides=*/{{1, 1, 1, 1}},
         /*padding=*/"VALID", /*data_format=*/"NCHW");
  INFER_OK(op, "[1,1,5,5];[3,3,1,1]", "[d0_0,d1_3,1,3]");

  // 3x3 filter, 2x1 dilations, 2x1 strides
  set_op(/*dilations=*/{{1, 1, 2, 1}}, /*strides=*/{{1, 1, 2, 1}},
         /*padding=*/"VALID", /*data_format=*/"NCHW");
  INFER_OK(op, "[1,1,5,5];[3,3,1,1]", "[d0_0,d1_3,1,3]");

  // 3x3 filter, 1x2 dilations, 2x2 strides
  set_op(/*dilations=*/{{1, 1, 1, 2}}, /*strides=*/{{1, 1, 2, 2}},
         /*padding=*/"VALID", /*data_format=*/"NCHW");
  INFER_OK(op, "[1,1,5,5];[3,3,1,1]", "[d0_0,d1_3,2,1]");

  // Some tests for "SAME" padding

  // 4x4 input, 1x1 filter, 2x1 dilations, 1x1 stride
  set_op(/*dilations=*/{{1, 2, 1, 1}}, /*strides=*/{{1, 1, 1, 1}},
         /*padding=*/"SAME", /*data_format=*/"NHWC");
  INFER_OK(op, "[1,4,4,1];[1,1,1,1]", "[d0_0,d0_1,d0_2,d1_3]");

  // 3x3 input, 2x2 filter, 2x2 dilations, 1x1 stride
  set_op(/*dilations=*/{{1, 2, 2, 1}}, /*strides=*/{{1, 1, 1, 1}},
         /*padding=*/"SAME", /*data_format=*/"NHWC");
  INFER_OK(op, "[1,3,3,1];[2,2,1,1]", "[d0_0,d0_1,d0_2,d1_3]");

  // 4x4 input, 2x2 filter, 1x2 dilations, 2x2 stride
  set_op(/*dilations=*/{{1, 1, 2, 1}}, /*strides=*/{{1, 2, 2, 1}},
         /*padding=*/"SAME", /*data_format=*/"NHWC");
  INFER_OK(op, "[1,4,4,1];[2,2,1,1]", "[d0_0,2,2,d1_3]");

  // 4x4 input, 2x2 filter, 2x2 dilations, 1x1 stride
  set_op(/*dilations=*/{{1, 2, 2, 1}}, /*strides=*/{{1, 1, 1, 1}},
         /*padding=*/"SAME", /*data_format=*/"NHWC");
  INFER_OK(op, "[1,4,4,1];[2,2,1,1]", "[d0_0,d0_1,d0_2,d1_3]");

  // Some tests for "EXPLICIT" padding

  // 4x4 input, 1x1 filter, 2x1 dilations, 1x1 stride, [0, 2, 1, 4] padding
  set_op(/*dilations=*/{{1, 2, 1, 1}}, /*strides=*/{{1, 1, 1, 1}},
         /*padding=*/"EXPLICIT", /*data_format=*/"NHWC",
         /*explicit_paddings=*/{0, 0, 0, 2, 1, 4, 0, 0});
  INFER_OK(op, "[1,4,4,1];[1,1,1,1]", "[d0_0,6,9,d1_3]");

  // 3x3 input, 2x2 filter, 2x2 dilations, 1x1 stride, [1, 0, 1, 2] padding
  set_op(/*dilations=*/{{1, 2, 2, 1}}, /*strides=*/{{1, 1, 1, 1}},
         /*padding=*/"EXPLICIT", /*data_format=*/"NHWC",
         /*explicit_paddings=*/{0, 0, 1, 0, 1, 2, 0, 0});
  INFER_OK(op, "[1,3,3,1];[2,2,1,1]", "[d0_0,2,4,d1_3]");

  // 4x4 input, 2x2 filter, 1x2 dilations, 2x2 stride, [3, 2, 1, 0] padding
  set_op(/*dilations=*/{{1, 1, 2, 1}}, /*strides=*/{{1, 2, 2, 1}},
         /*padding=*/"EXPLICIT", /*data_format=*/"NHWC",
         /*explicit_paddings=*/{0, 0, 3, 2, 1, 0, 0, 0});
  INFER_OK(op, "[1,4,4,1];[2,2,1,1]", "[d0_0,4,2,d1_3]");

  // 4x4 input, 2x2 filter, 2x2 dilations, 1x1 stride, [1, 1, 2, 2] padding
  set_op(/*dilations=*/{{1, 2, 2, 1}}, /*strides=*/{{1, 1, 1, 1}},
         /*padding=*/"EXPLICIT", /*data_format=*/"NHWC",
         /*explicit_paddings=*/{0, 0, 1, 1, 2, 2, 0, 0});
  INFER_OK(op, "[1,4,4,1];[2,2,1,1]", "[d0_0,4,6,d1_3]");
}

TEST(CommonShapeFnsTest, Conv3DShapeRankTest) {
  ShapeInferenceTestOp op("Conv3D");
  // Invalid rank for input.
  INFER_ERROR("must be rank 5", op, "[4,4];[2,1,1,1]");
  // Invalid rank for filter.
  INFER_ERROR("must be rank 5", op, "[1,4,4,1];[2,1,1]");
}

TEST(CommonShapeFnsTest, Conv3DGroupsTest) {
  ShapeInferenceTestOp op("Conv3D");
  auto set_op = [&op](const std::vector<int32>& strides,
                      const string& padding) {
    TF_CHECK_OK(NodeDefBuilder("test", "Conv3D")
                    .Input("input", 0, DT_FLOAT)
                    .Input("filter", 0, DT_FLOAT)
                    .Attr("strides", strides)
                    .Attr("padding", padding)
                    .Finalize(&op.node_def));
  };

  set_op(/*strides=*/{{1, 1, 1, 1, 1}}, /*padding=*/"VALID");
  // Input depth must be multiple of filter depth for group convolutions.
  INFER_ERROR(
      "Depth of input (10) is not a multiple of input depth of filter (6)", op,
      "[1,2,2,2,10];[1,1,1,6,20]");

  // Output dimensions must be multiple of group number.
  INFER_ERROR(
      "Depth of output (1) is not a multiple of the number of groups (2)", op,
      "[1,2,2,2,10];[1,1,1,5,1]");

  set_op(/*strides=*/{{1, 1, 1, 1, 1}}, /*padding=*/"SAME");
  // 4x4 input of depth 10, 2x2 filter with depth 5, 1x1 stride.
  INFER_OK(op, "[1,4,4,4,10];[2,2,2,5,2]", "[d0_0,d0_1,d0_2,d0_3,d1_4]");

  // Test output multiple of group size is ok:
  // 4x4 input of depth 10, 2x2 filter with depth 5, 1x1 stride.
  INFER_OK(op, "[1,4,4,4,10];[2,2,2,5,2]", "[d0_0,d0_1,d0_2,d0_3,d1_4]");

  // Depthwise convolution first step:
  // 4x4 input of depth 10, 2x2 filter with depth 1, 1x1 stride.
  INFER_OK(op, "[1,4,4,4,10];[2,2,2,1,10]", "[d0_0,d0_1,d0_2,d0_3,d1_4]");
}

INSTANTIATE_TEST_SUITE_P(CommonShapeFnsTest, Conv2DShapeTest,
                         ::testing::Values("Conv2D", "Conv"));

class Conv3DShapeTest : public ::testing::TestWithParam<string> {};

TEST_P(Conv3DShapeTest, Conv3DShapeTest) {
  const string op_name = GetParam();
  ShapeInferenceTestOp op(op_name);
  auto set_op = [&op](const std::vector<int32>& strides,
                      const string& padding) {
    TF_CHECK_OK(NodeDefBuilder("test", op.name)
                    .Input("input", 0, DT_FLOAT)
                    .Input("filter", 0, DT_FLOAT)
                    .Attr("strides", strides)
                    .Attr("padding", padding)
                    .Finalize(&op.node_def));
  };

  // Invalid value for strides
  set_op(/*strides=*/{{1, 1, 1, 0, 1}}, /*padding=*/"VALID");
  INFER_ERROR("must be > 0", op, "[1,2,2,2,1];[1,1,1,1,1]");

  // 1x1x1 filter
  set_op(/*strides=*/{{1, 1, 1, 1, 1}}, /*padding=*/"VALID");
  INFER_OK(op, "[1,2,2,2,1];[1,1,1,1,1]", "[d0_0,2,2,2,d1_4]");

  // unknown dims in the critical fields give partial inference.
  INFER_OK(op, "[1,2,2,2,1];[1,1,1,1,1]", "[d0_0,2,2,2,d1_4]");
  INFER_OK(op, "[1,?,2,2,1];[1,1,1,1,1]", "[d0_0,?,2,2,d1_4]");
  INFER_OK(op, "[1,2,?,2,1];[1,1,1,1,1]", "[d0_0,2,?,2,d1_4]");
  INFER_OK(op, "[1,2,2,?,1];[1,1,1,1,1]", "[d0_0,2,2,?,d1_4]");
  INFER_OK(op, "[1,2,2,2,1];[?,1,1,1,1]", "[d0_0,?,2,2,d1_4]");
  INFER_OK(op, "[1,2,2,2,1];[1,?,1,1,1]", "[d0_0,2,?,2,d1_4]");
  INFER_OK(op, "[1,2,2,2,1];[1,1,?,1,1]", "[d0_0,2,2,?,d1_4]");
  INFER_OK(op, "[1,2,2,2,1];[1,1,1,?,1]", "[d0_0,2,2,2,d1_4]");
  INFER_OK(op, "[1,2,2,2,1];[1,1,1,1,?]", "[d0_0,2,2,2,d1_4]");

  // 2x2x2 filter
  set_op(/*strides=*/{{1, 1, 1, 1, 1}}, /*padding=*/"VALID");
  INFER_OK(op, "[1,2,2,2,1];[2,2,2,1,1]", "[d0_0,1,1,1,d1_4]");

  // 3x3 input, 1x1 filter, 2x2 stride
  set_op(/*strides=*/{{1, 2, 2, 2, 1}}, /*padding=*/"VALID");
  INFER_OK(op, "[1,3,3,3,1];[1,1,1,1,1]", "[d0_0,2,2,2,d1_4]");

  // 3x3 input, 1x1 filter, 2x1x1 stride
  set_op(/*strides=*/{{1, 2, 1, 1, 1}}, /*padding=*/"VALID");
  INFER_OK(op, "[1,3,3,3,1];[1,1,1,1,1]", "[d0_0,2,3,3,d1_4]");

  // 4x4 input, 2x2 filter, 1x1 stride
  set_op(/*strides=*/{{1, 1, 1, 1, 1}}, /*padding=*/"SAME");
  INFER_OK(op, "[1,4,4,4,1];[2,2,2,1,1]", "[d0_0,d0_1,d0_2,d0_3,d1_4]");

  // with SAME, filter doesn't matter except for last dim.
  set_op(/*strides=*/{{1, 1, 1, 1, 1}}, /*padding=*/"SAME");
  INFER_OK(op, "[?,4,4,4,1];[2,2,2,1,1]", "[d0_0,d0_1,d0_2,d0_3,d1_4]");
  INFER_OK(op, "[1,?,4,4,1];[2,2,2,1,1]", "[d0_0,d0_1,d0_2,d0_3,d1_4]");
  INFER_OK(op, "[1,4,?,4,1];[2,2,2,1,1]", "[d0_0,d0_1,d0_2,d0_3,d1_4]");
  INFER_OK(op, "[1,4,4,?,1];[2,2,2,1,1]", "[d0_0,d0_1,d0_2,d0_3,d1_4]");
  INFER_OK(op, "[1,4,4,4,?];[2,2,2,1,1]", "[d0_0,d0_1,d0_2,d0_3,d1_4]");
  INFER_OK(op, "[1,4,4,4,1];[?,2,2,1,1]", "[d0_0,d0_1,d0_2,d0_3,d1_4]");
  INFER_OK(op, "[1,4,4,4,1];[2,?,2,1,1]", "[d0_0,d0_1,d0_2,d0_3,d1_4]");
  INFER_OK(op, "[1,4,4,4,1];[2,2,?,1,1]", "[d0_0,d0_1,d0_2,d0_3,d1_4]");
  INFER_OK(op, "[1,4,4,4,1];[2,2,2,?,1]", "[d0_0,d0_1,d0_2,d0_3,d1_4]");
  INFER_OK(op, "[1,4,4,4,1];[2,2,2,1,?]", "[d0_0,d0_1,d0_2,d0_3,d1_4]");

  // with SAME, and stride != 1, division happens to produce output.
  set_op(/*strides=*/{{1, 2, 3, 4, 1}}, /*padding=*/"SAME");
  INFER_OK(op, "[1,4,9,4,1];[2,2,2,1,1]", "[d0_0,2,3,1,d1_4]");
  INFER_OK(op, "[?,4,9,4,1];[2,2,2,1,1]", "[d0_0,2,3,1,d1_4]");
  INFER_OK(op, "[1,?,9,4,1];[2,2,2,1,1]", "[d0_0,?,3,1,d1_4]");
  INFER_OK(op, "[1,4,?,4,1];[2,2,2,1,1]", "[d0_0,2,?,1,d1_4]");
  INFER_OK(op, "[1,4,9,?,1];[2,2,2,1,1]", "[d0_0,2,3,?,d1_4]");
  INFER_OK(op, "[1,4,9,4,?];[2,2,2,1,1]", "[d0_0,2,3,1,d1_4]");
  INFER_OK(op, "[1,4,9,4,1];[?,2,2,1,1]", "[d0_0,2,3,1,d1_4]");
  INFER_OK(op, "[1,4,9,4,1];[2,?,2,1,1]", "[d0_0,2,3,1,d1_4]");
  INFER_OK(op, "[1,4,9,4,1];[2,2,?,1,1]", "[d0_0,2,3,1,d1_4]");
  INFER_OK(op, "[1,4,9,4,1];[2,2,2,?,1]", "[d0_0,2,3,1,d1_4]");
  INFER_OK(op, "[1,4,9,4,1];[2,2,2,1,?]", "[d0_0,2,3,1,d1_4]");
}

TEST_P(Conv3DShapeTest, Conv3DDilatedShapeTest) {
  const string op_name = GetParam();
  ShapeInferenceTestOp op(op_name);
  auto set_op = [&op](const std::vector<int32>& dilations,
                      const std::vector<int32>& strides,
                      const string& padding) {
    TF_CHECK_OK(NodeDefBuilder("test", op.name)
                    .Input("input", 0, DT_FLOAT)
                    .Input("filter", 0, DT_FLOAT)
                    .Attr("dilations", dilations)
                    .Attr("strides", strides)
                    .Attr("padding", padding)
                    .Finalize(&op.node_def));
  };

  // Invalid rank for dilation
  set_op(/*dilations=*/{{1, 2, 1, 1}}, /*strides=*/{{1, 1, 1, 1, 1}},
         /*padding=*/"VALID");
  INFER_ERROR("contain 5 values", op, "[1,2,2,2,1];[1,1,1,1,1]");

  // Invalid value for dilation
  set_op(/*dilations=*/{{1, 2, 0, 1, 1}}, /*strides=*/{{1, 1, 1, 1, 1}},
         /*padding=*/"VALID");
  INFER_ERROR("must be >= 1", op, "[1,2,2,2,1];[1,1,1,1,1]");

  // 2x1x1 dilation 1x1x1 filter
  set_op(/*dilations=*/{{1, 2, 1, 1, 1}}, /*strides=*/{{1, 1, 1, 1, 1}},
         /*padding=*/"VALID");
  INFER_OK(op, "[1,2,2,2,1];[1,1,1,1,1]", "[d0_0,2,2,2,d1_4]");

  // 2x1x1 dilation 2x2x2 filter
  set_op(/*dilations=*/{{1, 2, 1, 1, 1}}, /*strides=*/{{1, 1, 1, 1, 1}},
         /*padding=*/"VALID");
  INFER_OK(op, "[1,3,2,2,1];[2,2,2,1,1]", "[d0_0,1,1,1,d1_4]");

  // 2x1x1 dilation 3x3x3 input, 1x1x1 filter, 2x2x2 stride
  set_op(/*dilations=*/{{1, 2, 1, 1, 1}}, /*strides=*/{{1, 2, 2, 2, 1}},
         /*padding=*/"VALID");
  INFER_OK(op, "[1,3,3,3,1];[1,1,1,1,1]", "[d0_0,2,2,2,d1_4]");

  // 2x1x1 dilation 3x3x3 input, 1x1x1 filter, 2x1x1 stride
  set_op(/*dilations=*/{{1, 2, 1, 1, 1}}, /*strides=*/{{1, 2, 1, 1, 1}},
         /*padding=*/"VALID");
  INFER_OK(op, "[1,3,3,3,1];[1,1,1,1,1]", "[d0_0,2,3,3,d1_4]");

  // 2x1x1 dilation 4x4x4 input, 2x2x2 filter, 1x1x1 stride
  set_op(/*dilations=*/{{1, 2, 1, 1, 1}}, /*strides=*/{{1, 1, 1, 1, 1}},
         /*padding=*/"SAME");
  INFER_OK(op, "[1,4,4,4,1];[2,2,2,1,1]", "[d0_0,d0_1,d0_2,d0_3,d1_4]");
}

INSTANTIATE_TEST_SUITE_P(CommonShapeFnsTest, Conv3DShapeTest,
                         ::testing::Values("Conv3D", "Conv"));

TEST(CommonShapeFnsTest, DepthwiseConv2DShapeTest) {
  ShapeInferenceTestOp op("DepthwiseConv2dNative");
  std::vector<int32> strides = {{1, 1, 1, 1}};
  TF_CHECK_OK(NodeDefBuilder("test", "DepthwiseConv2dNative")
                  .Input("input", 0, DT_FLOAT)
                  .Input("filter", 0, DT_FLOAT)
                  .Attr("strides", strides)
                  .Attr("padding", "VALID")
                  .Attr("data_format", "NHWC")
                  .Finalize(&op.node_def));

  // Most of DepthwiseConv2D is implicitly tested by Conv2D, so
  // we test only the very-specific differences here.

  // 1x1 filter, depth multiplication
  INFER_OK(op, "[1,2,2,3];[1,1,3,4]", "[d0_0,2,2,12]");

  // Input depths not compatible
  INFER_ERROR("Dimensions must be equal, but are 3 and 12", op,
              "[1,2,2,3];[1,1,12,4]");

  // No unknown dims in the critical fields.
  INFER_OK(op, "[1,2,2,3];[1,1,3,4]", "[d0_0,2,2,12]");
  INFER_OK(op, "[1,?,2,3];[1,1,3,4]", "[d0_0,?,2,12]");
  INFER_OK(op, "[1,2,?,3];[1,1,3,4]", "[d0_0,2,?,12]");
  INFER_OK(op, "[1,2,2,3];[?,1,3,4]", "[d0_0,?,2,12]");
  INFER_OK(op, "[1,2,2,3];[1,?,3,4]", "[d0_0,2,?,12]");
  INFER_OK(op, "[1,2,2,3];[1,1,?,4]", "[d0_0,2,2,12]");
  INFER_OK(op, "[1,2,2,?];[1,1,?,4]", "[d0_0,2,2,?]");
  INFER_OK(op, "[1,2,2,3];[1,1,3,?]", "[d0_0,2,2,?]");

  // Test for NCHW format.
  TF_CHECK_OK(NodeDefBuilder("test", "DepthwiseConv2dNative")
                  .Input("input", 0, DT_FLOAT)
                  .Input("filter", 0, DT_FLOAT)
                  .Attr("strides", strides)
                  .Attr("padding", "VALID")
                  .Attr("data_format", "NCHW")
                  .Finalize(&op.node_def));

  // 1x1 filter, depth multiplication
  INFER_OK(op, "[1,3,2,2];[1,1,3,4]", "[d0_0,12,2,2]");
}

TEST(CommonShapeFnsTest, AvgPool2DShapeTest) {
  ShapeInferenceTestOp op("AvgPool");
  auto set_op = [&op](const std::vector<int32>& strides,
                      const std::vector<int32>& ksizes, const string& padding,
                      const string& data_format) {
    TF_CHECK_OK(NodeDefBuilder("test", "AvgPool")
                    .Input("input", 0, DT_FLOAT)
                    .Attr("strides", strides)
                    .Attr("ksize", ksizes)
                    .Attr("padding", padding)
                    .Attr("data_format", data_format)
                    .Finalize(&op.node_def));
  };

  // Most of the functionality is tested by conv-like shapes,
  // so we check the very-specific avgpooling features here.

  // 1x1 filter, 1x1 stride
  set_op({1, 1, 1, 1}, {1, 1, 1, 1}, "VALID", "NHWC");
  INFER_OK(op, "[1,2,2,1]", "[d0_0,2,2,d0_3]");

  // 4x4 input, 2x1 ksize, 1x2 stride
  set_op({1, 1, 2, 1}, {1, 2, 1, 1}, "VALID", "NHWC");
  INFER_OK(op, "[1,4,4,1]", "[d0_0,3,2,d0_3]");

  // 4x4 input, 2x1 ksize, 1x2 stride
  // unknown dims in the critical fields lead to partial inference.
  // Assumes NHWC format.
  INFER_OK(op, "[1,?,4,1]", "[d0_0,?,2,d0_3]");
  INFER_OK(op, "[1,4,?,1]", "[d0_0,3,?,d0_3]");

  // 4x4 input, 2x1 ksize, 1x2 stride, NCHW format
  set_op({{1, 1, 1, 2}}, {1, 1, 2, 1}, "VALID", "NCHW");
  INFER_OK(op, "[1,1,4,4]", "[d0_0,d0_1,3,2]");

  // 5x7 input, 2x2 ksize, 1x1 stride, NCHW_VECT_C test
  set_op({{1, 1, 1, 1}}, {1, 1, 2, 2}, "VALID", "NCHW_VECT_C");
  INFER_OK(op, "[2,3,5,7,4]", "[d0_0,d0_1,4,6,4]");
  INFER_OK(op, "[5,7,?,?,4]", "[d0_0,d0_1,?,?,4]");
  INFER_OK(op, "[?,?,?,?,4]", "[d0_0,d0_1,?,?,4]");
  INFER_ERROR("must be 4 or 32, but is 3", op, "[2,5,7,11,3]");

  // Invalid rank for input
  INFER_ERROR("Shape must be rank", op, "[4,4]");
}

TEST(CommonShapeFnsTest, MaxPool2DShapeTest) {
  ShapeInferenceTestOp op("MaxPool");
  auto set_op = [&op](const std::vector<int32>& strides,
                      const std::vector<int32>& ksizes, const string& padding,
                      const string& data_format) {
    TF_CHECK_OK(NodeDefBuilder("test", "MaxPool")
                    .Input("input", 0, DT_FLOAT)
                    .Attr("strides", strides)
                    .Attr("ksize", ksizes)
                    .Attr("padding", padding)
                    .Attr("data_format", data_format)
                    .Finalize(&op.node_def));
  };

  // Most of the functionality is tested by conv-like shapes,
  // so we check the very-specific maxpooling features here,
  // namely depthwise kernel and striding.

  // all 1 strides, depth 2 filter
  set_op({1, 1, 1, 1}, {1, 1, 1, 2}, "VALID", "NHWC");
  INFER_OK(op, "[1,2,2,2]", "[d0_0,2,2,1]");

  // depth 3 stride, 1x1x1 filter, NCHW
  set_op({1, 3, 1, 1}, {1, 1, 1, 1}, "VALID", "NCHW");
  INFER_OK(op, "[1,7,5,5]", "[d0_0,3,5,5]");

  // 5x7 input, 2x2 ksize, 1x1 stride, NCHW_VECT_C tests
  set_op({{1, 1, 1, 1}}, {1, 1, 2, 2}, "SAME", "NCHW_VECT_C");
  INFER_OK(op, "[2,3,5,7,4]", "[d0_0,d0_1,d0_2,d0_3,4]");
  INFER_OK(op, "[5,7,?,?,4]", "[d0_0,d0_1,d0_2,d0_3,4]");
  INFER_OK(op, "[?,?,?,?,4]", "[d0_0,d0_1,d0_2,d0_3,4]");
  INFER_ERROR("must be 4 or 32, but is 8", op, "[2,3,5,7,8]");
}

TEST(CommonShapeFnsTest, MaxPoolV22DShapeTest) {
  ShapeInferenceTestOp op("MaxPoolV2");
  Tensor ksizes_tensor, strides_tensor;
  auto set_op = [&op, &ksizes_tensor, &strides_tensor](
                    const std::vector<int32>& strides,
                    const std::vector<int32>& ksizes, const string& padding,
                    const string& data_format) {
    TF_CHECK_OK(NodeDefBuilder("test", "MaxPoolV2")
                    .Input("input", 0, DT_FLOAT)
                    .Input("ksize", 1, DT_INT32)
                    .Input("strides", 2, DT_INT32)
                    .Attr("padding", padding)
                    .Attr("data_format", data_format)
                    .Finalize(&op.node_def));
    ksizes_tensor = test::AsTensor<int32>(ksizes);
    op.input_tensors.resize(3);
    op.input_tensors[0] = nullptr;
    op.input_tensors[1] = &ksizes_tensor;
    strides_tensor = test::AsTensor<int32>(strides);
    op.input_tensors[2] = &strides_tensor;
  };

  // Most of the functionality is tested by conv-like shapes,
  // so we check the very-specific maxpooling features here,
  // namely depthwise kernel and striding.

  // all 1 strides, depth 2 filter
  set_op({1, 1, 1, 1}, {1, 1, 1, 2}, "VALID", "NHWC");
  INFER_OK(op, "[1,2,2,2];[4];[4]", "[d0_0,2,2,1]");

  // depth 3 stride, 1x1x1 filter, NCHW
  set_op({1, 3, 1, 1}, {1, 1, 1, 1}, "VALID", "NCHW");
  INFER_OK(op, "[1,7,5,5];[4];[4]", "[d0_0,3,5,5]");

  // 5x7 input, 2x2 ksize, 1x1 stride, NCHW_VECT_C tests
  set_op({{1, 1, 1, 1}}, {1, 1, 2, 2}, "SAME", "NCHW_VECT_C");
  INFER_OK(op, "[2,3,5,7,4];[4];[4]", "[d0_0,d0_1,d0_2,d0_3,4]");
  INFER_OK(op, "[5,7,?,?,4];[4];[4]", "[d0_0,d0_1,d0_2,d0_3,4]");
  INFER_OK(op, "[?,?,?,?,4];[4];[4]", "[d0_0,d0_1,d0_2,d0_3,4]");
  INFER_ERROR("must be 4 or 32, but is 8", op, "[2,3,5,7,8];[4];[4]");
}

TEST(CommonShapeFnsTest, Pool3DShapeTest) {
  ShapeInferenceTestOp op("MaxPool3D");
  auto set_op = [&op](const std::vector<int32>& strides,
                      const std::vector<int32>& ksizes, const string& padding) {
    TF_CHECK_OK(NodeDefBuilder("test", "MaxPool3D")
                    .Input("input", 0, DT_FLOAT)
                    .Attr("strides", strides)
                    .Attr("ksize", ksizes)
                    .Attr("padding", padding)
                    .Finalize(&op.node_def));
  };

  // Most of the functionality is tested by conv-like shapes,
  // so we check that we handle the extra dimension properly.

  // 2x3x4 stride, 1x1x1 filter.
  set_op({1, 2, 3, 4, 1}, {1, 1, 1, 1, 1}, "VALID");
  INFER_OK(op, "[1,24,24,24,1]", "[d0_0,12,8,6,d0_4]");

  // Test partially known dimensions
  set_op({1, 1, 3, 4, 1}, {1, 1, 1, 1, 1}, "VALID");
  INFER_OK(op, "[1,?,24,24,1]", "[d0_0,?,8,6,d0_4]");
}

TEST(CommonShapeFnsTest, UnknownShapeTest) {
  {
    // Single output
    ShapeInferenceTestOp op("QueueDequeue");
    TF_CHECK_OK(NodeDefBuilder("test", "QueueDequeue")
                    .Input("handle", 0, DT_STRING_REF)
                    .Attr("component_types", {DT_FLOAT})
                    .Finalize(&op.node_def));
    INFER_OK(op, "[1]", "?");
  }

  {
    // Multiple outputs
    ShapeInferenceTestOp op("QueueDequeue");
    TF_CHECK_OK(NodeDefBuilder("test", "QueueDequeue")
                    .Input("handle", 0, DT_STRING_REF)
                    .Attr("component_types", {DT_FLOAT, DT_FLOAT, DT_STRING})
                    .Finalize(&op.node_def));
    INFER_OK(op, "[1]", "?;?;?");
  }
}

TEST(CommonShapeFnsTest, Reduce_ShapeFn) {
  ShapeInferenceTestOp op("Sum");
  op.input_tensors.resize(2);

  TF_ASSERT_OK(NodeDefBuilder("test", "Sum")
                   .Input("input", 0, DT_FLOAT)
                   .Input("reduction_indices", 1, DT_INT32)
                   .Attr("keep_dims", false)
                   .Finalize(&op.node_def));

  // Reduction indices not available, so output is unknown.
  INFER_OK(op, "[2,4,5];[2]", "?");
  INFER_OK(op, "?;[2]", "?");

  Tensor indices = test::AsTensor<int32>({1, 2});
  op.input_tensors[1] = &indices;

  // Reduction indices available
  INFER_OK(op, "[2,4,5];[2]", "[d0_0]");

  // Wrapped indices
  indices = test::AsTensor<int32>({-1, -2});
  op.input_tensors[1] = &indices;
  INFER_OK(op, "[2,4,5];[2]", "[d0_0]");

  // Scalar
  indices = test::AsScalar<int32>(0);
  op.input_tensors[1] = &indices;
  INFER_OK(op, "[2,4,5];[]", "[d0_1,d0_2]");

  indices = test::AsScalar<int32>(-4);
  op.input_tensors[1] = &indices;
  INFER_ERROR("Invalid reduction dimension", op, "[2,4,5];[]");

  // Empty reduction indices
  indices = test::AsTensor<int32>({});
  op.input_tensors[1] = &indices;
  INFER_OK(op, "[2,4,5];[0]", "[d0_0,d0_1,d0_2]");

  // Keep dims = true
  TF_ASSERT_OK(NodeDefBuilder("test", "Sum")
                   .Input("input", 0, DT_FLOAT)
                   .Input("reduction_indices", 1, DT_INT32)
                   .Attr("keep_dims", true)
                   .Finalize(&op.node_def));
  indices = test::AsTensor<int32>({-1, -2});
  op.input_tensors[1] = &indices;
  INFER_OK(op, "[2,4,5];[2]", "[d0_0, 1, 1]");

  // input rank is known, but reduction indices are not (with keep_dim=true).
  // The output rank matches input rank (because of keep_dims=true).
  op.input_tensors[1] = nullptr;
  INFER_OK(op, "[?,?,?];?", "[?,?,?]");
  INFER_OK(op, "[?,?,?];[2]", "[?,?,?]");

  // Reduction indices with too many dimensions.
  INFER_ERROR("must be at most rank 1 but is rank 2", op, "[?,?,?];[?,?]");
  // With older graph-def version, this is allowed.
  op.graph_def_version = 20;
  INFER_OK(op, "[?,?,?];[?,?]", "[?,?,?]");
  // And when the tensor is specified, it's still allowed.
  op.input_tensors[1] = &indices;
  indices = test::AsTensor<int32>({-1, -2}, TensorShape({2, 1}));
  INFER_OK(op, "[2,4,5];[2,1]", "[d0_0, 1, 1]");
  indices = test::AsTensor<int32>({-1, -2}, TensorShape({1, 2}));
  INFER_OK(op, "[2,4,5];[1,2]", "[d0_0, 1, 1]");
}

TEST(CommonShapeFnsTest, ValidateSparseTensor_UnknownShapes) {
  NodeDef def;
  InferenceContext c(TF_GRAPH_DEF_VERSION, def, MakeOpDef(3, 1),
                     {Unknown(), Unknown(), Unknown()}, {}, {}, {});
  EXPECT_EQ(3, c.num_inputs());
  EXPECT_EQ(1, c.num_outputs());

  auto indices = c.input(0);
  auto values = c.input(1);
  auto shape = c.input(2);
  TF_EXPECT_OK(ValidateSparseTensor(&c, indices, values, shape));
}

TEST(CommonShapeFnsTest, ValidateSparseTensor_UnknownDims) {
  NodeDef def;
  InferenceContext c(TF_GRAPH_DEF_VERSION, def, MakeOpDef(3, 1),
                     {S({-1, -1}), S({-1}), S({-1})}, {}, {}, {});
  EXPECT_EQ(3, c.num_inputs());
  EXPECT_EQ(1, c.num_outputs());

  auto indices = c.input(0);
  auto values = c.input(1);
  auto shape = c.input(2);
  TF_EXPECT_OK(ValidateSparseTensor(&c, indices, values, shape));
}

TEST(CommonShapeFnsTest, ValidateSparseTensor_InvalidIndicesRank) {
  NodeDef def;
  InferenceContext c(TF_GRAPH_DEF_VERSION, def, MakeOpDef(3, 1),
                     {S({-1}), S({-1}), S({-1})}, {}, {}, {});
  EXPECT_EQ(3, c.num_inputs());
  EXPECT_EQ(1, c.num_outputs());

  auto indices = c.input(0);
  auto values = c.input(1);
  auto shape = c.input(2);
  EXPECT_EQ(error::INVALID_ARGUMENT,
            ValidateSparseTensor(&c, indices, values, shape).code());
}

TEST(CommonShapeFnsTest, ValidateSparseTensor_InvalidNumElements) {
  NodeDef def;
  InferenceContext c(TF_GRAPH_DEF_VERSION, def, MakeOpDef(3, 1),
                     {S({5, 3}), S({4}), S({3})}, {}, {}, {});
  EXPECT_EQ(3, c.num_inputs());
  EXPECT_EQ(1, c.num_outputs());

  auto indices = c.input(0);
  auto values = c.input(1);
  auto shape = c.input(2);
  EXPECT_EQ(error::INVALID_ARGUMENT,
            ValidateSparseTensor(&c, indices, values, shape).code());
}

TEST(CommonShapeFnsTest, ValidateSparseTensor_InvalidRank) {
  NodeDef def;
  InferenceContext c(TF_GRAPH_DEF_VERSION, def, MakeOpDef(3, 1),
                     {S({5, 3}), S({5}), S({4})}, {}, {}, {});
  EXPECT_EQ(3, c.num_inputs());
  EXPECT_EQ(1, c.num_outputs());

  auto indices = c.input(0);
  auto values = c.input(1);
  auto shape = c.input(2);
  EXPECT_EQ(error::INVALID_ARGUMENT,
            ValidateSparseTensor(&c, indices, values, shape).code());
}

TEST(CommonShapeFnsTest, ValidateSparseTensor_UnknownNumIndexElements) {
  NodeDef def;
  InferenceContext c(TF_GRAPH_DEF_VERSION, def, MakeOpDef(3, 1),
                     {S({-1, 3}), S({5}), S({3})}, {}, {}, {});
  EXPECT_EQ(3, c.num_inputs());
  EXPECT_EQ(1, c.num_outputs());

  auto indices = c.input(0);
  auto values = c.input(1);
  auto shape = c.input(2);
  TF_EXPECT_OK(ValidateSparseTensor(&c, indices, values, shape));
}

TEST(CommonShapeFnsTest, ValidateSparseTensor_UnknownNumValueElements) {
  NodeDef def;
  InferenceContext c(TF_GRAPH_DEF_VERSION, def, MakeOpDef(3, 1),
                     {S({5, 3}), S({-1}), S({3})}, {}, {}, {});
  EXPECT_EQ(3, c.num_inputs());
  EXPECT_EQ(1, c.num_outputs());

  auto indices = c.input(0);
  auto values = c.input(1);
  auto shape = c.input(2);
  TF_EXPECT_OK(ValidateSparseTensor(&c, indices, values, shape));
}

TEST(CommonShapeFnsTest, ValidateSparseTensor_UnknownIndexRank) {
  NodeDef def;
  InferenceContext c(TF_GRAPH_DEF_VERSION, def, MakeOpDef(3, 1),
                     {S({5, -1}), S({5}), S({3})}, {}, {}, {});
  EXPECT_EQ(3, c.num_inputs());
  EXPECT_EQ(1, c.num_outputs());

  auto indices = c.input(0);
  auto values = c.input(1);
  auto shape = c.input(2);
  TF_EXPECT_OK(ValidateSparseTensor(&c, indices, values, shape));
}

TEST(CommonShapeFnsTest, ValidateSparseTensor_UnknownShapeRank) {
  NodeDef def;
  InferenceContext c(TF_GRAPH_DEF_VERSION, def, MakeOpDef(3, 1),
                     {S({5, 3}), S({5}), S({-1})}, {}, {}, {});
  EXPECT_EQ(3, c.num_inputs());
  EXPECT_EQ(1, c.num_outputs());

  auto indices = c.input(0);
  auto values = c.input(1);
  auto shape = c.input(2);
  TF_EXPECT_OK(ValidateSparseTensor(&c, indices, values, shape));
}

TEST(CommonShapeFnsTest, ValidateSparseTensor) {
  NodeDef def;
  InferenceContext c(TF_GRAPH_DEF_VERSION, def, MakeOpDef(3, 1),
                     {S({5, 3}), S({5}), S({3})}, {}, {}, {});
  EXPECT_EQ(3, c.num_inputs());
  EXPECT_EQ(1, c.num_outputs());

  auto indices = c.input(0);
  auto values = c.input(1);
  auto shape = c.input(2);
  TF_EXPECT_OK(ValidateSparseTensor(&c, indices, values, shape));
}

TEST(CommonShapeFnsTest, ReduceScatterSuccess) {
  OpRegistrationData op_reg_data;
  TF_CHECK_OK(OpDefBuilder("XlaReduceScatter")
                  .Input("input: float")
                  .Input("group_assignment: int32")
                  .Input("scatter_dimension: int32")
                  .Output("output: float")
                  .Finalize(&op_reg_data));
  OpDef op_def = op_reg_data.op_def;

  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("test", "XlaReduceScatter")
                  .Input("input", 0, DT_FLOAT)
                  .Input("group_assignment", 0, DT_INT32)
                  .Input("scatter_dimension", 0, DT_INT32)
                  .Finalize(&def));
  const Tensor scatter_dimension = Tensor(0);
  InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def,
                     {S({2, 2}), S({1, 2}), S({1})},
                     {nullptr, nullptr, &scatter_dimension}, {}, {});
  EXPECT_EQ(3, c.num_inputs());
  EXPECT_EQ(1, c.num_outputs());

  TF_EXPECT_OK(ReduceScatterShape(&c));
  ShapeHandle output = c.output(0);
  EXPECT_EQ(1, c.Value(c.Dim(output, 0)));
  EXPECT_EQ(2, c.Value(c.Dim(output, 1)));
}

TEST(CommonShapeFnsTest, ReduceScatter_MissingScatterDimension) {
  OpRegistrationData op_reg_data;
  TF_CHECK_OK(OpDefBuilder("XlaReduceScatter")
                  .Input("input: float")
                  .Input("group_assignment: int32")
                  .Input("scatter_dimension: int32")
                  .Output("output: float")
                  .Finalize(&op_reg_data));
  OpDef op_def = op_reg_data.op_def;

  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("test", "XlaReduceScatter")
                  .Input("input", 0, DT_FLOAT)
                  .Input("group_assignment", 0, DT_INT32)
                  .Input("scatter_dimension", 0, DT_INT32)
                  .Finalize(&def));
  InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def,
                     {S({2, 2}), S({1, 2}), S({1})}, {}, {}, {});
  EXPECT_EQ(3, c.num_inputs());
  EXPECT_EQ(1, c.num_outputs());

  TF_EXPECT_OK(ReduceScatterShape(&c));
  ShapeHandle output = c.output(0);
  EXPECT_FALSE(c.ValueKnown(c.Dim(output, 0)));
  EXPECT_FALSE(c.ValueKnown(c.Dim(output, 1)));
}

TEST(CommonShapeFnsTest, ReduceScatter_NotEvenlyDivisible) {
  OpRegistrationData op_reg_data;
  TF_CHECK_OK(OpDefBuilder("XlaReduceScatter")
                  .Input("input: float")
                  .Input("group_assignment: int32")
                  .Input("scatter_dimension: int32")
                  .Output("output: float")
                  .Finalize(&op_reg_data));
  OpDef op_def = op_reg_data.op_def;

  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("test", "XlaReduceScatter")
                  .Input("input", 0, DT_FLOAT)
                  .Input("group_assignment", 0, DT_INT32)
                  .Input("scatter_dimension", 0, DT_INT32)
                  .Finalize(&def));
  const Tensor scatter_dimension = Tensor(0);
  InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def,
                     {S({3, 3}), S({1, 2}), S({1})},
                     {nullptr, nullptr, &scatter_dimension}, {}, {});
  EXPECT_EQ(3, c.num_inputs());
  EXPECT_EQ(1, c.num_outputs());
  EXPECT_THAT(ReduceScatterShape(&c),
              absl_testing::StatusIs(
                  error::INVALID_ARGUMENT,
                  "Dimension size must be evenly divisible by 2 but is 3"));
}

TEST(CommonShapeFnsTest, ReduceScatter_INVALID_GROUP_ASSIGNMENT) {
  OpRegistrationData op_reg_data;
  TF_CHECK_OK(OpDefBuilder("XlaReduceScatter")
                  .Input("input: float")
                  .Input("group_assignment: int32")
                  .Input("scatter_dimension: int32")
                  .Output("output: float")
                  .Finalize(&op_reg_data));
  OpDef op_def = op_reg_data.op_def;

  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("test", "XlaReduceScatter")
                  .Input("input", 0, DT_FLOAT)
                  .Input("group_assignment", 0, DT_INT32)
                  .Input("scatter_dimension", 0, DT_INT32)
                  .Finalize(&def));
  const Tensor scatter_dimension = Tensor(0);
  InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def,
                     {S({3, 3}), S({2}), S({1})},
                     {nullptr, nullptr, &scatter_dimension}, {}, {});
  EXPECT_EQ(3, c.num_inputs());
  EXPECT_EQ(1, c.num_outputs());
  EXPECT_THAT(ReduceScatterShape(&c),
              absl_testing::StatusIs(
                  error::INVALID_ARGUMENT,
                  "ReduceScatter group_assignment should be rank 2"));
}

}  // namespace shape_inference
}  // namespace tensorflow
