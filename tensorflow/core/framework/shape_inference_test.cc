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
#include "tensorflow/core/framework/shape_inference.h"

#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace shape_inference {
namespace {

OpDef MakeOpDefWithLists() {
  OpRegistrationData op_reg_data;
  OpDefBuilder b("dummy");
  b.Input(strings::StrCat("input: N * float"));
  b.Output(strings::StrCat("output: N * float"));
  CHECK(b.Attr("N:int >= 1").Finalize(&op_reg_data).ok());
  return op_reg_data.op_def;
}

TensorShapeProto S(std::initializer_list<int64> dims) {
  PartialTensorShape shape(dims);
  TensorShapeProto ret;
  shape.AsProto(&ret);
  return ret;
}

TensorShapeProto Unknown() {
  PartialTensorShape shape;
  TensorShapeProto ret;
  shape.AsProto(&ret);
  return ret;
}

}  // namespace

class ShapeInferenceTest : public ::testing::Test {
 protected:
  // These give access to private functions of DimensionHandle and ShapeHandle.
  bool SameHandle(DimensionHandle a, DimensionHandle b) {
    return a.SameHandle(b);
  }
  bool SameHandle(ShapeHandle a, ShapeHandle b) { return a.SameHandle(b); }
  bool IsSet(DimensionHandle d) { return d.IsSet(); }
  bool IsSet(ShapeHandle s) { return s.IsSet(); }
};

TEST_F(ShapeInferenceTest, InputOutputByName) {
  // Setup test to contain an input tensor list of size 3.
  OpDef op_def = MakeOpDefWithLists();
  NodeDef def;
  auto s = NodeDefBuilder("dummy", &op_def)
               .Attr("N", 3)
               .Input(FakeInput(DT_FLOAT))
               .Finalize(&def);
  InferenceContext c(&def, op_def, {S({1, 5}), S({2, 5}), S({1, 3})}, {}, {},
                     {}, {});

  EXPECT_EQ("5", c.DebugString(c.NumElements(c.input(0))));
  EXPECT_EQ("10", c.DebugString(c.NumElements(c.input(1))));
  EXPECT_EQ("3", c.DebugString(c.NumElements(c.input(2))));
  // Test getters.
  std::vector<ShapeHandle> shapes;
  EXPECT_FALSE(c.input("nonexistent", &shapes).ok());
  TF_EXPECT_OK(c.input("input", &shapes));
  EXPECT_EQ("[1,5]", c.DebugString(shapes[0]));
  EXPECT_EQ("[2,5]", c.DebugString(shapes[1]));
  EXPECT_EQ("[1,3]", c.DebugString(shapes[2]));

  // Test setters.
  EXPECT_FALSE(c.set_output("nonexistent", shapes).ok());
  TF_EXPECT_OK(c.set_output("output", shapes));
  EXPECT_EQ("5", c.DebugString(c.NumElements(c.output(0))));
  EXPECT_EQ("10", c.DebugString(c.NumElements(c.output(1))));
  EXPECT_EQ("3", c.DebugString(c.NumElements(c.output(2))));
}

static OpDef MakeOpDef(int num_inputs, int num_outputs) {
  OpRegistrationData op_reg_data;
  OpDefBuilder b("dummy");
  for (int i = 0; i < num_inputs; ++i) {
    b.Input(strings::StrCat("i", i, ": float"));
  }
  for (int i = 0; i < num_outputs; ++i) {
    b.Output(strings::StrCat("o", i, ": float"));
  }
  CHECK(b.Attr("foo:string").Finalize(&op_reg_data).ok());
  return op_reg_data.op_def;
}

TEST_F(ShapeInferenceTest, DimensionOrConstant) {
  NodeDef def;
  InferenceContext c(&def, MakeOpDef(1, 1), {Unknown()}, {}, {}, {}, {});
  EXPECT_EQ(InferenceContext::kUnknownDim,
            c.Value(InferenceContext::kUnknownDim));
  EXPECT_EQ(1, c.Value(1));

#ifndef NDEBUG
  // Only run death test if DCHECKS are enabled.
  EXPECT_DEATH(c.Value(-7), "Dimension must be non\\-negative or equal to");
#endif
}

TEST_F(ShapeInferenceTest, Run) {
  NodeDef def;
  def.set_name("foo");
  def.set_op("foo_op");
  InferenceContext c(&def, MakeOpDef(1, 2), {S({1})}, {}, {}, {}, {});
  TF_ASSERT_OK(c.construction_status());

  {
    auto fn = [](InferenceContext* c) {
      ShapeHandle h;
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(0), 6, &h));
      c->set_output(0, c->input(0));
      return Status::OK();
    };
    TF_ASSERT_OK(c.Run(fn));
  }

  {
    auto fn = [](InferenceContext* c) {
      ShapeHandle h;
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(0), 0, &h));
      c->set_output(0, c->input(0));
      return Status::OK();
    };
    Status s = c.Run(fn);
    // Extra error message is attached when Run fails.
    EXPECT_TRUE(StringPiece(s.ToString())
                    .contains("Shape must be at most rank 0 but "
                              "is rank 1 for 'foo' (op: "
                              "'foo_op')"))
        << s;
  }
}

// Tests different context data added when Run returns error.
TEST_F(ShapeInferenceTest, AttachContext) {
  NodeDef def;
  def.set_name("foo");
  def.set_op("foo_op");
  // Error when no constant tensors were requested.
  {
    InferenceContext c(&def, MakeOpDef(1, 2), {S({1, 2, 3})}, {}, {}, {}, {});
    TF_ASSERT_OK(c.construction_status());
    auto fn = [](InferenceContext* c) {
      ShapeHandle h;
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(0), 0, &h));
      c->set_output(0, c->input(0));
      return Status::OK();
    };
    EXPECT_EQ(
        "Invalid argument: Shape must be at most rank 0 but is rank 3 for "
        "'foo' (op: 'foo_op') with input shapes: [1,2,3].",
        c.Run(fn).ToString());
  }

  // Error when a constant tensor value was requested.
  {
    Tensor input_t =
        ::tensorflow::test::AsTensor<float>({1.1, 2.2, 3.3, 4.4, 5.5});
    InferenceContext c(&def, MakeOpDef(2, 2), {S({1, 2, 3}), S({4, 5})},
                       {nullptr, &input_t}, {}, {}, {});
    TF_ASSERT_OK(c.construction_status());
    auto fn = [](InferenceContext* c) {
      c->input_tensor(0);  // get this one, but it's null - won't be in error.
      c->input_tensor(1);  // get this one, will now be in error.
      ShapeHandle h;
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(0), 0, &h));
      c->set_output(0, c->input(0));
      return Status::OK();
    };
    EXPECT_EQ(
        "Invalid argument: Shape must be at most rank 0 but is rank 3 for "
        "'foo' (op: 'foo_op') with input shapes: [1,2,3], [4,5] and with "
        "computed input tensors: input[1] = <1.1 2.2 3.3 4.4 5.5>.",
        c.Run(fn).ToString());
  }

  // Error when a constant tensor value as shape was requested, but no partial
  // shapes provided.
  {
    Tensor input_t = ::tensorflow::test::AsTensor<int32>({1, 2, 3, 4, 5});
    InferenceContext c(&def, MakeOpDef(2, 2), {S({3}), S({4})},
                       {nullptr, &input_t}, {}, {}, {});
    TF_ASSERT_OK(c.construction_status());
    auto fn = [](InferenceContext* c) {
      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &s));
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(1, &s));
      ShapeHandle h;
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(0), 0, &h));
      c->set_output(0, c->input(0));
      return Status::OK();
    };
    EXPECT_EQ(
        "Invalid argument: Shape must be at most rank 0 but is rank 1 for "
        "'foo' (op: 'foo_op') with input shapes: [3], [4] and with computed "
        "input tensors: input[1] = <1 2 3 4 5>.",
        c.Run(fn).ToString());
  }

  // Error when a constant tensor value as shape was requested, and a partial
  // shape was provided.
  {
    Tensor input_t = ::tensorflow::test::AsTensor<int32>({1, 2, 3, 4, 5});
    InferenceContext c(&def, MakeOpDef(2, 2), {S({3}), S({4})},
                       {nullptr, &input_t}, {S({10, -1, 5}), Unknown()}, {},
                       {});
    TF_ASSERT_OK(c.construction_status());
    auto fn = [](InferenceContext* c) {
      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &s));
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(1, &s));
      ShapeHandle h;
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(0), 0, &h));
      c->set_output(0, c->input(0));
      return Status::OK();
    };
    EXPECT_EQ(
        "Invalid argument: Shape must be at most rank 0 but is rank 1 for "
        "'foo' (op: 'foo_op') with input shapes: [3], [4] and with computed "
        "input tensors: input[1] = <1 2 3 4 5> and with input tensors computed "
        "as partial shapes: input[0] = [10,?,5].",
        c.Run(fn).ToString());
  }
}

TEST_F(ShapeInferenceTest, RankAndDimInspection) {
  NodeDef def;
  InferenceContext c(&def, MakeOpDef(3, 2), {Unknown(), S({1, -1, 3}), S({})},
                     {}, {}, {}, {});
  EXPECT_EQ(3, c.num_inputs());
  EXPECT_EQ(2, c.num_outputs());

  auto in0 = c.input(0);
  EXPECT_EQ("?", c.DebugString(in0));
  EXPECT_FALSE(c.RankKnown(in0));
  EXPECT_EQ(InferenceContext::kUnknownRank, c.Rank(in0));
  EXPECT_EQ("?", c.DebugString(c.Dim(in0, 0)));
  EXPECT_EQ("?", c.DebugString(c.Dim(in0, -1)));
  EXPECT_EQ("?", c.DebugString(c.Dim(in0, 1000)));

  auto in1 = c.input(1);
  EXPECT_EQ("[1,?,3]", c.DebugString(in1));
  EXPECT_TRUE(c.RankKnown(in1));
  EXPECT_EQ(3, c.Rank(in1));
  auto d = c.Dim(in1, 0);
  EXPECT_EQ(1, c.Value(d));
  EXPECT_TRUE(SameHandle(d, c.Dim(in1, -3)));
  EXPECT_TRUE(c.ValueKnown(d));
  EXPECT_EQ("1", c.DebugString(d));
  d = c.Dim(in1, 1);
  EXPECT_EQ(InferenceContext::kUnknownDim, c.Value(d));
  EXPECT_FALSE(c.ValueKnown(d));
  EXPECT_TRUE(SameHandle(d, c.Dim(in1, -2)));
  EXPECT_EQ("?", c.DebugString(d));
  d = c.Dim(in1, 2);
  EXPECT_EQ(3, c.Value(d));
  EXPECT_TRUE(SameHandle(d, c.Dim(in1, -1)));
  EXPECT_TRUE(c.ValueKnown(d));
  EXPECT_EQ("3", c.DebugString(d));

  auto in2 = c.input(2);
  EXPECT_EQ("[]", c.DebugString(in2));
  EXPECT_TRUE(c.RankKnown(in2));
  EXPECT_EQ(0, c.Rank(in2));
}

TEST_F(ShapeInferenceTest, NumElements) {
  NodeDef def;
  InferenceContext c(&def, MakeOpDef(3, 2),
                     {Unknown(), S({1, -1, 3}), S({5, 4, 3, 2})}, {}, {}, {},
                     {});

  EXPECT_EQ("?", c.DebugString(c.NumElements(c.input(0))));
  EXPECT_EQ("?", c.DebugString(c.NumElements(c.input(1))));

  // Different handles (not the same unknown value).
  EXPECT_FALSE(SameHandle(c.Dim(c.input(1), 1), c.NumElements(c.input(1))));

  EXPECT_EQ("120", c.DebugString(c.NumElements(c.input(2))));
}

TEST_F(ShapeInferenceTest, WithRank) {
  NodeDef def;
  InferenceContext c(&def, MakeOpDef(2, 2), {Unknown(), S({1, -1, 3})}, {}, {},
                     {}, {});

  auto in0 = c.input(0);
  auto in1 = c.input(1);
  ShapeHandle s1;
  ShapeHandle s2;

  // WithRank on a shape with unknown dimensionality always succeeds.
  EXPECT_TRUE(c.WithRank(in0, 1, &s1).ok());
  EXPECT_EQ("[?]", c.DebugString(s1));

  EXPECT_TRUE(c.WithRank(in0, 2, &s2).ok());
  EXPECT_EQ("[?,?]", c.DebugString(s2));
  EXPECT_FALSE(SameHandle(s1, s2));
  EXPECT_FALSE(SameHandle(c.Dim(s2, 0), c.Dim(s2, 1)));

  EXPECT_TRUE(c.WithRank(in0, 1, &s2).ok());
  EXPECT_EQ("[?]", c.DebugString(s2));
  EXPECT_FALSE(SameHandle(s1, s2));

  EXPECT_TRUE(c.WithRank(in0, 0, &s1).ok());
  EXPECT_EQ("[]", c.DebugString(s1));

  // WithRank on shape with known dimensionality.
  s1 = in1;
  EXPECT_EQ("Invalid argument: Shape must be rank 2 but is rank 3",
            c.WithRank(in1, 2, &s1).ToString());
  EXPECT_FALSE(IsSet(s1));
  EXPECT_TRUE(c.WithRank(in1, 3, &s1).ok());
  EXPECT_TRUE(SameHandle(s1, in1));

  // Inputs are unchanged.
  EXPECT_EQ("?", c.DebugString(in0));
  EXPECT_EQ("[1,?,3]", c.DebugString(in1));
}

TEST_F(ShapeInferenceTest, WithRankAtMost) {
  NodeDef def;
  InferenceContext c(&def, MakeOpDef(2, 2), {Unknown(), S({1, -1, 3})}, {}, {},
                     {}, {});

  auto in0 = c.input(0);
  auto in1 = c.input(1);
  ShapeHandle s1;
  ShapeHandle s2;

  // WithRankAtMost on a shape with unknown dimensionality always succeeds.
  EXPECT_TRUE(c.WithRankAtMost(in0, 1, &s1).ok());
  EXPECT_EQ("?", c.DebugString(s1));
  EXPECT_FALSE(SameHandle(in0, s1));

  EXPECT_TRUE(c.WithRankAtMost(in0, 2, &s2).ok());
  EXPECT_EQ("?", c.DebugString(s2));
  EXPECT_FALSE(SameHandle(s1, s2));

  // WithRankAtMost on shape with known dimensionality.
  s1 = in1;
  EXPECT_TRUE(
      StringPiece(c.WithRankAtMost(in1, 2, &s1).ToString())
          .contains(
              "Invalid argument: Shape must be at most rank 2 but is rank 3"));

  EXPECT_FALSE(IsSet(s1));
  EXPECT_TRUE(c.WithRankAtMost(in1, 3, &s1).ok());
  EXPECT_TRUE(SameHandle(s1, in1));
  EXPECT_TRUE(c.WithRankAtMost(in1, 4, &s1).ok());
  EXPECT_TRUE(SameHandle(s1, in1));
  EXPECT_TRUE(c.WithRankAtMost(in1, 5, &s1).ok());
  EXPECT_TRUE(SameHandle(s1, in1));

  // Inputs are unchanged.
  EXPECT_EQ("?", c.DebugString(in0));
  EXPECT_EQ("[1,?,3]", c.DebugString(in1));
}

TEST_F(ShapeInferenceTest, WithRankAtLeast) {
  NodeDef def;
  InferenceContext c(&def, MakeOpDef(2, 2), {Unknown(), S({1, -1, 3})}, {}, {},
                     {}, {});

  auto in0 = c.input(0);
  auto in1 = c.input(1);
  ShapeHandle s1;
  ShapeHandle s2;

  // WithRankAtLeast on a shape with unknown dimensionality always succeeds.
  EXPECT_TRUE(c.WithRankAtLeast(in0, 1, &s1).ok());
  EXPECT_EQ("?", c.DebugString(s1));
  EXPECT_FALSE(SameHandle(in0, s1));

  EXPECT_TRUE(c.WithRankAtLeast(in0, 2, &s2).ok());
  EXPECT_EQ("?", c.DebugString(s2));
  EXPECT_FALSE(SameHandle(s1, s2));

  // WithRankAtLeast on shape with known dimensionality.
  s1 = in1;
  EXPECT_TRUE(
      StringPiece(c.WithRankAtLeast(in1, 4, &s1).ToString())
          .contains(
              "Invalid argument: Shape must be at least rank 4 but is rank 3"));

  EXPECT_FALSE(IsSet(s1));
  EXPECT_TRUE(c.WithRankAtLeast(in1, 3, &s1).ok());
  EXPECT_TRUE(SameHandle(s1, in1));
  EXPECT_TRUE(c.WithRankAtLeast(in1, 2, &s1).ok());
  EXPECT_TRUE(SameHandle(s1, in1));
  EXPECT_TRUE(c.WithRankAtLeast(in1, 0, &s1).ok());
  EXPECT_TRUE(SameHandle(s1, in1));

  // Inputs are unchanged.
  EXPECT_EQ("?", c.DebugString(in0));
  EXPECT_EQ("[1,?,3]", c.DebugString(in1));
}

TEST_F(ShapeInferenceTest, WithValue) {
  NodeDef def;
  InferenceContext c(&def, MakeOpDef(1, 2), {S({1, -1})}, {}, {}, {}, {});

  auto d0 = c.Dim(c.input(0), 0);
  auto d1 = c.Dim(c.input(0), 1);
  DimensionHandle out1;
  DimensionHandle out2;

  // WithValue on a dimension with unknown value always succeeds.
  EXPECT_TRUE(c.WithValue(d1, 1, &out1).ok());
  EXPECT_EQ(1, c.Value(out1));

  EXPECT_TRUE(c.WithValue(d1, 2, &out2).ok());
  EXPECT_EQ(2, c.Value(out2));
  EXPECT_FALSE(SameHandle(out1, out2));
  EXPECT_FALSE(SameHandle(out1, d1));

  EXPECT_TRUE(c.WithValue(d1, 1, &out2).ok());
  EXPECT_EQ(1, c.Value(out2));
  EXPECT_FALSE(SameHandle(out1, out2));

  // WithValue on dimension with known size.
  out1 = d0;

  EXPECT_TRUE(StringPiece(c.WithValue(d0, 0, &out1).ToString())
                  .contains("Invalid argument: Dimension must be 0 but is 1"));
  EXPECT_FALSE(IsSet(out1));
  out1 = d0;
  EXPECT_TRUE(StringPiece(c.WithValue(d0, 2, &out1).ToString())
                  .contains("Invalid argument: Dimension must be 2 but is 1"));

  EXPECT_FALSE(IsSet(out1));
  EXPECT_TRUE(c.WithValue(d0, 1, &out1).ok());
  EXPECT_TRUE(SameHandle(d0, out1));

  // Inputs are unchanged.
  EXPECT_EQ("1", c.DebugString(d0));
  EXPECT_EQ("?", c.DebugString(d1));
}

TEST_F(ShapeInferenceTest, MergeDim) {
  NodeDef def;
  InferenceContext c(&def, MakeOpDef(1, 2), {S({2, -1, 2, 1, -1})}, {}, {}, {},
                     {});

  auto d2 = c.Dim(c.input(0), 0);
  auto d_unknown = c.Dim(c.input(0), 1);
  auto d2_b = c.Dim(c.input(0), 2);
  auto d1 = c.Dim(c.input(0), 3);
  auto d_unknown_b = c.Dim(c.input(0), 4);
  DimensionHandle out;

  // Merging anything with unknown returns the same pointer.
  EXPECT_TRUE(c.Merge(d2, d_unknown, &out).ok());
  EXPECT_TRUE(SameHandle(d2, out));
  EXPECT_TRUE(c.Merge(d_unknown, d2, &out).ok());
  EXPECT_TRUE(SameHandle(d2, out));
  EXPECT_TRUE(c.Merge(d_unknown, d_unknown_b, &out).ok());
  EXPECT_TRUE(SameHandle(d_unknown, out));

  // Merging with self returns self.
  EXPECT_TRUE(c.Merge(d2, d2, &out).ok());
  EXPECT_TRUE(SameHandle(d2, out));
  EXPECT_TRUE(c.Merge(d_unknown, d_unknown, &out).ok());
  EXPECT_TRUE(SameHandle(d_unknown, out));

  // Merging equal values returns first one.
  EXPECT_TRUE(c.Merge(d2, d2_b, &out).ok());
  EXPECT_TRUE(SameHandle(d2, out));
  EXPECT_TRUE(c.Merge(d2_b, d2, &out).ok());
  EXPECT_TRUE(SameHandle(d2_b, out));

  // Merging inequal values is an error.
  EXPECT_TRUE(
      StringPiece(c.Merge(d2, d1, &out).ToString())
          .contains(
              "Invalid argument: Dimensions must be equal, but are 2 and 1"));

  EXPECT_FALSE(IsSet(out));
  EXPECT_TRUE(
      StringPiece(c.Merge(d1, d2, &out).ToString())
          .contains(
              "Invalid argument: Dimensions must be equal, but are 1 and 2"));

  EXPECT_FALSE(IsSet(out));
}

TEST_F(ShapeInferenceTest, MergeShape) {
  NodeDef def;
  InferenceContext c(&def, MakeOpDef(7, 2),
                     {Unknown(), S({1, 2}), S({-1, 2}), S({1, -1}), S({1, 3}),
                      Unknown(), S({1})},
                     {}, {}, {}, {});

  auto s_unknown = c.input(0);
  auto s_1_2 = c.input(1);
  auto s_u_2 = c.input(2);
  auto s_1_u = c.input(3);
  auto s_1_3 = c.input(4);
  auto s_unknown_b = c.input(5);
  auto s_1 = c.input(6);
  ShapeHandle out;

  // Merging any shape with unknown returns the shape.
  EXPECT_TRUE(c.Merge(s_unknown, s_1_2, &out).ok());
  EXPECT_TRUE(SameHandle(s_1_2, out));
  EXPECT_TRUE(c.Merge(s_u_2, s_unknown, &out).ok());
  EXPECT_TRUE(SameHandle(s_u_2, out));
  EXPECT_TRUE(c.Merge(s_unknown, s_unknown_b, &out).ok());
  EXPECT_TRUE(SameHandle(s_unknown, out));

  // Merging with self returns self.
  EXPECT_TRUE(c.Merge(s_1_2, s_1_2, &out).ok());
  EXPECT_TRUE(SameHandle(out, s_1_2));

  // Merging where one of the inputs is the right answer - return that input.
  out = ShapeHandle();
  EXPECT_TRUE(c.Merge(s_1_2, s_u_2, &out).ok());
  EXPECT_TRUE(SameHandle(s_1_2, out));
  out = ShapeHandle();
  EXPECT_TRUE(c.Merge(s_u_2, s_1_2, &out).ok());
  EXPECT_TRUE(SameHandle(s_1_2, out));

  // Merging where neither input is the right answer.
  EXPECT_TRUE(c.Merge(s_u_2, s_1_u, &out).ok());
  EXPECT_FALSE(SameHandle(out, s_u_2));
  EXPECT_FALSE(SameHandle(out, s_1_u));
  EXPECT_EQ("[1,2]", c.DebugString(out));
  EXPECT_TRUE(SameHandle(c.Dim(s_1_u, 0), c.Dim(out, 0)));
  EXPECT_TRUE(SameHandle(c.Dim(s_u_2, 1), c.Dim(out, 1)));

  // Incompatible merges give errors and set out to nullptr.
  out = s_unknown;
  EXPECT_TRUE(
      StringPiece(c.Merge(s_u_2, s_1_3, &out).ToString())
          .contains(
              "Invalid argument: Dimension 1 in both shapes must be equal, but "
              "are 2 and 3"));

  EXPECT_FALSE(IsSet(out));
  out = s_unknown;
  EXPECT_TRUE(
      StringPiece(c.Merge(s_1_3, s_u_2, &out).ToString())
          .contains(
              "Invalid argument: Dimension 1 in both shapes must be equal, but "
              "are 3 and 2"));

  EXPECT_FALSE(IsSet(out));
  out = s_unknown;
  EXPECT_TRUE(
      StringPiece(c.Merge(s_1, s_1_2, &out).ToString())
          .contains(
              "Invalid argument: Shapes must be equal rank, but are 1 and 2"));

  EXPECT_FALSE(IsSet(out));
}

TEST_F(ShapeInferenceTest, MergePrefix) {
  NodeDef def;
  InferenceContext c(&def, MakeOpDef(4, 2),
                     {
                         Unknown(), S({-1, 2}), S({1, -1, 3}), S({2, 4}),
                     },
                     {}, {}, {}, {});

  auto s_unknown = c.input(0);
  auto s_u_2 = c.input(1);
  auto s_1_u_3 = c.input(2);
  auto s_2_4 = c.input(3);

  ShapeHandle s_out;
  ShapeHandle s_prefix_out;

  // Merging with unknown returns the inputs.
  EXPECT_TRUE(c.MergePrefix(s_unknown, s_u_2, &s_out, &s_prefix_out).ok());
  EXPECT_TRUE(SameHandle(s_out, s_unknown));
  EXPECT_TRUE(SameHandle(s_prefix_out, s_u_2));
  EXPECT_TRUE(c.MergePrefix(s_1_u_3, s_unknown, &s_out, &s_prefix_out).ok());
  EXPECT_TRUE(SameHandle(s_out, s_1_u_3));
  EXPECT_TRUE(SameHandle(s_prefix_out, s_unknown));

  EXPECT_TRUE(c.MergePrefix(s_1_u_3, s_u_2, &s_out, &s_prefix_out).ok());
  EXPECT_FALSE(SameHandle(s_out, s_1_u_3));
  EXPECT_EQ("[1,2]", c.DebugString(s_prefix_out));
  EXPECT_EQ("[1,2,3]", c.DebugString(s_out));
  EXPECT_TRUE(SameHandle(c.Dim(s_prefix_out, 0), c.Dim(s_out, 0)));
  EXPECT_TRUE(SameHandle(c.Dim(s_out, 0), c.Dim(s_1_u_3, 0)));
  EXPECT_TRUE(SameHandle(c.Dim(s_prefix_out, 1), c.Dim(s_out, 1)));
  EXPECT_TRUE(SameHandle(c.Dim(s_prefix_out, 1), c.Dim(s_u_2, 1)));

  // Incompatible merges give errors and set outs to nullptr.
  s_out = s_unknown;
  s_prefix_out = s_unknown;
  EXPECT_TRUE(
      StringPiece(
          c.MergePrefix(s_1_u_3, s_2_4, &s_out, &s_prefix_out).ToString())
          .contains(
              "Invalid argument: Dimensions must be equal, but are 1 and 2"));

  EXPECT_FALSE(IsSet(s_out));
  EXPECT_FALSE(IsSet(s_prefix_out));

  s_out = s_unknown;
  s_prefix_out = s_unknown;
  EXPECT_TRUE(
      StringPiece(
          c.MergePrefix(s_2_4, s_1_u_3, &s_out, &s_prefix_out).ToString())
          .contains(
              "Invalid argument: Shape must be at least rank 3 but is rank 2"));
  EXPECT_FALSE(IsSet(s_out));
  EXPECT_FALSE(IsSet(s_prefix_out));
}

TEST_F(ShapeInferenceTest, Subshape) {
  NodeDef def;
  InferenceContext c(&def, MakeOpDef(2, 2), {S({1, 2, 3, -1, 5}), Unknown()},
                     {}, {}, {}, {});

  ShapeHandle unknown = c.input(1);
  ShapeHandle out;
  EXPECT_TRUE(c.Subshape(unknown, 0, &out).ok());
  EXPECT_EQ("?", c.DebugString(out));
  EXPECT_TRUE(SameHandle(out, unknown));
  EXPECT_TRUE(c.Subshape(unknown, 1, &out).ok());
  EXPECT_EQ("?", c.DebugString(out));
  EXPECT_FALSE(SameHandle(out, unknown));
  EXPECT_TRUE(c.Subshape(unknown, 200, &out).ok());
  EXPECT_EQ("?", c.DebugString(out));
  EXPECT_FALSE(SameHandle(out, unknown));

  const int kFullRank = 5;
  ShapeHandle out_arr[4];
  auto in0 = c.input(0);
  EXPECT_TRUE(c.Subshape(in0, 0, &out).ok());
  EXPECT_EQ("[1,2,3,?,5]", c.DebugString(out));
  EXPECT_TRUE(SameHandle(out, in0));
  EXPECT_EQ(kFullRank, c.Rank(out));
  for (int start = 0; start <= kFullRank + 1; ++start) {
    for (int end = start; end <= kFullRank + 1; ++end) {
      // Get subshapes using different start and end values that give the same
      // range.
      const int neg_start =
          start >= kFullRank ? kFullRank : (start - kFullRank);
      const int neg_end = end >= kFullRank ? kFullRank : (end - kFullRank);
      ASSERT_TRUE(c.Subshape(in0, start, end, &out_arr[0]).ok());
      ASSERT_TRUE(c.Subshape(in0, neg_start, end, &out_arr[1]).ok());
      ASSERT_TRUE(c.Subshape(in0, start, neg_end, &out_arr[2]).ok());
      ASSERT_TRUE(c.Subshape(in0, neg_start, neg_end, &out_arr[3]).ok());

      // Verify all computed subshapes.
      for (int arr_idx = 0; arr_idx < 4; ++arr_idx) {
        out = out_arr[arr_idx];
        ASSERT_EQ(std::min(kFullRank, end) - std::min(kFullRank, start),
                  c.Rank(out))
            << "start: " << start << " end: " << end << " arr_idx: " << arr_idx
            << " in0: " << c.DebugString(in0) << " out: " << c.DebugString(out);
        for (int d = 0; d < c.Rank(out); ++d) {
          EXPECT_TRUE(SameHandle(c.Dim(in0, start + d), c.Dim(out, d)))
              << "arr_idx: " << arr_idx;
        }
      }
    }
  }

  // Errors.
  out = unknown;
  EXPECT_TRUE(StringPiece(c.Subshape(in0, 6, -3, &out).ToString())
                  .contains("Invalid argument: Subshape must have computed "
                            "start <= end, but is 5 "
                            "and 2 (computed from start 6 and end -3 over "
                            "shape with rank 5)"));
  EXPECT_FALSE(IsSet(out));
  out = unknown;
  EXPECT_TRUE(StringPiece(c.Subshape(in0, -50, 100, &out).ToString())
                  .contains("Invalid argument: Subshape start out of "
                            "bounds: -50, for shape with "
                            "rank 5"));

  EXPECT_FALSE(IsSet(out));
  out = unknown;
  EXPECT_TRUE(StringPiece(c.Subshape(in0, 0, -50, &out).ToString())
                  .contains("Invalid argument: Subshape end out of bounds: "
                            "-50, for shape with rank "
                            "5"));

  EXPECT_FALSE(IsSet(out));
}

TEST_F(ShapeInferenceTest, Concatenate) {
  NodeDef def;
  InferenceContext c(&def, MakeOpDef(3, 2),
                     {S({1, -1, 3}), S({4, 5}), Unknown()}, {}, {}, {}, {});

  auto in0 = c.input(0);
  auto in1 = c.input(1);
  ShapeHandle unknown = c.input(2);
  ShapeHandle out;
  EXPECT_TRUE(c.Concatenate(unknown, unknown, &out).ok());
  EXPECT_EQ("?", c.DebugString(out));
  EXPECT_FALSE(SameHandle(out, unknown));
  EXPECT_TRUE(c.Concatenate(unknown, in0, &out).ok());
  EXPECT_EQ("?", c.DebugString(out));
  EXPECT_FALSE(SameHandle(out, unknown));

  EXPECT_TRUE(c.Concatenate(in0, in1, &out).ok());
  EXPECT_EQ("[1,?,3,4,5]", c.DebugString(out));
  int out_i = 0;
  for (int i = 0; i < c.Rank(in0); ++i, ++out_i) {
    EXPECT_TRUE(SameHandle(c.Dim(in0, i), c.Dim(out, out_i)));
  }
  for (int i = 0; i < c.Rank(in1); ++i, ++out_i) {
    EXPECT_TRUE(SameHandle(c.Dim(in1, i), c.Dim(out, out_i)));
  }
}

TEST_F(ShapeInferenceTest, ReplaceDim) {
  NodeDef def;
  InferenceContext c(&def, MakeOpDef(2, 0), {S({1, 2, 3}), Unknown()}, {}, {},
                     {}, {});

  auto in = c.input(0);
  auto unknown = c.input(1);

  ShapeHandle replaced;
  EXPECT_TRUE(c.ReplaceDim(in, 0, c.Dim(in, 1), &replaced).ok());
  EXPECT_EQ("[2,2,3]", c.DebugString(replaced));
  EXPECT_TRUE(c.ReplaceDim(in, 2, c.Dim(in, 1), &replaced).ok());
  EXPECT_EQ("[1,2,2]", c.DebugString(replaced));
  EXPECT_TRUE(c.ReplaceDim(in, 1, c.Dim(in, 2), &replaced).ok());
  EXPECT_EQ("[1,3,3]", c.DebugString(replaced));
  EXPECT_TRUE(c.ReplaceDim(unknown, 0, c.Dim(in, 1), &replaced).ok());
  EXPECT_EQ("?", c.DebugString(replaced));

  // Negative indexing.
  EXPECT_TRUE(c.ReplaceDim(in, -1, c.Dim(in, 1), &replaced).ok());
  EXPECT_EQ("[1,2,2]", c.DebugString(replaced));
  EXPECT_TRUE(c.ReplaceDim(unknown, -1, c.Dim(in, 1), &replaced).ok());
  EXPECT_EQ("?", c.DebugString(replaced));

  // out of range indexing.
  EXPECT_FALSE(c.ReplaceDim(in, 3, c.Dim(in, 1), &replaced).ok());
  EXPECT_FALSE(IsSet(replaced));
  replaced = in;
  EXPECT_FALSE(c.ReplaceDim(in, -4, c.Dim(in, 1), &replaced).ok());
  EXPECT_FALSE(IsSet(replaced));
}

TEST_F(ShapeInferenceTest, MakeShape) {
  NodeDef def;
  InferenceContext c(&def, MakeOpDef(1, 2), {S({1, 2, 3, -1, 5})}, {}, {}, {},
                     {});

  std::vector<DimensionHandle> dims;
  auto in0 = c.input(0);
  const int rank = c.Rank(in0);
  for (int i = 0; i < rank; ++i) {
    dims.push_back(c.Dim(in0, rank - i - 1));
  }

  auto s = c.MakeShape(dims);
  EXPECT_EQ("[5,?,3,2,1]", c.DebugString(s));
  EXPECT_TRUE(SameHandle(c.Dim(s, 0), c.Dim(in0, rank - 1)));

  auto s2 = c.MakeShape(dims);
  EXPECT_FALSE(SameHandle(s, s2));
  EXPECT_TRUE(SameHandle(c.Dim(s2, 0), c.Dim(in0, rank - 1)));

  auto s3 = c.MakeShape({1, 2, dims[2]});
  EXPECT_FALSE(SameHandle(s, s3));
  EXPECT_EQ("[1,2,3]", c.DebugString(s3));
}

TEST_F(ShapeInferenceTest, UnknownShape) {
  NodeDef def;
  std::vector<ShapeHandle> empty;
  InferenceContext c(&def, MakeOpDef(0, 2), empty, {}, {}, {}, {});

  auto u0 = c.UnknownShape();
  auto u1 = c.UnknownShape();
  EXPECT_EQ("?", c.DebugString(u0));
  EXPECT_EQ("?", c.DebugString(u1));
  EXPECT_FALSE(SameHandle(u0, u1));
}

TEST_F(ShapeInferenceTest, Scalar) {
  NodeDef def;
  std::vector<ShapeHandle> empty;
  InferenceContext c(&def, MakeOpDef(0, 2), empty, {}, {}, {}, {});

  auto s0 = c.Scalar();
  EXPECT_EQ("[]", c.DebugString(s0));
  auto s1 = c.Scalar();
  EXPECT_EQ("[]", c.DebugString(s1));
}

TEST_F(ShapeInferenceTest, Vector) {
  NodeDef def;
  std::vector<ShapeHandle> empty;
  InferenceContext c(&def, MakeOpDef(0, 2), empty, {}, {}, {}, {});

  auto s0 = c.Vector(1);
  EXPECT_EQ("[1]", c.DebugString(s0));
  auto s1 = c.Vector(InferenceContext::kUnknownDim);
  EXPECT_EQ("[?]", c.DebugString(s1));

  auto d1 = c.UnknownDim();
  auto s2 = c.Vector(d1);
  EXPECT_EQ("[?]", c.DebugString(s2));
  EXPECT_TRUE(SameHandle(d1, c.Dim(s2, 0)));
}

TEST_F(ShapeInferenceTest, Matrix) {
  NodeDef def;
  std::vector<ShapeHandle> empty;
  InferenceContext c(&def, MakeOpDef(0, 2), empty, {}, {}, {}, {});

  auto s0 = c.Matrix(1, 2);
  EXPECT_EQ("[1,2]", c.DebugString(s0));
  auto s1 = c.Matrix(0, InferenceContext::kUnknownDim);
  EXPECT_EQ("[0,?]", c.DebugString(s1));

  auto d1 = c.UnknownDim();
  auto d2 = c.UnknownDim();
  auto s2 = c.Matrix(d1, d2);
  EXPECT_EQ("[?,?]", c.DebugString(s2));
  EXPECT_TRUE(SameHandle(d1, c.Dim(s2, 0)));
  EXPECT_TRUE(SameHandle(d2, c.Dim(s2, 1)));

  auto s3 = c.Matrix(d1, 100);
  EXPECT_EQ("[?,100]", c.DebugString(s3));
  EXPECT_TRUE(SameHandle(d1, c.Dim(s2, 0)));
}

TEST_F(ShapeInferenceTest, MakeShapeFromShapeTensor) {
  auto create = [&](Tensor* t) {
    NodeDef def;
    InferenceContext c(&def, MakeOpDef(1, 0), {Unknown()}, {t}, {}, {}, {});
    ShapeHandle out;
    Status s = c.MakeShapeFromShapeTensor(0, &out);
    if (s.ok()) {
      return c.DebugString(out);
    } else {
      EXPECT_FALSE(IsSet(out));
      return s.error_message();
    }
  };

  Tensor t;
  EXPECT_EQ("?", create(nullptr));

  t = ::tensorflow::test::AsTensor<int32>({1, 2, 3});
  EXPECT_EQ("[1,2,3]", create(&t));

  t = ::tensorflow::test::AsTensor<int64>({3, 2, 1});
  EXPECT_EQ("[3,2,1]", create(&t));

  t = ::tensorflow::test::AsTensor<int64>({3, -1, 1});
  EXPECT_EQ("[3,?,1]", create(&t));

  t = ::tensorflow::test::AsTensor<int64>({});
  EXPECT_EQ("[]", create(&t));

  t = ::tensorflow::test::AsTensor<float>({1, 2, 3});
  EXPECT_TRUE(
      StringPiece(create(&t))
          .contains("Input tensor must be int32 or int64, but was float"));

  t = ::tensorflow::test::AsScalar<int32>(1);
  EXPECT_TRUE(StringPiece(create(&t))
                  .contains("Input tensor must be rank 1, but was rank 0"));

  t = ::tensorflow::test::AsTensor<int32>({1, 2}, TensorShape{2, 1});
  EXPECT_TRUE(StringPiece(create(&t))
                  .contains("Input tensor must be rank 1, but was rank 2"));

  // Test negative values for the dims.
  t = ::tensorflow::test::AsTensor<int64>({3, -2, 1});
  EXPECT_TRUE(StringPiece(create(&t))
                  .contains("Invalid value in tensor used for shape: -2"));

  // Test negative values for the dims.
  t = ::tensorflow::test::AsTensor<int32>({3, -2, 1});
  EXPECT_TRUE(StringPiece(create(&t))
                  .contains("Invalid value in tensor used for shape: -2"));

  // Test when the input shape is wrong.
  {
    NodeDef def;
    InferenceContext c(&def, MakeOpDef(1, 0), {S({1, -1})}, {nullptr}, {}, {},
                       {});
    ShapeHandle out;
    EXPECT_EQ("Shape must be rank 1 but is rank 2",
              c.MakeShapeFromShapeTensor(0, &out).error_message());
  }
}

TEST_F(ShapeInferenceTest, MakeShapeFromShapeProto) {
  NodeDef def;
  std::vector<ShapeHandle> empty;
  InferenceContext c(&def, MakeOpDef(0, 2), empty, {}, {}, {}, {});
  TensorShapeProto proto;

  // With a set unknown rank.
  ShapeHandle out;
  proto.set_unknown_rank(true);
  EXPECT_TRUE(c.MakeShapeFromShapeProto(proto, &out).ok());
  EXPECT_EQ("?", c.DebugString(out));
  proto.add_dim()->set_size(0);
  EXPECT_TRUE(
      StringPiece(c.MakeShapeFromShapeProto(proto, &out).error_message())
          .contains("An unknown shape must not have any dimensions set."));
  EXPECT_FALSE(IsSet(out));

  // With known rank.
  proto.set_unknown_rank(false);
  EXPECT_TRUE(c.MakeShapeFromShapeProto(proto, &out).ok());
  EXPECT_EQ("[0]", c.DebugString(out));
  proto.add_dim()->set_size(-1);
  proto.add_dim()->set_size(1000);
  EXPECT_TRUE(c.MakeShapeFromShapeProto(proto, &out).ok());
  EXPECT_EQ("[0,?,1000]", c.DebugString(out));

  // With invalid dimension value.
  proto.add_dim()->set_size(-2);
  EXPECT_TRUE(
      StringPiece(c.MakeShapeFromShapeProto(proto, &out).error_message())
          .contains("Shape [0,?,1000,-2] has dimensions with values below -1 "
                    "(where -1 means unknown)"));

  EXPECT_FALSE(IsSet(out));
}

TEST_F(ShapeInferenceTest, MakeDim) {
  NodeDef def;
  std::vector<ShapeHandle> empty;
  InferenceContext c(&def, MakeOpDef(0, 2), empty, {}, {}, {}, {});

  auto d0 = c.MakeDim(1);
  auto d1 = c.MakeDim(1);
  auto d2 = c.MakeDim(2);
  EXPECT_EQ("1", c.DebugString(d0));
  EXPECT_EQ("1", c.DebugString(d1));
  EXPECT_FALSE(SameHandle(d0, d1));
  EXPECT_EQ("2", c.DebugString(d2));
}

TEST_F(ShapeInferenceTest, UnknownDim) {
  NodeDef def;
  std::vector<ShapeHandle> empty;
  InferenceContext c(&def, MakeOpDef(0, 2), empty, {}, {}, {}, {});

  auto d0 = c.UnknownDim();
  auto d1 = c.UnknownDim();
  EXPECT_EQ("?", c.DebugString(d0));
  EXPECT_EQ("?", c.DebugString(d1));
  EXPECT_FALSE(SameHandle(d0, d1));
}

TEST_F(ShapeInferenceTest, UnknownShapeOfRank) {
  NodeDef def;
  std::vector<ShapeHandle> empty;
  InferenceContext c(&def, MakeOpDef(0, 2), empty, {}, {}, {}, {});

  auto unknown_shape_of_rank_3 = c.UnknownShapeOfRank(3);
  EXPECT_EQ("[?,?,?]", c.DebugString(unknown_shape_of_rank_3));

  auto unknown_shape_of_rank_0 = c.UnknownShapeOfRank(0);
  EXPECT_EQ("[]", c.DebugString(unknown_shape_of_rank_0));
}

TEST_F(ShapeInferenceTest, InputTensors) {
  const Tensor t1 = tensorflow::test::AsTensor<float>({10});
  const Tensor t2 = tensorflow::test::AsTensor<float>({20, 30});
  NodeDef def;
  InferenceContext c(&def, MakeOpDef(3, 2), {S({1}), S({2}), S({3})},
                     {&t1, &t2}, {}, {}, {});

  EXPECT_TRUE(c.input_tensor(0) == &t1);
  EXPECT_TRUE(c.input_tensor(1) == &t2);
  EXPECT_TRUE(c.input_tensor(2) == nullptr);
}

TEST_F(ShapeInferenceTest, MakeDimForScalarInput) {
  Tensor t1 = tensorflow::test::AsScalar<int32>(20);
  Tensor t2 = tensorflow::test::AsScalar<int32>(-1);
  NodeDef def;
  InferenceContext c(&def, MakeOpDef(2, 2), {S({}), S({})}, {&t1, &t2}, {}, {},
                     {});

  DimensionHandle d;
  EXPECT_TRUE(c.MakeDimForScalarInput(0, &d).ok());
  EXPECT_EQ("20", c.DebugString(d));

  EXPECT_TRUE(StringPiece(c.MakeDimForScalarInput(1, &d).error_message())
                  .contains("Dimension size, given by scalar input 1, must "
                            "be non-negative but is -1"));

  // Same tests, with int64 values.
  t1 = tensorflow::test::AsScalar<int64>(20);
  t2 = tensorflow::test::AsScalar<int64>(-1);
  EXPECT_TRUE(c.MakeDimForScalarInput(0, &d).ok());
  EXPECT_EQ("20", c.DebugString(d));

  EXPECT_TRUE(StringPiece(c.MakeDimForScalarInput(1, &d).error_message())
                  .contains("Dimension size, given by scalar input 1, must "
                            "be non-negative but is -1"));
}

TEST_F(ShapeInferenceTest, GetAttr) {
  OpRegistrationData op_reg_data;
  op_reg_data.op_def = MakeOpDef(0, 2);
  NodeDef def;
  CHECK(NodeDefBuilder("dummy", &op_reg_data.op_def)
            .Attr("foo", "bar")
            .Finalize(&def)
            .ok());

  std::vector<ShapeHandle> empty;
  InferenceContext c(&def, op_reg_data.op_def, empty, {}, {}, {}, {});
  string value;
  EXPECT_TRUE(c.GetAttr("foo", &value).ok());
  EXPECT_EQ("bar", value);
}

TEST_F(ShapeInferenceTest, Divide) {
  NodeDef def;
  InferenceContext c(&def, MakeOpDef(1, 2), {S({6, -1, 1, 2, 0})}, {}, {}, {},
                     {});

  auto s = c.input(0);
  auto d_6 = c.Dim(s, 0);
  auto d_unknown = c.Dim(s, 1);
  auto d_1 = c.Dim(s, 2);
  auto d_2 = c.Dim(s, 3);
  auto d_0 = c.Dim(s, 4);
  bool evenly_divisible = true;

  // Dividing unknown by non-1 gives new unknown.
  DimensionHandle out;
  EXPECT_TRUE(c.Divide(d_unknown, 2, evenly_divisible, &out).ok());
  EXPECT_EQ("?", c.DebugString(out));
  EXPECT_FALSE(SameHandle(out, d_unknown));

  // Dividing anything by 1 returns the input.
  EXPECT_TRUE(c.Divide(d_unknown, 1, evenly_divisible, &out).ok());
  EXPECT_TRUE(SameHandle(out, d_unknown));
  EXPECT_TRUE(c.Divide(d_6, 1, evenly_divisible, &out).ok());
  EXPECT_TRUE(SameHandle(out, d_6));
  EXPECT_TRUE(c.Divide(d_unknown, d_1, evenly_divisible, &out).ok());
  EXPECT_TRUE(SameHandle(out, d_unknown));
  EXPECT_TRUE(c.Divide(d_6, d_1, evenly_divisible, &out).ok());
  EXPECT_TRUE(SameHandle(out, d_6));

  EXPECT_TRUE(c.Divide(d_6, 2, evenly_divisible, &out).ok());
  EXPECT_EQ("3", c.DebugString(out));
  EXPECT_TRUE(c.Divide(d_6, d_2, evenly_divisible, &out).ok());
  EXPECT_EQ("3", c.DebugString(out));

  EXPECT_TRUE(
      StringPiece(c.Divide(d_6, 5, evenly_divisible, &out).error_message())
          .contains("Dimension size must be evenly divisible by 5 but is 6"));

  EXPECT_TRUE(
      StringPiece(c.Divide(d_6, 0, evenly_divisible, &out).error_message())
          .contains("Divisor must be positive but is 0"));
  EXPECT_TRUE(
      StringPiece(c.Divide(d_6, d_0, evenly_divisible, &out).error_message())
          .contains("Divisor must be positive but is 0"));

  EXPECT_TRUE(
      StringPiece(c.Divide(d_6, -1, evenly_divisible, &out).error_message())
          .contains("Divisor must be positive but is -1"));

  // Repeat error cases above with evenly_divisible=false.
  evenly_divisible = false;
  EXPECT_TRUE(c.Divide(d_6, 5, evenly_divisible, &out).ok());
  EXPECT_EQ("1", c.DebugString(out));

  EXPECT_TRUE(
      StringPiece(c.Divide(d_6, 0, evenly_divisible, &out).error_message())
          .contains("Divisor must be positive but is 0"));

  EXPECT_TRUE(
      StringPiece(c.Divide(d_6, -1, evenly_divisible, &out).error_message())
          .contains("Divisor must be positive but is -1"));
}

TEST_F(ShapeInferenceTest, Add) {
  NodeDef def;
  InferenceContext c(&def, MakeOpDef(1, 2), {S({6, -1, 0})}, {}, {}, {}, {});

  auto s = c.input(0);
  auto d_6 = c.Dim(s, 0);
  auto d_unknown = c.Dim(s, 1);
  auto d_0 = c.Dim(s, 2);

  // Adding non-zero to unknown gives new unknown.
  DimensionHandle out;
  EXPECT_TRUE(c.Add(d_unknown, 1, &out).ok());
  EXPECT_EQ("?", c.DebugString(out));
  EXPECT_FALSE(SameHandle(out, d_unknown));

  // Adding 0 to anything gives input.
  EXPECT_TRUE(c.Add(d_unknown, 0, &out).ok());
  EXPECT_TRUE(SameHandle(out, d_unknown));
  EXPECT_TRUE(c.Add(d_6, 0, &out).ok());
  EXPECT_TRUE(SameHandle(out, d_6));

  // Adding dimension with value 0 to anything gives input.
  EXPECT_TRUE(c.Add(d_unknown, c.MakeDim(0ll), &out).ok());
  EXPECT_TRUE(SameHandle(out, d_unknown));
  EXPECT_TRUE(c.Add(d_6, c.MakeDim(0ll), &out).ok());
  EXPECT_TRUE(SameHandle(out, d_6));

  // Test addition.
  EXPECT_TRUE(c.Add(d_6, 2, &out).ok());
  EXPECT_EQ("8", c.DebugString(out));
  EXPECT_TRUE(c.Add(d_6, std::numeric_limits<int64>::max() - 6, &out).ok());
  EXPECT_EQ(std::numeric_limits<int64>::max(), c.Value(out));

  // Test addition using dimension as second value.
  EXPECT_TRUE(c.Add(d_6, c.MakeDim(2), &out).ok());
  EXPECT_EQ("8", c.DebugString(out));
  EXPECT_TRUE(
      c.Add(d_6, c.MakeDim(std::numeric_limits<int64>::max() - 6), &out).ok());
  EXPECT_EQ(std::numeric_limits<int64>::max(), c.Value(out));
  EXPECT_TRUE(c.Add(d_6, c.UnknownDim(), &out).ok());
  EXPECT_EQ("?", c.DebugString(out));
  EXPECT_TRUE(c.Add(d_0, d_6, &out).ok());
  EXPECT_TRUE(SameHandle(out, d_6));

  EXPECT_TRUE(
      StringPiece(c.Add(d_6, std::numeric_limits<int64>::max() - 5, &out)
                      .error_message())
          .contains(
              "Dimension size overflow from adding 6 and 9223372036854775802"));
}

TEST_F(ShapeInferenceTest, Subtract) {
  NodeDef def;
  InferenceContext c(&def, MakeOpDef(1, 2), {S({6, -1, 0, 5})}, {}, {}, {}, {});

  auto s = c.input(0);
  auto d_6 = c.Dim(s, 0);
  auto d_unknown = c.Dim(s, 1);
  auto d_0 = c.Dim(s, 2);
  auto d_5 = c.Dim(s, 3);

  // Subtracting non-zero from unknown gives new unknown.
  DimensionHandle out;
  EXPECT_TRUE(c.Subtract(d_unknown, 1, &out).ok());
  EXPECT_EQ("?", c.DebugString(out));
  EXPECT_FALSE(SameHandle(out, d_unknown));

  // Subtracting 0 from anything gives input.
  EXPECT_TRUE(c.Subtract(d_unknown, 0ll, &out).ok());
  EXPECT_TRUE(SameHandle(out, d_unknown));
  EXPECT_TRUE(c.Subtract(d_6, 0ll, &out).ok());
  EXPECT_TRUE(SameHandle(out, d_6));

  // Subtracting dimension with value 0 from anything gives input.
  EXPECT_TRUE(c.Subtract(d_unknown, c.MakeDim(0ll), &out).ok());
  EXPECT_TRUE(SameHandle(out, d_unknown));
  EXPECT_TRUE(c.Subtract(d_6, c.MakeDim(0ll), &out).ok());
  EXPECT_TRUE(SameHandle(out, d_6));

  // Test subtraction.
  EXPECT_TRUE(c.Subtract(d_6, 2, &out).ok());
  EXPECT_EQ("4", c.DebugString(out));
  EXPECT_TRUE(c.Subtract(d_6, 6, &out).ok());
  EXPECT_EQ("0", c.DebugString(out));

  // Test subtraction using dimension as second value.
  EXPECT_TRUE(c.Subtract(d_6, c.MakeDim(2), &out).ok());
  EXPECT_EQ("4", c.DebugString(out));
  EXPECT_TRUE(c.Subtract(d_6, d_5, &out).ok());
  EXPECT_EQ("1", c.DebugString(out));
  EXPECT_TRUE(c.Subtract(d_6, c.UnknownDim(), &out).ok());
  EXPECT_EQ("?", c.DebugString(out));
  EXPECT_TRUE(c.Subtract(d_6, d_0, &out).ok());
  EXPECT_TRUE(SameHandle(out, d_6));

  EXPECT_TRUE(
      StringPiece(c.Subtract(d_5, d_6, &out).error_message())
          .contains("Negative dimension size caused by subtracting 6 from 5"));
}

TEST_F(ShapeInferenceTest, Multiply) {
  NodeDef def;
  InferenceContext c(&def, MakeOpDef(1, 2), {S({6, -1, 0, 1})}, {}, {}, {}, {});

  auto s = c.input(0);
  auto d_6 = c.Dim(s, 0);
  auto d_unknown = c.Dim(s, 1);
  auto d_0 = c.Dim(s, 2);
  auto d_1 = c.Dim(s, 3);

  // Multiplying non-zero to unknown gives new unknown.
  DimensionHandle out;
  EXPECT_TRUE(c.Multiply(d_unknown, 2, &out).ok());
  EXPECT_EQ("?", c.DebugString(out));

  // Multiplying 0 to anything gives 0.
  EXPECT_TRUE(c.Multiply(d_unknown, 0, &out).ok());
  EXPECT_EQ("0", c.DebugString(out));
  EXPECT_TRUE(c.Multiply(d_unknown, d_0, &out).ok());
  EXPECT_EQ("0", c.DebugString(out));
  EXPECT_TRUE(c.Multiply(d_0, d_unknown, &out).ok());
  EXPECT_EQ("0", c.DebugString(out));

  // Multiplying 1 to anything gives the original.
  // (unknown -> unknown)
  EXPECT_TRUE(c.Multiply(d_unknown, 1, &out).ok());
  EXPECT_TRUE(SameHandle(d_unknown, out));
  EXPECT_TRUE(c.Multiply(d_unknown, d_1, &out).ok());
  EXPECT_TRUE(SameHandle(d_unknown, out));
  EXPECT_TRUE(c.Multiply(d_1, d_unknown, &out).ok());
  EXPECT_TRUE(SameHandle(d_unknown, out));
  // (known -> known)
  EXPECT_TRUE(c.Multiply(d_6, 1, &out).ok());
  EXPECT_TRUE(SameHandle(d_6, out));
  EXPECT_TRUE(c.Multiply(d_6, d_1, &out).ok());
  EXPECT_TRUE(SameHandle(d_6, out));
  EXPECT_TRUE(c.Multiply(d_1, d_6, &out).ok());
  EXPECT_TRUE(SameHandle(d_6, out));

  // Test multiplication.
  EXPECT_TRUE(c.Multiply(d_6, 2, &out).ok());
  EXPECT_EQ("12", c.DebugString(out));
  EXPECT_TRUE(c.Multiply(d_6, 6, &out).ok());
  EXPECT_EQ("36", c.DebugString(out));

  // Test multiplication using dimension as second value.
  EXPECT_TRUE(c.Multiply(d_6, c.MakeDim(2), &out).ok());
  EXPECT_EQ("12", c.DebugString(out));
  EXPECT_TRUE(c.Multiply(d_6, c.UnknownDim(), &out).ok());
  EXPECT_EQ("?", c.DebugString(out));
}

TEST_F(ShapeInferenceTest, FullyDefined) {
  NodeDef def;
  std::vector<ShapeHandle> empty;
  InferenceContext c(&def, MakeOpDef(0, 2), empty, {}, {}, {}, {});

  // No rank or missing dimension information should return false.
  EXPECT_FALSE(c.FullyDefined(c.UnknownShape()));
  EXPECT_FALSE(c.FullyDefined(c.Matrix(c.MakeDim(1), c.UnknownDim())));

  // Return true if all information exists.
  EXPECT_TRUE(c.FullyDefined(c.Matrix(c.MakeDim(1), c.MakeDim(2))));
  EXPECT_TRUE(c.FullyDefined(c.Scalar()));
}

TEST_F(ShapeInferenceTest, Min) {
  NodeDef def;
  InferenceContext c(&def, MakeOpDef(1, 2), {S({1, 2, -1, 0})}, {}, {}, {}, {});

  auto s = c.input(0);
  auto d_1 = c.Dim(s, 0);
  auto d_2 = c.Dim(s, 1);
  auto d_unknown = c.Dim(s, 2);
  auto d_0 = c.Dim(s, 3);

  // Minimum involving zero and unknown returns zero.
  DimensionHandle out;
  EXPECT_TRUE(c.Min(d_0, d_unknown, &out).ok());
  EXPECT_TRUE(SameHandle(d_0, out));
  EXPECT_TRUE(c.Min(d_unknown, d_0, &out).ok());
  EXPECT_TRUE(SameHandle(d_0, out));
  EXPECT_TRUE(c.Min(c.MakeDim(0ll), d_unknown, &out).ok());
  EXPECT_EQ("0", c.DebugString(out));
  EXPECT_TRUE(c.Min(d_unknown, 0ll, &out).ok());
  EXPECT_EQ("0", c.DebugString(out));

  // Minimum involving unknowns and non-zeros gives new unknown.
  EXPECT_TRUE(c.Min(d_unknown, d_unknown, &out).ok());
  EXPECT_EQ("?", c.DebugString(out));
  EXPECT_TRUE(c.Min(d_unknown, 1, &out).ok());
  EXPECT_EQ("?", c.DebugString(out));
  EXPECT_TRUE(c.Min(d_1, d_unknown, &out).ok());
  EXPECT_EQ("?", c.DebugString(out));

  // Minimum with constant second arg.
  EXPECT_TRUE(c.Min(d_1, 1, &out).ok());
  EXPECT_TRUE(SameHandle(d_1, out));
  EXPECT_TRUE(c.Min(d_1, 3, &out).ok());
  EXPECT_TRUE(SameHandle(d_1, out));
  EXPECT_TRUE(c.Min(d_2, 1, &out).ok());
  EXPECT_EQ("1", c.DebugString(out));

  // Minimum with two dimensions.
  EXPECT_TRUE(c.Min(d_1, d_1, &out).ok());
  EXPECT_TRUE(SameHandle(d_1, out));
  EXPECT_TRUE(c.Min(d_1, d_2, &out).ok());
  EXPECT_TRUE(SameHandle(d_1, out));
  EXPECT_TRUE(c.Min(d_2, d_1, &out).ok());
  EXPECT_TRUE(SameHandle(d_1, out));
  EXPECT_TRUE(c.Min(d_2, d_2, &out).ok());
  EXPECT_TRUE(SameHandle(d_2, out));
}

TEST_F(ShapeInferenceTest, Max) {
  NodeDef def;
  InferenceContext c(&def, MakeOpDef(1, 2), {S({1, 2, -1})}, {}, {}, {}, {});

  auto s = c.input(0);
  auto d_1 = c.Dim(s, 0);
  auto d_2 = c.Dim(s, 1);
  auto d_unknown = c.Dim(s, 2);

  // Maximum involving unknowns gives new unknown.
  DimensionHandle out;
  EXPECT_TRUE(c.Max(d_unknown, d_unknown, &out).ok());
  EXPECT_EQ("?", c.DebugString(out));
  EXPECT_TRUE(c.Max(d_unknown, 1, &out).ok());
  EXPECT_EQ("?", c.DebugString(out));
  EXPECT_TRUE(c.Max(d_1, d_unknown, &out).ok());
  EXPECT_EQ("?", c.DebugString(out));

  // Maximum with constant second arg.
  EXPECT_TRUE(c.Max(d_1, 1, &out).ok());
  EXPECT_TRUE(SameHandle(d_1, out));
  EXPECT_TRUE(c.Max(d_2, 1, &out).ok());
  EXPECT_TRUE(SameHandle(d_2, out));
  EXPECT_TRUE(c.Max(d_2, 3, &out).ok());
  EXPECT_EQ("3", c.DebugString(out));

  // Maximum with two dimensions.
  EXPECT_TRUE(c.Max(d_1, d_1, &out).ok());
  EXPECT_TRUE(SameHandle(d_1, out));
  EXPECT_TRUE(c.Max(d_1, d_2, &out).ok());
  EXPECT_TRUE(SameHandle(d_2, out));
  EXPECT_TRUE(c.Max(d_2, d_1, &out).ok());
  EXPECT_TRUE(SameHandle(d_2, out));
  EXPECT_TRUE(c.Max(d_2, d_2, &out).ok());
  EXPECT_TRUE(SameHandle(d_2, out));
}

}  // namespace shape_inference
}  // namespace tensorflow
