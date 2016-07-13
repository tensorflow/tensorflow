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

#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace shape_inference {

TEST(ShapeInferenceTest, RankAndDimInspection) {
  NodeDef def;
  InferenceContext c(&def, {"?", "[1,?,3]", "[]"}, 2 /* num_outputs */, {});
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
  EXPECT_TRUE(d == c.Dim(in1, -3));
  EXPECT_TRUE(c.ValueKnown(d));
  EXPECT_EQ("1", c.DebugString(d));
  d = c.Dim(in1, 1);
  EXPECT_EQ(InferenceContext::kUnknownDim, c.Value(d));
  EXPECT_FALSE(c.ValueKnown(d));
  EXPECT_TRUE(d == c.Dim(in1, -2));
  EXPECT_EQ("?", c.DebugString(d));
  d = c.Dim(in1, 2);
  EXPECT_EQ(3, c.Value(d));
  EXPECT_TRUE(d == c.Dim(in1, -1));
  EXPECT_TRUE(c.ValueKnown(d));
  EXPECT_EQ("3", c.DebugString(d));

  auto in2 = c.input(2);
  EXPECT_EQ("[]", c.DebugString(in2));
  EXPECT_TRUE(c.RankKnown(in2));
  EXPECT_EQ(0, c.Rank(in2));
}

TEST(ShapeInferenceTest, WithRank) {
  NodeDef def;
  InferenceContext c(&def, {"?", "[1,?,3]"}, 2 /* num_outputs */, {});

  auto in0 = c.input(0);
  auto in1 = c.input(1);
  const Shape* s1 = nullptr;
  const Shape* s2 = nullptr;

  // WithRank on a shape with unknown dimensionality always succeeds.
  EXPECT_TRUE(c.WithRank(in0, 1, &s1).ok());
  EXPECT_EQ("[?]", c.DebugString(s1));

  EXPECT_TRUE(c.WithRank(in0, 2, &s2).ok());
  EXPECT_EQ("[?,?]", c.DebugString(s2));
  EXPECT_TRUE(s1 != s2);                      // different pointers
  EXPECT_TRUE(c.Dim(s2, 0) != c.Dim(s2, 1));  // different pointers.

  EXPECT_TRUE(c.WithRank(in0, 1, &s2).ok());
  EXPECT_EQ("[?]", c.DebugString(s2));
  EXPECT_TRUE(s1 != s2);  // different pointers

  EXPECT_TRUE(c.WithRank(in0, 0, &s1).ok());
  EXPECT_EQ("[]", c.DebugString(s1));

  // WithRank on shape with known dimensionality.
  s1 = in1;
  EXPECT_EQ("Invalid argument: Shape must be rank 2 but is rank 3",
            c.WithRank(in1, 2, &s1).ToString());
  EXPECT_TRUE(s1 == nullptr);
  EXPECT_TRUE(c.WithRank(in1, 3, &s1).ok());
  EXPECT_TRUE(s1 == in1);  // same pointers

  // Inputs are unchanged.
  EXPECT_EQ("?", c.DebugString(in0));
  EXPECT_EQ("[1,?,3]", c.DebugString(in1));
}

TEST(ShapeInferenceTest, WithRankAtMost) {
  NodeDef def;
  InferenceContext c(&def, {"?", "[1,?,3]"}, 2 /* num_outputs */, {});

  auto in0 = c.input(0);
  auto in1 = c.input(1);
  const Shape* s1 = nullptr;
  const Shape* s2 = nullptr;

  // WithRankAtMost on a shape with unknown dimensionality always succeeds.
  EXPECT_TRUE(c.WithRankAtMost(in0, 1, &s1).ok());
  EXPECT_EQ("?", c.DebugString(s1));
  EXPECT_TRUE(in0 != s1);  // different pointers

  EXPECT_TRUE(c.WithRankAtMost(in0, 2, &s2).ok());
  EXPECT_EQ("?", c.DebugString(s2));
  EXPECT_TRUE(s1 != s2);  // different pointers

  // WithRankAtMost on shape with known dimensionality.
  s1 = in1;
  EXPECT_EQ("Invalid argument: Shape must be at most rank 2 but is rank 3",
            c.WithRankAtMost(in1, 2, &s1).ToString());
  EXPECT_TRUE(s1 == nullptr);
  EXPECT_TRUE(c.WithRankAtMost(in1, 3, &s1).ok());
  EXPECT_TRUE(s1 == in1);  // same pointers
  EXPECT_TRUE(c.WithRankAtMost(in1, 4, &s1).ok());
  EXPECT_TRUE(s1 == in1);  // same pointers
  EXPECT_TRUE(c.WithRankAtMost(in1, 5, &s1).ok());
  EXPECT_TRUE(s1 == in1);  // same pointers

  // Inputs are unchanged.
  EXPECT_EQ("?", c.DebugString(in0));
  EXPECT_EQ("[1,?,3]", c.DebugString(in1));
}

TEST(ShapeInferenceTest, WithRankAtLeast) {
  NodeDef def;
  InferenceContext c(&def, {"?", "[1,?,3]"}, 2 /* num_outputs */, {});

  auto in0 = c.input(0);
  auto in1 = c.input(1);
  const Shape* s1 = nullptr;
  const Shape* s2 = nullptr;

  // WithRankAtLeast on a shape with unknown dimensionality always succeeds.
  EXPECT_TRUE(c.WithRankAtLeast(in0, 1, &s1).ok());
  EXPECT_EQ("?", c.DebugString(s1));
  EXPECT_TRUE(in0 != s1);  // different pointers

  EXPECT_TRUE(c.WithRankAtLeast(in0, 2, &s2).ok());
  EXPECT_EQ("?", c.DebugString(s2));
  EXPECT_TRUE(s1 != s2);  // different pointers

  // WithRankAtLeast on shape with known dimensionality.
  s1 = in1;
  EXPECT_EQ("Invalid argument: Shape must be at least rank 4 but is rank 3",
            c.WithRankAtLeast(in1, 4, &s1).ToString());
  EXPECT_TRUE(s1 == nullptr);
  EXPECT_TRUE(c.WithRankAtLeast(in1, 3, &s1).ok());
  EXPECT_TRUE(s1 == in1);  // same pointers
  EXPECT_TRUE(c.WithRankAtLeast(in1, 2, &s1).ok());
  EXPECT_TRUE(s1 == in1);  // same pointers
  EXPECT_TRUE(c.WithRankAtLeast(in1, 0, &s1).ok());
  EXPECT_TRUE(s1 == in1);  // same pointers

  // Inputs are unchanged.
  EXPECT_EQ("?", c.DebugString(in0));
  EXPECT_EQ("[1,?,3]", c.DebugString(in1));
}

TEST(ShapeInferenceTest, WithValue) {
  NodeDef def;
  InferenceContext c(&def, {"[1,?]"}, 2 /* num_outputs */, {});

  auto d0 = c.Dim(c.input(0), 0);
  auto d1 = c.Dim(c.input(0), 1);
  const Dimension* out1 = nullptr;
  const Dimension* out2 = nullptr;

  // WithValue on a dimension with unknown value always succeeds.
  EXPECT_TRUE(c.WithValue(d1, 1, &out1).ok());
  EXPECT_EQ(1, c.Value(out1));

  EXPECT_TRUE(c.WithValue(d1, 2, &out2).ok());
  EXPECT_EQ(2, c.Value(out2));
  EXPECT_TRUE(out1 != out2);  // different pointers
  EXPECT_TRUE(out1 != d1);    // different pointers

  EXPECT_TRUE(c.WithValue(d1, 1, &out2).ok());
  EXPECT_EQ(1, c.Value(out2));
  EXPECT_TRUE(out1 != out2);  // different pointers

  // WithValue on dimension with known size.
  out1 = d0;
  EXPECT_EQ("Invalid argument: Dimension must be 0 but is 1",
            c.WithValue(d0, 0, &out1).ToString());
  EXPECT_TRUE(out1 == nullptr);
  out1 = d0;
  EXPECT_EQ("Invalid argument: Dimension must be 2 but is 1",
            c.WithValue(d0, 2, &out1).ToString());
  EXPECT_TRUE(out1 == nullptr);
  EXPECT_TRUE(c.WithValue(d0, 1, &out1).ok());
  EXPECT_TRUE(d0 == out1);  // same pointers

  // Inputs are unchanged.
  EXPECT_EQ("1", c.DebugString(d0));
  EXPECT_EQ("?", c.DebugString(d1));
}

TEST(ShapeInferenceTest, MergeDim) {
  NodeDef def;
  InferenceContext c(&def, {"[2,?,2,1,?]"}, 2 /* num_outputs */, {});

  auto d2 = c.Dim(c.input(0), 0);
  auto d_unknown = c.Dim(c.input(0), 1);
  auto d2_b = c.Dim(c.input(0), 2);
  auto d1 = c.Dim(c.input(0), 3);
  auto d_unknown_b = c.Dim(c.input(0), 4);
  const Dimension* out = nullptr;

  // Merging anything with unknown returns the same pointer.
  EXPECT_TRUE(c.Merge(d2, d_unknown, &out).ok());
  EXPECT_TRUE(d2 == out);
  EXPECT_TRUE(c.Merge(d_unknown, d2, &out).ok());
  EXPECT_TRUE(d2 == out);
  EXPECT_TRUE(c.Merge(d_unknown, d_unknown_b, &out).ok());
  EXPECT_TRUE(d_unknown == out);

  // Merging with self returns self.
  EXPECT_TRUE(c.Merge(d2, d2, &out).ok());
  EXPECT_TRUE(d2 == out);
  EXPECT_TRUE(c.Merge(d_unknown, d_unknown, &out).ok());
  EXPECT_TRUE(d_unknown == out);

  // Merging equal values returns first one.
  EXPECT_TRUE(c.Merge(d2, d2_b, &out).ok());
  EXPECT_TRUE(d2 == out);
  EXPECT_TRUE(c.Merge(d2_b, d2, &out).ok());
  EXPECT_TRUE(d2_b == out);

  // Merging inequal values is an error.
  EXPECT_EQ("Invalid argument: Dimensions must be equal, but are 2 and 1",
            c.Merge(d2, d1, &out).ToString());
  EXPECT_TRUE(out == nullptr);
  EXPECT_EQ("Invalid argument: Dimensions must be equal, but are 1 and 2",
            c.Merge(d1, d2, &out).ToString());
  EXPECT_TRUE(out == nullptr);
}

TEST(ShapeInferenceTest, MergeShape) {
  NodeDef def;
  InferenceContext c(&def,
                     {"?", "[1,2]", "[?,2]", "[1,?]", "[1,3]", "?", "[1]"},
                     2 /* num_outputs */, {});

  auto s_unknown = c.input(0);
  auto s_1_2 = c.input(1);
  auto s_u_2 = c.input(2);
  auto s_1_u = c.input(3);
  auto s_1_3 = c.input(4);
  auto s_unknown_b = c.input(5);
  auto s_1 = c.input(6);
  const Shape* out = nullptr;

  // Merging any shape with unknown returns the shape.
  EXPECT_TRUE(c.Merge(s_unknown, s_1_2, &out).ok());
  EXPECT_TRUE(s_1_2 == out);
  EXPECT_TRUE(c.Merge(s_u_2, s_unknown, &out).ok());
  EXPECT_TRUE(s_u_2 == out);
  EXPECT_TRUE(c.Merge(s_unknown, s_unknown_b, &out).ok());
  EXPECT_TRUE(s_unknown == out);

  // Merging with self returns self.
  EXPECT_TRUE(c.Merge(s_1_2, s_1_2, &out).ok());
  EXPECT_TRUE(out == s_1_2);

  // Merging where one of the inputs is the right answer - return that input.
  out = nullptr;
  EXPECT_TRUE(c.Merge(s_1_2, s_u_2, &out).ok());
  EXPECT_TRUE(s_1_2 == out);
  out = nullptr;
  EXPECT_TRUE(c.Merge(s_u_2, s_1_2, &out).ok());
  EXPECT_TRUE(s_1_2 == out);

  // Merging where neither input is the right answer.
  EXPECT_TRUE(c.Merge(s_u_2, s_1_u, &out).ok());
  EXPECT_TRUE(out != s_u_2);
  EXPECT_TRUE(out != s_1_u);
  EXPECT_EQ("[1,2]", c.DebugString(out));
  EXPECT_TRUE(c.Dim(s_1_u, 0) == c.Dim(out, 0));  // same pointers
  EXPECT_TRUE(c.Dim(s_u_2, 1) == c.Dim(out, 1));  // same pointers

  // Incompatible merges give errors and set out to nullptr.
  out = s_unknown;
  EXPECT_EQ(("Invalid argument: Dimension 1 in both shapes must be equal, but "
             "are 2 and 3"),
            c.Merge(s_u_2, s_1_3, &out).ToString());
  EXPECT_TRUE(out == nullptr);
  out = s_unknown;
  EXPECT_EQ(("Invalid argument: Dimension 1 in both shapes must be equal, but "
             "are 3 and 2"),
            c.Merge(s_1_3, s_u_2, &out).ToString());
  EXPECT_TRUE(out == nullptr);
  out = s_unknown;
  EXPECT_EQ("Invalid argument: Shapes must be equal rank, but are 1 and 2",
            c.Merge(s_1, s_1_2, &out).ToString());
  EXPECT_TRUE(out == nullptr);
}

TEST(ShapeInferenceTest, MergePrefix) {
  NodeDef def;
  InferenceContext c(&def, {"?", "[?,2]", "[1,?,3]", "[2,4]"},
                     2 /* num_outputs */, {});

  auto s_unknown = c.input(0);
  auto s_u_2 = c.input(1);
  auto s_1_u_3 = c.input(2);
  auto s_2_4 = c.input(3);

  const Shape* s_out = nullptr;
  const Shape* s_prefix_out = nullptr;

  // Merging with unknown returns the inputs.
  EXPECT_TRUE(c.MergePrefix(s_unknown, s_u_2, &s_out, &s_prefix_out).ok());
  EXPECT_TRUE(s_out == s_unknown);
  EXPECT_TRUE(s_prefix_out == s_u_2);
  EXPECT_TRUE(c.MergePrefix(s_1_u_3, s_unknown, &s_out, &s_prefix_out).ok());
  EXPECT_TRUE(s_out == s_1_u_3);
  EXPECT_TRUE(s_prefix_out == s_unknown);

  EXPECT_TRUE(c.MergePrefix(s_1_u_3, s_u_2, &s_out, &s_prefix_out).ok());
  EXPECT_TRUE(s_out != s_1_u_3);
  EXPECT_EQ("[1,2]", c.DebugString(s_prefix_out));
  EXPECT_EQ("[1,2,3]", c.DebugString(s_out));
  EXPECT_TRUE(c.Dim(s_prefix_out, 0) == c.Dim(s_out, 0));
  EXPECT_TRUE(c.Dim(s_out, 0) == c.Dim(s_1_u_3, 0));
  EXPECT_TRUE(c.Dim(s_prefix_out, 1) == c.Dim(s_out, 1));
  EXPECT_TRUE(c.Dim(s_prefix_out, 1) == c.Dim(s_u_2, 1));

  // Incompatible merges give errors and set outs to nullptr.
  s_out = s_unknown;
  s_prefix_out = s_unknown;
  EXPECT_EQ(("Invalid argument: Dimensions must be equal, but are 1 and 2"),
            c.MergePrefix(s_1_u_3, s_2_4, &s_out, &s_prefix_out).ToString());
  EXPECT_TRUE(s_out == nullptr);
  EXPECT_TRUE(s_prefix_out == nullptr);

  s_out = s_unknown;
  s_prefix_out = s_unknown;
  EXPECT_EQ(("Invalid argument: Shape must be at least rank 3 but is rank 2"),
            c.MergePrefix(s_2_4, s_1_u_3, &s_out, &s_prefix_out).ToString());
  EXPECT_TRUE(s_out == nullptr);
  EXPECT_TRUE(s_prefix_out == nullptr);
}

TEST(ShapeInferenceTest, Subshape) {
  NodeDef def;
  InferenceContext c(&def, {"[1,2,3,?,5]", "?"}, 2 /* num_outputs */, {});

  const Shape* unknown = c.input(1);
  const Shape* out;
  EXPECT_TRUE(c.Subshape(unknown, 0, &out).ok());
  EXPECT_EQ("?", c.DebugString(out));
  EXPECT_TRUE(out == unknown);
  EXPECT_TRUE(c.Subshape(unknown, 1, &out).ok());
  EXPECT_EQ("?", c.DebugString(out));
  EXPECT_TRUE(out != unknown);
  EXPECT_TRUE(c.Subshape(unknown, 200, &out).ok());
  EXPECT_EQ("?", c.DebugString(out));
  EXPECT_TRUE(out != unknown);

  const int kFullRank = 5;
  const Shape* out_arr[4];
  auto in0 = c.input(0);
  EXPECT_TRUE(c.Subshape(in0, 0, &out).ok());
  EXPECT_EQ("[1,2,3,?,5]", c.DebugString(out));
  EXPECT_TRUE(out == in0);
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
          EXPECT_TRUE(c.Dim(in0, start + d) == c.Dim(out, d)) << "arr_idx: "
                                                              << arr_idx;
        }
      }
    }
  }

  // Errors.
  out = unknown;
  EXPECT_EQ(
      "Invalid argument: Subshape must have computed start <= end, but is 5 "
      "and 2 (computed from start 6 and end -3 over shape with rank 5)",
      c.Subshape(in0, 6, -3, &out).ToString());
  EXPECT_TRUE(out == nullptr);
  out = unknown;
  EXPECT_EQ(
      "Invalid argument: Subshape start out of bounds: -50, for shape with "
      "rank 5",
      c.Subshape(in0, -50, 100, &out).ToString());
  EXPECT_TRUE(out == nullptr);
  out = unknown;
  EXPECT_EQ(
      "Invalid argument: Subshape end out of bounds: -50, for shape with rank "
      "5",
      c.Subshape(in0, 0, -50, &out).ToString());
  EXPECT_TRUE(out == nullptr);
}

TEST(ShapeInferenceTest, Concatenate) {
  NodeDef def;
  InferenceContext c(&def, {"[1,?,3]", "[4,5]", "?"}, 2 /* num_outputs */, {});

  auto in0 = c.input(0);
  auto in1 = c.input(1);
  const Shape* unknown = c.input(2);
  const Shape* out;
  EXPECT_TRUE(c.Concatenate(unknown, unknown, &out).ok());
  EXPECT_EQ("?", c.DebugString(out));
  EXPECT_TRUE(out != unknown);
  EXPECT_TRUE(c.Concatenate(unknown, in0, &out).ok());
  EXPECT_EQ("?", c.DebugString(out));
  EXPECT_TRUE(out != unknown);

  EXPECT_TRUE(c.Concatenate(in0, in1, &out).ok());
  EXPECT_EQ("[1,?,3,4,5]", c.DebugString(out));
  int out_i = 0;
  for (int i = 0; i < c.Rank(in0); ++i, ++out_i) {
    EXPECT_TRUE(c.Dim(in0, i) == c.Dim(out, out_i));
  }
  for (int i = 0; i < c.Rank(in1); ++i, ++out_i) {
    EXPECT_TRUE(c.Dim(in1, i) == c.Dim(out, out_i));
  }
}

TEST(ShapeInferenceTest, CreateShape) {
  NodeDef def;
  InferenceContext c(&def, {"[1,2,3,?,5]"}, 2 /* num_outputs */, {});

  std::vector<const Dimension*> dims;
  auto in0 = c.input(0);
  const int rank = c.Rank(in0);
  for (int i = 0; i < rank; ++i) {
    dims.push_back(c.Dim(in0, rank - i - 1));
  }

  auto s = c.CreateShape(dims);
  EXPECT_EQ("[5,?,3,2,1]", c.DebugString(s));
  EXPECT_TRUE(c.Dim(s, 0) == c.Dim(in0, rank - 1));

  auto s2 = c.CreateShape(dims);
  EXPECT_TRUE(s != s2);  // different pointers
  EXPECT_TRUE(c.Dim(s2, 0) == c.Dim(in0, rank - 1));
}

TEST(ShapeInferenceTest, CreateUnknownShape) {
  NodeDef def;
  InferenceContext c(&def, {}, 2 /* num_outputs */, {});

  auto u0 = c.CreateUnknownShape();
  auto u1 = c.CreateUnknownShape();
  EXPECT_EQ("?", c.DebugString(u0));
  EXPECT_EQ("?", c.DebugString(u1));
  EXPECT_TRUE(u0 != u1);  // different pointers
}

TEST(ShapeInferenceTest, CreateShapeFromShapeTensor) {
  auto create = [](Tensor* t) {
    NodeDef def;
    InferenceContext c(&def, {"?"}, 0 /* num_outputs */, {t});
    const Shape* out;
    Status s = c.CreateShapeFromShapeTensor(0, &out);
    if (s.ok()) {
      return c.DebugString(out);
    } else {
      EXPECT_TRUE(out == nullptr);
      return s.error_message();
    }
  };

  Tensor t;
  EXPECT_EQ("?", create(nullptr));

  t = ::tensorflow::test::AsTensor<int32>({1, 2, 3});
  EXPECT_EQ("[1,2,3]", create(&t));

  t = ::tensorflow::test::AsTensor<int64>({3, 2, 1});
  EXPECT_EQ("[3,2,1]", create(&t));

  t = ::tensorflow::test::AsTensor<int64>({});
  EXPECT_EQ("[]", create(&t));

  t = ::tensorflow::test::AsTensor<float>({1, 2, 3});
  EXPECT_EQ("Input tensor must be int32 or int64, but was float", create(&t));

  t = ::tensorflow::test::AsScalar<int32>(1);
  EXPECT_EQ("Input tensor must be rank 1, but was rank 0", create(&t));

  t = ::tensorflow::test::AsTensor<int32>({1, 2}, TensorShape{2, 1});
  EXPECT_EQ("Input tensor must be rank 1, but was rank 2", create(&t));

  // Test when the input shape is wrong.
  {
    NodeDef def;
    InferenceContext c(&def, {"[1,?]"}, 0 /* num_outputs */, {nullptr});
    const Shape* out;
    EXPECT_EQ("Shape must be rank 1 but is rank 2",
              c.CreateShapeFromShapeTensor(0, &out).error_message());
  }
}

TEST(ShapeInferenceTest, CreateDim) {
  NodeDef def;
  InferenceContext c(&def, {}, 2 /* num_outputs */, {});

  auto* d0 = c.CreateDim(1);
  auto* d1 = c.CreateDim(1);
  auto* d2 = c.CreateDim(2);
  EXPECT_EQ("1", c.DebugString(d0));
  EXPECT_EQ("1", c.DebugString(d1));
  EXPECT_TRUE(d0 != d1);  // different pointers
  EXPECT_EQ("2", c.DebugString(d2));
}

TEST(ShapeInferenceTest, CreateUnknownDim) {
  NodeDef def;
  InferenceContext c(&def, {}, 2 /* num_outputs */, {});

  auto* d0 = c.CreateUnknownDim();
  auto* d1 = c.CreateUnknownDim();
  EXPECT_EQ("?", c.DebugString(d0));
  EXPECT_EQ("?", c.DebugString(d1));
  EXPECT_TRUE(d0 != d1);  // different pointers
}

TEST(ShapeInferenceTest, InputTensors) {
  const Tensor t1 = tensorflow::test::AsTensor<float>({10});
  const Tensor t2 = tensorflow::test::AsTensor<float>({20, 30});
  NodeDef def;
  InferenceContext c(&def, {"[1]", "[2]", "[3]"}, 2 /* num_outputs */,
                     {&t1, &t2});

  EXPECT_TRUE(c.input_tensor(0) == &t1);
  EXPECT_TRUE(c.input_tensor(1) == &t2);
  EXPECT_TRUE(c.input_tensor(2) == nullptr);
}

TEST(ShapeInferenceTest, CreateDimForScalarInput) {
  Tensor t1 = tensorflow::test::AsScalar<int32>(20);
  Tensor t2 = tensorflow::test::AsScalar<int32>(-1);
  NodeDef def;
  InferenceContext c(&def, {"[]", "[]"}, 2 /* num_outputs */, {&t1, &t2});

  const Dimension* d;
  EXPECT_TRUE(c.CreateDimForScalarInput(0, &d).ok());
  EXPECT_EQ("20", c.DebugString(d));

  EXPECT_EQ(
      "Dimension size, given by scalar input 1, must be non-negative but is -1",
      c.CreateDimForScalarInput(1, &d).error_message());

  // Same tests, with int64 values.
  t1 = tensorflow::test::AsScalar<int64>(20);
  t2 = tensorflow::test::AsScalar<int64>(-1);
  EXPECT_TRUE(c.CreateDimForScalarInput(0, &d).ok());
  EXPECT_EQ("20", c.DebugString(d));

  EXPECT_EQ(
      "Dimension size, given by scalar input 1, must be non-negative but is -1",
      c.CreateDimForScalarInput(1, &d).error_message());
}

TEST(ShapeInferenceTest, GetAttr) {
  OpRegistrationData op_reg_data;
  CHECK(OpDefBuilder("dummy").Attr("foo:string").Finalize(&op_reg_data).ok());
  NodeDef def;
  CHECK(NodeDefBuilder("dummy", &op_reg_data.op_def)
            .Attr("foo", "bar")
            .Finalize(&def)
            .ok());

  InferenceContext c(&def, {}, 2 /* num_outputs */, {});
  string value;
  EXPECT_TRUE(c.GetAttr("foo", &value).ok());
  EXPECT_EQ("bar", value);
}

}  // namespace shape_inference
}  // namespace tensorflow
