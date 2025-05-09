/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/shape.h"

#include <cstdint>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/hash/hash_testing.h"
#include "absl/strings/str_cat.h"
#include "xla/hlo/testlib/test.h"
#include "xla/layout.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

class ShapeTest : public ::testing::Test {
 protected:
  const Shape opaque_ = ShapeUtil::MakeOpaqueShape();
  const Shape token_ = ShapeUtil::MakeTokenShape();
  const Shape scalar_ = ShapeUtil::MakeShape(F32, {});
  const Shape scalar_with_tile_ =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {}, {}, {Tile({256})});
  const Shape matrix_ = ShapeUtil::MakeShape(U32, {1, 2});
  const Shape matrix2_ =
      ShapeUtil::MakeShapeWithDenseLayout(S32, {3, 4}, {0, 1});
  const Shape tuple_ =
      ShapeUtil::MakeTupleShape({opaque_, scalar_, matrix_, matrix2_});
  const Shape nested_tuple_ =
      ShapeUtil::MakeTupleShape({tuple_, matrix_, token_});
  const Shape dynamic_matrix_ =
      ShapeUtil::MakeShape(S32, {5, 2}, {true, false});
  const Shape unbounded_ =
      ShapeUtil::MakeShape(F32, {Shape::kUnboundedSize, 784}, {true, false});
};

TEST_F(ShapeTest, Empty) {
  Shape shape;
  EXPECT_EQ(shape.element_type(), PRIMITIVE_TYPE_INVALID);
  EXPECT_FALSE(shape.IsArray());
  EXPECT_FALSE(shape.IsTuple());
  EXPECT_FALSE(shape.IsToken());
  EXPECT_FALSE(shape.IsOpaque());
}

TEST_F(ShapeTest, Array) {
  Shape shape(F32, {1, 2, 3}, {});
  EXPECT_EQ(shape.element_type(), F32);
  EXPECT_TRUE(shape.IsArray());

  shape.Clear();
  EXPECT_EQ(shape.element_type(), PRIMITIVE_TYPE_INVALID);
  EXPECT_FALSE(shape.IsArray());
}

TEST_F(ShapeTest, MutationRemovesSharing) {
  Shape s1(F32, {1, 2, 3}, {});
  Shape s2 = s1;

  s1.set_element_type(S32);
  EXPECT_EQ(s1.element_type(), S32);

  // s2 should not change due to s1 mutation.
  EXPECT_EQ(s2.element_type(), F32);
}

TEST_F(ShapeTest, Copy) {
  Shape s1(F32, {1, 2, 3}, {});
  Shape s2 = s1;
  EXPECT_EQ(s2.element_type(), F32);
  EXPECT_EQ(s2.dimensions(), std::vector<int64_t>({1, 2, 3}));

  Shape s3;
  s3 = s1;
  EXPECT_EQ(s3.element_type(), F32);
  EXPECT_EQ(s3.dimensions(), std::vector<int64_t>({1, 2, 3}));
}

TEST_F(ShapeTest, Move) {
  // NOLINTBEGIN(bugprone-use-after-move): intentionally test source after move
  Shape s1(F32, {1, 2, 3}, {});

  // Move s1 to s2
  Shape s2(std::move(s1));
  EXPECT_EQ(s1.element_type(), PRIMITIVE_TYPE_INVALID);
  EXPECT_FALSE(s1.IsArray());
  EXPECT_EQ(s2.element_type(), F32);
  EXPECT_EQ(s2.dimensions(), std::vector<int64_t>({1, 2, 3}));

  // Move s2 to s3.
  Shape s3;
  s3 = std::move(s2);
  EXPECT_EQ(s2.element_type(), PRIMITIVE_TYPE_INVALID);
  EXPECT_FALSE(s2.IsArray());
  EXPECT_EQ(s3.element_type(), F32);
  EXPECT_EQ(s3.dimensions(), std::vector<int64_t>({1, 2, 3}));
  // NOLINTEND(bugprone-use-after-move)
}

// Tests that if the dynamic_dimensions parameter empty in the Shape
// constructor, it's treated as all dimensions are static.
TEST(Shape, ArrayCtorTreatsEmptyDynamicDimensionsAsAllStatic) {
  const Shape shape(F32, {1, 2, 3}, {});
  EXPECT_TRUE(shape.is_static());
  EXPECT_TRUE(shape.is_static_dimension(0));
  EXPECT_TRUE(shape.is_static_dimension(1));
  EXPECT_TRUE(shape.is_static_dimension(2));
}

TEST_F(ShapeTest, ShapeToFromProto) {
  for (const Shape& shape :
       {opaque_, token_, scalar_, matrix_, matrix2_, tuple_, nested_tuple_,
        dynamic_matrix_, unbounded_}) {
    auto shape_copy = Shape::FromProto(shape.ToProto());
    TF_ASSERT_OK(shape_copy);
    EXPECT_TRUE(ShapeUtil::Equal(shape, *shape_copy))
        << shape << " != " << *shape_copy;
  }
}

TEST_F(ShapeTest, ShapeToString) {
  EXPECT_EQ("opaque[]", opaque_.ToString());
  EXPECT_EQ("token[]", token_.ToString());
  EXPECT_EQ("f32[]", scalar_.ToString());
  EXPECT_EQ("u32[1,2]", matrix_.ToString());
  EXPECT_EQ("s32[3,4]", matrix2_.ToString());
  EXPECT_EQ("(opaque[], f32[], u32[1,2], s32[3,4])", tuple_.ToString());
  EXPECT_EQ("((opaque[], f32[], u32[1,2], s32[3,4]), u32[1,2], token[])",
            nested_tuple_.ToString());

  EXPECT_EQ("opaque[]", opaque_.ToString(/*print_layout=*/true));
  EXPECT_EQ("f32[]", scalar_.ToString(/*print_layout=*/true));
  EXPECT_EQ("f32[]{:T(256)}",
            scalar_with_tile_.ToString(/*print_layout=*/true));
  EXPECT_EQ("u32[1,2]{1,0}", matrix_.ToString(/*print_layout=*/true));
  EXPECT_EQ("s32[3,4]{0,1}", matrix2_.ToString(/*print_layout=*/true));
  EXPECT_EQ("(opaque[], f32[], u32[1,2]{1,0}, s32[3,4]{0,1})",
            tuple_.ToString(/*print_layout=*/true));
  EXPECT_EQ(
      "((opaque[], f32[], u32[1,2]{1,0}, s32[3,4]{0,1}), u32[1,2]{1,0}, "
      "token[])",
      nested_tuple_.ToString(/*print_layout=*/true));
}

TEST_F(ShapeTest, DynamicShapeToString) {
  Shape array_shape =
      ShapeUtil::MakeShape(F32, {23, 44, 55}, {true, false, true});
  EXPECT_EQ("f32[<=23,44,<=55]", array_shape.ToString());

  array_shape.set_dynamic_dimension(2, false);
  EXPECT_EQ("f32[<=23,44,55]", array_shape.ToString());

  EXPECT_EQ("f32[?,784]", unbounded_.ToString());
}

TEST_F(ShapeTest, DeleteDimensions) {
  Shape shape = ShapeUtil::MakeShapeWithDenseLayout(F32, {5, 3, 2, 7, 9},
                                                    {2, 0, 1, 4, 3});
  shape.DeleteDimensions({1, 2, 3});
  EXPECT_EQ(shape, ShapeUtil::MakeShapeWithDenseLayout(F32, {5, 9}, {0, 1}));
}

TEST_F(ShapeTest, DeleteDimensionsUnordered) {
  Shape shape = ShapeUtil::MakeShapeWithDenseLayout(F32, {5, 3, 2, 7, 9},
                                                    {2, 0, 1, 4, 3});
  shape.DeleteDimensions({3, 1, 2});
  EXPECT_EQ(shape, ShapeUtil::MakeShapeWithDenseLayout(F32, {5, 9}, {0, 1}));
}

TEST_F(ShapeTest, EqualityTest) {
  // Different layouts.
  EXPECT_NE(ShapeUtil::MakeShapeWithDenseLayout(F32, {23, 44}, {1, 0}),
            ShapeUtil::MakeShapeWithDenseLayout(F32, {23, 44}, {0, 1}));

  // Different dims.
  EXPECT_NE(ShapeUtil::MakeShapeWithDenseLayout(F32, {44, 23}, {1, 0}),
            ShapeUtil::MakeShapeWithDenseLayout(F32, {23, 44}, {1, 0}));

  // Different elements.
  EXPECT_NE(ShapeUtil::MakeShapeWithDenseLayout(S32, {44, 23}, {1, 0}),
            ShapeUtil::MakeShapeWithDenseLayout(F32, {23, 44}, {1, 0}));

  // Equal shapes.
  EXPECT_EQ(ShapeUtil::MakeShapeWithDenseLayout(F32, {23, 44}, {1, 0}),
            ShapeUtil::MakeShapeWithDenseLayout(F32, {23, 44}, {1, 0}));
}

TEST_F(ShapeTest, AreAllLeavesIntegers) {
  EXPECT_FALSE(opaque_.AreAllLeavesIntegers());
  EXPECT_FALSE(token_.AreAllLeavesIntegers());
  EXPECT_TRUE(matrix_.AreAllLeavesIntegers());
  EXPECT_FALSE(tuple_.AreAllLeavesIntegers());
  EXPECT_FALSE(nested_tuple_.AreAllLeavesIntegers());

  Shape u32_shape = ShapeUtil::MakeShape(U32, {1});
  EXPECT_TRUE(u32_shape.AreAllLeavesIntegers());

  Shape f32_shape = ShapeUtil::MakeShape(F32, {1});
  EXPECT_FALSE(f32_shape.AreAllLeavesIntegers());

  Shape integer_tuple = ShapeUtil::MakeTupleShape({u32_shape, u32_shape});
  EXPECT_TRUE(integer_tuple.AreAllLeavesIntegers());

  Shape mixed_type_tuple = ShapeUtil::MakeTupleShape({u32_shape, f32_shape});
  EXPECT_FALSE(mixed_type_tuple.AreAllLeavesIntegers());
}

TEST_F(ShapeTest, IsStatic) {
  EXPECT_TRUE(opaque_.is_static());
  EXPECT_TRUE(token_.is_static());
  EXPECT_TRUE(matrix_.is_static());
  EXPECT_TRUE(tuple_.is_static());
  EXPECT_TRUE(nested_tuple_.is_static());

  Shape dynamic_matrix = matrix_;
  EXPECT_TRUE(dynamic_matrix.is_static());
  dynamic_matrix.set_dynamic_dimension(1, true);
  EXPECT_FALSE(dynamic_matrix.is_static());

  Shape dynamic_tuple = tuple_;
  EXPECT_TRUE(dynamic_tuple.is_static());
  ShapeUtil::GetMutableSubshape(&dynamic_tuple, {2})
      ->set_dynamic_dimension(1, true);
  EXPECT_FALSE(dynamic_tuple.is_static());

  EXPECT_FALSE(unbounded_.is_static());
}

TEST_F(ShapeTest, IsDynamic) {
  EXPECT_FALSE(matrix_.is_dynamic());
  EXPECT_FALSE(matrix_.is_unbounded_dynamic());

  EXPECT_TRUE(dynamic_matrix_.is_dynamic());
  EXPECT_FALSE(dynamic_matrix_.is_unbounded_dynamic());

  EXPECT_TRUE(unbounded_.is_dynamic());
  EXPECT_TRUE(unbounded_.is_unbounded_dynamic());

  Shape unbounded_tuple = tuple_;
  EXPECT_FALSE(unbounded_tuple.is_unbounded_dynamic());
  ShapeUtil::GetMutableSubshape(&unbounded_tuple, {2})
      ->set_dynamic_dimension(1, true);
  EXPECT_FALSE(unbounded_tuple.is_unbounded_dynamic());
  ShapeUtil::GetMutableSubshape(&unbounded_tuple, {2})
      ->set_dimensions(1, Shape::kUnboundedSize, /*is_dynamic=*/true);
  EXPECT_TRUE(unbounded_tuple.is_unbounded_dynamic());
}

TEST_F(ShapeTest, IsDynamicDimension) {
  Shape dynamic_matrix = matrix_;
  dynamic_matrix.set_dynamic_dimension(1, true);
  EXPECT_FALSE(dynamic_matrix.is_dynamic_dimension(0));
  EXPECT_TRUE(dynamic_matrix.is_dynamic_dimension(1));

  Shape dynamic_tuple = tuple_;
  EXPECT_TRUE(dynamic_tuple.is_static());
  ShapeUtil::GetMutableSubshape(&dynamic_tuple, {2})
      ->set_dynamic_dimension(1, true);
  EXPECT_FALSE(dynamic_tuple.is_static());

  EXPECT_TRUE(unbounded_.is_dynamic_dimension(0));
  EXPECT_FALSE(unbounded_.is_dynamic_dimension(1));
}

TEST_F(ShapeTest, IsStaticDimension) {
  Shape dynamic_matrix = matrix_;
  dynamic_matrix.set_dynamic_dimension(1, true);
  EXPECT_TRUE(dynamic_matrix.is_static_dimension(0));
  EXPECT_FALSE(dynamic_matrix.is_static_dimension(1));
  EXPECT_FALSE(unbounded_.is_static_dimension(0));
  EXPECT_TRUE(unbounded_.is_static_dimension(1));
}

TEST_F(ShapeTest, ProgramShapeToFromProto) {
  ProgramShape program_shape;
  program_shape.AddParameter(ShapeUtil::MakeShape(F32, {1, 2, 3}), "foo");
  program_shape.AddParameter(ShapeUtil::MakeTokenShape(), "bar");
  program_shape.AddParameter(ShapeUtil::MakeShape(S64, {}), "baz");
  program_shape.AddParameter(
      ShapeUtil::MakeTupleShape(
          {ShapeUtil::MakeShape(S32, {}),
           ShapeUtil::MakeTupleShape({ShapeUtil::MakeTokenShape()}),
           ShapeUtil::MakeShape(F32, {42, 42})}),
      "qux qux");

  *program_shape.mutable_result() = ShapeUtil::MakeShape(F32, {7});

  // Create a copy of the program shape by round-tripping through a proto.
  TF_ASSERT_OK_AND_ASSIGN(auto program_shape_copy,
                          ProgramShape::FromProto(program_shape.ToProto()));
  ASSERT_EQ(program_shape.parameters_size(),
            program_shape_copy.parameters_size());
  for (int i = 0; i < program_shape.parameters_size(); ++i) {
    EXPECT_TRUE(ShapeUtil::Equal(program_shape.parameters(i),
                                 program_shape_copy.parameters(i)));
  }

  EXPECT_TRUE(
      ShapeUtil::Equal(program_shape.result(), program_shape_copy.result()));

  ASSERT_EQ(program_shape.parameters_size(),
            program_shape_copy.parameters_size());
  for (int i = 0; i < program_shape.parameters_size(); ++i) {
    EXPECT_EQ(program_shape.parameter_names(i),
              program_shape_copy.parameter_names(i));
  }
}

TEST_F(ShapeTest, ProgramShapeToString) {
  ProgramShape prog = ShapeUtil::MakeProgramShape(
      {opaque_, scalar_, matrix_, matrix2_, tuple_, nested_tuple_},
      nested_tuple_);
  EXPECT_EQ(
      "((unknown): opaque[], "
      "(unknown): f32[], "
      "(unknown): u32[1,2], "
      "(unknown): s32[3,4], "
      "(unknown): (opaque[], f32[], u32[1,2], s32[3,4]), "
      "(unknown): ((opaque[], f32[], u32[1,2], s32[3,4]), u32[1,2], token[])) "
      "-> "
      "((opaque[], f32[], u32[1,2], s32[3,4]), u32[1,2], token[])",
      prog.ToString());

  prog.set_parameter_names(0, "arg0");
  prog.set_parameter_names(1, "scalar");
  prog.set_parameter_names(2, "matrix");
  prog.set_parameter_names(3, "matrix2");
  prog.set_parameter_names(4, "tuple");
  prog.set_parameter_names(5, "nested_tuple");
  EXPECT_EQ(
      "(arg0: opaque[], "
      "scalar: f32[], "
      "matrix: u32[1,2], "
      "matrix2: s32[3,4], "
      "tuple: (opaque[], f32[], u32[1,2], s32[3,4]), "
      "nested_tuple: ((opaque[], f32[], u32[1,2], s32[3,4]), u32[1,2], "
      "token[])) "
      "-> "
      "((opaque[], f32[], u32[1,2], s32[3,4]), u32[1,2], token[])",
      prog.ToString());
}

TEST_F(ShapeTest, IgnoreSplitsComparison) {
  Shape shape = ShapeUtil::MakeShapeWithDenseLayout(F32, {256, 256}, {1, 0});
  Shape other_shape = shape;
  SplitConfig split_config(/*dimension=*/0, {128});
  other_shape.mutable_layout()->add_split_configs(split_config);

  EXPECT_TRUE(Shape::Equal().IgnoreSplitConfigInLayout()(shape, other_shape));
}

TEST_F(ShapeTest, SupportsAbslHash) {
  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly(
      {opaque_, token_, scalar_, scalar_with_tile_, matrix_, matrix2_, tuple_,
       nested_tuple_, dynamic_matrix_}));
}

TEST_F(ShapeTest, Fingerprint) {
  // Make a list of unique shapes.
  std::vector<Shape> shapes;
  shapes.push_back(Shape());
  shapes.push_back(Shape(F32, {1, 2, 3}, {}));
  shapes.push_back(Shape(U32, {1, 2, 3}, {}));
  shapes.push_back(Shape(U32, {1}, {}));
  shapes.push_back(Shape(U32, {1}, {true}));
  shapes.push_back(ShapeUtil::MakeTupleShape({shapes[1], shapes[2]}));
  shapes.push_back(ShapeUtil::MakeTupleShape({shapes[1], shapes[3]}));
  shapes.push_back(ShapeUtil::MakeTupleShape({shapes[1], shapes.back()}));
  shapes.push_back(Shape(TOKEN));
  shapes.push_back(Shape(OPAQUE_TYPE));

  for (const Shape& s1 : shapes) {
    SCOPED_TRACE(absl::StrCat("s=", ShapeUtil::HumanString(s1)));
    const uint64_t fp1 = s1.FingerprintWithLayout();

    // Check that copy has same fingerprint.
    Shape copy = s1;
    EXPECT_EQ(fp1, copy.FingerprintWithLayout()) << s1;

    // Check that converting to/from proto produces same fingerprint.
    auto decoded = Shape::FromProto(s1.ToProto());
    TF_ASSERT_OK(decoded.status());
    EXPECT_EQ(s1, *decoded);
    EXPECT_EQ(fp1, decoded->FingerprintWithLayout()) << s1;

    // Check that fingerprints match iff shapes are same.
    for (const Shape& s2 : shapes) {
      const uint64_t fp2 = s2.FingerprintWithLayout();
      if (&s1 == &s2) {
        EXPECT_EQ(fp1, fp2) << s1 << " vs. " << s2;
      } else {
        EXPECT_NE(fp1, fp2) << s1 << " vs. " << s2;
      }
    }
  }
}

TEST_F(ShapeTest, FingerprintEmpty) {
  Shape shape;
  const uint64_t fp = shape.FingerprintWithLayout();

  shape = Shape(F32, {1}, {});
  ASSERT_NE(fp, shape.FingerprintWithLayout());

  shape.Clear();
  ASSERT_EQ(fp, shape.FingerprintWithLayout());

  auto decoded = Shape::FromProto(Shape().ToProto());
  TF_ASSERT_OK(decoded.status());
  ASSERT_EQ(fp, decoded->FingerprintWithLayout());
}

TEST_F(ShapeTest, MutationChangesFingerprint) {
  Shape shape(F32, {1, 2, 3}, {});
  const uint64_t fp1 = shape.FingerprintWithLayout();
  Shape copy = shape;
  EXPECT_EQ(fp1, copy.FingerprintWithLayout());

  // Changing copy changes its fingerprint but not shape's fingerprint.
  copy.set_dimensions(1, 5);
  EXPECT_EQ(fp1, shape.FingerprintWithLayout());
  EXPECT_NE(fp1, copy.FingerprintWithLayout());

  // Changing copy back restores expected fingerprint.
  copy.set_dimensions(1, 2);
  EXPECT_EQ(fp1, shape.FingerprintWithLayout());
  EXPECT_EQ(fp1, copy.FingerprintWithLayout());
}

TEST_F(ShapeTest, AllMutationsChangeFingerprint) {
  Shape s(F32, {1, 2, 3, 4}, {});
  Shape last = s;

  // Apply a sequence of mutations and verify that each one changes the
  // fingerprint.
  auto apply = [&](auto mutation) {
    mutation();
    EXPECT_NE(s.FingerprintWithLayout(), last.FingerprintWithLayout());
    last = s;
  };

  // Array shape mutations.
  apply([&] { s.set_unbounded_dynamic_dimension(1); });
  apply([&] { s.set_dynamic_dimension(2, true); });
  apply([&] { s.set_dynamic_dimension(2, false); });
  apply([&] { s.DeleteDimension(2); });
  apply([&] { s.DeleteDimensions({2, 3}); });
  apply([&] { s.set_element_type(U32); });
  apply([&] { s.set_dimensions(0, 500); });
  apply([&] { s.clear_dimensions(); });
  apply([&] { s.add_dimensions(100); });
  apply([&] { s.set_dynamic_dimension(0, true); });
  apply([&] { s.clear_dynamic_dimensions(); });
  apply([&] { s.clear_dimensions(); });
  apply([&] { s.add_dimensions(100); });

  // Tuple shape mutations.
  apply([&] { *s.mutable_layout() = Layout({0}); });
  apply([&] { s = ShapeUtil::MakeTupleShape({s, s}); });
  apply([&] { s.mutable_tuple_shapes(0)->add_dimensions(200); });
  apply([&] { s.mutable_tuple_shapes(1)->set_dimensions(0, 300); });
  apply([&] { *s.add_tuple_shapes() = Shape(F32, {1}, {}); });

  apply([&] { s.Clear(); });
}

static const int kDistinctShapes = 4;

static Shape MakeShapeHelper(int id) {
  Shape shape;
  switch (id % kDistinctShapes) {
    case 0: {
      // Shape()
      break;
    }
    case 1: {
      // f32[1,2,2]{2,1,0}
      shape = Shape(F32, {1, 2, 2}, {false, false, false});
      *shape.mutable_layout() = Layout({2, 1, 0});
      break;
    }
    case 2: {
      // f32[1,2,2]{2,1,0:T(2,128)}
      shape = Shape(F32, {1, 2, 2}, {false, false, false});
      *shape.mutable_layout() = Layout({2, 1, 0}, {}, {Tile({2, 128})});
      break;
    }
    default: {
      // f32[1,2,2]{2,1,0}
      shape = Shape(F32, {1024, 1024, 128}, {});
    }
  }
  return shape;
}

void BM_ShapeProtoCopy(::testing::benchmark::State& state) {
  // Create different shapes based on benchmark parameters:
  Shape shape = MakeShapeHelper(state.range(0));
  state.SetLabel(shape.ToString(true));

  for (auto s : state) {
    auto copy = Shape::FromProto(shape.ToProto());
    TF_ASSERT_OK(copy);
    CHECK(ShapeUtil::Equal(shape, *copy));
  }
}
BENCHMARK(BM_ShapeProtoCopy)->Arg(0)->Arg(1)->Arg(2);

void BM_ShapeCopy(::testing::benchmark::State& state) {
  // Create different shapes based on benchmark parameters:
  const int n_shapes = state.range(0);
  std::vector<Shape> shapes(n_shapes);
  bool share = (state.range(1) != 0);
  for (int i = 0; i < n_shapes; i++) {
    if (share && (i >= kDistinctShapes)) {
      shapes[i] = shapes[i % kDistinctShapes];
    } else {
      shapes[i] = MakeShapeHelper(i);
    }
  }
  state.SetLabel(absl::StrCat("Working set: ", n_shapes, " shapes",
                              share ? " shared" : ""));

  int64_t iter = 0;
  Shape copy;
  for (auto s : state) {
    copy = shapes[iter];
    iter++;
    if (iter == n_shapes) {
      iter = 0;
    }
  }
}
BENCHMARK(BM_ShapeCopy)
    ->ArgPair(1, 0)
    ->ArgPair(1, 1)
    ->ArgPair(1000, 0)
    ->ArgPair(1000, 1)
    ->ArgPair(100000, 0)
    ->ArgPair(100000, 1);

}  // namespace
}  // namespace xla
