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

// Tests that our utility functions for dealing with literals are correctly
// implemented.

#include "xla/tests/literal_test_util.h"

#include <vector>

#include "absl/strings/str_join.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/literal.h"
#include "tsl/platform/env.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/path.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

TEST(LiteralTestUtilTest, ComparesEqualTuplesEqual) {
  Literal literal = LiteralUtil::MakeTupleFromSlices({
      LiteralUtil::CreateR0<int32_t>(42),
      LiteralUtil::CreateR0<int32_t>(64),
  });
  EXPECT_TRUE(LiteralTestUtil::Equal(literal, literal));
}

TEST(LiteralTestUtilTest, ComparesEqualComplex64TuplesEqual) {
  Literal literal = LiteralUtil::MakeTupleFromSlices({
      LiteralUtil::CreateR0<complex64>({42.0, 64.0}),
      LiteralUtil::CreateR0<complex64>({64.0, 42.0}),
  });
  EXPECT_TRUE(LiteralTestUtil::Equal(literal, literal));
}

TEST(LiteralTestUtilTest, ComparesEqualComplex128TuplesEqual) {
  Literal literal = LiteralUtil::MakeTupleFromSlices({
      LiteralUtil::CreateR0<complex128>({42.0, 64.0}),
      LiteralUtil::CreateR0<complex128>({64.0, 42.0}),
  });
  EXPECT_TRUE(LiteralTestUtil::Equal(literal, literal));
}

TEST(LiteralTestUtilTest, ComparesUnequalComplex64TuplesUnequal) {
  Literal literal0 = LiteralUtil::MakeTupleFromSlices({
      LiteralUtil::CreateR0<complex64>({42.0, 64.0}),
      LiteralUtil::CreateR0<complex64>({64.0, 42.0}),
  });
  Literal literal1 = LiteralUtil::MakeTupleFromSlices({
      LiteralUtil::CreateR0<complex64>({64.0, 42.0}),
      LiteralUtil::CreateR0<complex64>({42.0, 64.0}),
  });
  Literal literal2 = LiteralUtil::MakeTupleFromSlices({
      LiteralUtil::CreateR0<complex64>({42.42, 64.0}),
      LiteralUtil::CreateR0<complex64>({64.0, 42.0}),
  });
  Literal literal3 = LiteralUtil::MakeTupleFromSlices({
      LiteralUtil::CreateR0<complex64>({42.0, 64.0}),
      LiteralUtil::CreateR0<complex64>({64.0, 42.42}),
  });
  EXPECT_FALSE(LiteralTestUtil::Equal(literal0, literal1));
  EXPECT_FALSE(LiteralTestUtil::Equal(literal0, literal2));
  EXPECT_FALSE(LiteralTestUtil::Equal(literal0, literal3));
  EXPECT_FALSE(LiteralTestUtil::Equal(literal2, literal3));
}

TEST(LiteralTestUtilTest, ComparesUnequalComplex128TuplesUnequal) {
  Literal literal0 = LiteralUtil::MakeTupleFromSlices({
      LiteralUtil::CreateR0<complex128>({42.0, 64.0}),
      LiteralUtil::CreateR0<complex128>({64.0, 42.0}),
  });
  Literal literal1 = LiteralUtil::MakeTupleFromSlices({
      LiteralUtil::CreateR0<complex128>({64.0, 42.0}),
      LiteralUtil::CreateR0<complex128>({42.0, 64.0}),
  });
  Literal literal2 = LiteralUtil::MakeTupleFromSlices({
      LiteralUtil::CreateR0<complex128>({42.42, 64.0}),
      LiteralUtil::CreateR0<complex128>({64.0, 42.0}),
  });
  Literal literal3 = LiteralUtil::MakeTupleFromSlices({
      LiteralUtil::CreateR0<complex128>({42.0, 64.0}),
      LiteralUtil::CreateR0<complex128>({64.0, 42.42}),
  });
  EXPECT_FALSE(LiteralTestUtil::Equal(literal0, literal1));
  EXPECT_FALSE(LiteralTestUtil::Equal(literal0, literal2));
  EXPECT_FALSE(LiteralTestUtil::Equal(literal0, literal3));
  EXPECT_FALSE(LiteralTestUtil::Equal(literal2, literal3));
}

TEST(LiteralTestUtilTest, ComparesUnequalTuplesUnequal) {
  // Implementation note: we have to use a death test here, because you can't
  // un-fail an assertion failure. The CHECK-failure is death, so we can make a
  // death assertion.
  auto unequal_things_are_equal = [] {
    Literal lhs = LiteralUtil::MakeTupleFromSlices({
        LiteralUtil::CreateR0<int32_t>(42),
        LiteralUtil::CreateR0<int32_t>(64),
    });
    Literal rhs = LiteralUtil::MakeTupleFromSlices({
        LiteralUtil::CreateR0<int32_t>(64),
        LiteralUtil::CreateR0<int32_t>(42),
    });
    CHECK(LiteralTestUtil::Equal(lhs, rhs)) << "LHS and RHS are unequal";
  };
  ASSERT_DEATH(unequal_things_are_equal(), "LHS and RHS are unequal");
}

TEST(LiteralTestUtilTest, ExpectNearFailurePlacesResultsInTemporaryDirectory) {
  auto dummy_lambda = [] {
    auto two = LiteralUtil::CreateR0<float>(2);
    auto four = LiteralUtil::CreateR0<float>(4);
    ErrorSpec error(0.001);
    CHECK(LiteralTestUtil::Near(two, four, error)) << "two is not near four";
  };

  tsl::Env* env = tsl::Env::Default();

  std::string outdir;
  if (!tsl::io::GetTestUndeclaredOutputsDir(&outdir)) {
    outdir = tsl::testing::TmpDir();
  }
  std::string pattern = tsl::io::JoinPath(outdir, "tempfile-*.pb");
  std::vector<std::string> files;
  TF_CHECK_OK(env->GetMatchingPaths(pattern, &files));
  for (const auto& f : files) {
    TF_CHECK_OK(env->DeleteFile(f)) << f;
  }

  ASSERT_DEATH(dummy_lambda(), "two is not near four");

  // Now check we wrote temporary files to the temporary directory that we can
  // read.
  std::vector<std::string> results;
  TF_CHECK_OK(env->GetMatchingPaths(pattern, &results));

  LOG(INFO) << "results: [" << absl::StrJoin(results, ", ") << "]";
  EXPECT_EQ(3, results.size());
  for (const std::string& result : results) {
    LiteralProto literal_proto;
    TF_CHECK_OK(
        tsl::ReadBinaryProto(tsl::Env::Default(), result, &literal_proto));
    Literal literal = Literal::CreateFromProto(literal_proto).value();
    if (result.find("expected") != std::string::npos) {
      EXPECT_EQ("f32[] 2", literal.ToString());
    } else if (result.find("actual") != std::string::npos) {
      EXPECT_EQ("f32[] 4", literal.ToString());
    } else if (result.find("mismatches") != std::string::npos) {
      EXPECT_EQ("pred[] true", literal.ToString());
    } else {
      FAIL() << "unknown file in temporary directory: " << result;
    }
  }
}

TEST(LiteralTestUtilTest, NotEqualHasValuesInMessage) {
  auto expected = LiteralUtil::CreateR1<int32_t>({1, 2, 3});
  auto actual = LiteralUtil::CreateR1<int32_t>({4, 5, 6});
  ::testing::AssertionResult result = LiteralTestUtil::Equal(expected, actual);
  EXPECT_THAT(result.message(),
              ::testing::HasSubstr("Expected literal:\ns32[3] {1, 2, 3}"));
  EXPECT_THAT(result.message(),
              ::testing::HasSubstr("Actual literal:\ns32[3] {4, 5, 6}"));
}

TEST(LiteralTestUtilTest, NearComparatorR1) {
  auto a = LiteralUtil::CreateR1<float>(
      {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8});
  auto b = LiteralUtil::CreateR1<float>(
      {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8});
  EXPECT_TRUE(LiteralTestUtil::Near(a, b, ErrorSpec{0.0001}));
}

TEST(LiteralTestUtilTest, NearComparatorR1Complex64) {
  auto a = LiteralUtil::CreateR1<complex64>({{0.0, 1.0},
                                             {0.1, 1.1},
                                             {0.2, 1.2},
                                             {0.3, 1.3},
                                             {0.4, 1.4},
                                             {0.5, 1.5},
                                             {0.6, 1.6},
                                             {0.7, 1.7},
                                             {0.8, 1.8}});
  auto b = LiteralUtil::CreateR1<complex64>({{0.0, 1.0},
                                             {0.1, 1.1},
                                             {0.2, 1.2},
                                             {0.3, 1.3},
                                             {0.4, 1.4},
                                             {0.5, 1.5},
                                             {0.6, 1.6},
                                             {0.7, 1.7},
                                             {0.8, 1.8}});
  auto c = LiteralUtil::CreateR1<complex64>({{0.0, 1.0},
                                             {0.1, 1.1},
                                             {0.2, 1.2},
                                             {0.3, 1.3},
                                             {0.4, 1.4},
                                             {0.5, 1.5},
                                             {0.6, 1.6},
                                             {0.7, 1.7},
                                             {0.9, 1.8}});
  auto d = LiteralUtil::CreateR1<complex64>({{0.0, 1.0},
                                             {0.1, 1.1},
                                             {0.2, 1.2},
                                             {0.3, 1.3},
                                             {0.4, 1.4},
                                             {0.5, 1.5},
                                             {0.6, 1.6},
                                             {0.7, 1.7},
                                             {0.8, 1.9}});
  EXPECT_TRUE(LiteralTestUtil::Near(a, b, ErrorSpec{0.0001}));
  EXPECT_FALSE(LiteralTestUtil::Near(a, c, ErrorSpec{0.0001}));
  EXPECT_FALSE(LiteralTestUtil::Near(a, d, ErrorSpec{0.0001}));
  EXPECT_FALSE(LiteralTestUtil::Near(c, d, ErrorSpec{0.0001}));
}

TEST(LiteralTestUtilTest, NearComparatorR1Complex128) {
  auto a = LiteralUtil::CreateR1<complex128>({{0.0, 1.0},
                                              {0.1, 1.1},
                                              {0.2, 1.2},
                                              {0.3, 1.3},
                                              {0.4, 1.4},
                                              {0.5, 1.5},
                                              {0.6, 1.6},
                                              {0.7, 1.7},
                                              {0.8, 1.8}});
  auto b = LiteralUtil::CreateR1<complex128>({{0.0, 1.0},
                                              {0.1, 1.1},
                                              {0.2, 1.2},
                                              {0.3, 1.3},
                                              {0.4, 1.4},
                                              {0.5, 1.5},
                                              {0.6, 1.6},
                                              {0.7, 1.7},
                                              {0.8, 1.8}});
  auto c = LiteralUtil::CreateR1<complex128>({{0.0, 1.0},
                                              {0.1, 1.1},
                                              {0.2, 1.2},
                                              {0.3, 1.3},
                                              {0.4, 1.4},
                                              {0.5, 1.5},
                                              {0.6, 1.6},
                                              {0.7, 1.7},
                                              {0.9, 1.8}});
  auto d = LiteralUtil::CreateR1<complex128>({{0.0, 1.0},
                                              {0.1, 1.1},
                                              {0.2, 1.2},
                                              {0.3, 1.3},
                                              {0.4, 1.4},
                                              {0.5, 1.5},
                                              {0.6, 1.6},
                                              {0.7, 1.7},
                                              {0.8, 1.9}});
  EXPECT_TRUE(LiteralTestUtil::Near(a, b, ErrorSpec{0.0001}));
  EXPECT_FALSE(LiteralTestUtil::Near(a, c, ErrorSpec{0.0001}));
  EXPECT_FALSE(LiteralTestUtil::Near(a, d, ErrorSpec{0.0001}));
  EXPECT_FALSE(LiteralTestUtil::Near(c, d, ErrorSpec{0.0001}));
}

TEST(LiteralTestUtilTest, NearComparatorR1Nan) {
  auto a = LiteralUtil::CreateR1<float>(
      {0.0, 0.1, 0.2, 0.3, NAN, 0.5, 0.6, 0.7, 0.8});
  auto b = LiteralUtil::CreateR1<float>(
      {0.0, 0.1, 0.2, 0.3, NAN, 0.5, 0.6, 0.7, 0.8});
  EXPECT_TRUE(LiteralTestUtil::Near(a, b, ErrorSpec{0.0001}));
}

TEST(LiteralTestUtil, NearComparatorDifferentLengths) {
  auto a = LiteralUtil::CreateR1<float>(
      {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8});
  auto b =
      LiteralUtil::CreateR1<float>({0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7});
  EXPECT_FALSE(LiteralTestUtil::Near(a, b, ErrorSpec{0.0001}));
  EXPECT_FALSE(LiteralTestUtil::Near(b, a, ErrorSpec{0.0001}));
}

TEST(LiteralTestUtilTest, ExpectNearDoubleOutsideFloatValueRange) {
  auto two_times_float_max =
      LiteralUtil::CreateR0<double>(2.0 * std::numeric_limits<float>::max());
  ErrorSpec error(0.001);
  EXPECT_TRUE(
      LiteralTestUtil::Near(two_times_float_max, two_times_float_max, error));
}

TEST(LiteralTestUtilTest, DynamicEqualityR1) {
  auto literal1 = Literal(ShapeUtil::MakeValidatedShape(U32, {10}).value());
  literal1.PopulateR1<uint32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  literal1.SetDynamicSize(0, 5);
  auto literal2 = Literal(ShapeUtil::MakeValidatedShape(U32, {10}).value());
  literal2.PopulateR1<uint32_t>({1, 2, 3, 4, 5, 99, 99, 99, 99, 99});
  literal2.SetDynamicSize(0, 5);
  EXPECT_TRUE(LiteralTestUtil::Equal(literal1, literal2));
}

TEST(LiteralTestUtilTest, DynamicEqualityR2Dim) {
  auto literal1 = Literal(ShapeUtil::MakeValidatedShape(U32, {3, 3}).value());
  literal1.PopulateR2<uint32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  literal1.SetDynamicSize(0, 2);
  auto literal2 = Literal(ShapeUtil::MakeValidatedShape(U32, {3, 3}).value());
  literal2.PopulateR2<uint32_t>({{1, 2, 3}, {4, 5, 6}, {99, 99, 99}});
  literal2.SetDynamicSize(0, 2);
  EXPECT_TRUE(LiteralTestUtil::Equal(literal1, literal2));
}

TEST(LiteralTestUtilTest, DynamicEqualityR2Dim1) {
  auto literal1 = Literal(ShapeUtil::MakeValidatedShape(U32, {3, 3}).value());
  literal1.PopulateR2<uint32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  literal1.SetDynamicSize(1, 2);
  auto literal2 = Literal(ShapeUtil::MakeValidatedShape(U32, {3, 3}).value());
  literal2.PopulateR2<uint32_t>({{1, 2, 99}, {4, 5, 99}, {7, 8, 99}});
  literal2.SetDynamicSize(1, 2);
  EXPECT_TRUE(LiteralTestUtil::Equal(literal1, literal2));
}

TEST(LiteralTestUtilTest, DynamicNearEqualityR1) {
  auto literal1 = Literal(ShapeUtil::MakeValidatedShape(F32, {10}).value());
  literal1.PopulateR1<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  literal1.SetDynamicSize(0, 5);
  auto literal2 = Literal(ShapeUtil::MakeValidatedShape(F32, {10}).value());
  literal2.PopulateR1<float>({1, 2, 3, 4, 5, 99, 99, 99, 99, 99});
  literal2.SetDynamicSize(0, 5);
  ErrorSpec error(0.001);
  EXPECT_TRUE(LiteralTestUtil::Near(literal1, literal2, error));
}

TEST(LiteralTestUtilTest, DynamicNearEqualityR2Dim) {
  auto literal1 = Literal(ShapeUtil::MakeValidatedShape(F32, {3, 3}).value());
  literal1.PopulateR2<float>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  literal1.SetDynamicSize(0, 2);
  auto literal2 = Literal(ShapeUtil::MakeValidatedShape(F32, {3, 3}).value());
  literal2.PopulateR2<float>({{1, 2, 3}, {4, 5, 6}, {99, 99, 99}});
  literal2.SetDynamicSize(0, 2);
  ErrorSpec error(0.001);
  EXPECT_TRUE(LiteralTestUtil::Near(literal1, literal2, error));
}

TEST(LiteralTestUtilTest, DynamicNearEqualityR2Dim1) {
  auto literal1 = Literal(ShapeUtil::MakeValidatedShape(F32, {3, 3}).value());
  literal1.PopulateR2<float>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  literal1.SetDynamicSize(1, 2);
  auto literal2 = Literal(ShapeUtil::MakeValidatedShape(F32, {3, 3}).value());
  literal2.PopulateR2<float>({{1, 2, 99}, {4, 5, 99}, {7, 8, 99}});
  literal2.SetDynamicSize(1, 2);
  ErrorSpec error(0.001);
  EXPECT_TRUE(LiteralTestUtil::Near(literal1, literal2, error));
}

TEST(LiteralTestUtilTest, UnequalDynamicDimensionsR1) {
  auto literal1 = Literal(ShapeUtil::MakeValidatedShape(U32, {10}).value());
  literal1.PopulateR1<uint32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  literal1.SetDynamicSize(0, 5);
  auto literal2 = Literal(ShapeUtil::MakeValidatedShape(U32, {10}).value());
  literal2.PopulateR1<uint32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  literal2.SetDynamicSize(0, 6);
  // Dynamic sizes do not match.
  EXPECT_FALSE(LiteralTestUtil::Equal(literal1, literal2));
}

TEST(LiteralTestUtilTest, UnequalDynamicDimensionsR1_F32) {
  auto literal1 = Literal(ShapeUtil::MakeValidatedShape(F32, {10}).value());
  literal1.PopulateR1<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  literal1.SetDynamicSize(0, 5);
  auto literal2 = Literal(ShapeUtil::MakeValidatedShape(F32, {10}).value());
  literal2.PopulateR1<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  literal2.SetDynamicSize(0, 6);
  // Dynamic sizes do not match.
  EXPECT_FALSE(LiteralTestUtil::Near(literal1, literal2, ErrorSpec{0.0001}));
}

TEST(LiteralTestUtilTest, ExpectedIsDynamicActualIsNotR1) {
  auto literal1 = Literal(ShapeUtil::MakeValidatedShape(U32, {10}).value());
  literal1.PopulateR1<uint32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  literal1.SetDynamicSize(0, 5);
  auto literal2 = Literal(ShapeUtil::MakeValidatedShape(U32, {10}).value());
  literal2.PopulateR1<uint32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  // Only literal1 is dynamic.
  EXPECT_FALSE(LiteralTestUtil::Equal(literal1, literal2));
}

TEST(LiteralTestUtilTest, ExpectedIsDynamicActualIsNotR1_F32) {
  auto literal1 = Literal(ShapeUtil::MakeValidatedShape(F32, {10}).value());
  literal1.PopulateR1<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  literal1.SetDynamicSize(0, 5);
  auto literal2 = Literal(ShapeUtil::MakeValidatedShape(F32, {10}).value());
  literal2.PopulateR1<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  // Only literal1 is dynamic.
  EXPECT_FALSE(LiteralTestUtil::Near(literal1, literal2, ErrorSpec{0.0001}));
}

TEST(LiteralTestUtilTest, ActualIsDynamicExpectedIsNotR1) {
  auto literal1 = Literal(ShapeUtil::MakeValidatedShape(U32, {10}).value());
  literal1.PopulateR1<uint32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  auto literal2 = Literal(ShapeUtil::MakeValidatedShape(U32, {10}).value());
  literal2.PopulateR1<uint32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  literal2.SetDynamicSize(0, 5);
  // Only literal2 is dynamic.
  EXPECT_FALSE(LiteralTestUtil::Equal(literal1, literal2));
}

TEST(LiteralTestUtilTest, ActualIsDynamicExpectedIsNotR1_F32) {
  auto literal1 = Literal(ShapeUtil::MakeValidatedShape(F32, {10}).value());
  literal1.PopulateR1<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  auto literal2 = Literal(ShapeUtil::MakeValidatedShape(F32, {10}).value());
  literal2.PopulateR1<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  literal2.SetDynamicSize(0, 5);
  // Only literal2 is dynamic.
  EXPECT_FALSE(LiteralTestUtil::Near(literal1, literal2, ErrorSpec{0.0001}));
}

TEST(LiteralTestUtilTest, UnequalDynamicDimensionsR2Dim0) {
  auto literal1 = Literal(ShapeUtil::MakeValidatedShape(U32, {3, 3}).value());
  literal1.PopulateR2<uint32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  literal1.SetDynamicSize(0, 2);
  auto literal2 = Literal(ShapeUtil::MakeValidatedShape(U32, {3, 3}).value());
  literal2.PopulateR2<uint32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  literal2.SetDynamicSize(0, 3);
  // Dynamic sizes do not match.
  EXPECT_FALSE(LiteralTestUtil::Equal(literal1, literal2));
}

TEST(LiteralTestUtilTest, UnequalDynamicDimensionsR2Dim0_F32) {
  auto literal1 = Literal(ShapeUtil::MakeValidatedShape(F32, {3, 3}).value());
  literal1.PopulateR2<float>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  literal1.SetDynamicSize(0, 2);
  auto literal2 = Literal(ShapeUtil::MakeValidatedShape(F32, {3, 3}).value());
  literal2.PopulateR2<float>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  literal2.SetDynamicSize(0, 3);
  // Dynamic sizes do not match.
  EXPECT_FALSE(LiteralTestUtil::Near(literal1, literal2, ErrorSpec{0.0001}));
}

TEST(LiteralTestUtilTest, UnequalDynamicDimensionsR2Dim1) {
  auto literal1 = Literal(ShapeUtil::MakeValidatedShape(U32, {3, 3}).value());
  literal1.PopulateR2<uint32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  literal1.SetDynamicSize(1, 2);
  auto literal2 = Literal(ShapeUtil::MakeValidatedShape(U32, {3, 3}).value());
  literal2.PopulateR2<uint32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  literal2.SetDynamicSize(1, 3);
  // Dynamic sizes do not match.
  EXPECT_FALSE(LiteralTestUtil::Equal(literal1, literal2));
}

TEST(LiteralTestUtilTest, UnequalDynamicDimensionsR2Dim1_F32) {
  auto literal1 = Literal(ShapeUtil::MakeValidatedShape(F32, {3, 3}).value());
  literal1.PopulateR2<float>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  literal1.SetDynamicSize(1, 2);
  auto literal2 = Literal(ShapeUtil::MakeValidatedShape(F32, {3, 3}).value());
  literal2.PopulateR2<float>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  literal2.SetDynamicSize(1, 3);
  // Dynamic sizes do not match.
  EXPECT_FALSE(LiteralTestUtil::Near(literal1, literal2, ErrorSpec{0.0001}));
}

TEST(LiteralTestUtilTest, UnequalDynamicDimensionsR2DifferentDimensions) {
  auto literal1 = Literal(ShapeUtil::MakeValidatedShape(U32, {3, 3}).value());
  literal1.PopulateR2<uint32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  literal1.SetDynamicSize(1, 2);
  auto literal2 = Literal(ShapeUtil::MakeValidatedShape(U32, {3, 3}).value());
  literal2.PopulateR2<uint32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  literal2.SetDynamicSize(0, 2);
  // Different dimensions were set as dynamic.
  EXPECT_FALSE(LiteralTestUtil::Equal(literal1, literal2));
}

TEST(LiteralTestUtilTest, UnequalDynamicDimensionsR2DifferentDimensions_F32) {
  auto literal1 = Literal(ShapeUtil::MakeValidatedShape(F32, {3, 3}).value());
  literal1.PopulateR2<float>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  literal1.SetDynamicSize(1, 2);
  auto literal2 = Literal(ShapeUtil::MakeValidatedShape(F32, {3, 3}).value());
  literal2.PopulateR2<float>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  literal2.SetDynamicSize(0, 2);
  // Different dimensions were set as dynamic.
  EXPECT_FALSE(LiteralTestUtil::Near(literal1, literal2, ErrorSpec{0.0001}));
}

TEST(LiteralTestUtilTest, DynamicTuplesAreEqual) {
  auto literal1 = Literal(
      ShapeUtil::MakeValidatedTupleShape(
          {ShapeUtil::MakeShape(U32, {5}), ShapeUtil::MakeShape(U32, {5})})
          .value());
  auto literal2 = Literal(
      ShapeUtil::MakeValidatedTupleShape(
          {ShapeUtil::MakeShape(U32, {5}), ShapeUtil::MakeShape(U32, {5})})
          .value());
  MutableBorrowingLiteral(&literal1, /*view_root=*/{0})
      .PopulateR1<uint32_t>({1, 2, 3, 4, 5});
  MutableBorrowingLiteral(&literal1, /*view_root=*/{1})
      .PopulateR1<uint32_t>({1, 2, 3, 4, 5});
  literal1.SetDynamicSize(0, {0}, 5);
  MutableBorrowingLiteral(&literal2, /*view_root=*/{0})
      .PopulateR1<uint32_t>({1, 2, 3, 4, 5});
  MutableBorrowingLiteral(&literal2, /*view_root=*/{1})
      .PopulateR1<uint32_t>({1, 2, 3, 4, 5});
  literal2.SetDynamicSize(0, {0}, 5);
  EXPECT_TRUE(LiteralTestUtil::Equal(literal1, literal2));
}

TEST(LiteralTestUtilTest, DynamicTuplesAreNear) {
  auto literal1 = Literal(
      ShapeUtil::MakeValidatedTupleShape(
          {ShapeUtil::MakeShape(F32, {5}), ShapeUtil::MakeShape(F32, {5})})
          .value());
  auto literal2 = Literal(
      ShapeUtil::MakeValidatedTupleShape(
          {ShapeUtil::MakeShape(F32, {5}), ShapeUtil::MakeShape(F32, {5})})
          .value());
  MutableBorrowingLiteral(&literal1, /*view_root=*/{0})
      .PopulateR1<float>({1, 2, 3, 4, 5});
  MutableBorrowingLiteral(&literal1, /*view_root=*/{1})
      .PopulateR1<float>({1, 2, 3, 4, 5});
  literal1.SetDynamicSize(0, {0}, 5);
  MutableBorrowingLiteral(&literal2, /*view_root=*/{0})
      .PopulateR1<float>({1, 2, 3, 4, 5});
  MutableBorrowingLiteral(&literal2, /*view_root=*/{1})
      .PopulateR1<float>({1, 2, 3, 4, 5});
  literal2.SetDynamicSize(0, {0}, 5);
  EXPECT_TRUE(LiteralTestUtil::Near(literal1, literal2, ErrorSpec{0.0001}));
}

TEST(LiteralTestUtilTest, DynamicTuplesAreEqualWithinDynamicBounds) {
  auto literal1 = Literal(
      ShapeUtil::MakeValidatedTupleShape(
          {ShapeUtil::MakeShape(U32, {5}), ShapeUtil::MakeShape(U32, {5})})
          .value());
  auto literal2 = Literal(
      ShapeUtil::MakeValidatedTupleShape(
          {ShapeUtil::MakeShape(U32, {5}), ShapeUtil::MakeShape(U32, {5})})
          .value());
  MutableBorrowingLiteral(&literal1, /*view_root=*/{0})
      .PopulateR1<uint32_t>({1, 2, 3, 4, 5});
  MutableBorrowingLiteral(&literal1, /*view_root=*/{1})
      .PopulateR1<uint32_t>({1, 2, 3, 4, 5});
  literal1.SetDynamicSize(0, {0}, 3);
  MutableBorrowingLiteral(&literal2, /*view_root=*/{0})
      .PopulateR1<uint32_t>({1, 2, 3, 99, 99});
  MutableBorrowingLiteral(&literal2, /*view_root=*/{1})
      .PopulateR1<uint32_t>({1, 2, 3, 4, 5});
  literal2.SetDynamicSize(0, {0}, 3);
  EXPECT_TRUE(LiteralTestUtil::Equal(literal1, literal2));
}

TEST(LiteralTestUtilTest, DynamicTuplesAreNearWithinDynamicBounds) {
  auto literal1 = Literal(
      ShapeUtil::MakeValidatedTupleShape(
          {ShapeUtil::MakeShape(F32, {5}), ShapeUtil::MakeShape(F32, {5})})
          .value());
  auto literal2 = Literal(
      ShapeUtil::MakeValidatedTupleShape(
          {ShapeUtil::MakeShape(F32, {5}), ShapeUtil::MakeShape(F32, {5})})
          .value());
  MutableBorrowingLiteral(&literal1, /*view_root=*/{0})
      .PopulateR1<float>({1, 2, 3, 4, 5});
  MutableBorrowingLiteral(&literal1, /*view_root=*/{1})
      .PopulateR1<float>({1, 2, 3, 4, 5});
  literal1.SetDynamicSize(0, {0}, 3);
  MutableBorrowingLiteral(&literal2, /*view_root=*/{0})
      .PopulateR1<float>({1, 2, 3, 99, 99});
  MutableBorrowingLiteral(&literal2, /*view_root=*/{1})
      .PopulateR1<float>({1, 2, 3, 4, 5});
  literal2.SetDynamicSize(0, {0}, 3);
  EXPECT_TRUE(LiteralTestUtil::Near(literal1, literal2, ErrorSpec{0.0001}));
}

TEST(LiteralTestUtilTest, DynamicTuplesHaveDifferentDynamicSizes) {
  auto literal1 = Literal(
      ShapeUtil::MakeValidatedTupleShape(
          {ShapeUtil::MakeShape(U32, {5}), ShapeUtil::MakeShape(U32, {5})})
          .value());
  auto literal2 = Literal(
      ShapeUtil::MakeValidatedTupleShape(
          {ShapeUtil::MakeShape(U32, {5}), ShapeUtil::MakeShape(U32, {5})})
          .value());
  MutableBorrowingLiteral(&literal1, /*view_root=*/{0})
      .PopulateR1<uint32_t>({1, 2, 3, 4, 5});
  MutableBorrowingLiteral(&literal1, /*view_root=*/{1})
      .PopulateR1<uint32_t>({1, 2, 3, 4, 5});
  literal1.SetDynamicSize(0, {0}, 5);
  MutableBorrowingLiteral(&literal2, /*view_root=*/{0})
      .PopulateR1<uint32_t>({1, 2, 3, 4, 5});
  MutableBorrowingLiteral(&literal2, /*view_root=*/{1})
      .PopulateR1<uint32_t>({1, 2, 3, 4, 5});
  literal2.SetDynamicSize(0, {0}, 4);
  // Dynamic sizes are not equal.
  EXPECT_FALSE(LiteralTestUtil::Equal(literal1, literal2));
}

TEST(LiteralTestUtilTest, DynamicTuplesHaveDifferentDynamicSizes_F32) {
  auto literal1 = Literal(
      ShapeUtil::MakeValidatedTupleShape(
          {ShapeUtil::MakeShape(F32, {5}), ShapeUtil::MakeShape(F32, {5})})
          .value());
  auto literal2 = Literal(
      ShapeUtil::MakeValidatedTupleShape(
          {ShapeUtil::MakeShape(F32, {5}), ShapeUtil::MakeShape(F32, {5})})
          .value());
  MutableBorrowingLiteral(&literal1, /*view_root=*/{0})
      .PopulateR1<float>({1, 2, 3, 4, 5});
  MutableBorrowingLiteral(&literal1, /*view_root=*/{1})
      .PopulateR1<float>({1, 2, 3, 4, 5});
  literal1.SetDynamicSize(0, {0}, 5);
  MutableBorrowingLiteral(&literal2, /*view_root=*/{0})
      .PopulateR1<float>({1, 2, 3, 4, 5});
  MutableBorrowingLiteral(&literal2, /*view_root=*/{1})
      .PopulateR1<float>({1, 2, 3, 4, 5});
  literal2.SetDynamicSize(0, {0}, 4);
  // Dynamic sizes are not equal.
  EXPECT_FALSE(LiteralTestUtil::Near(literal1, literal2, ErrorSpec{0.0001}));
}

TEST(LiteralTestUtilTest, OneTupleDynamicOneIsNot) {
  auto literal1 = Literal(
      ShapeUtil::MakeValidatedTupleShape(
          {ShapeUtil::MakeShape(U32, {5}), ShapeUtil::MakeShape(U32, {5})})
          .value());
  auto literal2 = Literal(
      ShapeUtil::MakeValidatedTupleShape(
          {ShapeUtil::MakeShape(U32, {5}), ShapeUtil::MakeShape(U32, {5})})
          .value());
  MutableBorrowingLiteral(&literal1, /*view_root=*/{0})
      .PopulateR1<uint32_t>({1, 2, 3, 4, 5});
  MutableBorrowingLiteral(&literal1, /*view_root=*/{1})
      .PopulateR1<uint32_t>({1, 2, 3, 4, 5});
  literal1.SetDynamicSize(0, {0}, 5);
  MutableBorrowingLiteral(&literal2, /*view_root=*/{0})
      .PopulateR1<uint32_t>({1, 2, 3, 4, 5});
  MutableBorrowingLiteral(&literal2, /*view_root=*/{1})
      .PopulateR1<uint32_t>({1, 2, 3, 4, 5});
  // Only one of the tuples is dynamic.
  EXPECT_FALSE(LiteralTestUtil::Equal(literal1, literal2));
}

TEST(LiteralTestUtilTest, OneTupleDynamicOneIsNot_F32) {
  auto literal1 = Literal(
      ShapeUtil::MakeValidatedTupleShape(
          {ShapeUtil::MakeShape(F32, {5}), ShapeUtil::MakeShape(F32, {5})})
          .value());
  auto literal2 = Literal(
      ShapeUtil::MakeValidatedTupleShape(
          {ShapeUtil::MakeShape(F32, {5}), ShapeUtil::MakeShape(F32, {5})})
          .value());
  MutableBorrowingLiteral(&literal1, /*view_root=*/{0})
      .PopulateR1<float>({1, 2, 3, 4, 5});
  MutableBorrowingLiteral(&literal1, /*view_root=*/{1})
      .PopulateR1<float>({1, 2, 3, 4, 5});
  literal1.SetDynamicSize(0, {0}, 5);
  MutableBorrowingLiteral(&literal2, /*view_root=*/{0})
      .PopulateR1<float>({1, 2, 3, 4, 5});
  MutableBorrowingLiteral(&literal2, /*view_root=*/{1})
      .PopulateR1<float>({1, 2, 3, 4, 5});
  // Only one of the tuples is dynamic.
  EXPECT_FALSE(LiteralTestUtil::Near(literal1, literal2, ErrorSpec{0.0001}));
}
}  // namespace
}  // namespace xla
