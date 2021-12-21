/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/tests/literal_test_util.h"

#include <vector>

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/test.h"

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

  tensorflow::Env* env = tensorflow::Env::Default();

  std::string outdir;
  if (!tensorflow::io::GetTestUndeclaredOutputsDir(&outdir)) {
    outdir = tensorflow::testing::TmpDir();
  }
  std::string pattern = tensorflow::io::JoinPath(outdir, "tempfile-*.pb");
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
    TF_CHECK_OK(tensorflow::ReadBinaryProto(tensorflow::Env::Default(), result,
                                            &literal_proto));
    Literal literal =
        Literal::CreateFromProto(literal_proto).ConsumeValueOrDie();
    if (result.find("expected") != string::npos) {
      EXPECT_EQ("f32[] 2", literal.ToString());
    } else if (result.find("actual") != string::npos) {
      EXPECT_EQ("f32[] 4", literal.ToString());
    } else if (result.find("mismatches") != string::npos) {
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

}  // namespace
}  // namespace xla
