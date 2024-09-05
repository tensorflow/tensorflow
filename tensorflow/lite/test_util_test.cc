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
#include "tensorflow/lite/test_util.h"

#include <sstream>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>
#include "absl/strings/str_format.h"
#include "tensorflow/lite/array.h"
#include "tensorflow/lite/c/common.h"

using ::testing::FloatNear;

namespace tflite {
namespace {

template <class T>
class TfArrayIsTest : public testing::Test {};

using TfArrayIsTestTypes = ::testing::Types<int, float>;

TYPED_TEST_SUITE(TfArrayIsTest, TfArrayIsTestTypes);

TYPED_TEST(TfArrayIsTest, AbslStringifyWorks) {
  const TfLiteArrayUniquePtr<TypeParam> array =
      BuildTfLiteArray<TypeParam>({1, 2, 3});
  EXPECT_EQ(absl::StrFormat("%v", *array), "[1, 2, 3]");
  EXPECT_EQ(absl::StrFormat("%v", array), "[1, 2, 3]");
  EXPECT_EQ(absl::StrFormat("%v", array.get()), "[1, 2, 3]");
  TfLiteArrayUniquePtr<TypeParam> null_array_ptr(nullptr);
  EXPECT_EQ(absl::StrFormat("%v", null_array_ptr), "nullptr");
  EXPECT_EQ(absl::StrFormat("%v", null_array_ptr.get()), "nullptr");
}

TYPED_TEST(TfArrayIsTest, CompilesWithExpectedTfLiteArray) {
  const TfLiteArrayUniquePtr<TypeParam> array =
      BuildTfLiteArray<TypeParam>({1, 2, 3});
  EXPECT_THAT(array, TfLiteArrayIs(*array));
}

TYPED_TEST(TfArrayIsTest, CompilesWithExpectedTfLiteArrayRawPtr) {
  const TfLiteArrayUniquePtr<TypeParam> array =
      BuildTfLiteArray<TypeParam>({1, 2, 3});
  EXPECT_THAT(array, TfLiteArrayIs(array.get()));
}

TYPED_TEST(TfArrayIsTest, CompilesWithExpectedTfLiteArraySmartPtr) {
  const TfLiteArrayUniquePtr<TypeParam> array =
      BuildTfLiteArray<TypeParam>({1, 2, 3});
  EXPECT_THAT(array, TfLiteArrayIs(array));
}

TYPED_TEST(TfArrayIsTest, CompilesWithExpectedIntitializerList) {
  const TfLiteArrayUniquePtr<TypeParam> array =
      BuildTfLiteArray<TypeParam>({1, 2, 3});
  EXPECT_THAT(array, TfLiteArrayIs({1, 2, 3}));
}

TYPED_TEST(TfArrayIsTest, CompilesWithExpectedVector) {
  const std::vector<TypeParam> values = {1, 2, 3};
  const TfLiteArrayUniquePtr<TypeParam> array =
      BuildTfLiteArray<TypeParam>(values);
  EXPECT_THAT(array, TfLiteArrayIs(values));
}

TYPED_TEST(TfArrayIsTest, CanMatchReference) {
  const std::vector<TypeParam> values = {1, 2, 3};
  const TfLiteArrayUniquePtr<TypeParam> array =
      BuildTfLiteArray<TypeParam>(values);
  EXPECT_THAT(*array, TfLiteArrayIs(values));
}

TYPED_TEST(TfArrayIsTest, CanMatchRawPointer) {
  const std::vector<TypeParam> values = {1, 2, 3};
  const TfLiteArrayUniquePtr<TypeParam> array =
      BuildTfLiteArray<TypeParam>(values);
  EXPECT_THAT(array.get(), TfLiteArrayIs(values));
}

TYPED_TEST(TfArrayIsTest, FailsWithDifferentArray) {
  const std::vector<TypeParam> expected = {1, 2, 3};
  const std::vector<TypeParam> provided = {2, 2, 3};
  ASSERT_NE(expected, provided);
  const TfLiteArrayUniquePtr<TypeParam> array =
      BuildTfLiteArray<TypeParam>(provided);
  EXPECT_NONFATAL_FAILURE(EXPECT_THAT(array, TfLiteArrayIs(expected)), "");
}

TYPED_TEST(TfArrayIsTest, FailsWithNullPointer) {
  const TfLiteIntArray* const ptr = nullptr;
  EXPECT_NONFATAL_FAILURE(EXPECT_THAT(ptr, TfLiteArrayIs({1, 2, 3})), "");
}

TYPED_TEST(TfArrayIsTest, FailsWithLongerExpected) {
  const std::vector<TypeParam> expected = {1, 2, 3, 4};
  const std::vector<TypeParam> provided = {1, 2, 3};
  const TfLiteArrayUniquePtr<TypeParam> array = BuildTfLiteArray(provided);
  EXPECT_NONFATAL_FAILURE(EXPECT_THAT(*array, TfLiteArrayIs(expected)), "");
}

TYPED_TEST(TfArrayIsTest, FailsWithShorterExpected) {
  const std::vector<TypeParam> expected = {1, 2};
  const std::vector<TypeParam> provided = {1, 2, 3};
  const TfLiteArrayUniquePtr<TypeParam> array = BuildTfLiteArray(provided);
  EXPECT_NONFATAL_FAILURE(EXPECT_THAT(*array, TfLiteArrayIs(expected)), "");
}

TYPED_TEST(TfArrayIsTest, SucceedsWithFloatNearMatcherWithinExpectedTolerance) {
  const std::vector<TypeParam> expected = {1, 2, 3};
  const std::vector<TypeParam> provided = {2, 1, 4};
  ASSERT_NE(expected, provided);
  const TfLiteArrayUniquePtr<TypeParam> array = BuildTfLiteArray(provided);
  EXPECT_THAT(array, TfLiteArrayIs(FloatNear(1), expected));
}

TYPED_TEST(TfArrayIsTest, FailsWithFloatNearMatcherOutsideExpectedTolerance) {
  const std::vector<TypeParam> expected = {1, 2, 3};
  const std::vector<TypeParam> provided = {1, 2, 5};
  ASSERT_NE(expected, provided);
  const TfLiteArrayUniquePtr<TypeParam> array = BuildTfLiteArray(provided);
  EXPECT_NONFATAL_FAILURE(
      EXPECT_THAT(array, TfLiteArrayIs(FloatNear(1), expected)), "");
}

// Returns the reason why x matches, or doesn't match, m.
template <typename MatcherType, typename Value>
std::string Explain(const Value& x, const MatcherType& m) {
  ::testing::StringMatchResultListener listener;
  ExplainMatchResult(m, x, &listener);
  return listener.str();
}

TYPED_TEST(TfArrayIsTest, SuccessHasNoExplanation) {
  const std::vector<TypeParam> expected = {1, 2, 3};
  const TfLiteArrayUniquePtr<TypeParam> array = BuildTfLiteArray(expected);
  EXPECT_THAT(Explain(array, TfLiteArrayIs(expected)), "");
}

TYPED_TEST(TfArrayIsTest, TooShortFailureHasExplanation) {
  const std::vector<TypeParam> expected{1, 2, 3};
  const std::vector<TypeParam> provided{1, 2};
  const TfLiteArrayUniquePtr<TypeParam> array = BuildTfLiteArray(provided);
  EXPECT_THAT(Explain(array, TfLiteArrayIs(expected)),
              "does not have the right size. Expected 3, got 2");
}

TYPED_TEST(TfArrayIsTest, TooLongFailureHasExplanation) {
  const std::vector<TypeParam> expected{1, 2, 3};
  const std::vector<TypeParam> provided{1, 2, 3, 4};
  const TfLiteArrayUniquePtr<TypeParam> array = BuildTfLiteArray(provided);
  EXPECT_THAT(Explain(array, TfLiteArrayIs(expected)),
              "does not have the right size. Expected 3, got 4");
}

TYPED_TEST(TfArrayIsTest, MismatchFailureHasExplanation) {
  const std::vector<TypeParam> expected{1, 2, 3};
  const std::vector<TypeParam> provided{1, 2, 4};
  const TfLiteArrayUniquePtr<TypeParam> array = BuildTfLiteArray(provided);
  EXPECT_THAT(Explain(array, TfLiteArrayIs(expected)),
              "the values (4, 3) at index 2 do not match.");
}

TYPED_TEST(TfArrayIsTest, DescribeToIsCorrect) {
  std::stringstream sstr;
  ::testing::Matcher<const TfLiteIntArray&> m = TfLiteArrayIs({1, 2, 3});
  m.DescribeTo(&sstr);
  EXPECT_THAT(sstr.str(),
              "has each element and its corresponding value in [1, 2, 3] that "
              "are an equal pair");
}

TYPED_TEST(TfArrayIsTest, DescribeNegationToIsCorrect) {
  std::stringstream sstr;
  ::testing::Matcher<const TfLiteIntArray&> m = TfLiteArrayIs({1, 2, 3});
  m.DescribeNegationTo(&sstr);
  EXPECT_THAT(sstr.str(),
              "has at least one element and its corresponding value in [1, "
              "2, 3] that "
              "aren't an equal pair");
}

}  // namespace
}  // namespace tflite
