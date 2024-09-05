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
#include "tensorflow/lite/array.h"

#include <algorithm>
#include <initializer_list>
#include <type_traits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "tensorflow/lite/core/c/common.h"

using testing::ElementsAreArray;
using testing::Eq;

namespace tflite {
namespace {

absl::Span<int> GetSpan(TfLiteIntArray& array) {
  return {array.data, static_cast<size_t>(array.size)};
}

absl::Span<float> GetSpan(TfLiteFloatArray& array) {
  return {array.data, static_cast<size_t>(array.size)};
}

template <class T>
class TfLiteArrayTest : public testing::Test {
  static_assert(
      std::is_same_v<TfLiteIntArray, TfLiteArrayUniquePtr<int>::element_type>,
      "TfLiteArrayUniquePtr<int>::element_type should be TfLiteIntArray");
  static_assert(
      std::is_same_v<TfLiteFloatArray,
                     TfLiteArrayUniquePtr<float>::element_type>,
      "TfLiteArrayUniquePtr<float>::element_type should be TfLiteFloatArray");
};

using ArrayTypes = testing::Types<int, float>;

TYPED_TEST_SUITE(TfLiteArrayTest, ArrayTypes);

TYPED_TEST(TfLiteArrayTest, BuildArrayWithSize) {
  constexpr int size = 3;
  TfLiteArrayUniquePtr<TypeParam> array = BuildTfLiteArray<TypeParam>(size);
  ASSERT_NE(array, nullptr);
  EXPECT_THAT(array->size, Eq(size));
  // Touch data to check that allocation is not too small.
  std::fill_n(array->data, size, static_cast<TypeParam>(1));
}

TYPED_TEST(TfLiteArrayTest, BuildFromDynamicArray) {
  constexpr int size = 4;
  constexpr TypeParam values[size] = {1, 2, 3, 4};
  TfLiteArrayUniquePtr<TypeParam> array = BuildTfLiteArray(size, values);
  ASSERT_NE(array, nullptr);
  EXPECT_THAT(array->size, Eq(size));
  EXPECT_THAT(GetSpan(*array), ElementsAreArray(values));
}

TYPED_TEST(TfLiteArrayTest, BuildFromCArray) {
  TypeParam values[] = {1, 2, 3, 4};
  TfLiteArrayUniquePtr<TypeParam> array = BuildTfLiteArray(values);
  ASSERT_NE(array, nullptr);
  EXPECT_THAT(array->size, Eq(sizeof(values) / sizeof(TypeParam)));
  EXPECT_THAT(GetSpan(*array), ElementsAreArray(values));
}

TYPED_TEST(TfLiteArrayTest, BuildFromVector) {
  std::vector<TypeParam> values = {1, 2, 3, 4};
  TfLiteArrayUniquePtr<TypeParam> array = BuildTfLiteArray(values);
  ASSERT_NE(array, nullptr);
  EXPECT_THAT(array->size, Eq(values.size()));
  EXPECT_THAT(GetSpan(*array), ElementsAreArray(values));
}

TYPED_TEST(TfLiteArrayTest, BuildFromVectorForceType) {
  using DifferentType =
      std::conditional_t<std::is_same_v<TypeParam, int>, float, int>;
  std::vector<DifferentType> values = {1, 2, 3, 4};
  TfLiteArrayUniquePtr<TypeParam> array = BuildTfLiteArray<TypeParam>(values);
  ASSERT_NE(array, nullptr);
  EXPECT_THAT(array->size, Eq(values.size()));
  EXPECT_THAT(GetSpan(*array), ElementsAreArray(values));
}

TYPED_TEST(TfLiteArrayTest, BuildFromSpan) {
  std::vector<TypeParam> values = {1, 2, 3, 4};
  TfLiteArrayUniquePtr<TypeParam> array =
      BuildTfLiteArray(absl::Span<const TypeParam>(values));
  ASSERT_NE(array, nullptr);
  EXPECT_THAT(array->size, Eq(values.size()));
  EXPECT_THAT(GetSpan(*array), ElementsAreArray(values));
}

TYPED_TEST(TfLiteArrayTest, BuildFromInitializerList) {
  std::initializer_list<TypeParam> values{1, 2, 3, 4};
  TfLiteArrayUniquePtr<TypeParam> array = BuildTfLiteArray(values);
  ASSERT_NE(array, nullptr);
  EXPECT_THAT(array->size, Eq(values.size()));
  EXPECT_THAT(GetSpan(*array), ElementsAreArray(values));
}

TYPED_TEST(TfLiteArrayTest, BuildUsingSingleElementInitializerList) {
  constexpr TypeParam value = 42;
  TfLiteArrayUniquePtr<TypeParam> array = BuildTfLiteArray({value});
  ASSERT_NE(array, nullptr);
  EXPECT_THAT(array->size, Eq(1));
  EXPECT_THAT(array->data[0], Eq(value));
}

TYPED_TEST(TfLiteArrayTest, BuildFromTfLiteArray) {
  std::initializer_list<TypeParam> values{1, 2, 3, 4};
  const auto ref = BuildTfLiteArray(values);
  TfLiteArrayUniquePtr<TypeParam> array = BuildTfLiteArray(*ref);
  ASSERT_NE(array, nullptr);
  EXPECT_THAT(array->size, Eq(values.size()));
  EXPECT_THAT(GetSpan(*array), ElementsAreArray(values));
}

}  // namespace
}  // namespace tflite
