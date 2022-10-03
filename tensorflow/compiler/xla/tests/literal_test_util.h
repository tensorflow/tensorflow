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

#ifndef TENSORFLOW_COMPILER_XLA_TESTS_LITERAL_TEST_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_TESTS_LITERAL_TEST_UTIL_H_

#include <initializer_list>
#include <memory>
#include <optional>
#include <random>
#include <string>

#include "absl/base/attributes.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/array3d.h"
#include "tensorflow/compiler/xla/array4d.h"
#include "tensorflow/compiler/xla/error_spec.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/tsl/platform/test.h"

namespace xla {

// Utility class for making expectations/assertions related to XLA literals.
class LiteralTestUtil {
 public:
  // Asserts that the given shapes have the same rank, dimension sizes, and
  // primitive types.
  [[nodiscard]] static ::testing::AssertionResult EqualShapes(
      const Shape& expected, const Shape& actual);

  // Asserts that the provided shapes are equal as defined in AssertEqualShapes
  // and that they have the same layout.
  [[nodiscard]] static ::testing::AssertionResult EqualShapesAndLayouts(
      const Shape& expected, const Shape& actual);

  [[nodiscard]] static ::testing::AssertionResult Equal(
      const LiteralSlice& expected, const LiteralSlice& actual);

  // Asserts the given literal are (bitwise) equal to given expected values.
  template <typename NativeT>
  static void ExpectR0Equal(NativeT expected, const LiteralSlice& actual);

  template <typename NativeT>
  static void ExpectR1Equal(absl::Span<const NativeT> expected,
                            const LiteralSlice& actual);
  template <typename NativeT>
  static void ExpectR2Equal(
      std::initializer_list<std::initializer_list<NativeT>> expected,
      const LiteralSlice& actual);

  template <typename NativeT>
  static void ExpectR3Equal(
      std::initializer_list<
          std::initializer_list<std::initializer_list<NativeT>>>
          expected,
      const LiteralSlice& actual);

  // Asserts the given literal are (bitwise) equal to given array.
  template <typename NativeT>
  static void ExpectR2EqualArray2D(const Array2D<NativeT>& expected,
                                   const LiteralSlice& actual);
  template <typename NativeT>
  static void ExpectR3EqualArray3D(const Array3D<NativeT>& expected,
                                   const LiteralSlice& actual);
  template <typename NativeT>
  static void ExpectR4EqualArray4D(const Array4D<NativeT>& expected,
                                   const LiteralSlice& actual);

  // Decorates literal_comparison::Near() with an AssertionResult return type.
  //
  // See comment on literal_comparison::Near().
  [[nodiscard]] static ::testing::AssertionResult Near(
      const LiteralSlice& expected, const LiteralSlice& actual,
      const ErrorSpec& error_spec,
      std::optional<bool> detailed_message = std::nullopt);

  // Asserts the given literal are within the given error bound of the given
  // expected values. Only supported for floating point values.
  template <typename NativeT>
  static void ExpectR0Near(NativeT expected, const LiteralSlice& actual,
                           const ErrorSpec& error);

  template <typename NativeT>
  static void ExpectR1Near(absl::Span<const NativeT> expected,
                           const LiteralSlice& actual, const ErrorSpec& error);

  template <typename NativeT>
  static void ExpectR2Near(
      std::initializer_list<std::initializer_list<NativeT>> expected,
      const LiteralSlice& actual, const ErrorSpec& error);

  template <typename NativeT>
  static void ExpectR3Near(
      std::initializer_list<
          std::initializer_list<std::initializer_list<NativeT>>>
          expected,
      const LiteralSlice& actual, const ErrorSpec& error);

  template <typename NativeT>
  static void ExpectR4Near(
      std::initializer_list<std::initializer_list<
          std::initializer_list<std::initializer_list<NativeT>>>>
          expected,
      const LiteralSlice& actual, const ErrorSpec& error);

  // Asserts the given literal are within the given error bound to the given
  // array. Only supported for floating point values.
  template <typename NativeT>
  static void ExpectR2NearArray2D(const Array2D<NativeT>& expected,
                                  const LiteralSlice& actual,
                                  const ErrorSpec& error);

  template <typename NativeT>
  static void ExpectR3NearArray3D(const Array3D<NativeT>& expected,
                                  const LiteralSlice& actual,
                                  const ErrorSpec& error);

  template <typename NativeT>
  static void ExpectR4NearArray4D(const Array4D<NativeT>& expected,
                                  const LiteralSlice& actual,
                                  const ErrorSpec& error);

  // If the error spec is given, returns whether the expected and the actual are
  // within the error bound; otherwise, returns whether they are equal. Tuples
  // will be compared recursively.
  [[nodiscard]] static ::testing::AssertionResult NearOrEqual(
      const LiteralSlice& expected, const LiteralSlice& actual,
      const std::optional<ErrorSpec>& error);

 private:
  LiteralTestUtil(const LiteralTestUtil&) = delete;
  LiteralTestUtil& operator=(const LiteralTestUtil&) = delete;
};

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR0Equal(NativeT expected,
                                                 const LiteralSlice& actual) {
  EXPECT_TRUE(Equal(LiteralUtil::CreateR0<NativeT>(expected), actual));
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR1Equal(
    absl::Span<const NativeT> expected, const LiteralSlice& actual) {
  EXPECT_TRUE(Equal(LiteralUtil::CreateR1<NativeT>(expected), actual));
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR2Equal(
    std::initializer_list<std::initializer_list<NativeT>> expected,
    const LiteralSlice& actual) {
  EXPECT_TRUE(Equal(LiteralUtil::CreateR2<NativeT>(expected), actual));
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR3Equal(
    std::initializer_list<std::initializer_list<std::initializer_list<NativeT>>>
        expected,
    const LiteralSlice& actual) {
  EXPECT_TRUE(Equal(LiteralUtil::CreateR3<NativeT>(expected), actual));
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR2EqualArray2D(
    const Array2D<NativeT>& expected, const LiteralSlice& actual) {
  EXPECT_TRUE(Equal(LiteralUtil::CreateR2FromArray2D(expected), actual));
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR3EqualArray3D(
    const Array3D<NativeT>& expected, const LiteralSlice& actual) {
  EXPECT_TRUE(Equal(LiteralUtil::CreateR3FromArray3D(expected), actual));
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR4EqualArray4D(
    const Array4D<NativeT>& expected, const LiteralSlice& actual) {
  EXPECT_TRUE(Equal(LiteralUtil::CreateR4FromArray4D(expected), actual));
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR0Near(NativeT expected,
                                                const LiteralSlice& actual,
                                                const ErrorSpec& error) {
  EXPECT_TRUE(Near(LiteralUtil::CreateR0<NativeT>(expected), actual, error));
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR1Near(
    absl::Span<const NativeT> expected, const LiteralSlice& actual,
    const ErrorSpec& error) {
  EXPECT_TRUE(Near(LiteralUtil::CreateR1<NativeT>(expected), actual, error));
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR2Near(
    std::initializer_list<std::initializer_list<NativeT>> expected,
    const LiteralSlice& actual, const ErrorSpec& error) {
  EXPECT_TRUE(Near(LiteralUtil::CreateR2<NativeT>(expected), actual, error));
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR3Near(
    std::initializer_list<std::initializer_list<std::initializer_list<NativeT>>>
        expected,
    const LiteralSlice& actual, const ErrorSpec& error) {
  EXPECT_TRUE(Near(LiteralUtil::CreateR3<NativeT>(expected), actual, error));
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR4Near(
    std::initializer_list<std::initializer_list<
        std::initializer_list<std::initializer_list<NativeT>>>>
        expected,
    const LiteralSlice& actual, const ErrorSpec& error) {
  EXPECT_TRUE(Near(LiteralUtil::CreateR4<NativeT>(expected), actual, error));
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR2NearArray2D(
    const Array2D<NativeT>& expected, const LiteralSlice& actual,
    const ErrorSpec& error) {
  EXPECT_TRUE(Near(LiteralUtil::CreateR2FromArray2D(expected), actual, error));
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR3NearArray3D(
    const Array3D<NativeT>& expected, const LiteralSlice& actual,
    const ErrorSpec& error) {
  EXPECT_TRUE(Near(LiteralUtil::CreateR3FromArray3D(expected), actual, error));
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR4NearArray4D(
    const Array4D<NativeT>& expected, const LiteralSlice& actual,
    const ErrorSpec& error) {
  EXPECT_TRUE(Near(LiteralUtil::CreateR4FromArray4D(expected), actual, error));
}

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_TESTS_LITERAL_TEST_UTIL_H_
