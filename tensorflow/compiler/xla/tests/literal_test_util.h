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
#include <random>
#include <string>

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/array3d.h"
#include "tensorflow/compiler/xla/array4d.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/optional.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// Structure describing permissible absolute and relative error bounds.
struct ErrorSpec {
  explicit ErrorSpec(float aabs, float arel = 0, bool relaxed_nans = false)
      : abs(aabs), rel(arel), relaxed_nans(relaxed_nans) {}

  float abs;  // Absolute error bound.
  float rel;  // Relative error bound.

  // If relaxed_nans is true then any result is valid if we are expecting NaNs.
  // In effect, this allows the tested operation to produce incorrect results
  // for inputs outside its mathematical domain.
  bool relaxed_nans;
};

// Utility class for making expectations/assertions related to XLA literals.
class LiteralTestUtil {
 public:
  // Asserts that the given shapes have the same rank, dimension sizes, and
  // primitive types.
  static ::testing::AssertionResult EqualShapes(
      const Shape& expected, const Shape& actual) MUST_USE_RESULT;

  // Asserts that the provided shapes are equal as defined in AssertEqualShapes
  // and that they have the same layout.
  static ::testing::AssertionResult EqualShapesAndLayouts(
      const Shape& expected, const Shape& actual) MUST_USE_RESULT;

  static ::testing::AssertionResult Equal(const LiteralSlice& expected,
                                          const LiteralSlice& actual)
      TF_MUST_USE_RESULT;

  // Asserts the given literal are (bitwise) equal to given expected values.
  template <typename NativeT>
  static void ExpectR0Equal(NativeT expected, const LiteralSlice& actual);

  template <typename NativeT>
  static void ExpectR1Equal(tensorflow::gtl::ArraySlice<NativeT> expected,
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

  // Asserts that the expected and actual literals are within the given error
  // bound for all elements. Also, asserts that the rank, dimensions sizes, and
  // bounds are equivalent.
  //
  // Tuples are matched recursively.  When comparing tensors of
  // non-floating-point type, checks for exact equality, ignoring the ErrorSpec.
  //
  // If the shape of the literals is neither a complex/floating-point tensor nor
  // a tuple which contains a complex/floating-point tensor, Near() is
  // equivalent to Equal().  We don't raise an error in this case, because we
  // want to allow callers to call Near() even if they have no preconceptions
  // about the shapes being compared.
  //
  // If detailed_message is true, then the error message in the assertion result
  // will contain a more detailed breakdown of mismatches.
  static ::testing::AssertionResult Near(
      const LiteralSlice& expected, const LiteralSlice& actual,
      const ErrorSpec& error, bool detailed_message = false) TF_MUST_USE_RESULT;

  // Asserts the given literal are within the given error bound of the given
  // expected values. Only supported for floating point values.
  template <typename NativeT>
  static void ExpectR0Near(NativeT expected, const LiteralSlice& actual,
                           const ErrorSpec& error);

  template <typename NativeT>
  static void ExpectR1Near(tensorflow::gtl::ArraySlice<NativeT> expected,
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
  static ::testing::AssertionResult NearOrEqual(
      const LiteralSlice& expected, const LiteralSlice& actual,
      const tensorflow::gtl::optional<ErrorSpec>& error) TF_MUST_USE_RESULT;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(LiteralTestUtil);
};

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR0Equal(NativeT expected,
                                                 const LiteralSlice& actual) {
  EXPECT_TRUE(Equal(*Literal::CreateR0<NativeT>(expected), actual));
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR1Equal(
    tensorflow::gtl::ArraySlice<NativeT> expected, const LiteralSlice& actual) {
  EXPECT_TRUE(Equal(*Literal::CreateR1<NativeT>(expected), actual));
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR2Equal(
    std::initializer_list<std::initializer_list<NativeT>> expected,
    const LiteralSlice& actual) {
  EXPECT_TRUE(Equal(*Literal::CreateR2<NativeT>(expected), actual));
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR3Equal(
    std::initializer_list<std::initializer_list<std::initializer_list<NativeT>>>
        expected,
    const LiteralSlice& actual) {
  EXPECT_TRUE(Equal(*Literal::CreateR3<NativeT>(expected), actual));
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR2EqualArray2D(
    const Array2D<NativeT>& expected, const LiteralSlice& actual) {
  EXPECT_TRUE(Equal(*Literal::CreateR2FromArray2D(expected), actual));
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR3EqualArray3D(
    const Array3D<NativeT>& expected, const LiteralSlice& actual) {
  EXPECT_TRUE(Equal(*Literal::CreateR3FromArray3D(expected), actual));
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR4EqualArray4D(
    const Array4D<NativeT>& expected, const LiteralSlice& actual) {
  EXPECT_TRUE(Equal(*Literal::CreateR4FromArray4D(expected), actual));
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR0Near(NativeT expected,
                                                const LiteralSlice& actual,
                                                const ErrorSpec& error) {
  EXPECT_TRUE(Near(*Literal::CreateR0<NativeT>(expected), actual, error));
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR1Near(
    tensorflow::gtl::ArraySlice<NativeT> expected, const LiteralSlice& actual,
    const ErrorSpec& error) {
  EXPECT_TRUE(Near(*Literal::CreateR1<NativeT>(expected), actual, error));
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR2Near(
    std::initializer_list<std::initializer_list<NativeT>> expected,
    const LiteralSlice& actual, const ErrorSpec& error) {
  EXPECT_TRUE(Near(*Literal::CreateR2<NativeT>(expected), actual, error));
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR3Near(
    std::initializer_list<std::initializer_list<std::initializer_list<NativeT>>>
        expected,
    const LiteralSlice& actual, const ErrorSpec& error) {
  EXPECT_TRUE(Near(*Literal::CreateR3<NativeT>(expected), actual, error));
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR4Near(
    std::initializer_list<std::initializer_list<
        std::initializer_list<std::initializer_list<NativeT>>>>
        expected,
    const LiteralSlice& actual, const ErrorSpec& error) {
  EXPECT_TRUE(Near(*Literal::CreateR4<NativeT>(expected), actual, error));
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR2NearArray2D(
    const Array2D<NativeT>& expected, const LiteralSlice& actual,
    const ErrorSpec& error) {
  EXPECT_TRUE(Near(*Literal::CreateR2FromArray2D(expected), actual, error));
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR3NearArray3D(
    const Array3D<NativeT>& expected, const LiteralSlice& actual,
    const ErrorSpec& error) {
  EXPECT_TRUE(Near(*Literal::CreateR3FromArray3D(expected), actual, error));
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR4NearArray4D(
    const Array4D<NativeT>& expected, const LiteralSlice& actual,
    const ErrorSpec& error) {
  EXPECT_TRUE(Near(*Literal::CreateR4FromArray4D(expected), actual, error));
}

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_TESTS_LITERAL_TEST_UTIL_H_
