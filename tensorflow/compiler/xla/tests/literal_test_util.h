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
#include <string>

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/array3d.h"
#include "tensorflow/compiler/xla/array4d.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// Structure describing permissible absolute and relative error bounds.
struct ErrorSpec {
  explicit ErrorSpec(float aabs, float arel = 0) : abs(aabs), rel(arel) {}

  float abs;  // Absolute error bound.
  float rel;  // Relative error bound.
};

// Utility class for making expectations/assertions related to XLA literals.
class LiteralTestUtil {
 public:
  // Asserts that the given shapes have the same rank, dimension sizes, and
  // primitive types.
  static void AssertEqualShapes(const Shape& expected, const Shape& actual);

  // Asserts that the provided shapes are equal as defined in AssertEqualShapes
  // and that they have the same layout.
  static void AssertEqualShapesAndLayouts(const Shape& expected,
                                          const Shape& actual);

  // Asserts that the expected and actual literals are (bitwise) equal for all
  // elements in the literal. Also, asserts that the rank, dimensions sizes, and
  // primitive type are equal.
  static testing::AssertionResult Equal(
      const Literal& expected, const Literal& actual) TF_MUST_USE_RESULT;

  // Expects that expected and actual are Equal.
  static void ExpectEqual(const Literal& expected, const Literal& actual);

  // Expects that expected and actual are Not Equal.
  static void ExpectNotEqual(const Literal& expected, const Literal& actual);

  // Asserts the given literal are (bitwise) equal to given expected values.
  template <typename NativeT>
  static void ExpectR0Equal(NativeT expected, const Literal& actual);
  template <typename NativeT>
  static void ExpectR1Equal(tensorflow::gtl::ArraySlice<NativeT> expected,
                            const Literal& actual);
  template <typename NativeT>
  static void ExpectR2Equal(
      std::initializer_list<std::initializer_list<NativeT>> expected,
      const Literal& actual);
  template <typename NativeT>
  static void ExpectR3Equal(
      std::initializer_list<
          std::initializer_list<std::initializer_list<NativeT>>>
          expected,
      const Literal& actual);

  // Asserts the given literal are (bitwise) equal to given array.
  template <typename NativeT>
  static void ExpectR2EqualArray2D(const Array2D<NativeT>& expected,
                                   const Literal& actual);
  template <typename NativeT>
  static void ExpectR3EqualArray3D(const Array3D<NativeT>& expected,
                                   const Literal& actual);
  template <typename NativeT>
  static void ExpectR4EqualArray4D(const Array4D<NativeT>& expected,
                                   const Literal& actual);

  // Expects that the values of the elements in the expected and actual tuples
  // are equal. Tuples are matched recursively.
  static void ExpectEqualTuple(const Literal& expected, const Literal& actual);

  // Asserts that the expected and actual literals are within the given error
  // bound for all elements. Also, asserts that the rank, dimensions sizes, and
  // bounds are equivalent. Only supported for floating point values.
  static testing::AssertionResult Near(
      const Literal& expected, const Literal& actual,
      const ErrorSpec& error) TF_MUST_USE_RESULT;

  // Expects expected and actual to be Near with the given error.
  static void ExpectNear(const Literal& expected, const Literal& actual,
                         const ErrorSpec& error);

  // Asserts the given literal are within the given error bound of the given
  // expected values. Only supported for floating point values.
  template <typename NativeT>
  static void ExpectR0Near(NativeT expected, const Literal& actual,
                           const ErrorSpec& error);
  template <typename NativeT>
  static void ExpectR1Near(tensorflow::gtl::ArraySlice<NativeT> expected,
                           const Literal& actual, const ErrorSpec& error);
  template <typename NativeT>
  static void ExpectR2Near(
      std::initializer_list<std::initializer_list<NativeT>> expected,
      const Literal& actual, const ErrorSpec& error);
  template <typename NativeT>
  static void ExpectR3Near(
      std::initializer_list<
          std::initializer_list<std::initializer_list<NativeT>>>
          expected,
      const Literal& actual, const ErrorSpec& error);

  // Asserts the given literal are within the given error bound to the given
  // array. Only supported for floating point values.
  template <typename NativeT>
  static void ExpectR2NearArray2D(const Array2D<NativeT>& expected,
                                  const Literal& actual,
                                  const ErrorSpec& error);
  template <typename NativeT>
  static void ExpectR3NearArray3D(const Array3D<NativeT>& expected,
                                  const Literal& actual,
                                  const ErrorSpec& error);
  template <typename NativeT>
  static void ExpectR4NearArray4D(const Array4D<NativeT>& expected,
                                  const Literal& actual,
                                  const ErrorSpec& error);

  // Returns whether the values of the elements in the expected and actual
  // tuples are within the given error bound. Tuples are matched recursively.
  // If the elements of the tuple are not floating-point types, the error spec
  // is ignored and exact equality is checked.
  static testing::AssertionResult NearTuple(
      const Literal& expected, const Literal& actual,
      const ErrorSpec& error) TF_MUST_USE_RESULT;

  // Expects that the expected and actual values are near.
  static void ExpectNearTuple(const Literal& expected, const Literal& actual,
                              const ErrorSpec& error);

  // Returns a multi-dimensional index as a string. For example: '{7, 8}' will
  // be returned for a 2-dimensional index with dimension 0 index equal to 7,
  // dimension 1 equal to 8.
  static string MultiIndexAsString(
      tensorflow::gtl::ArraySlice<int64> multi_index);

  // Creates a literal with a new shape with the given new dimensions using the
  // data in the given input literal. For reshaping purposes the (flat) data
  // buffer of the input literal is assumed to have the given minor_to_major
  // layout order.
  static std::unique_ptr<Literal> Reshape(
      tensorflow::gtl::ArraySlice<int64> new_dimensions,
      tensorflow::gtl::ArraySlice<int64> minor_to_major,
      const Literal& literal);

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(LiteralTestUtil);
};

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR0Equal(NativeT expected,
                                                 const Literal& actual) {
  ExpectEqual(*LiteralUtil::CreateR0<NativeT>(expected), actual);
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR1Equal(
    tensorflow::gtl::ArraySlice<NativeT> expected, const Literal& actual) {
  ExpectEqual(*LiteralUtil::CreateR1<NativeT>(expected), actual);
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR2Equal(
    std::initializer_list<std::initializer_list<NativeT>> expected,
    const Literal& actual) {
  ExpectEqual(*LiteralUtil::CreateR2<NativeT>(expected), actual);
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR3Equal(
    std::initializer_list<std::initializer_list<std::initializer_list<NativeT>>>
        expected,
    const Literal& actual) {
  ExpectEqual(*LiteralUtil::CreateR3<NativeT>(expected), actual);
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR2EqualArray2D(
    const Array2D<NativeT>& expected, const Literal& actual) {
  ExpectEqual(*LiteralUtil::CreateR2FromArray2D(expected), actual);
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR3EqualArray3D(
    const Array3D<NativeT>& expected, const Literal& actual) {
  ExpectEqual(*LiteralUtil::CreateR3FromArray3D(expected), actual);
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR4EqualArray4D(
    const Array4D<NativeT>& expected, const Literal& actual) {
  ExpectEqual(*LiteralUtil::CreateR4FromArray4D(expected), actual);
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR0Near(NativeT expected,
                                                const Literal& actual,
                                                const ErrorSpec& error) {
  ExpectNear(*LiteralUtil::CreateR0<NativeT>(expected), actual, error);
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR1Near(
    tensorflow::gtl::ArraySlice<NativeT> expected, const Literal& actual,
    const ErrorSpec& error) {
  ExpectNear(*LiteralUtil::CreateR1<NativeT>(expected), actual, error);
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR2Near(
    std::initializer_list<std::initializer_list<NativeT>> expected,
    const Literal& actual, const ErrorSpec& error) {
  ExpectNear(*LiteralUtil::CreateR2<NativeT>(expected), actual, error);
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR3Near(
    std::initializer_list<std::initializer_list<std::initializer_list<NativeT>>>
        expected,
    const Literal& actual, const ErrorSpec& error) {
  ExpectNear(*LiteralUtil::CreateR3<NativeT>(expected), actual, error);
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR2NearArray2D(
    const Array2D<NativeT>& expected, const Literal& actual,
    const ErrorSpec& error) {
  ExpectNear(*LiteralUtil::CreateR2FromArray2D(expected), actual, error);
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR3NearArray3D(
    const Array3D<NativeT>& expected, const Literal& actual,
    const ErrorSpec& error) {
  ExpectNear(*LiteralUtil::CreateR3FromArray3D(expected), actual, error);
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR4NearArray4D(
    const Array4D<NativeT>& expected, const Literal& actual,
    const ErrorSpec& error) {
  ExpectNear(*LiteralUtil::CreateR4FromArray4D(expected), actual, error);
}

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_TESTS_LITERAL_TEST_UTIL_H_
