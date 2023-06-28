/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_TEST_UTIL_H_
#define TENSORFLOW_LITE_TEST_UTIL_H_

#include <ostream>
#include <sstream>
#include <string>
#include <type_traits>

#include "gtest/gtest.h"
#include "absl/strings/str_join.h"
#include "tensorflow/lite/array.h"
#include "tensorflow/lite/c/test_util.h"

namespace tflite {
namespace testing {

class Test : public ::testing::Test {
 public:
  void SetUp() override {
    ASSERT_EQ(TfLiteInitializeShimsForTest(), 0);
  }
};

}  // namespace testing

namespace test_util_internal {

template <class T>
struct TfLiteArrayDataTypeImpl;

template <>
struct TfLiteArrayDataTypeImpl<TfLiteIntArray> {
  using Type = int;
};

template <>
struct TfLiteArrayDataTypeImpl<TfLiteArrayUniquePtr<int>> {
  using Type = int;
};

template <>
struct TfLiteArrayDataTypeImpl<TfLiteFloatArray> {
  using Type = float;
};

template <>
struct TfLiteArrayDataTypeImpl<TfLiteArrayUniquePtr<float>> {
  using Type = float;
};

// Maps from the array type to its data type.
template <class T>
using TfLiteArrayDataType =
    typename test_util_internal::TfLiteArrayDataTypeImpl<
        std::remove_const_t<std::remove_pointer_t<std::decay_t<T>>>>::Type;

// Matches TFLite array values against the expected values.
template <class LhsType, class RhsType, class TfLiteArrayType>
bool MatchAndExplainTfLiteArray(
    const TfLiteArrayType& array, const std::vector<RhsType>& rhs,
    const ::testing::Matcher<std::tuple<LhsType, RhsType>>& m,
    ::testing::MatchResultListener* listener) {
  if (static_cast<size_t>(array.size) != rhs.size()) {
    *listener << "does not have the right size. Expected " << rhs.size()
              << ", got " << array.size;
    return false;
  }
  for (int i = 0; i < array.size; ++i) {
    ::testing::StringMatchResultListener inner_listener;
    if (!m.MatchAndExplain(std::tuple<LhsType, RhsType>(array.data[i], rhs[i]),
                           &inner_listener)) {
      *listener << "the values (" << array.data[i] << ", " << rhs[i]
                << ") at index " << i << " do not match.";
      const std::string matcher_explanation = inner_listener.str();
      if (!matcher_explanation.empty()) {
        *listener << matcher_explanation;
      }
      return false;
    }
  }
  return true;
}

// Matches TFLite array pointee with the expected values.
template <class LhsType, class RhsType, class TfLiteArrayType>
bool MatchAndExplainTfLiteArray(
    TfLiteArrayType* const array, const std::vector<RhsType>& rhs,
    const ::testing::Matcher<std::tuple<LhsType, RhsType>>& m,
    ::testing::MatchResultListener* listener) {
  if (array == nullptr) {
    *listener << "does not point to an array.";
    return false;
  }
  return MatchAndExplainTfLiteArray(*array, rhs, m, listener);
}

// Matches TFLite array pointee with the expected values.
template <class LhsType, class RhsType>
bool MatchAndExplainTfLiteArray(
    const TfLiteArrayUniquePtr<LhsType>& array, const std::vector<RhsType>& rhs,
    const ::testing::Matcher<std::tuple<LhsType, RhsType>>& m,
    ::testing::MatchResultListener* listener) {
  return MatchAndExplainTfLiteArray(array.get(), rhs, m, listener);
}

}  // namespace test_util_internal

// Matches a TFLite array against the provided expected values.
//
// This matcher allows checking TFLite array references, raw and smart
// pointers against the expected result.
//
// Raw pointers make for the vast majority of TFLite array use cases. This
// matchers aims to simplify testing those.
//
// A multi-argument matcher may be provided for tolerance, otherwise
// `testing::Eq` is used. This should be most useful for float arrays in
// combination with `testing::FloatNear` but will also work with int arrays.
template <class TupleMatcher, class RhsType>
class TfArrayMatcher {
 public:
  using RhsDataContainer = std::vector<RhsType>;

  TfArrayMatcher(const TupleMatcher& tuple_matcher,
                 absl::Span<const RhsType> rhs)
      : tuple_matcher_(tuple_matcher), rhs_(rhs.begin(), rhs.end()) {}

  // This has to be implicit. GTest expects the matcher factory functions to
  // return an object of type `testing::Matcher<Lhs>`. This matcher is
  // polymorphic: it does not know the LHS beforehand, so we need to be able to
  // implicitely convert it to the expected type when it is called.
  //
  // Note that this is the canonical way of implementing this kind of matcher.
  // See `testing::PointwiseMatcher` for an example.
  template <class ArrayType>
  operator ::testing::Matcher<ArrayType>() const {
    return ::testing::Matcher<ArrayType>(
        new Impl<ArrayType>(tuple_matcher_, rhs_));
  }

  template <class TfLiteArrayType>
  class Impl : public ::testing::MatcherInterface<TfLiteArrayType> {
   public:
    using LhsType = test_util_internal::TfLiteArrayDataType<TfLiteArrayType>;
    using TupleArg = std::tuple<LhsType, RhsType>;

    Impl(const TupleMatcher& tuple_matcher, const RhsDataContainer& rhs)
        : m_(::testing::SafeMatcherCast<TupleArg>(tuple_matcher)), rhs_(rhs) {}

    // Because of the 2 level template matcher implementation, we do not have
    // much control on the signature of MatchAndexplain. We use a helper
    // function to handle pointer and value semantics at the same time.
    bool MatchAndExplain(
        const TfLiteArrayType& array,
        ::testing::MatchResultListener* listener) const override {
      return test_util_internal::MatchAndExplainTfLiteArray(array, rhs_, m_,
                                                            listener);
    }

    void DescribeTo(std::ostream* os) const override {
      *os << "has each element and its corresponding value in [";
      *os << absl::StrJoin(rhs_, ", ");
      *os << "] that ";
      m_.DescribeTo(os);
    }

    void DescribeNegationTo(std::ostream* os) const override {
      *os << "has at least one element and its corresponding value in [";
      *os << absl::StrJoin(rhs_, ", ");
      *os << "] that ";
      m_.DescribeNegationTo(os);
    }

   private:
    ::testing::Matcher<TupleArg> m_;
    RhsDataContainer rhs_;
  };

  TupleMatcher tuple_matcher_;
  RhsDataContainer rhs_;
};

namespace test_util_internal {

// Deduces the type and the matcher to build a TfArrayMatcher.
template <class TupleMatcher, class T>
TfArrayMatcher<TupleMatcher, T> TfLiteArrayIsFactory(
    const TupleMatcher& m, absl::Span<const T> reference_data) {
  return TfArrayMatcher<TupleMatcher, T>(m, reference_data);
}

// Creates an `abls::Span` pointing to the container data.
//
// The container needs to implement the interface that is accepted by
// `absl::Span`'s constructor.
//
// We need this intermediate function to create spans for the TFLite array forms
// that we may encounter. Because the arrays are C structs, we can add a
// convertion operator. This also helps deducing the data type of the array.
template <class Container, class T = std::decay_t<
                               decltype(*(std::declval<Container>().begin()))>>
absl::Span<const T> AsSpan(const Container& v) {
  return absl::Span<const T>(v);
}

// Creates an `abls::Span` pointing to the array data.
inline absl::Span<const int> AsSpan(const TfLiteIntArray* v) {
  return absl::Span<const int>(v->data, v->size);
}

// Creates an `abls::Span` pointing to the array data.
inline absl::Span<const float> AsSpan(const TfLiteFloatArray* v) {
  return absl::Span<const float>(v->data, v->size);
}

// Creates an `abls::Span` pointing to the array data.
inline absl::Span<const int> AsSpan(const TfLiteIntArray& v) {
  return absl::Span<const int>(v.data, v.size);
}

// Creates an `abls::Span` pointing to the array data.
inline absl::Span<const float> AsSpan(const TfLiteFloatArray& v) {
  return absl::Span<const float>(v.data, v.size);
}

// Creates an `abls::Span` pointing to the array data.
inline absl::Span<const int> AsSpan(const TfLiteArrayUniquePtr<int>& v) {
  return absl::Span<const int>(v->data, v->size);
}

// Creates an `abls::Span` pointing to the array data.
inline absl::Span<const float> AsSpan(const TfLiteArrayUniquePtr<float>& v) {
  return absl::Span<const float>(v->data, v->size);
}

// Creates an `abls::Span` pointing to the list data.
template <class T>
absl::Span<const T> AsSpan(std::initializer_list<T> v) {
  return v;
}

}  // namespace test_util_internal

// Matches a TFLite array value, pointer or smart pointer against the expected
// container using the testing::Eq matcher.
template <class Container>
auto TfLiteArrayIs(const Container& expected) {
  return test_util_internal::TfLiteArrayIsFactory(
      ::testing::Eq(), test_util_internal::AsSpan(expected));
}

// Matches a TFLite array value, pointer or smart pointer against the expected
// container using the provided matcher.
template <class TupleMatcher, class Container>
auto TfLiteArrayIs(const TupleMatcher& m, const Container& expected) {
  return test_util_internal::TfLiteArrayIsFactory(
      m, test_util_internal::AsSpan(expected));
}

// Matches a TFLite array value, pointer or smart pointer against the expected
// container using the testing::Eq matcher.
//
// This overload is needed to handle intializer lists.
template <class T>
auto TfLiteArrayIs(std::initializer_list<T> expected) {
  return test_util_internal::TfLiteArrayIsFactory(
      ::testing::Eq(), test_util_internal::AsSpan(expected));
}

// Matches a TFLite array value, pointer or smart pointer against the expected
// container using the testing::Eq matcher.
//
// This overload is needed to handle intializer lists.
template <class TupleMatcher, class T>
auto TfLiteArrayIs(const TupleMatcher& m, std::initializer_list<T> expected) {
  return test_util_internal::TfLiteArrayIsFactory(
      m, test_util_internal::AsSpan(expected));
}

}  // namespace tflite

// Abseil printing machinery for TFLite array GTest error printing and
// debugging.
//
// TfLiteIntArray and TfLiteFloat array are C structs and therefore defined in
// at the global namespace level. If we want GTest to find the printers, we need
// to define them there too.
//
// We explicitely speel out the different overloads to avoid catching other
// types by inadvertence.

namespace tflite {
namespace test_util_internal {

// Writes the contents of the span to the absl sink.
template <class Sink, class T>
void WriteToSink(Sink& sink, absl::Span<const T> span) {
  sink.Append("[");
  if (!span.empty()) {
    absl::Format(&sink, "%v", span[0]);
  }
  for (size_t i = 1; i < span.size(); ++i) {
    absl::Format(&sink, ", %v", span[i]);
  }
  sink.Append("]");
}

// Checks whether the given value is null if it is a pointer. Falls back on the
// other overload otherwise.
template <class Sink, class T>
bool CheckPointer(Sink& sink, T ptr, std::true_type) {
  if (ptr == nullptr) {
    sink.Append("nullptr");
    return false;
  }
  return true;
}

// Non pointer types always return true.
template <class Sink, class T>
bool CheckPointer(Sink& sink, const T& ptr, std::false_type) {
  return true;
}

// Implements the absl stringification of TFLite arrays.
//
// If the given type is an array, this will also check that it is not null.
template <class Sink, class TfLiteArrayType>
void AbslStringifyImpl(Sink& sink, const TfLiteArrayType& array) {
  if (CheckPointer(sink, array, std::is_pointer<TfLiteArrayType>())) {
    WriteToSink(sink, AsSpan(array));
  }
}

}  // namespace test_util_internal
}  // namespace tflite

// Implements the printing of custom values in GTest for TfLiteIntArray.
template <class Sink>
void AbslStringify(Sink& sink, const TfLiteIntArray& arr) {
  tflite::test_util_internal::AbslStringifyImpl(sink, arr);
}

// Implements the printing of custom values in GTest for TfLiteIntArray
// pointers.
template <class Sink>
void AbslStringify(Sink& sink, const TfLiteIntArray* const arr) {
  tflite::test_util_internal::AbslStringifyImpl(sink, arr);
}

// Implements the printing of custom values in GTest for TfLiteIntArray
// smart pointers.
template <class Sink>
void AbslStringify(Sink& sink, const tflite::TfLiteArrayUniquePtr<int>& arr) {
  AbslStringify(sink, arr.get());
}

// Implements the printing of custom values in GTest for TfLiteFloatArray.
template <class Sink>
void AbslStringify(Sink& sink, const TfLiteFloatArray& arr) {
  tflite::test_util_internal::AbslStringifyImpl(sink, arr);
}

// Implements the printing of custom values in GTest for TfLiteFloatArray
// pointers.
template <class Sink>
void AbslStringify(Sink& sink, const TfLiteFloatArray* const arr) {
  tflite::test_util_internal::AbslStringifyImpl(sink, arr);
}

// Implements the printing of custom values in GTest for TfLiteFloatArray
// smart pointers.
template <class Sink>
void AbslStringify(Sink& sink, const tflite::TfLiteArrayUniquePtr<float>& arr) {
  AbslStringify(sink, arr.get());
}

#endif  // TENSORFLOW_LITE_TEST_UTIL_H_
