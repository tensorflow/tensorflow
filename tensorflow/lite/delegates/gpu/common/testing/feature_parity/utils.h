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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TESTING_FEATURE_PARITY_UTILS_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TESTING_FEATURE_PARITY_UTILS_H_

#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include "absl/status/status.h"
#include "tensorflow/lite/model.h"

namespace tflite {

// These two functions implement usability printing for TfLiteTensor dimensions
// and coordinates. By default dimensions are interpreted depending on the size:
// 1:Linear, 2:HW, 3: HWC, 4:BHWC. If there are more than 4 dimensions,
// absl::nullopt will be returned.
absl::optional<std::string> ShapeToString(TfLiteIntArray* shape);
absl::optional<std::string> CoordinateToString(TfLiteIntArray* shape,
                                               int linear);

template <typename TupleMatcher>
class TensorEqMatcher {
 public:
  TensorEqMatcher(const TupleMatcher& tuple_matcher, const TfLiteTensor& rhs)
      : tuple_matcher_(tuple_matcher), rhs_(rhs) {}

  // Make TensorEqMatcher movable only (The copy operations are implicitly
  // deleted).
  TensorEqMatcher(TensorEqMatcher&& other) = default;
  TensorEqMatcher& operator=(TensorEqMatcher&& other) = default;

  template <typename T>
  operator testing::Matcher<T>() const {  // NOLINT
    return testing::Matcher<T>(new Impl(tuple_matcher_, rhs_));
  }

  class Impl : public testing::MatcherInterface<TfLiteTensor> {
   public:
    typedef ::std::tuple<float, float> InnerMatcherArg;

    Impl(const TupleMatcher& tuple_matcher, const TfLiteTensor& rhs)
        : mono_tuple_matcher_(
              testing::SafeMatcherCast<InnerMatcherArg>(tuple_matcher)),
          rhs_(rhs) {}

    // Make Impl movable only (The copy operations are implicitly deleted).
    Impl(Impl&& other) = default;
    Impl& operator=(Impl&& other) = default;

    // Define what gtest framework will print for the Expected field.
    void DescribeTo(std::ostream* os) const override {
      std::string shape;
      absl::optional<std::string> result = ShapeToString(rhs_.dims);
      if (result.has_value()) {
        shape = std::move(result.value());
      } else {
        shape = "[error: unsupported number of dimensions]";
      }
      *os << "tensor which has the shape of " << shape
          << ", where each value and its corresponding expected value ";
      mono_tuple_matcher_.DescribeTo(os);
    }

    bool MatchAndExplain(
        TfLiteTensor lhs,
        testing::MatchResultListener* listener) const override {
      // 1. Check that TfLiteTensor data type is supported.
      // Support for other data types will be added on demand.
      if (lhs.type != kTfLiteFloat32 || rhs_.type != kTfLiteFloat32) {
        *listener << "which data type is not float32, which is not currently "
                     "supported.";
        return false;
      }

      // 2. Check that dimensions' sizes match. Otherwise, we are not able to
      // compare tensors.
      if (lhs.dims->size != rhs_.dims->size) {
        *listener << "which is different from the expected shape of size "
                  << rhs_.dims->size;
        return false;
      }
      // 3. Check that dimensions' values are equal as well. We are not able to
      // compare tensors of different shapes, even if the total elements count
      // matches.
      bool dims_are_equal = true;
      for (int i = 0; i < lhs.dims->size; i++) {
        dims_are_equal &= lhs.dims->data[i] == rhs_.dims->data[i];
      }
      if (!dims_are_equal) {
        std::string shape;
        absl::optional<std::string> result = ShapeToString(rhs_.dims);
        if (result.has_value()) {
          shape = std::move(result.value());
        } else {
          shape = "[error: unsupported number of dimensions]";
        }
        *listener << "which is different from the expected shape " << shape;
        return false;
      }

      // 4. Proceed to data comparison. Iterate throught elements as they lay
      // flat. If some pair of elements don't match, deduct the coordinate
      // basing on the dimensions, then return.
      absl::Span<float> lhs_span(lhs.data.f, lhs.bytes / sizeof(float));
      absl::Span<float> rhs_span(rhs_.data.f, rhs_.bytes / sizeof(float));

      auto left = lhs_span.begin();
      auto right = rhs_span.begin();
      for (size_t i = 0; i != lhs_span.size(); ++i, ++left, ++right) {
        if (listener->IsInterested()) {
          testing::StringMatchResultListener inner_listener;
          if (!mono_tuple_matcher_.MatchAndExplain({*left, *right},
                                                   &inner_listener)) {
            *listener << "where the value pair (";
            testing::internal::UniversalPrint(*left, listener->stream());
            *listener << ", ";
            testing::internal::UniversalPrint(*right, listener->stream());
            std::string coordinate;
            absl::optional<std::string> result =
                CoordinateToString(lhs.dims, i);
            if (result.has_value()) {
              coordinate = std::move(result.value());
            } else {
              coordinate = "[error: unsupported number of dimensions]";
            }
            *listener << ") with coordinate " << coordinate << " don't match";
            testing::internal::PrintIfNotEmpty(inner_listener.str(),
                                               listener->stream());
            return false;
          }
        } else {
          if (!mono_tuple_matcher_.Matches({*left, *right})) return false;
        }
      }

      return true;
    }

   private:
    const testing::Matcher<InnerMatcherArg> mono_tuple_matcher_;
    const TfLiteTensor rhs_;
  };

 private:
  const TupleMatcher tuple_matcher_;
  const TfLiteTensor rhs_;
};

// Builds intepreter for a model, allocates tensors.
absl::Status BuildInterpreter(const Model* model,
                              std::unique_ptr<Interpreter>* interpreter);

// Allocates tensors for a given interpreter.
absl::Status AllocateTensors(std::unique_ptr<Interpreter>* interpreter);

// Modifies graph with given delegate.
absl::Status ModifyGraphWithDelegate(std::unique_ptr<Interpreter>* interpreter,
                                     TfLiteDelegate* delegate);

// Initializes inputs with consequent values of some fixed range.
void InitializeInputs(int left, int right,
                      std::unique_ptr<Interpreter>* interpreter);

// Invokes a prebuilt interpreter.
absl::Status Invoke(std::unique_ptr<Interpreter>* interpreter);

// Usability structure, which is used to pass parameters data to parametrized
// tests.
struct TestParams {
  // A gtest name, which will be used for a generated tests.
  std::string name;

  // Function, which returns a TFLite model, associated with this test name.
  std::vector<uint8_t> model;
};

// Defines how the TestParams should be printed into the command line if
// something fails during testing.
std::ostream& operator<<(std::ostream& os, const TestParams& param);

}  // namespace tflite

// Gtest framework uses this function to describe TfLiteTensor if something
// fails. TfLiteTensor is defined in global namespace, same should be done for
// streaming operator.
std::ostream& operator<<(std::ostream& os, const TfLiteTensor& tensor);

// Defines a matcher to compare two TfLiteTensors pointwise using the given
// tuple matcher for comparing their values.
template <typename TupleMatcherT>
inline tflite::TensorEqMatcher<TupleMatcherT> TensorEq(
    const TupleMatcherT& matcher, const TfLiteTensor& rhs) {
  return tflite::TensorEqMatcher<TupleMatcherT>(matcher, rhs);
}

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TESTING_FEATURE_PARITY_UTILS_H_
