/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_CODEGEN_MATH_TEST_MATCHERS_H_
#define XLA_CODEGEN_MATH_TEST_MATCHERS_H_

#include <cstdint>
#include <ostream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/fp_util.h"

namespace xla::codegen::math {

// A matcher that checks if a floating point value is within a given number of
// ULPs (Units in the Last Place) of a given expected value.
// We typically want all of our math functions to be within 1 ULP of truth.
template <typename T>
class FloatingPointNearUlpsMatcherWithExpected
    : public testing::MatcherInterface<T> {
 public:
  FloatingPointNearUlpsMatcherWithExpected(T expected_value, int max_ulps)
      : expected_value_(expected_value), max_ulps_(max_ulps) {}

  void DescribeTo(std::ostream* os) const override {
    *os << "is within " << max_ulps_ << " ULPs of "
        << testing::PrintToString(expected_value_);
  }

  void DescribeNegationTo(std::ostream* os) const override {
    *os << "is not within " << max_ulps_ << " ULPs of "
        << testing::PrintToString(expected_value_);
  }

  bool MatchAndExplain(T actual_value,
                       testing::MatchResultListener* listener) const override {
    // Handle NaN/Inf separately for clearer error messages if they don't match
    // expectations
    if (std::isnan(expected_value_) || std::isnan(actual_value)) {
      if (std::isnan(expected_value_) && std::isnan(actual_value)) {
        return true;  // Both NaN, considered a match
      }
      if (listener->IsInterested()) {
        *listener << "Expected " << testing::PrintToString(expected_value_)
                  << " and actual " << testing::PrintToString(actual_value)
                  << ". One is NaN, the other is not.";
      }
      return false;
    }

    if (std::isinf(expected_value_) || std::isinf(actual_value)) {
      if (expected_value_ == actual_value) {  // Both Inf of same sign
        return true;
      }
      if (listener->IsInterested()) {
        *listener << "Expected " << testing::PrintToString(expected_value_)
                  << " and actual " << testing::PrintToString(actual_value)
                  << ". One is Inf, the other is not, or signs differ.";
      }
      return false;
    }

    // Use your CalculateDistanceInFloats function
    int64_t ulps_diff =
        CalculateDistanceInFloats(expected_value_, actual_value);

    if (listener->IsInterested()) {
      *listener << "Distance in ULPs: " << ulps_diff;
    }

    return ulps_diff <= max_ulps_;
  }

 private:
  T expected_value_;
  int max_ulps_;
};

template <typename T>
testing::Matcher<T> NearUlps(T expected_value, int max_ulps) {
  return testing::MakeMatcher(new FloatingPointNearUlpsMatcherWithExpected<T>(
      expected_value, max_ulps));
}
}  // namespace xla::codegen::math

#endif  // XLA_CODEGEN_MATH_TEST_MATCHERS_H_
