// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_TEST_MATCHERS_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_TEST_MATCHERS_H_

#include <optional>
#include <ostream>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"

// Is equivalent to `ASSERT_THAT(expr, testing::litert::IsOk())`
#define LITERT_ASSERT_OK(EXPR) ASSERT_THAT((EXPR), ::testing::litert::IsOk())

// Is equivalent to `EXPECT_THAT(expr, testing::litert::IsOk())`
#define LITERT_EXPECT_OK(EXPR) EXPECT_THAT((EXPR), ::testing::litert::IsOk())

// Checks that the result of `EXPR` (a `litert::Expected` object) is not an
// error and assigns the value it holds to `DECL` as if:
// ```
// DECL = std::move(EXPR.Value());
// ```
//
// ```cpp
// Expected<Something> BuildSomething();
//
// Will fail the test if `BuildSomething()`'s returned value holds an error.
// Otherwise defines and assigns the returned `Something` value to `smth`
// ASSERT_OK_AND_ASSIGN(Something smth, BuildSomething());
// ```
#define LITERT_ASSERT_OK_AND_ASSIGN(DECL, EXPR) \
  LITERT_ASSERT_OK_AND_ASSIGN_HELPER2(__LINE__, DECL, EXPR)

#define LITERT_ASSERT_OK_AND_ASSIGN_HELPER1(LINE, DECL, EXPR) \
  auto&& litert_expected_value_or_error_##LINE = (EXPR);      \
  LITERT_ASSERT_OK(litert_expected_value_or_error_##LINE);    \
  DECL = std::move(litert_expected_value_or_error_##LINE.Value());

#define LITERT_ASSERT_OK_AND_ASSIGN_HELPER2(LINE, DECL, EXPR) \
  LITERT_ASSERT_OK_AND_ASSIGN_HELPER1(LINE, DECL, EXPR)

namespace testing::litert {

// Matches `litert::Expected` values that hold a success value and
// `LiteRtStatusOk`.
//
// See `IsOk()` function below for usage examples.
class IsOkMatcher {
 public:
  // Implicitly builds and wraps the matcher implementation in a GTest
  // Matcher object.
  template <class T>
  // NOLINTNEXTLINE(*-explicit-constructor): This needs to be implicit.
  operator testing::Matcher<T>() const {
    return testing::Matcher<T>(new Impl<const T&>());
  }

  template <class V>
  class Impl : public testing::MatcherInterface<V> {
    template <class T>
    bool MatchAndExplainImpl(const ::litert::Expected<T>& value,
                             testing::MatchResultListener* listener) const {
      return value.HasValue();
    }

    bool MatchAndExplainImpl(const ::litert::Unexpected& unexpected,
                             testing::MatchResultListener* listener) const {
      return false;
    }

    bool MatchAndExplainImpl(const ::litert::Error& e,
                             testing::MatchResultListener* listener) const {
      return false;
    }

    bool MatchAndExplainImpl(const LiteRtStatus& status,
                             testing::MatchResultListener* listener) const {
      if (status != kLiteRtStatusOk) {
        *listener << "status is " << LiteRtGetStatusString(status);
        return false;
      }
      return true;
    }

   public:
    using is_gtest_matcher = void;

    bool MatchAndExplain(
        V value, testing::MatchResultListener* listener) const override {
      return MatchAndExplainImpl(value, listener);
    }

    void DescribeTo(std::ostream* os) const override {
      if (os) {
        *os << "is ok.";
      }
    }

    void DescribeNegationTo(std::ostream* os) const override {
      if (os) {
        *os << "is not ok.";
      }
    }
  };
};

// Matches `litert::Expected` values that hold a success value and
// `LiteRtStatusOk`.
//
// Note: you might want to use the convenience macros:
//   - `LITERT_EXPECT_OK(expr)`
//   - `LITERT_ASSERT_OK(expr)`
//   - `ASSERT_OK_AND_ASSIGN(type var, expr)`
//
// ```cpp
// LiteRtStatus DoSomething();
//
// // Will fail the test if DoSomething() doesn't return kLiteRtStatusOk.
// EXPECT_THAT(DoSomething(), IsOk());
// ```
//
// This also works for `Expected` objects.
//
// Note: You probably want `ASSERT_OK_AND_ASSIGN` when working with `Expected`.
//
// ```cpp
// Expected<Something> BuildSomething();
//
// // Will fail the test if BuildSomething()'s returned value holds an error.
// // Note that the returned value is unused.
// EXPECT_THAT(BuildSomething(), IsOk());
// ```
inline IsOkMatcher IsOk() { return IsOkMatcher(); }

// Matches `litert::Expected` values that hold an error and
// `LiteRtStatusError*` values.
//
// See `IsError(...)` functions below for usage examples.
class IsErrorMatcher {
 public:
  IsErrorMatcher(std::optional<LiteRtStatus> status,
                 std::optional<std::string> msg)
      : impl_(status, msg) {}

  // Implicitly builds and wraps the matcher implementation in a GTest
  // Matcher object.
  template <class T>
  // NOLINTNEXTLINE(*-explicit-constructor): This needs to be implicit.
  operator testing::Matcher<T>() const {
    return testing::Matcher<T>(new Impl<const T&>(impl_));
  }

 private:
  class ImplBase {
   public:
    ImplBase() = default;

    explicit ImplBase(std::optional<LiteRtStatus> status,
                      std::optional<std::string> msg)
        : status_(status), msg_(std::move(msg)) {};

   protected:
    bool MatchAndExplainImpl(const LiteRtStatus status,
                             const absl::string_view msg,
                             testing::MatchResultListener* listener) const {
      if (status == kLiteRtStatusOk ||
          (status_.has_value() && status != status_.value())) {
        if (listener) {
          *listener << "status doesn't match";
        }
        return false;
      }
      if (msg_.has_value() && msg != msg_.value()) {
        if (listener) {
          *listener << "message doesn't match";
        }
        return false;
      }
      return true;
    }

    template <class T>
    bool MatchAndExplainImpl(const ::litert::Expected<T>& value,
                             testing::MatchResultListener* listener) const {
      if (value.HasValue()) {
        *listener << "expected holds a value (but should hold an error)";
        return false;
      }
      return MatchAndExplainImpl(value.Error(), listener);
    }

    bool MatchAndExplainImpl(const ::litert::Unexpected& e,
                             testing::MatchResultListener* listener) const {
      return MatchAndExplainImpl(e.Error().Status(), e.Error().Message(),
                                 listener);
    }

    bool MatchAndExplainImpl(const ::litert::Error& e,
                             testing::MatchResultListener* listener) const {
      return MatchAndExplainImpl(e.Status(), e.Message(), listener);
    }

    bool MatchAndExplainImpl(const LiteRtStatus& status,
                             testing::MatchResultListener* listener) const {
      return MatchAndExplainImpl(status, {}, listener);
    }

    void DescribeImpl(std::ostream* os, const bool negation) const {
      if (os) {
        *os << "is" << (negation ? " not" : "") << " an error";
        const char* sep = " with ";
        if (status_.has_value()) {
          *os << sep << "status " << LiteRtGetStatusString(status_.value());
          sep = " and ";
        }
        if (msg_.has_value()) {
          *os << sep << "message matching: '" << msg_.value() << "'";
        }
        *os << ".";
      }
    }

   private:
    std::optional<LiteRtStatus> status_;
    std::optional<std::string> msg_;
  };

  template <class V>
  class Impl : public testing::MatcherInterface<V>, ImplBase {
   public:
    using is_gtest_matcher = void;

    Impl() = default;
    explicit Impl(const ImplBase& base) : ImplBase(base) {}

    bool MatchAndExplain(
        V value, testing::MatchResultListener* listener) const override {
      return MatchAndExplainImpl(value, listener);
    }

    void DescribeTo(std::ostream* os) const override {
      DescribeImpl(os, /*negation=*/false);
    }

    void DescribeNegationTo(std::ostream* os) const override {
      DescribeImpl(os, /*negation=*/true);
    }
  };

  ImplBase impl_;
};

// Matches `litert::Expected`, `litert::Unexpected`, `litert::Error` and
// `LiteRtStatus` values that hold an error.
//
// Note: This will always match `true` for `litert::Unexpected` and
// `litert::Error`. This can be useful to test template code that might always
// return an error for certain specialisations.
//
// ```cpp
// LiteRtStatus DoSomething();
//
// // Will fail the test if `DoSomething()` returns `kLiteRtStatusOk`.
// EXPECT_THAT(DoSomething(), IsError());
// ```
//
// This also works for `Expected` objects.
//
// ```cpp
// Expected<Something> BuildSomething();
//
// // Will fail the test if BuildSomething()'s returned object holds a value.
// EXPECT_THAT(BuildSomething(), IsError());
// ```
inline IsErrorMatcher IsError() {
  return IsErrorMatcher(/*status=*/std::nullopt, /*msg=*/std::nullopt);
}

// Matches `litert::Expected`, `litert::Unexpected`, `litert::Error` and
// `LiteRtStatus` values that hold a specific error status.
//
// ```cpp
// Expected<Something> BuildSomething();
//
// // Will fail the test if BuildSomething()'s returned object holds a value or
// // if the error status is not `kLiteRtStatusErrorSystemError`.
// EXPECT_THAT(BuildSomething(), IsError(kLiteRtStatusErrorSystemError));
// ```
inline IsErrorMatcher IsError(LiteRtStatus status) {
  return IsErrorMatcher(status, /*msg=*/std::nullopt);
}

// Matches `litert::Expected` and `LiteRtStatus` values that have a specific
// error status and error message.
//
// Warning: This will always return `false` for `LiteRtStatus` objects as those
// do not convey a message.
//
// ```cpp
// Expected<Something> BuildSomething();
//
// // Will fail the test if BuildSomething()'s returned object holds a value.
// EXPECT_THAT(BuildSomething(), IsError(kLiteRtStatusErrorSystemError,
//                                       "System is not initialised"));
// ```
inline IsErrorMatcher IsError(LiteRtStatus status, std::string msg) {
  return IsErrorMatcher(status, std::move(msg));
}

}  // namespace testing::litert

// GTest doesn't use `AbslStringify` if `GTEST_USE_ABSL` is not defined. This
// provides a fallback implementation.
//
// This is defined here instead of with `litert::Expected` because those
// functions should only be used for testing.
#if defined(LITERT_DEFINE_GTEST_STATUS_PRINTER) && !defined(GTEST_USE_ABSL)
#include "absl/strings/str_format.h"

// GTest documentation explicitly states that functions the those below must
// live in the same namespace as the classes they are used with so that GTest
// can find them through ADL.
namespace litert {

inline void PrintTo(const Error& error, std::ostream* os) {
  *os << absl::StrFormat("%v", error);
}

inline void PrintTo(const Unexpected& unexpected, std::ostream* os) {
  *os << absl::StrFormat("%v", unexpected);
}

template <class T>
void PrintTo(const Expected<T>& expected, std::ostream* os) {
  *os << absl::StrFormat("%v", expected);
}

}  // namespace litert

#endif

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_TEST_MATCHERS_H_
