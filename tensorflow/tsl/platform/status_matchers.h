/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_TSL_PLATFORM_STATUS_MATCHERS_H_
#define TENSORFLOW_TSL_PLATFORM_STATUS_MATCHERS_H_

#include <ostream>
#include <string>
#include <utility>

#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/statusor.h"
#include "tensorflow/tsl/platform/test.h"
#include "tensorflow/tsl/protobuf/error_codes.pb.h"

// Defines the following utilities:
//
// ===============
// IsOkAndHolds(m)
// ===============
//
// This matcher matches a StatusOr<T> value whose status is OK and whose inner
// value matches matcher m. Example:
//
//   using ::tsl::testing::IsOkAndHolds;
//   using ::testing::HasSubstr;
//   ...
//   StatusOr<std::string> status_or_message("Hello, world");
//   EXPECT_THAT(status_or_message, IsOkAndHolds("Hello, world")));
//   EXPECT_THAT(status_or_message, IsOkAndHolds(HasSubstr("Hello,")));
//
// ===============================
// StatusIs(status_code_matcher,
//          error_message_matcher)
// ===============================
//
// This matcher matches a Status or StatusOr<T> if the following are true:
//
//   - the status's code() matches status_code_matcher, and
//   - the status's error_message() matches error_message_matcher.
//
// Example:
//
//   using ::tsl::testing::StatusIs;
//   using ::testing::HasSubstr;
//   using ::testing::MatchesRegex;
//   using ::testing::Ne;
//   using ::testing::_;
//   StatusOr<std::string> GetMessage(int id);
//   ...
//
//   // The status code must be CANCELLED; the error message can be anything.
//   EXPECT_THAT(GetName(42),
//               StatusIs(tsl::error::CANCELLED, _));
//
//   // The status code can be anything; the error message must match the regex.
//   EXPECT_THAT(GetName(43),
//               StatusIs(_, MatchesRegex("server.*time-out")));
//
//   // The status code should not be CANCELLED; the error message can be
//   // anything with "Cancelled" in it.
//   EXPECT_THAT(GetName(44),
//               StatusIs(Ne(tsl::error::CANCELLED),
//                        HasSubstr("Cancelled"))));
//
// =============================
// StatusIs(status_code_matcher)
// =============================
//
// This is a shorthand for
//   StatusIs(status_code_matcher, ::testing::_)
//
// In other words, it's like the two-argument StatusIs(), except that it ignores
// error messages.
//
// ======
// IsOk()
// ======
//
// Matches a Status or StatusOr<T> whose status value is OK.
// Equivalent to 'StatusIs(error::OK)'.
//
// Example:
//   ...
//   StatusOr<std::string> message("Hello, world");
//   EXPECT_THAT(message, IsOk());
//   Status status = OkStatus();
//   EXPECT_THAT(status, IsOk());

namespace tensorflow {
namespace error {
// TODO(ddunleavy) Move this to TSL. This stays here until error_codes proto
// is moved to TSL due to an ADL issue
inline void PrintTo(const tsl::error::Code code, std::ostream* os) {
  *os << Code_Name(code);
}

}  // namespace error
}  // namespace tensorflow

namespace tsl {

template <typename T>
void PrintTo(const StatusOr<T>& status_or, std::ostream* os) {
  *os << ::testing::PrintToString(status_or.status());
  if (status_or.ok()) {
    *os << ": " << ::testing::PrintToString(status_or.value());
  }
}

namespace testing {
namespace internal_status {

inline const Status& GetStatus(const Status& status) { return status; }

template <typename T>
inline const Status& GetStatus(const StatusOr<T>& status) {
  return status.status();
}

////////////////////////////////////////////////////////////
// Implementation of IsOkAndHolds().
//
// Monomorphic implementation of matcher IsOkAndHolds(m). StatusOrType is a
// reference to StatusOr<T>.
template <typename StatusOrType>
class IsOkAndHoldsMatcherImpl
    : public ::testing::MatcherInterface<StatusOrType> {
 public:
  typedef
      typename std::remove_reference<StatusOrType>::type::value_type value_type;

  template <typename InnerMatcher>
  explicit IsOkAndHoldsMatcherImpl(InnerMatcher&& inner_matcher)
      : inner_matcher_(::testing::SafeMatcherCast<const value_type&>(
            std::forward<InnerMatcher>(inner_matcher))) {}

  void DescribeTo(std::ostream* os) const override {
    *os << "is OK and has a value that ";
    inner_matcher_.DescribeTo(os);
  }

  void DescribeNegationTo(std::ostream* os) const override {
    *os << "isn't OK or has a value that ";
    inner_matcher_.DescribeNegationTo(os);
  }

  bool MatchAndExplain(
      StatusOrType actual_value,
      ::testing::MatchResultListener* result_listener) const override {
    if (!actual_value.ok()) {
      *result_listener << "which has status " << actual_value.status();
      return false;
    }

    ::testing::StringMatchResultListener inner_listener;
    const bool matches =
        inner_matcher_.MatchAndExplain(*actual_value, &inner_listener);
    const std::string inner_explanation = inner_listener.str();
    if (!inner_explanation.empty()) {
      *result_listener << "which contains value "
                       << ::testing::PrintToString(*actual_value) << ", "
                       << inner_explanation;
    }
    return matches;
  }

 private:
  const ::testing::Matcher<const value_type&> inner_matcher_;
};

// Implements IsOkAndHolds(m) as a polymorphic matcher.
template <typename InnerMatcher>
class IsOkAndHoldsMatcher {
 public:
  explicit IsOkAndHoldsMatcher(InnerMatcher inner_matcher)
      : inner_matcher_(std::move(inner_matcher)) {}

  // Converts this polymorphic matcher to a monomorphic matcher of the given
  // type. StatusOrType can be either StatusOr<T> or a reference to StatusOr<T>.
  template <typename StatusOrType>
  operator ::testing::Matcher<StatusOrType>() const {  // NOLINT
    return ::testing::Matcher<StatusOrType>(
        new IsOkAndHoldsMatcherImpl<const StatusOrType&>(inner_matcher_));
  }

 private:
  const InnerMatcher inner_matcher_;
};

////////////////////////////////////////////////////////////
// Implementation of StatusIs().
//
// StatusIs() is a polymorphic matcher. This class is the common
// implementation of it shared by all types T where StatusIs() can be used as
// a Matcher<T>.

class StatusIsMatcherCommonImpl {
 public:
  StatusIsMatcherCommonImpl(
      ::testing::Matcher<const tsl::error::Code> code_matcher,
      ::testing::Matcher<const std::string&> message_matcher)
      : code_matcher_(std::move(code_matcher)),
        message_matcher_(std::move(message_matcher)) {}

  void DescribeTo(std::ostream* os) const;

  void DescribeNegationTo(std::ostream* os) const;

  bool MatchAndExplain(const Status& status,
                       ::testing::MatchResultListener* result_listener) const;

 private:
  const ::testing::Matcher<const tsl::error::Code> code_matcher_;
  const ::testing::Matcher<const std::string&> message_matcher_;
};

// Monomorphic implementation of matcher StatusIs() for a given type T. T can
// be Status, StatusOr<>, or a reference to either of them.
template <typename T>
class MonoStatusIsMatcherImpl : public ::testing::MatcherInterface<T> {
 public:
  explicit MonoStatusIsMatcherImpl(StatusIsMatcherCommonImpl common_impl)
      : common_impl_(std::move(common_impl)) {}

  void DescribeTo(std::ostream* os) const override {
    common_impl_.DescribeTo(os);
  }

  void DescribeNegationTo(std::ostream* os) const override {
    common_impl_.DescribeNegationTo(os);
  }

  bool MatchAndExplain(
      T actual_value,
      ::testing::MatchResultListener* result_listener) const override {
    return common_impl_.MatchAndExplain(GetStatus(actual_value),
                                        result_listener);
  }

 private:
  StatusIsMatcherCommonImpl common_impl_;
};

// Implements StatusIs() as a polymorphic matcher.
class StatusIsMatcher {
 public:
  StatusIsMatcher(::testing::Matcher<const tsl::error::Code> code_matcher,
                  ::testing::Matcher<const std::string&> message_matcher)
      : common_impl_(
            ::testing::MatcherCast<const tsl::error::Code>(code_matcher),
            ::testing::MatcherCast<const std::string&>(message_matcher)) {}

  // Converts this polymorphic matcher to a monomorphic matcher of the given
  // type. T can be StatusOr<>, Status, or a reference to either of them.
  template <typename T>
  operator ::testing::Matcher<T>() const {  // NOLINT
    return ::testing::MakeMatcher(new MonoStatusIsMatcherImpl<T>(common_impl_));
  }

 private:
  const StatusIsMatcherCommonImpl common_impl_;
};

// Monomorphic implementation of matcher IsOk() for a given type T.
// T can be Status, StatusOr<>, or a reference to either of them.
template <typename T>
class MonoIsOkMatcherImpl : public ::testing::MatcherInterface<T> {
 public:
  void DescribeTo(std::ostream* os) const override { *os << "is OK"; }
  void DescribeNegationTo(std::ostream* os) const override {
    *os << "is not OK";
  }
  bool MatchAndExplain(T actual_value,
                       ::testing::MatchResultListener*) const override {
    return GetStatus(actual_value).ok();
  }
};

// Implements IsOk() as a polymorphic matcher.
class IsOkMatcher {
 public:
  template <typename T>
  operator ::testing::Matcher<T>() const {  // NOLINT
    return ::testing::Matcher<T>(new MonoIsOkMatcherImpl<const T&>());
  }
};
}  // namespace internal_status

// Returns a matcher that matches a StatusOr<> whose status is OK and whose
// value matches the inner matcher.
template <typename InnerMatcher>
internal_status::IsOkAndHoldsMatcher<typename std::decay<InnerMatcher>::type>
IsOkAndHolds(InnerMatcher&& inner_matcher) {
  return internal_status::IsOkAndHoldsMatcher<
      typename std::decay<InnerMatcher>::type>(
      std::forward<InnerMatcher>(inner_matcher));
}

// Returns a matcher that matches a Status or StatusOr<> whose status code
// matches code_matcher, and whose error message matches message_matcher.
template <typename CodeMatcher, typename MessageMatcher>
internal_status::StatusIsMatcher StatusIs(CodeMatcher code_matcher,
                                          MessageMatcher message_matcher) {
  return internal_status::StatusIsMatcher(std::move(code_matcher),
                                          std::move(message_matcher));
}

// Returns a matcher that matches a Status or StatusOr<> whose status code
// matches code_matcher.
template <typename CodeMatcher>
internal_status::StatusIsMatcher StatusIs(CodeMatcher code_matcher) {
  return StatusIs(std::move(code_matcher), ::testing::_);
}

// Returns a matcher that matches a Status or StatusOr<> which is OK.
inline internal_status::IsOkMatcher IsOk() {
  return internal_status::IsOkMatcher();
}

}  // namespace testing
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PLATFORM_STATUS_MATCHERS_H_
