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
#ifndef TENSORFLOW_CORE_PLATFORM_STATUS_MATCHERS_H_
#define TENSORFLOW_CORE_PLATFORM_STATUS_MATCHERS_H_

#include <ostream>
#include <string>
#include <utility>

#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

// Defines the following utilities:
//
// =================
// TfIsOkAndHolds(m)
// =================
//
// This matcher matches a StatusOr<T> value whose status is OK and whose inner
// value matches matcher m. Example:
//
//   using ::tensorflow::testing::TfIsOkAndHolds;
//   using ::testing::HasSubstr;
//   ...
//   StatusOr<std::string> status_or_message = std::string("Hello, world");
//   EXPECT_THAT(status_or_message, TfIsOkAndHolds("Hello, world")));
//   EXPECT_THAT(status_or_message, TfIsOkAndHolds(HasSubstr("Hello,")));
//
// =================================
// TfStatusIs(status_code_matcher,
//            error_message_matcher)
// =================================
//
// This matcher matches a Status or StatusOr<T> if the following are true:
//
//   - the status's error_code() matches status_code_matcher, and
//   - the status's error_message() matches error_message_matcher.
//
// Example:
//
//   using ::tensorflow::testing::TfStatusIs;
//   using ::testing::HasSubstr;
//   using ::testing::MatchesRegex;
//   using ::testing::Ne;
//   using ::testing::_;
//   StatusOr<std::string> GetMessage(int id);
//   ...
//
//   // The status code must be CANCELLED; the error message can be anything.
//   EXPECT_THAT(GetName(42),
//               TfStatusIs(tensorflow::error::CANCELLED, _));
//
//   // The status code can be anything; the error message must match the regex.
//   EXPECT_THAT(GetName(43),
//               TfStatusIs(_, MatchesRegex("server.*time-out")));
//
//   // The status code should not be CANCELLED; the error message can be
//   // anything with "Cancelled" in it.
//   EXPECT_THAT(GetName(44),
//               TfStatusIs(Ne(tensorflow::error::CANCELLED),
//                          HasSubstr("Cancelled"))));
//
// ===============================
// TfStatusIs(status_code_matcher)
// ===============================
//
// This is a shorthand for
//   TfStatusIs(status_code_matcher, ::testing::_)
//
// In other words, it's like the two-argument TfStatusIs(), except that it
// ignores error message.
//
// ========
// TfIsOk()
// ========
//
// Matches a Status or StatusOr<T> whose status value is OK.
// Equivalent to 'TfStatusIs(error::OK)'.
//
// Example:
//   ...
//   StatusOr<std::string> message = std::string("Hello, world");
//   EXPECT_THAT(message, TfIsOk());
//   Status status = Status::OK();
//   EXPECT_THAT(status, TfIsOk());

namespace tensorflow {

template <typename T>
void PrintTo(const StatusOr<T>& status_or, std::ostream* os) {
  *os << status_or.status();
  if (status_or.ok()) {
    *os << ": " << ::testing::PrintToString(status_or.ValueOrDie());
  }
}

namespace error {
inline void PrintTo(const tensorflow::error::Code code, std::ostream* os) {
  *os << Code_Name(code);
}
}  // namespace error

namespace testing {
namespace internal_status {

inline const Status& GetStatus(const Status& status) { return status; }

template <typename T>
inline const Status& GetStatus(const StatusOr<T>& status) {
  return status.status();
}

////////////////////////////////////////////////////////////
// Implementation of TfIsOkAndHolds().
//
// Monomorphic implementation of matcher TfIsOkAndHolds(m). StatusOrType is a
// reference to StatusOr<T>.
template <typename StatusOrType>
class TfIsOkAndHoldsMatcherImpl
    : public ::testing::MatcherInterface<StatusOrType> {
 public:
  typedef
      typename std::remove_reference<StatusOrType>::type::value_type value_type;

  template <typename InnerMatcher>
  explicit TfIsOkAndHoldsMatcherImpl(InnerMatcher&& inner_matcher)
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

// Implements TfIsOkAndHolds(m) as a polymorphic matcher.
template <typename InnerMatcher>
class TfIsOkAndHoldsMatcher {
 public:
  explicit TfIsOkAndHoldsMatcher(InnerMatcher inner_matcher)
      : inner_matcher_(std::move(inner_matcher)) {}

  // Converts this polymorphic matcher to a monomorphic matcher of the given
  // type. StatusOrType can be either StatusOr<T> or a reference to StatusOr<T>.
  template <typename StatusOrType>
  operator ::testing::Matcher<StatusOrType>() const {  // NOLINT
    return ::testing::Matcher<StatusOrType>(
        new TfIsOkAndHoldsMatcherImpl<const StatusOrType&>(inner_matcher_));
  }

 private:
  const InnerMatcher inner_matcher_;
};

////////////////////////////////////////////////////////////
// Implementation of TfStatusIs().
//
// TfStatusIs() is a polymorphic matcher. This class is the common
// implementation of it shared by all types T where TfStatusIs() can be used as
// a Matcher<T>.

class TfStatusIsMatcherCommonImpl {
 public:
  TfStatusIsMatcherCommonImpl(
      ::testing::Matcher<const tensorflow::error::Code> code_matcher,
      ::testing::Matcher<const std::string&> message_matcher)
      : code_matcher_(std::move(code_matcher)),
        message_matcher_(std::move(message_matcher)) {}

  void DescribeTo(std::ostream* os) const;

  void DescribeNegationTo(std::ostream* os) const;

  bool MatchAndExplain(const Status& status,
                       ::testing::MatchResultListener* result_listener) const;

 private:
  const ::testing::Matcher<const tensorflow::error::Code> code_matcher_;
  const ::testing::Matcher<const std::string&> message_matcher_;
};

// Monomorphic implementation of matcher TfStatusIs() for a given type T. T can
// be Status, StatusOr<>, or a reference to either of them.
template <typename T>
class TfMonoStatusIsMatcherImpl : public ::testing::MatcherInterface<T> {
 public:
  explicit TfMonoStatusIsMatcherImpl(TfStatusIsMatcherCommonImpl common_impl)
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
  TfStatusIsMatcherCommonImpl common_impl_;
};

// Implements TfStatusIs() as a polymorphic matcher.
class TfStatusIsMatcher {
 public:
  TfStatusIsMatcher(
      ::testing::Matcher<const tensorflow::error::Code> code_matcher,
      ::testing::Matcher<const std::string&> message_matcher)
      : common_impl_(
            ::testing::MatcherCast<const tensorflow::error::Code>(code_matcher),
            ::testing::MatcherCast<const std::string&>(message_matcher)) {}

  // Converts this polymorphic matcher to a monomorphic matcher of the given
  // type. T can be StatusOr<>, Status, or a reference to either of them.
  template <typename T>
  operator ::testing::Matcher<T>() const {  // NOLINT
    return ::testing::MakeMatcher(
        new TfMonoStatusIsMatcherImpl<T>(common_impl_));
  }

 private:
  const TfStatusIsMatcherCommonImpl common_impl_;
};

// Monomorphic implementation of matcher TfIsOk() for a given type T.
// T can be Status, StatusOr<>, or a reference to either of them.
template <typename T>
class TfMonoIsOkMatcherImpl : public ::testing::MatcherInterface<T> {
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

// Implements TfIsOk() as a polymorphic matcher.
class TfIsOkMatcher {
 public:
  template <typename T>
  operator ::testing::Matcher<T>() const {  // NOLINT
    return ::testing::Matcher<T>(new TfMonoIsOkMatcherImpl<const T&>());
  }
};
}  // namespace internal_status

// Returns a matcher that matches a StatusOr<> whose status is OK and whose
// value matches the inner matcher.
template <typename InnerMatcher>
internal_status::TfIsOkAndHoldsMatcher<typename std::decay<InnerMatcher>::type>
TfIsOkAndHolds(InnerMatcher&& inner_matcher) {
  return internal_status::TfIsOkAndHoldsMatcher<
      typename std::decay<InnerMatcher>::type>(
      std::forward<InnerMatcher>(inner_matcher));
}

// Returns a matcher that matches a Status or StatusOr<> whose status code
// matches code_matcher, and whose error message matches message_matcher.
template <typename CodeMatcher, typename MessageMatcher>
internal_status::TfStatusIsMatcher TfStatusIs(CodeMatcher code_matcher,
                                              MessageMatcher message_matcher) {
  return internal_status::TfStatusIsMatcher(std::move(code_matcher),
                                            std::move(message_matcher));
}

// Returns a matcher that matches a Status or StatusOr<> whose status code
// matches code_matcher.
template <typename CodeMatcher>
internal_status::TfStatusIsMatcher TfStatusIs(CodeMatcher code_matcher) {
  return TfStatusIs(std::move(code_matcher), ::testing::_);
}

// Returns a matcher that matches a Status or StatusOr<> which is OK.
inline internal_status::TfIsOkMatcher TfIsOk() {
  return internal_status::TfIsOkMatcher();
}

}  // namespace testing
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_STATUS_MATCHERS_H_
