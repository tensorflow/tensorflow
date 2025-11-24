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
#ifndef XLA_TSL_PLATFORM_STATUS_MATCHERS_H_
#define XLA_TSL_PLATFORM_STATUS_MATCHERS_H_

#include "absl/base/macros.h"
#include "absl/status/status_matchers.h"

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

namespace tsl {
namespace testing {

// Returns a matcher that matches a StatusOr<> whose status is OK and whose
// value matches the inner matcher.
template <typename InnerMatcherT>
ABSL_DEPRECATE_AND_INLINE()
auto IsOkAndHolds(InnerMatcherT&& inner_matcher) {
  return absl_testing::IsOkAndHolds(inner_matcher);
}

// Returns a matcher that matches a Status or StatusOr<> whose status code
// matches code_matcher, and whose error message matches message_matcher.
template <typename StatusCodeMatcherT, typename StatusMessageMatcherT>
ABSL_DEPRECATE_AND_INLINE()
auto StatusIs(StatusCodeMatcherT&& code_matcher,
              StatusMessageMatcherT&& message_matcher) {
  return absl_testing::StatusIs(code_matcher, message_matcher);
}

// Returns a matcher that matches a Status or StatusOr<> whose status code
// matches code_matcher.
template <typename StatusCodeMatcherT>
ABSL_DEPRECATE_AND_INLINE()
auto StatusIs(StatusCodeMatcherT&& code_matcher) {
  return absl_testing::StatusIs(code_matcher);
}

// Returns a matcher that matches a Status or StatusOr<> which is OK.
ABSL_DEPRECATE_AND_INLINE()
inline auto IsOk() { return absl_testing::IsOk(); }

}  // namespace testing
}  // namespace tsl

#endif  // XLA_TSL_PLATFORM_STATUS_MATCHERS_H_
