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
#include "tsl/platform/status_matchers.h"

#include <ostream>
#include <string>

#include "tsl/platform/status.h"
#include "tsl/platform/test.h"
#include "tsl/protobuf/error_codes.pb.h"

namespace tsl {
namespace testing {
namespace internal_status {

void StatusIsMatcherCommonImpl::DescribeTo(std::ostream* os) const {
  *os << "has a status code that ";
  code_matcher_.DescribeTo(os);
  *os << ", and has an error message that ";
  message_matcher_.DescribeTo(os);
}

void StatusIsMatcherCommonImpl::DescribeNegationTo(std::ostream* os) const {
  *os << "has a status code that ";
  code_matcher_.DescribeNegationTo(os);
  *os << ", or has an error message that ";
  message_matcher_.DescribeNegationTo(os);
}

bool StatusIsMatcherCommonImpl::MatchAndExplain(
    const absl::Status& status,
    ::testing::MatchResultListener* result_listener) const {
  ::testing::StringMatchResultListener inner_listener;

  inner_listener.Clear();
  if (!code_matcher_.MatchAndExplain(
          static_cast<absl::StatusCode>(status.code()), &inner_listener)) {
    *result_listener << (inner_listener.str().empty()
                             ? "whose status code is wrong"
                             : "which has a status code " +
                                   inner_listener.str());
    return false;
  }

  if (!message_matcher_.Matches(std::string(status.message()))) {
    *result_listener << "whose error message is wrong";
    return false;
  }

  return true;
}

}  // namespace internal_status
}  // namespace testing
}  // namespace tsl
