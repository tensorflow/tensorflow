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
#include "tensorflow/core/platform/status_matchers.h"

#include <string>
#include <vector>

#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace testing {
namespace {

using ::testing::_;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::MatchesRegex;
using ::testing::Ne;
using ::testing::Not;

TEST(StatusMatchersTest, TfIsOkAndHolds) {
  StatusOr<std::string> status_or_message = std::string("Hello, world");
  EXPECT_THAT(status_or_message, TfIsOkAndHolds("Hello, world"));
  EXPECT_THAT(status_or_message, TfIsOkAndHolds(HasSubstr("Hello,")));
}

TEST(StatusMatchersTest, TfIsOkAndHoldsContainer) {
  StatusOr<std::vector<std::string>> status_or_messages =
      std::vector<std::string>{"Hello, world", "Hello, tf"};
  EXPECT_THAT(status_or_messages,
              TfIsOkAndHolds(ElementsAre("Hello, world", "Hello, tf")));
  EXPECT_THAT(
      status_or_messages,
      TfIsOkAndHolds(ElementsAre(HasSubstr("Hello,"), HasSubstr("Hello,"))));
}

TEST(StatusMatchersTest, TfStatusIsOk) {
  EXPECT_THAT(Status::OK(), TfStatusIs(error::OK));
  EXPECT_THAT(errors::DeadlineExceeded("Deadline exceeded"),
              Not(TfStatusIs(error::OK)));

  StatusOr<std::string> message = std::string("Hello, world");
  EXPECT_THAT(message, TfStatusIs(error::OK));
  StatusOr<std::string> status = errors::NotFound("Not found");
  EXPECT_THAT(status, Not(TfStatusIs(error::OK)));
}

TEST(StatusMatchersTest, TfStatusIsCancelled) {
  Status s = errors::Cancelled("Cancelled");
  EXPECT_THAT(s, TfStatusIs(error::CANCELLED));
  EXPECT_THAT(s, TfStatusIs(error::CANCELLED, "Cancelled"));
  EXPECT_THAT(s, TfStatusIs(_, "Cancelled"));
  EXPECT_THAT(s, TfStatusIs(error::CANCELLED, _));
  EXPECT_THAT(s, TfStatusIs(Ne(error::INVALID_ARGUMENT), _));
  EXPECT_THAT(s, TfStatusIs(error::CANCELLED, HasSubstr("Can")));
  EXPECT_THAT(s, TfStatusIs(error::CANCELLED, MatchesRegex("Can.*")));
}

TEST(StatusMatchersTest, TfStatusOrStatusIs) {
  StatusOr<int> s = errors::InvalidArgument("Invalid Argument");
  EXPECT_THAT(s, TfStatusIs(error::INVALID_ARGUMENT));
  EXPECT_THAT(s, TfStatusIs(error::INVALID_ARGUMENT, "Invalid Argument"));
  EXPECT_THAT(s, TfStatusIs(_, "Invalid Argument"));
  EXPECT_THAT(s, TfStatusIs(error::INVALID_ARGUMENT, _));
  EXPECT_THAT(s, TfStatusIs(Ne(error::CANCELLED), _));
  EXPECT_THAT(s, TfStatusIs(error::INVALID_ARGUMENT, HasSubstr("Argument")));
  EXPECT_THAT(s,
              TfStatusIs(error::INVALID_ARGUMENT, MatchesRegex(".*Argument")));
}

TEST(StatusMatchersTest, TfIsOk) {
  EXPECT_THAT(Status::OK(), TfIsOk());
  EXPECT_THAT(errors::DeadlineExceeded("Deadline exceeded"), Not(TfIsOk()));

  StatusOr<std::string> message = std::string("Hello, world");
  EXPECT_THAT(message, TfIsOk());
  StatusOr<std::string> status = errors::NotFound("Not found");
  EXPECT_THAT(status, Not(TfIsOk()));
}

}  // namespace
}  // namespace testing
}  // namespace tensorflow
