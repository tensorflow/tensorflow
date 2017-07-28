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

#include "tensorflow/compiler/xla/status_macros.h"

#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {

Status RetCheckFail() {
  TF_RET_CHECK(2 > 3);
  return Status::OK();
}

Status RetCheckFailWithExtraMessage() {
  TF_RET_CHECK(2 > 3) << "extra message";
  return Status::OK();
}

Status RetCheckSuccess() {
  TF_RET_CHECK(3 > 2);
  return Status::OK();
}

TEST(StatusMacros, RetCheckFailing) {
  Status status = RetCheckFail();
  EXPECT_EQ(status.code(), tensorflow::error::INTERNAL);
  EXPECT_THAT(status.error_message(),
              ::testing::ContainsRegex("RET_CHECK failure.*2 > 3"));
}

TEST(StatusMacros, RetCheckFailingWithExtraMessage) {
  Status status = RetCheckFailWithExtraMessage();
  EXPECT_EQ(status.code(), tensorflow::error::INTERNAL);
  EXPECT_THAT(status.error_message(),
              ::testing::ContainsRegex("RET_CHECK.*2 > 3 extra message"));
}

TEST(StatusMacros, RetCheckSucceeding) {
  Status status = RetCheckSuccess();
  EXPECT_IS_OK(status);
}

StatusOr<int> CreateIntSuccessfully() { return 42; }

StatusOr<int> CreateIntUnsuccessfully() {
  return tensorflow::errors::Internal("foobar");
}

TEST(StatusMacros, AssignOrAssertOnOK) {
  TF_ASSERT_OK_AND_ASSIGN(int result, CreateIntSuccessfully());
  EXPECT_EQ(42, result);
}

Status ReturnStatusOK() { return Status::OK(); }

Status ReturnStatusError() { return (tensorflow::errors::Internal("foobar")); }

using StatusReturningFunction = std::function<Status()>;

StatusOr<int> CallStatusReturningFunction(const StatusReturningFunction& func) {
  TF_RETURN_IF_ERROR(func());
  return 42;
}

TEST(StatusMacros, ReturnIfErrorOnOK) {
  StatusOr<int> rc = CallStatusReturningFunction(ReturnStatusOK);
  EXPECT_IS_OK(rc);
  EXPECT_EQ(42, rc.ConsumeValueOrDie());
}

TEST(StatusMacros, ReturnIfErrorOnError) {
  StatusOr<int> rc = CallStatusReturningFunction(ReturnStatusError);
  EXPECT_FALSE(rc.ok());
  EXPECT_EQ(rc.status().code(), tensorflow::error::INTERNAL);
}

TEST(StatusMacros, AssignOrReturnSuccessufully) {
  Status status = []() {
    TF_ASSIGN_OR_RETURN(int value, CreateIntSuccessfully());
    EXPECT_EQ(value, 42);
    return Status::OK();
  }();
  EXPECT_IS_OK(status);
}

TEST(StatusMacros, AssignOrReturnUnsuccessfully) {
  Status status = []() {
    TF_ASSIGN_OR_RETURN(int value, CreateIntUnsuccessfully());
    (void)value;
    return Status::OK();
  }();
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), tensorflow::error::INTERNAL);
}

}  // namespace xla
