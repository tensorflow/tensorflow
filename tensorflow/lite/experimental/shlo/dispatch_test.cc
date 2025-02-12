/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/experimental/shlo/dispatch.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/status_matcher.h"

namespace {

void VoidFunction() {}

TEST(DispatchTest, ReturnAbslOkIfVoidCompiles) {
  auto f = []() -> absl::Status { RETURN_OK_STATUS_IF_VOID(VoidFunction()); };
  EXPECT_OK(f());
}

TEST(DispatchTest, AbslOkStatusCompiles) {
  auto f = []() -> absl::Status { RETURN_OK_STATUS_IF_VOID(absl::OkStatus()); };
  EXPECT_OK(f());
}

TEST(DispatchTest, AbslErrorCompiles) {
  auto f = []() -> absl::Status {
    RETURN_OK_STATUS_IF_VOID(absl::UnknownError("error message"));
  };
  EXPECT_EQ(f(), absl::UnknownError("error message"));
}

}  // namespace
