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

#include "tsl/platform/errors.h"

#include "absl/status/status.h"
#include "tsl/platform/test.h"

namespace tsl {

TEST(AppendToMessageTest, PayloadsAreCopied) {
  absl::Status status = errors::Aborted("Aborted Error Message");
  status.SetPayload("payload_key", absl::Cord("payload_value"));
  errors::AppendToMessage(&status, "Appended Message");

  EXPECT_EQ(status.message(), "Aborted Error Message\n\tAppended Message");
  EXPECT_EQ(status.GetPayload("payload_key"), absl::Cord("payload_value"));
}

TEST(Status, GetAllPayloads) {
  absl::Status s_error(absl::StatusCode::kInternal, "Error message");
  s_error.SetPayload("Error key", absl::Cord("foo"));
  auto payloads_error_status = errors::GetPayloads(s_error);
  ASSERT_EQ(payloads_error_status.size(), 1);
  ASSERT_EQ(payloads_error_status["Error key"], "foo");

  absl::Status s_ok = absl::Status();
  auto payloads_ok_status = errors::GetPayloads(s_ok);
  ASSERT_TRUE(payloads_ok_status.empty());
}

TEST(Status, OKStatusInsertPayloadsFromErrorStatus) {
  // An OK status will should not change after InsertPayloads() calls.
  absl::Status s_error(absl::StatusCode::kInternal, "Error message");
  s_error.SetPayload("Error key", absl::Cord("foo"));
  absl::Status s_ok = absl::Status();

  errors::InsertPayloads(s_ok, errors::GetPayloads(s_error));
  auto payloads_ok_status = errors::GetPayloads(s_ok);
  ASSERT_TRUE(payloads_ok_status.empty());
}

TEST(Status, ErrorStatusInsertPayloadsFromOKStatus) {
  // An InsertPayloads() call should not take effect from empty inputs.
  absl::Status s_error(absl::StatusCode::kInternal, "Error message");
  s_error.SetPayload("Error key", absl::Cord("foo"));
  absl::Status s_ok = absl::Status();

  errors::InsertPayloads(s_error, errors::GetPayloads(s_ok));
  ASSERT_EQ(s_error.GetPayload("Error key"), "foo");
}

TEST(Status, ErrorStatusInsertPayloadsFromErrorStatus) {
  absl::Status s_error1(absl::StatusCode::kInternal, "Error message");
  s_error1.SetPayload("Error key 1", absl::Cord("foo"));
  s_error1.SetPayload("Error key 2", absl::Cord("bar"));
  absl::Status s_error2(absl::StatusCode::kInternal, "Error message");
  s_error2.SetPayload("Error key", absl::Cord("bar"));
  ASSERT_EQ(s_error2.GetPayload("Error key"), "bar");

  errors::InsertPayloads(s_error2, errors::GetPayloads(s_error1));
  ASSERT_EQ(s_error2.GetPayload("Error key 1"), "foo");
  ASSERT_EQ(s_error2.GetPayload("Error key 2"), "bar");
  auto payloads_error_status = errors::GetPayloads(s_error2);
  ASSERT_EQ(payloads_error_status.size(), 3);
}

#if defined(PLATFORM_GOOGLE)

absl::Status GetError() {
  return absl::InvalidArgumentError("An invalid argument error");
}

absl::Status PropagateError() {
  TF_RETURN_IF_ERROR(GetError());
  return absl::OkStatus();
}

absl::Status PropagateError2() {
  TF_RETURN_IF_ERROR(PropagateError());
  return absl::OkStatus();
}

TEST(Status, StackTracePropagation) {
  absl::Status s = PropagateError2();
  auto sources = s.GetSourceLocations();
  ASSERT_EQ(sources.size(), 3);

  for (int i = 0; i < 3; ++i) {
    ASSERT_EQ(sources[i].file_name(),
              "third_party/tensorflow/tsl/platform/errors_test.cc");
  }
}

TEST(Status, SourceLocationsPreservedByAppend) {
  absl::Status s = PropagateError2();
  ASSERT_EQ(s.GetSourceLocations().size(), 3);
  errors::AppendToMessage(&s, "A new message.");
  ASSERT_EQ(s.GetSourceLocations().size(), 3);
}

TEST(Status, SourceLocationsPreservedByUpdate) {
  absl::Status s = PropagateError2();
  ASSERT_EQ(s.GetSourceLocations().size(), 3);
  absl::Status s2 = errors::CreateWithUpdatedMessage(s, "New message.");
  ASSERT_EQ(s2.GetSourceLocations().size(), 3);
}

#endif

}  // namespace tsl
