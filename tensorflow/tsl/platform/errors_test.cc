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

#include "tensorflow/tsl/platform/errors.h"

#include "tensorflow/tsl/platform/test.h"

namespace tsl {

TEST(AppendToMessageTest, PayloadsAreCopied) {
  Status status = errors::Aborted("Aborted Error Message");
  status.SetPayload("payload_key", "payload_value");
  errors::AppendToMessage(&status, "Appended Message");

  EXPECT_EQ(status.error_message(),
            "Aborted Error Message\n\tAppended Message");
  EXPECT_EQ(status.GetPayload("payload_key"), "payload_value");
}

TEST(Status, GetAllPayloads) {
  Status s_error(error::INTERNAL, "Error message");
  s_error.SetPayload("Error key", "foo");
  auto payloads_error_status = errors::GetPayloads(s_error);
  ASSERT_EQ(payloads_error_status.size(), 1);
  ASSERT_EQ(payloads_error_status["Error key"], "foo");

  Status s_ok = Status();
  auto payloads_ok_status = errors::GetPayloads(s_ok);
  ASSERT_TRUE(payloads_ok_status.empty());
}

TEST(Status, OKStatusInsertPayloadsFromErrorStatus) {
  // An OK status will should not change after InsertPayloads() calls.
  Status s_error(error::INTERNAL, "Error message");
  s_error.SetPayload("Error key", "foo");
  Status s_ok = Status();

  errors::InsertPayloads(s_ok, errors::GetPayloads(s_error));
  auto payloads_ok_status = errors::GetPayloads(s_ok);
  ASSERT_TRUE(payloads_ok_status.empty());
}

TEST(Status, ErrorStatusInsertPayloadsFromOKStatus) {
  // An InsertPayloads() call should not take effect from empty inputs.
  Status s_error(error::INTERNAL, "Error message");
  s_error.SetPayload("Error key", "foo");
  Status s_ok = Status();

  errors::InsertPayloads(s_error, errors::GetPayloads(s_ok));
  ASSERT_EQ(s_error.GetPayload("Error key"), "foo");
}

TEST(Status, ErrorStatusInsertPayloadsFromErrorStatus) {
  Status s_error1(error::INTERNAL, "Error message");
  s_error1.SetPayload("Error key 1", "foo");
  s_error1.SetPayload("Error key 2", "bar");
  Status s_error2(error::INTERNAL, "Error message");
  s_error2.SetPayload("Error key", "bar");
  ASSERT_EQ(s_error2.GetPayload("Error key"), "bar");

  errors::InsertPayloads(s_error2, errors::GetPayloads(s_error1));
  ASSERT_EQ(s_error2.GetPayload("Error key 1"), "foo");
  ASSERT_EQ(s_error2.GetPayload("Error key 2"), "bar");
  auto payloads_error_status = errors::GetPayloads(s_error2);
  ASSERT_EQ(payloads_error_status.size(), 3);
}

}  // namespace tsl
