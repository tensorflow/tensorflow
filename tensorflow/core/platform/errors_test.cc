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

#include "tensorflow/core/platform/errors.h"

#include "absl/container/flat_hash_map.h"
#include "absl/strings/cord.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

TEST(GetPayloadsTest, OKStatusNoPayloadsAdded) {
  Status status = Status::OK();
  status.SetPayload("Error key", absl::Cord("foo"));
  ASSERT_TRUE(errors::GetPayloads(status).empty());
}

TEST(GetPayloadsTest, PayloadsExtracted) {
  Status status = errors::Aborted("Aborted Error Message");
  status.SetPayload("payload_key", absl::Cord("payload_value"));
  auto payloads = errors::GetPayloads(status);

  EXPECT_EQ(payloads.size(), 1);
  EXPECT_EQ(payloads["payload_key"], "payload_value");
}

TEST(CopyPayloadsTest, PayloadIsCopied) {
  Status status_a = errors::Aborted("Aborted Error Message");
  status_a.SetPayload("payload_key", absl::Cord("payload_value"));

  Status status_b = errors::Internal("Internal Error Message");
  errors::CopyPayloads(status_a, status_b);

  EXPECT_EQ(status_b.error_message(), "Internal Error Message");
  EXPECT_EQ(status_b.GetPayload("payload_key"), "payload_value");
}

TEST(CopyPayloadsTest, PayloadIsOverwritten) {
  Status status_a = errors::Aborted("Aborted Error Message");
  status_a.SetPayload("payload_key", absl::Cord("payload_value"));

  Status status_b = errors::Internal("Internal Error Message");
  status_b.SetPayload("payload_key", absl::Cord("payload_value_2"));
  errors::CopyPayloads(status_a, status_b);

  EXPECT_EQ(status_b.error_message(), "Internal Error Message");
  EXPECT_EQ(status_b.GetPayload("payload_key"), "payload_value");
}

TEST(InsertPayloadsTest, PayloadsAreInserted) {
  Status status = errors::Aborted("Aborted Error Message");
  absl::flat_hash_map<std::string, absl::Cord> payloads;
  payloads["key1"] = "value1";
  payloads["key2"] = "value2";
  payloads["key3"] = "value3";
  errors::InsertPayloads(status, payloads);

  EXPECT_EQ(status.GetPayload("key1"), "value1");
  EXPECT_EQ(status.GetPayload("key2"), "value2");
  EXPECT_EQ(status.GetPayload("key3"), "value3");
}

TEST(CreateWithUpdatedMessageTest, PayloadsAreCopied) {
  Status status = errors::Aborted("Aborted Error Message");
  status.SetPayload("payload_key", absl::Cord("payload_value"));
  status = errors::CreateWithUpdatedMessage(status, "New Message");

  EXPECT_EQ(status.error_message(), "New Message");
  EXPECT_EQ(status.GetPayload("payload_key"), "payload_value");
}

TEST(AppendToMessageTest, PayloadsAreCopied) {
  Status status = errors::Aborted("Aborted Error Message");
  status.SetPayload("payload_key", absl::Cord("payload_value"));
  errors::AppendToMessage(&status, "Appended Message");

  EXPECT_EQ(status.error_message(),
            "Aborted Error Message\n\tAppended Message");
  EXPECT_EQ(status.GetPayload("payload_key"), "payload_value");
}

TEST(DeclareErrorWithPayloads, PayloadsAccepted) {
  absl::flat_hash_map<std::string, absl::Cord> payloads;
  payloads["key1"] = "value1";
  payloads["key2"] = "value2";
  payloads["key3"] = "value3";
  Status status = errors::AbortedWithPayloads("message", payloads);

  EXPECT_EQ(status.GetPayload("key1"), "value1");
  EXPECT_EQ(status.GetPayload("key2"), "value2");
  EXPECT_EQ(status.GetPayload("key3"), "value3");
}
}  // namespace tensorflow
