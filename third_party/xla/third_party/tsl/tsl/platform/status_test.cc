
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
#include "tsl/platform/status.h"

#include <unordered_map>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_format.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/stack_frame.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/status_to_from_proto.h"
#include "tsl/platform/test.h"
#include "tsl/protobuf/error_codes.pb.h"
#include "tsl/protobuf/status.pb.h"

namespace tsl {
namespace {

using ::testing::IsEmpty;
using ::tsl::testing::IsOk;
using ::tsl::testing::StatusIs;

TEST(ToStringTest, PayloadsArePrinted) {
  Status status = errors::Aborted("Aborted Error Message");
  status.SetPayload("payload_key", absl::Cord(absl::StrFormat(
                                       "payload_value %c%c%c", 1, 2, 3)));

  EXPECT_EQ(status.ToString(),
            "ABORTED: Aborted Error Message [payload_key='payload_value "
            "\\x01\\x02\\x03']");
}

TEST(ToStringTest, MatchesAbslStatus) {
  Status status = errors::Aborted("Aborted Error Message");
  status.SetPayload("payload_key", absl::Cord(absl::StrFormat(
                                       "payload_value %c%c%c", 1, 2, 3)));

  absl::Status absl_status =
      absl::Status(absl::StatusCode::kAborted, status.message());
  absl_status.SetPayload("payload_key", absl::Cord(absl::StrFormat(
                                            "payload_value %c%c%c", 1, 2, 3)));

  EXPECT_EQ(status.ToString(), absl_status.ToString());
}

TEST(StackTrace, SerializeAndDeserializeCorrectly) {
  Status status = errors::Aborted("Aborted Error Message");
  std::vector<StackFrame> stack_trace;
  stack_trace.push_back(StackFrame("filename_1", 33, "func_name_1"));
  stack_trace.push_back(StackFrame("filename_2", 66, "func_name_2"));
  errors::SetStackTrace(status, stack_trace);

  std::vector<StackFrame> deserialized = errors::GetStackTrace(status);

  EXPECT_EQ(stack_trace.size(), deserialized.size());
  for (size_t i = 0; i < stack_trace.size(); ++i) {
    EXPECT_EQ(stack_trace[i], deserialized[i]);
  }
}

TEST(StatusGroupTest, DeterministicOrderWithoutPayloads) {
  Status status_a = errors::Aborted("Status A");
  Status status_b = errors::Aborted("Status B");
  Status status_c = errors::Aborted("Status C");

  Status combined =
      StatusGroup({status_a, status_b, status_c}).as_summary_status();

  EXPECT_EQ(combined,
            StatusGroup({status_a, status_b, status_c}).as_summary_status());
  EXPECT_EQ(combined,
            StatusGroup({status_a, status_c, status_b}).as_summary_status());
  EXPECT_EQ(combined,
            StatusGroup({status_b, status_a, status_c}).as_summary_status());
  EXPECT_EQ(combined,
            StatusGroup({status_b, status_c, status_a}).as_summary_status());
  EXPECT_EQ(combined,
            StatusGroup({status_c, status_a, status_b}).as_summary_status());
  EXPECT_EQ(combined,
            StatusGroup({status_c, status_b, status_a}).as_summary_status());
}

TEST(StatusGroupTest, DeterministicOrderWithPayloads) {
  Status status_a = errors::Aborted("Status A");
  status_a.SetPayload("payload_key", absl::Cord("payload_value_a"));
  Status status_b = errors::Aborted("Status B");
  status_b.SetPayload("payload_key", absl::Cord("payload_value_b"));
  Status status_c = errors::Aborted("Status C");
  status_c.SetPayload("payload_key", absl::Cord("payload_value_c"));

  Status combined =
      StatusGroup({status_a, status_b, status_c}).as_summary_status();
  ASSERT_TRUE(combined.GetPayload("payload_key").has_value());
  std::string payload(combined.GetPayload("payload_key").value());

  EXPECT_EQ(payload, StatusGroup({status_a, status_b, status_c})
                         .as_summary_status()
                         .GetPayload("payload_key"));
  EXPECT_EQ(payload, StatusGroup({status_a, status_c, status_b})
                         .as_summary_status()
                         .GetPayload("payload_key"));
  EXPECT_EQ(payload, StatusGroup({status_b, status_a, status_c})
                         .as_summary_status()
                         .GetPayload("payload_key"));
  EXPECT_EQ(payload, StatusGroup({status_b, status_c, status_a})
                         .as_summary_status()
                         .GetPayload("payload_key"));
  EXPECT_EQ(payload, StatusGroup({status_c, status_a, status_b})
                         .as_summary_status()
                         .GetPayload("payload_key"));
  EXPECT_EQ(payload, StatusGroup({status_c, status_b, status_a})
                         .as_summary_status()
                         .GetPayload("payload_key"));
}

TEST(StatusGroupTest, PayloadsMergedProperly) {
  Status status_a = errors::Aborted("Status A");
  status_a.SetPayload("payload_key_a",
                      absl::Cord(std::string("payload_value_a")));
  Status status_b = errors::Aborted("Status B");
  status_b.SetPayload("payload_key_b",
                      absl::Cord(std::string("payload_value_b")));
  Status status_c = errors::Aborted("Status C");
  status_c.SetPayload("payload_key_c",
                      absl::Cord(std::string("payload_value_c")));
  Status derived_status_c =
      StatusGroup::MakeDerived(errors::Aborted("Status C"));
  derived_status_c.SetPayload(
      "payload_key_c", absl::Cord(std::string("derived_payload_value_c")));

  StatusGroup status_group({status_a, status_b, status_c, derived_status_c});
  EXPECT_THAT(status_group.GetPayloads(), ::testing::SizeIs(3));

  Status combined = status_group.as_summary_status();
  EXPECT_EQ(combined.GetPayload("payload_key_a"), "payload_value_a");
  EXPECT_EQ(combined.GetPayload("payload_key_b"), "payload_value_b");
  EXPECT_EQ(combined.GetPayload("payload_key_c"), "payload_value_c");
}

TEST(Status, ErrorStatusForEachPayloadIteratesOverAll) {
  Status s(absl::StatusCode::kInternal, "Error message");
  s.SetPayload("key1", absl::Cord("value1"));
  s.SetPayload("key2", absl::Cord("value2"));
  s.SetPayload("key3", absl::Cord("value3"));

  std::unordered_map<std::string, absl::Cord> payloads;
  s.ForEachPayload([&payloads](StringPiece key, const absl::Cord& value) {
    payloads[std::string(key)] = value;
  });

  EXPECT_EQ(payloads.size(), 3);
  EXPECT_EQ(payloads["key1"], "value1");
  EXPECT_EQ(payloads["key2"], "value2");
  EXPECT_EQ(payloads["key3"], "value3");
}

TEST(Status, OkStatusForEachPayloadNoIteration) {
  Status s = OkStatus();
  s.SetPayload("key1", absl::Cord("value1"));
  s.SetPayload("key2", absl::Cord("value2"));
  s.SetPayload("key3", absl::Cord("value3"));

  std::unordered_map<std::string, absl::Cord> payloads;
  s.ForEachPayload([&payloads](StringPiece key, const absl::Cord& value) {
    payloads[std::string(key)] = value;
  });

  EXPECT_EQ(payloads.size(), 0);
}

TEST(Status, SaveOKStatusToProto) {
  tensorflow::StatusProto status_proto = StatusToProto(OkStatus());
  EXPECT_EQ(status_proto.code(), error::OK);
  EXPECT_THAT(status_proto.message(), IsEmpty());
}

TEST(Status, SaveErrorStatusToProto) {
  tensorflow::StatusProto status_proto =
      StatusToProto(errors::NotFound("Not found"));
  EXPECT_EQ(status_proto.code(), error::NOT_FOUND);
  EXPECT_EQ(status_proto.message(), "Not found");
}

TEST(Status, SaveEmptyStatusToProto) {
  tensorflow::StatusProto status_proto = StatusToProto(Status());
  EXPECT_EQ(status_proto.code(), error::OK);
  EXPECT_THAT(status_proto.message(), IsEmpty());
}

TEST(Status, MakeOKStatusFromProto) {
  tensorflow::StatusProto status_proto;
  status_proto.set_code(error::OK);
  EXPECT_THAT(StatusFromProto(status_proto), IsOk());
}

TEST(Status, MakeErrorStatusFromProto) {
  tensorflow::StatusProto status_proto;
  status_proto.set_code(error::INVALID_ARGUMENT);
  status_proto.set_message("Invalid argument");
  EXPECT_THAT(StatusFromProto(status_proto),
              StatusIs(error::INVALID_ARGUMENT, "Invalid argument"));
}

TEST(Status, MakeStatusFromEmptyProto) {
  EXPECT_THAT(StatusFromProto(tensorflow::StatusProto()), IsOk());
}

}  // namespace
}  // namespace tsl
