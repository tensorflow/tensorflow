/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/pjrt/pjrt_build_stamp_writer.h"

#include <cstdint>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/time/time.h"
#include "xla/pjrt/pjrt_build_stamp.h"
#include "xla/pjrt/proto/pjrt_build_stamp.pb.h"
#include "xla/tsl/builddata/builddata.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla {
namespace {

// Tests for `SetMosaicVersion` are in `pjrt_compatibility_versions_test.cc`.

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::tsl::proto_testing::EqualsProto;

TEST(PjRtBuildStampWriterTest, ProtoFromBuildDataHandlesStampingState) {
  PjRtBuildStampProto proto = PjRtBuildStampProtoFromBuildData();
  EXPECT_NE(proto.vcs_revision(), "<unknown>");
  EXPECT_NE(proto.vcs_revision(), "unknown");

  const int64_t timestamp_seconds = tsl::builddata::TimestampAsInt();
  if (timestamp_seconds <= 0) {
    EXPECT_FALSE(proto.has_build_timestamp());
    if (proto.vcs_revision().empty()) {
      EXPECT_THAT(PjRtBuildStamp::FromProto(proto),
                  StatusIs(absl::StatusCode::kInvalidArgument,
                           HasSubstr("at least one build identity field")));
    }
    proto.set_source_hash("test_hash");
    ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                         PjRtBuildStamp::FromProto(std::move(proto)));
    EXPECT_EQ(stamp.build_timestamp(), absl::InfinitePast());
    EXPECT_EQ(stamp.source_hash(), "test_hash");
  } else {
    EXPECT_TRUE(proto.has_build_timestamp());
    ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                         PjRtBuildStamp::FromProto(std::move(proto)));
    EXPECT_EQ(stamp.build_timestamp(),
              absl::FromUnixSeconds(timestamp_seconds));
    EXPECT_NE(stamp.vcs_revision(), "<unknown>");
    EXPECT_THAT(stamp.source_hash(), IsEmpty());
    EXPECT_THAT(stamp.ToProto().source_hash(), IsEmpty());
  }
}

TEST(PjRtBuildStampWriterTest, SetAttributeAddsAttribute) {
  PjRtBuildStampProto proto;
  ASSERT_OK(SetAttribute(proto, "key1", true));

  EXPECT_THAT(proto, EqualsProto(R"pb(attributes {
                                        key: "key1"
                                        value { bool_value: true }
                                      })pb"));
}

TEST(PjRtBuildStampWriterTest, SetAttributeFailsWhenKeyIsEmpty) {
  PjRtBuildStampProto proto;
  EXPECT_THAT(SetAttribute(proto, "", true),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Attribute key must not be empty")));
}

TEST(PjRtBuildStampWriterTest, SetAttributeFailsWhenKeyAlreadyExists) {
  PjRtBuildStampProto proto;
  ASSERT_OK(SetAttribute(proto, "key1", true));
  EXPECT_THAT(SetAttribute(proto, "key1", false),
              StatusIs(absl::StatusCode::kAlreadyExists,
                       HasSubstr("Attribute 'key1' is already set")));
}

}  // namespace
}  // namespace xla
