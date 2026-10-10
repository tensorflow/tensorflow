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

#include "xla/pjrt/pjrt_build_stamp.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "google/protobuf/timestamp.pb.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/proto/pjrt_build_stamp.pb.h"

namespace xla {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::Optional;

TEST(PjRtBuildStampTest, BasicAccessors) {
  PjRtBuildStampProto proto;
  proto.mutable_build_timestamp()->set_seconds(1700000000);
  proto.set_vcs_revision("git_rev_123");
  proto.set_build_label("target_foo");
  proto.set_source_hash("hash_abcdef01");
  SetMosaicVersion(proto, 12);

  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       PjRtBuildStamp::Create(std::move(proto)));

  EXPECT_EQ(stamp.build_timestamp(), absl::FromUnixSeconds(1700000000));
  EXPECT_EQ(stamp.vcs_revision(), "git_rev_123");
  EXPECT_EQ(stamp.build_label(), "target_foo");
  EXPECT_EQ(stamp.source_hash(), "hash_abcdef01");
  EXPECT_THAT(GetMosaicVersion(stamp), Optional(int64_t{12}));
  EXPECT_TRUE(stamp.components().empty());
  EXPECT_FALSE(stamp.attributes().empty());
  EXPECT_EQ(stamp.full_source_hash(), "hash_abcdef01");
}

TEST(PjRtBuildStampTest, DefaultBuildTimestamp) {
  // Verify default InfinitePast when build_timestamp is unset.
  PjRtBuildStampProto proto_no_timestamp;
  proto_no_timestamp.set_source_hash("hash_no_ts");
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp_no_ts,
                       PjRtBuildStamp::Create(std::move(proto_no_timestamp)));
  EXPECT_EQ(stamp_no_ts.build_timestamp(), absl::InfinitePast());
}

TEST(PjRtBuildStampTest, BuildTimestampWithNanos) {
  const absl::Time now =
      absl::FromUnixSeconds(1700000000) + absl::Nanoseconds(123456789);
  PjRtBuildStampProto proto;
  proto.mutable_build_timestamp()->set_seconds(1700000000);
  proto.mutable_build_timestamp()->set_nanos(123456789);
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       PjRtBuildStamp::Create(std::move(proto)));
  EXPECT_EQ(stamp.build_timestamp(), now);
  EXPECT_EQ(stamp.full_source_hash(), "ts:1700000000.123456789");
}

TEST(PjRtBuildStampTest, CreateFailsWhenMissingBuildIdentity) {
  EXPECT_THAT(PjRtBuildStamp::Create(PjRtBuildStampProto{}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("at least one build identity field")));
}

TEST(PjRtBuildStampTest, CreateFailsWhenComponentHasEmptyName) {
  PjRtBuildStampProto proto_empty_name;
  proto_empty_name.set_source_hash("hash");
  auto* c1 = proto_empty_name.add_components();
  c1->set_hash("fp");
  EXPECT_THAT(
      PjRtBuildStamp::Create(std::move(proto_empty_name)),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Component build stamp name and hash must not be empty")));
}

TEST(PjRtBuildStampTest, CreateFailsWhenComponentHasEmptyHash) {
  PjRtBuildStampProto proto_empty_hash;
  proto_empty_hash.set_source_hash("hash");
  auto* c2 = proto_empty_hash.add_components();
  c2->set_name("comp");
  EXPECT_THAT(
      PjRtBuildStamp::Create(std::move(proto_empty_hash)),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Component build stamp name and hash must not be empty")));
}

TEST(PjRtBuildStampTest,
     CreateFailsWhenDuplicateComponentHasConflictingHashes) {
  PjRtBuildStampProto proto_conflicting;
  proto_conflicting.set_source_hash("base_hash");
  auto* pc1 = proto_conflicting.add_components();
  pc1->set_kind("kind_a");
  pc1->set_name("name_a");
  pc1->set_owner("owner_a");
  pc1->set_hash("hash_1");
  auto* pc2 = proto_conflicting.add_components();
  pc2->set_kind("kind_a");
  pc2->set_name("name_a");
  pc2->set_owner("owner_a");
  pc2->set_hash("hash_2");
  EXPECT_THAT(
      PjRtBuildStamp::Create(std::move(proto_conflicting)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Duplicate component found with conflicting hashes")));
}

TEST(PjRtBuildStampTest, ProtoFromBuildData) {
  PjRtBuildStampProto proto = PjRtBuildStamp::BuildStampProtoFromBuildData();
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       PjRtBuildStamp::Create(std::move(proto)));
  EXPECT_NE(stamp.build_timestamp(), absl::InfinitePast());
}

TEST(PjRtBuildStampTest, ToProtoRoundTrip) {
  PjRtBuildStampProto proto1;
  proto1.set_vcs_revision("rev1");
  proto1.set_source_hash("hash1");
  SetMosaicVersion(proto1, 7);
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp build_stamp1,
                       PjRtBuildStamp::Create(proto1));

  PjRtBuildStampProto proto2 = build_stamp1.ToProto();
  EXPECT_EQ(proto2.vcs_revision(), "rev1");
  EXPECT_EQ(proto2.source_hash(), "hash1");
  EXPECT_EQ(proto2.full_source_hash(), "hash1");

  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp build_stamp2,
                       PjRtBuildStamp::Create(proto2));
  EXPECT_EQ(build_stamp1.source_hash(), build_stamp2.source_hash());
  EXPECT_EQ(build_stamp1.full_source_hash(), build_stamp2.full_source_hash());
  EXPECT_THAT(GetMosaicVersion(build_stamp1), Optional(int64_t{7}));
  EXPECT_THAT(GetMosaicVersion(build_stamp2), Optional(int64_t{7}));
}

TEST(PjRtBuildStampTest, ModifiedComponentsAffectFullSourceHash) {
  PjRtBuildStampProto proto1;
  proto1.set_source_hash("hash1");
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp build_stamp1,
                       PjRtBuildStamp::Create(proto1));
  EXPECT_EQ(build_stamp1.source_hash(), build_stamp1.full_source_hash());

  PjRtBuildStampProto proto2 = build_stamp1.ToProto();
  auto* comp = proto2.add_components();
  comp->set_kind("custom_call");
  comp->set_name("op_a");
  comp->set_owner("pkg:op_a");
  comp->set_hash("hash_a");

  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp build_stamp2,
                       PjRtBuildStamp::Create(std::move(proto2)));
  ASSERT_EQ(build_stamp2.components().size(), 1);
  EXPECT_EQ(build_stamp2.components()[0].kind(), "custom_call");
  EXPECT_EQ(build_stamp2.components()[0].name(), "op_a");
  EXPECT_EQ(build_stamp2.components()[0].owner(), "pkg:op_a");
  EXPECT_EQ(build_stamp2.components()[0].hash(), "hash_a");
  EXPECT_NE(build_stamp1.full_source_hash(), build_stamp2.full_source_hash());
}

TEST(PjRtBuildStampTest, AttributesAccessors) {
  PjRtBuildStampProto attr_proto;
  attr_proto.set_source_hash("hash1");
  SetMosaicVersion(attr_proto, 15);
  (*attr_proto.mutable_attributes())["vendor.candidate"] =
      PjRtValueTypeToProto(std::string("xla_tpu_candidate_1"));
  (*attr_proto.mutable_attributes())["is_experimental"] =
      PjRtValueTypeToProto(true);
  (*attr_proto.mutable_attributes())["int_attr"] =
      PjRtValueTypeToProto(int64_t{42});
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp attr_stamp,
                       PjRtBuildStamp::Create(std::move(attr_proto)));

  EXPECT_THAT(GetMosaicVersion(attr_stamp), Optional(int64_t{15}));
  EXPECT_THAT(attr_stamp.GetStringAttribute("vendor.candidate"),
              Optional(absl::string_view("xla_tpu_candidate_1")));
  EXPECT_THAT(attr_stamp.GetAttribute<std::string>("vendor.candidate"),
              Optional(std::string("xla_tpu_candidate_1")));
  EXPECT_THAT(attr_stamp.GetAttribute<bool>("is_experimental"), Optional(true));
  EXPECT_THAT(attr_stamp.GetAttribute<int64_t>("int_attr"),
              Optional(int64_t{42}));
  EXPECT_EQ(attr_stamp.GetAttribute<int64_t>("non_existent"), std::nullopt);
  EXPECT_EQ(attr_stamp.GetStringAttribute("non_existent"), std::nullopt);
  EXPECT_EQ(attr_stamp.GetAttribute<int64_t>("vendor.candidate"), std::nullopt);
  EXPECT_EQ(attr_stamp.GetAttribute<bool>("int_attr"), std::nullopt);
  EXPECT_EQ(attr_stamp.GetStringAttribute("int_attr"), std::nullopt);
  EXPECT_EQ(attr_stamp.GetStringAttribute("is_experimental"), std::nullopt);
}

TEST(PjRtBuildStampTest, AttributesDoNotAffectFullSourceHash) {
  PjRtBuildStampProto base_proto;
  base_proto.set_source_hash("hash1");
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp base_stamp,
                       PjRtBuildStamp::Create(base_proto));

  PjRtBuildStampProto attr_proto = base_stamp.ToProto();
  (*attr_proto.mutable_attributes())["vendor.candidate"] =
      PjRtValueTypeToProto(std::string("xla_tpu_candidate_1"));
  (*attr_proto.mutable_attributes())["is_experimental"] =
      PjRtValueTypeToProto(true);
  (*attr_proto.mutable_attributes())["int_attr"] =
      PjRtValueTypeToProto(int64_t{42});
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp attr_stamp,
                       PjRtBuildStamp::Create(std::move(attr_proto)));

  // Attributes must NOT factor into full_source_hash.
  EXPECT_EQ(base_stamp.full_source_hash(), attr_stamp.full_source_hash());
}

TEST(PjRtBuildStampTest, SourceHashFallbackToVcsRevision) {
  PjRtBuildStampProto vcs_proto;
  vcs_proto.set_vcs_revision("git_rev_456");
  vcs_proto.mutable_build_timestamp()->set_seconds(1700000000);
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp vcs_stamp,
                       PjRtBuildStamp::Create(std::move(vcs_proto)));
  EXPECT_TRUE(vcs_stamp.source_hash().empty());
  EXPECT_EQ(vcs_stamp.full_source_hash(), "vcs:git_rev_456");
}

TEST(PjRtBuildStampTest, SourceHashFallbackToTimestamp) {
  PjRtBuildStampProto ts_proto;
  ts_proto.mutable_build_timestamp()->set_seconds(1700000000);
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp ts_stamp,
                       PjRtBuildStamp::Create(std::move(ts_proto)));
  EXPECT_TRUE(ts_stamp.source_hash().empty());
  EXPECT_EQ(ts_stamp.full_source_hash(), "ts:1700000000.000000000");
}

TEST(PjRtBuildStampTest,
     CanonicalOrderAndDeduplicationProduceIdenticalFullSourceHash) {
  PjRtComponentBuildStampProto c1;
  c1.set_kind("xla.gpu.native_custom_call");
  c1.set_name("alpha");
  c1.set_owner("pkg:alpha");
  c1.set_hash("1111");
  PjRtComponentBuildStampProto c2;
  c2.set_kind("xla.ffi.handler");
  c2.set_name("beta");
  c2.set_owner("pkg:beta");
  c2.set_hash("2222");

  PjRtBuildStampProto proto1;
  proto1.set_source_hash("base_hash");
  *proto1.add_components() = c2;
  *proto1.add_components() = c1;
  *proto1.add_components() = c1;  // Duplicate component

  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp1,
                       PjRtBuildStamp::Create(std::move(proto1)));

  PjRtBuildStampProto proto2;
  proto2.set_source_hash("base_hash");
  *proto2.add_components() = c1;
  *proto2.add_components() = c2;

  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp2,
                       PjRtBuildStamp::Create(std::move(proto2)));

  EXPECT_THAT(stamp1.components(),
              ::testing::ElementsAre(::testing::EqualsProto(c2),
                                     ::testing::EqualsProto(c1)));
  EXPECT_THAT(stamp2.components(),
              ::testing::ElementsAre(::testing::EqualsProto(c2),
                                     ::testing::EqualsProto(c1)));
  EXPECT_EQ(stamp1.full_source_hash(), stamp2.full_source_hash());
}

TEST(PjRtBuildStampTest,
     DifferentComponentHashProducesDifferentFullSourceHash) {
  PjRtBuildStampProto proto1;
  proto1.set_source_hash("base_hash");
  auto* p1_c1 = proto1.add_components();
  p1_c1->set_kind("xla.gpu.native_custom_call");
  p1_c1->set_name("alpha");
  p1_c1->set_owner("pkg:alpha");
  p1_c1->set_hash("1111");

  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp1,
                       PjRtBuildStamp::Create(std::move(proto1)));

  PjRtBuildStampProto proto2;
  proto2.set_source_hash("base_hash");
  auto* p2_c1 = proto2.add_components();
  p2_c1->set_kind("xla.gpu.native_custom_call");
  p2_c1->set_name("alpha");
  p2_c1->set_owner("pkg:alpha");
  p2_c1->set_hash("2222");

  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp2,
                       PjRtBuildStamp::Create(std::move(proto2)));
  EXPECT_NE(stamp1.full_source_hash(), stamp2.full_source_hash());
}

}  // namespace
}  // namespace xla
