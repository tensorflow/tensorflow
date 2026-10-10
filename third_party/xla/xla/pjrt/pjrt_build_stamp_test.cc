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
#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/timestamp.pb.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/pjrt/pjrt_build_stamp_internal.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/proto/pjrt_build_stamp.pb.h"
#include "xla/pjrt/proto/pjrt_value_type.pb.h"

namespace xla {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::EqualsProto;
using ::testing::HasSubstr;
using ::testing::IsEmpty;

TEST(PjRtBuildStampTest, BasicAccessors) {
  PjRtBuildStampProto proto;
  proto.set_build_label("target_foo");
  proto.mutable_build_timestamp()->set_seconds(1700000000);
  proto.set_vcs_revision("git_rev_123");
  proto.set_source_hash("hash_abcdef01");

  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       PjRtBuildStamp::Create(std::move(proto)));

  EXPECT_EQ(stamp.build_label(), "target_foo");
  EXPECT_EQ(stamp.build_timestamp(), absl::FromUnixSeconds(1700000000));
  EXPECT_EQ(stamp.vcs_revision(), "git_rev_123");
  EXPECT_EQ(stamp.source_hash(), "hash_abcdef01");
}

TEST(PjRtBuildStampTest, DefaultBuildTimestampIsInfinitePast) {
  PjRtBuildStampProto proto;
  proto.set_source_hash("hash_no_ts");

  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       PjRtBuildStamp::Create(std::move(proto)));

  EXPECT_EQ(stamp.build_timestamp(), absl::InfinitePast());
}

TEST(PjRtBuildStampTest, BuildTimestampIncludesNanos) {
  PjRtBuildStampProto proto;
  proto.mutable_build_timestamp()->set_seconds(1700000000);
  proto.mutable_build_timestamp()->set_nanos(123456789);

  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       PjRtBuildStamp::Create(std::move(proto)));

  EXPECT_EQ(stamp.build_timestamp(),
            absl::FromUnixSeconds(1700000000) + absl::Nanoseconds(123456789));
}

TEST(PjRtBuildStampTest, BuildStampProtoFromBuildDataPopulatesTimestamp) {
  PjRtBuildStampProto proto = PjRtBuildStamp::BuildStampProtoFromBuildData();

  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       PjRtBuildStamp::Create(std::move(proto)));

  EXPECT_NE(stamp.build_timestamp(), absl::InfinitePast());
  EXPECT_THAT(stamp.source_hash(), IsEmpty());
  EXPECT_THAT(stamp.ToProto().source_hash(), IsEmpty());
}

TEST(PjRtBuildStampTest, ToProtoRoundTrip) {
  PjRtBuildStampProto proto;
  proto.set_build_label("label1");
  proto.set_vcs_revision("rev1");
  proto.set_source_hash("hash1");
  SetMosaicVersion(proto, 7, 1);

  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp1,
                       PjRtBuildStamp::Create(std::move(proto)));
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp2,
                       PjRtBuildStamp::Create(stamp1.ToProto()));

  EXPECT_THAT(stamp2.ToProto(), EqualsProto(stamp1.ToProto()));
}

TEST(PjRtBuildStampTest, CreateFailsWhenMissingBuildIdentity) {
  EXPECT_THAT(PjRtBuildStamp::Create(PjRtBuildStampProto{}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("at least one build identity field")));
}

TEST(PjRtBuildStampTest, CreateFailsWhenComponentHasEmptyName) {
  PjRtBuildStampProto proto;
  proto.set_source_hash("hash");
  proto.add_components()->set_hash("fp");

  EXPECT_THAT(
      PjRtBuildStamp::Create(std::move(proto)),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Component build stamp name and hash must not be empty")));
}

TEST(PjRtBuildStampTest, CreateFailsWhenComponentHasEmptyHash) {
  PjRtBuildStampProto proto;
  proto.set_source_hash("hash");
  proto.add_components()->set_name("comp");

  EXPECT_THAT(
      PjRtBuildStamp::Create(std::move(proto)),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Component build stamp name and hash must not be empty")));
}

TEST(PjRtBuildStampTest,
     CreateFailsWhenDuplicateComponentHasConflictingHashes) {
  PjRtBuildStampProto proto;
  proto.set_source_hash("base_hash");
  auto* c1 = proto.add_components();
  c1->set_kind("kind_a");
  c1->set_name("name_a");
  c1->set_owner("owner_a");
  c1->set_hash("hash_1");
  auto* c2 = proto.add_components();
  c2->set_kind("kind_a");
  c2->set_name("name_a");
  c2->set_owner("owner_a");
  c2->set_hash("hash_2");

  EXPECT_THAT(
      PjRtBuildStamp::Create(std::move(proto)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Duplicate component found with conflicting hashes")));
}

TEST(PjRtBuildStampTest, CreateSortsAndDeduplicatesComponents) {
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

  PjRtBuildStampProto proto;
  proto.set_source_hash("base_hash");
  *proto.add_components() = c1;
  *proto.add_components() = c2;
  *proto.add_components() = c1;

  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       PjRtBuildStamp::Create(std::move(proto)));

  EXPECT_THAT(stamp.components(),
              ElementsAre(EqualsProto(c2), EqualsProto(c1)));
}

TEST(PjRtBuildStampTest, CreateFailsWhenVersionHasEmptyName) {
  PjRtBuildStampProto proto;
  proto.set_source_hash("base_hash");
  auto* v = proto.add_versions();
  v->set_name("");
  v->set_major_version(1);

  EXPECT_THAT(PjRtBuildStamp::Create(std::move(proto)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Version name must not be empty")));
}

TEST(PjRtBuildStampTest, CreateFailsWhenMajorVersionIsNegative) {
  PjRtBuildStampProto proto;
  proto.set_source_hash("base_hash");
  auto* v = proto.add_versions();
  v->set_name("mosaic");
  v->set_major_version(-1);

  EXPECT_THAT(PjRtBuildStamp::Create(std::move(proto)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Version numbers must be non-negative")));
}

TEST(PjRtBuildStampTest, CreateFailsWhenMinorVersionIsNegative) {
  PjRtBuildStampProto proto;
  proto.set_source_hash("base_hash");
  auto* v = proto.add_versions();
  v->set_name("mosaic");
  v->set_major_version(1);
  v->set_minor_version(-1);

  EXPECT_THAT(PjRtBuildStamp::Create(std::move(proto)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Version numbers must be non-negative")));
}

TEST(PjRtBuildStampTest,
     CreateFailsWhenDuplicateVersionHasConflictingMajorVersion) {
  PjRtBuildStampProto proto;
  proto.set_source_hash("base_hash");
  auto* v1 = proto.add_versions();
  v1->set_name("mosaic");
  v1->set_major_version(12);
  auto* v2 = proto.add_versions();
  v2->set_name("mosaic");
  v2->set_major_version(13);

  EXPECT_THAT(
      PjRtBuildStamp::Create(std::move(proto)),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "Duplicate version found with conflicting values for 'mosaic'")));
}

TEST(PjRtBuildStampTest,
     CreateFailsWhenDuplicateVersionHasConflictingMinorVersion) {
  PjRtBuildStampProto proto;
  proto.set_source_hash("base_hash");
  auto* v1 = proto.add_versions();
  v1->set_name("mosaic");
  v1->set_major_version(12);
  v1->set_minor_version(0);
  auto* v2 = proto.add_versions();
  v2->set_name("mosaic");
  v2->set_major_version(12);
  v2->set_minor_version(1);

  EXPECT_THAT(
      PjRtBuildStamp::Create(std::move(proto)),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "Duplicate version found with conflicting values for 'mosaic'")));
}

TEST(PjRtBuildStampTest, CreateSortsAndDeduplicatesVersions) {
  PjRtVersionProto v1;
  v1.set_name("mosaic");
  v1.set_major_version(12);
  v1.set_minor_version(1);
  PjRtVersionProto v2;
  v2.set_name("tpu_runtime");
  v2.set_major_version(3);
  v2.set_minor_version(0);

  PjRtBuildStampProto proto;
  proto.set_source_hash("base_hash");
  *proto.add_versions() = v2;
  *proto.add_versions() = v1;
  *proto.add_versions() = v2;

  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       PjRtBuildStamp::Create(std::move(proto)));

  EXPECT_THAT(stamp.versions(), ElementsAre(EqualsProto(v1), EqualsProto(v2)));
}

TEST(PjRtBuildStampTest, CreateFailsWhenAttributeKeyIsEmpty) {
  PjRtBuildStampProto proto;
  proto.set_source_hash("base_hash");
  (*proto.mutable_attributes())[""] = PjRtValueTypeToProto(true);

  EXPECT_THAT(PjRtBuildStamp::Create(std::move(proto)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Attribute key must not be empty")));
}

TEST(PjRtBuildStampTest, CreateFailsWhenAttributeValueIsNotSet) {
  PjRtBuildStampProto proto;
  proto.set_source_hash("base_hash");
  (*proto.mutable_attributes())["empty_attr"] = PjRtValueTypeProto{};

  EXPECT_THAT(
      PjRtBuildStamp::Create(std::move(proto)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Attribute value for key='empty_attr' must be set")));
}

TEST(PjRtBuildStampTest,
     FullSourceHashEqualsSourceHashWhenNoComponentsOrVersions) {
  PjRtBuildStampProto proto;
  proto.set_source_hash("hash1");

  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       PjRtBuildStamp::Create(std::move(proto)));

  EXPECT_EQ(stamp.full_source_hash(), "hash1");
}

TEST(PjRtBuildStampTest, FullSourceHashFallsBackToVcsRevision) {
  PjRtBuildStampProto proto;
  proto.set_vcs_revision("git_rev_456");
  proto.mutable_build_timestamp()->set_seconds(1700000000);

  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       PjRtBuildStamp::Create(std::move(proto)));

  EXPECT_EQ(stamp.full_source_hash(), "vcs:git_rev_456");
  EXPECT_THAT(stamp.source_hash(), IsEmpty());
  EXPECT_THAT(stamp.ToProto().source_hash(), IsEmpty());
}

TEST(PjRtBuildStampTest, FullSourceHashFallsBackToTimestamp) {
  PjRtBuildStampProto proto;
  proto.mutable_build_timestamp()->set_seconds(1700000000);
  proto.mutable_build_timestamp()->set_nanos(12345);

  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       PjRtBuildStamp::Create(std::move(proto)));

  EXPECT_EQ(stamp.full_source_hash(), "ts:1700000000.000012345");
  EXPECT_THAT(stamp.source_hash(), IsEmpty());
  EXPECT_THAT(stamp.ToProto().source_hash(), IsEmpty());
}

TEST(PjRtBuildStampTest,
     FullSourceHashIncludesComponentsWhenFallingBackToVcsRevision) {
  PjRtBuildStampProto base_proto;
  base_proto.set_vcs_revision("git_rev_456");

  PjRtBuildStampProto with_comp = base_proto;
  auto* comp = with_comp.add_components();
  comp->set_kind("custom_call");
  comp->set_name("op_a");
  comp->set_owner("pkg:op_a");
  comp->set_hash("hash_a");

  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp base_stamp,
                       PjRtBuildStamp::Create(std::move(base_proto)));
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp comp_stamp,
                       PjRtBuildStamp::Create(std::move(with_comp)));

  EXPECT_EQ(base_stamp.full_source_hash(), "vcs:git_rev_456");
  EXPECT_NE(comp_stamp.full_source_hash(), "vcs:git_rev_456");
  EXPECT_NE(comp_stamp.full_source_hash(), base_stamp.full_source_hash());
  EXPECT_THAT(base_stamp.source_hash(), IsEmpty());
  EXPECT_THAT(base_stamp.ToProto().source_hash(), IsEmpty());
  EXPECT_THAT(comp_stamp.source_hash(), IsEmpty());
  EXPECT_THAT(comp_stamp.ToProto().source_hash(), IsEmpty());
}

TEST(PjRtBuildStampTest,
     FullSourceHashIncludesVersionsWhenFallingBackToTimestamp) {
  PjRtBuildStampProto base_proto;
  base_proto.mutable_build_timestamp()->set_seconds(1700000000);
  base_proto.mutable_build_timestamp()->set_nanos(12345);

  PjRtBuildStampProto with_ver = base_proto;
  SetMosaicVersion(with_ver, 1, 0);

  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp base_stamp,
                       PjRtBuildStamp::Create(std::move(base_proto)));
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp ver_stamp,
                       PjRtBuildStamp::Create(std::move(with_ver)));

  EXPECT_EQ(base_stamp.full_source_hash(), "ts:1700000000.000012345");
  EXPECT_NE(ver_stamp.full_source_hash(), "ts:1700000000.000012345");
  EXPECT_NE(ver_stamp.full_source_hash(), base_stamp.full_source_hash());
  EXPECT_THAT(base_stamp.source_hash(), IsEmpty());
  EXPECT_THAT(base_stamp.ToProto().source_hash(), IsEmpty());
  EXPECT_THAT(ver_stamp.source_hash(), IsEmpty());
  EXPECT_THAT(ver_stamp.ToProto().source_hash(), IsEmpty());
}

TEST(PjRtBuildStampTest, AddingComponentChangesFullSourceHash) {
  PjRtBuildStampProto proto1;
  proto1.set_source_hash("hash1");
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp1, PjRtBuildStamp::Create(proto1));

  PjRtBuildStampProto proto2 = proto1;
  auto* comp = proto2.add_components();
  comp->set_kind("custom_call");
  comp->set_name("op_a");
  comp->set_owner("pkg:op_a");
  comp->set_hash("hash_a");
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp2,
                       PjRtBuildStamp::Create(std::move(proto2)));

  EXPECT_NE(stamp1.full_source_hash(), stamp2.full_source_hash());
}

TEST(PjRtBuildStampTest,
     DifferentComponentHashProducesDifferentFullSourceHash) {
  PjRtBuildStampProto proto1;
  proto1.set_source_hash("base_hash");
  auto* c1 = proto1.add_components();
  c1->set_kind("xla.gpu.native_custom_call");
  c1->set_name("alpha");
  c1->set_owner("pkg:alpha");
  c1->set_hash("1111");
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp1,
                       PjRtBuildStamp::Create(std::move(proto1)));

  PjRtBuildStampProto proto2;
  proto2.set_source_hash("base_hash");
  auto* c2 = proto2.add_components();
  c2->set_kind("xla.gpu.native_custom_call");
  c2->set_name("alpha");
  c2->set_owner("pkg:alpha");
  c2->set_hash("2222");
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp2,
                       PjRtBuildStamp::Create(std::move(proto2)));

  EXPECT_NE(stamp1.full_source_hash(), stamp2.full_source_hash());
}

TEST(PjRtBuildStampTest,
     FullSourceHashIsInvariantToComponentOrderAndDuplicates) {
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
  *proto1.add_components() = c1;
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp1,
                       PjRtBuildStamp::Create(std::move(proto1)));

  PjRtBuildStampProto proto2;
  proto2.set_source_hash("base_hash");
  *proto2.add_components() = c1;
  *proto2.add_components() = c2;
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp2,
                       PjRtBuildStamp::Create(std::move(proto2)));

  EXPECT_EQ(stamp1.full_source_hash(), stamp2.full_source_hash());
}

TEST(PjRtBuildStampTest, DifferentVersionProducesDifferentFullSourceHash) {
  PjRtBuildStampProto proto1;
  proto1.set_source_hash("base_hash");
  SetMosaicVersion(proto1, 12, 0);
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp1,
                       PjRtBuildStamp::Create(std::move(proto1)));

  PjRtBuildStampProto proto2;
  proto2.set_source_hash("base_hash");
  SetMosaicVersion(proto2, 13, 0);
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp2,
                       PjRtBuildStamp::Create(std::move(proto2)));

  EXPECT_NE(stamp1.full_source_hash(), stamp2.full_source_hash());
}

TEST(PjRtBuildStampTest, FullSourceHashIsInvariantToVersionOrderAndDuplicates) {
  PjRtVersionProto v1;
  v1.set_name("mosaic");
  v1.set_major_version(12);
  v1.set_minor_version(1);
  PjRtVersionProto v2;
  v2.set_name("tpu_runtime");
  v2.set_major_version(3);
  v2.set_minor_version(0);

  PjRtBuildStampProto proto1;
  proto1.set_source_hash("base_hash");
  *proto1.add_versions() = v1;
  *proto1.add_versions() = v2;
  *proto1.add_versions() = v1;
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp1,
                       PjRtBuildStamp::Create(std::move(proto1)));

  PjRtBuildStampProto proto2;
  proto2.set_source_hash("base_hash");
  *proto2.add_versions() = v2;
  *proto2.add_versions() = v1;
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp2,
                       PjRtBuildStamp::Create(std::move(proto2)));

  EXPECT_EQ(stamp1.full_source_hash(), stamp2.full_source_hash());
}

TEST(PjRtBuildStampTest, AttributesDoNotAffectFullSourceHash) {
  PjRtBuildStampProto base_proto;
  base_proto.set_source_hash("hash1");
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp base_stamp,
                       PjRtBuildStamp::Create(base_proto));

  PjRtBuildStampProto attr_proto = base_proto;
  (*attr_proto.mutable_attributes())["vendor.candidate"] =
      PjRtValueTypeToProto(std::string("xla_tpu_candidate_1"));
  (*attr_proto.mutable_attributes())["is_experimental"] =
      PjRtValueTypeToProto(true);
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp attr_stamp,
                       PjRtBuildStamp::Create(std::move(attr_proto)));

  EXPECT_EQ(base_stamp.full_source_hash(), attr_stamp.full_source_hash());
}

TEST(PjRtBuildStampTest, ProvenanceMetadataDoesNotAffectFullSourceHash) {
  PjRtBuildStampProto proto1;
  proto1.set_source_hash("hash1");
  proto1.set_build_label("label_a");
  proto1.set_vcs_revision("rev_a");
  proto1.mutable_build_timestamp()->set_seconds(100);
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp1,
                       PjRtBuildStamp::Create(std::move(proto1)));

  PjRtBuildStampProto proto2;
  proto2.set_source_hash("hash1");
  proto2.set_build_label("label_b");
  proto2.set_vcs_revision("rev_b");
  proto2.mutable_build_timestamp()->set_seconds(200);
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp2,
                       PjRtBuildStamp::Create(std::move(proto2)));

  EXPECT_EQ(stamp1.full_source_hash(), stamp2.full_source_hash());
}

TEST(PjRtBuildStampTest, GetVersionReturnsMatchingVersion) {
  PjRtVersionProto expected;
  expected.set_name("tpu_runtime");
  expected.set_major_version(3);
  expected.set_minor_version(2);

  PjRtBuildStampProto proto;
  proto.set_source_hash("hash1");
  *proto.add_versions() = expected;
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       PjRtBuildStamp::Create(std::move(proto)));

  EXPECT_THAT(stamp.GetVersion("tpu_runtime"),
              IsOkAndHolds(EqualsProto(expected)));
}

TEST(PjRtBuildStampTest, GetVersionForMissingNameReturnsNotFound) {
  PjRtBuildStampProto proto;
  proto.set_source_hash("hash1");
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       PjRtBuildStamp::Create(std::move(proto)));

  EXPECT_THAT(stamp.GetVersion("non_existent"),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr("Version 'non_existent' not found.")));
}

TEST(PjRtBuildStampTest, SetAndGetMosaicVersion) {
  PjRtBuildStampProto proto;
  proto.set_source_hash("hash1");
  SetMosaicVersion(proto, 12, 3);
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       PjRtBuildStamp::Create(std::move(proto)));

  PjRtVersionProto expected;
  expected.set_name("mosaic");
  expected.set_major_version(12);
  expected.set_minor_version(3);
  EXPECT_THAT(GetMosaicVersion(stamp), IsOkAndHolds(EqualsProto(expected)));
}

TEST(PjRtBuildStampTest, SetMosaicVersionOverwritesExistingMosaicVersion) {
  PjRtBuildStampProto proto;
  proto.set_source_hash("hash1");
  SetMosaicVersion(proto, 12, 0);
  SetMosaicVersion(proto, 13, 1);
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       PjRtBuildStamp::Create(std::move(proto)));

  PjRtVersionProto expected;
  expected.set_name("mosaic");
  expected.set_major_version(13);
  expected.set_minor_version(1);
  EXPECT_THAT(GetMosaicVersion(stamp), IsOkAndHolds(EqualsProto(expected)));
}

TEST(PjRtBuildStampTest, GetMosaicVersionWhenUnsetReturnsNotFound) {
  PjRtBuildStampProto proto;
  proto.set_source_hash("hash1");
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       PjRtBuildStamp::Create(std::move(proto)));

  EXPECT_THAT(GetMosaicVersion(stamp),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr("Version 'mosaic' not found.")));
}

TEST(PjRtBuildStampTest, AttributesAccessorReturnsPopulatedMap) {
  PjRtBuildStampProto proto;
  proto.set_source_hash("hash1");
  (*proto.mutable_attributes())["is_experimental"] = PjRtValueTypeToProto(true);
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       PjRtBuildStamp::Create(std::move(proto)));

  ASSERT_EQ(stamp.attributes().size(), 1);
  EXPECT_TRUE(stamp.attributes().at("is_experimental").bool_value());
}

TEST(PjRtBuildStampTest, GetAttributeReturnsTypedValues) {
  PjRtBuildStampProto proto;
  proto.set_source_hash("hash1");
  (*proto.mutable_attributes())["str_attr"] =
      PjRtValueTypeToProto(std::string("xla_tpu_candidate_1"));
  (*proto.mutable_attributes())["bool_attr"] = PjRtValueTypeToProto(true);
  (*proto.mutable_attributes())["int_attr"] = PjRtValueTypeToProto(int64_t{42});
  (*proto.mutable_attributes())["float_attr"] =
      PjRtValueTypeToProto(float{3.5f});
  (*proto.mutable_attributes())["int_vec_attr"] =
      PjRtValueTypeToProto(std::vector<int64_t>{1, 2, 3});
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       PjRtBuildStamp::Create(std::move(proto)));

  EXPECT_THAT(stamp.GetAttribute<std::string>("str_attr"),
              IsOkAndHolds("xla_tpu_candidate_1"));
  EXPECT_THAT(stamp.GetAttribute<bool>("bool_attr"), IsOkAndHolds(true));
  EXPECT_THAT(stamp.GetAttribute<int64_t>("int_attr"),
              IsOkAndHolds(int64_t{42}));
  EXPECT_THAT(stamp.GetAttribute<float>("float_attr"),
              IsOkAndHolds(float{3.5f}));
  EXPECT_THAT(stamp.GetAttribute<std::vector<int64_t>>("int_vec_attr"),
              IsOkAndHolds(ElementsAre(1, 2, 3)));
}

TEST(PjRtBuildStampTest, GetAttributeForMissingKeyReturnsNotFound) {
  PjRtBuildStampProto proto;
  proto.set_source_hash("hash1");
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       PjRtBuildStamp::Create(std::move(proto)));

  EXPECT_THAT(stamp.GetAttribute<int64_t>("non_existent"),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr("Attribute 'non_existent' not found.")));
}

TEST(PjRtBuildStampTest, GetAttributeWithWrongTypeReturnsInvalidArgument) {
  PjRtBuildStampProto proto;
  proto.set_source_hash("hash1");
  (*proto.mutable_attributes())["bool_attr"] = PjRtValueTypeToProto(true);
  (*proto.mutable_attributes())["float_attr"] =
      PjRtValueTypeToProto(float{42.5});
  (*proto.mutable_attributes())["int_attr"] = PjRtValueTypeToProto(int64_t{42});
  (*proto.mutable_attributes())["int_vector_attr"] =
      PjRtValueTypeToProto(std::vector<int64_t>{1, 2, 3});
  (*proto.mutable_attributes())["string_attr"] =
      PjRtValueTypeToProto(std::string("xla_tpu_candidate_1"));
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       PjRtBuildStamp::Create(std::move(proto)));

  EXPECT_THAT(
      stamp.GetAttribute<std::string>("bool_attr"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Attribute 'bool_attr' has type bool, not the requested "
                    "type \"string\".")));
  EXPECT_THAT(
      stamp.GetAttribute<bool>("float_attr"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Attribute 'float_attr' has type float, not the requested "
                    "type \"bool\".")));
  EXPECT_THAT(
      stamp.GetAttribute<float>("int_attr"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Attribute 'int_attr' has type int64_t, not the requested "
                    "type \"float\".")));
  EXPECT_THAT(stamp.GetAttribute<bool>("int_vector_attr"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Attribute 'int_vector_attr' has type "
                                 "std::vector<int64_t>, not the requested type "
                                 "\"bool\".")));
  EXPECT_THAT(stamp.GetAttribute<int64_t>("string_attr"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Attribute 'string_attr' has type string, not "
                                 "the requested type \"int64_t\".")));
}

TEST(PjRtBuildStampTest, GetStringAttributeReturnsValue) {
  PjRtBuildStampProto proto;
  proto.set_source_hash("hash1");
  (*proto.mutable_attributes())["vendor.candidate"] =
      PjRtValueTypeToProto(std::string("xla_tpu_candidate_1"));
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       PjRtBuildStamp::Create(std::move(proto)));

  EXPECT_THAT(stamp.GetStringAttribute("vendor.candidate"),
              IsOkAndHolds("xla_tpu_candidate_1"));
}

TEST(PjRtBuildStampTest, GetStringAttributeForMissingKeyReturnsNotFound) {
  PjRtBuildStampProto proto;
  proto.set_source_hash("hash1");
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       PjRtBuildStamp::Create(std::move(proto)));

  EXPECT_THAT(stamp.GetStringAttribute("non_existent"),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr("Attribute 'non_existent' not found.")));
}

TEST(PjRtBuildStampTest,
     GetStringAttributeWithWrongTypeReturnsInvalidArgument) {
  PjRtBuildStampProto proto;
  proto.set_source_hash("hash1");
  (*proto.mutable_attributes())["bool_attr"] = PjRtValueTypeToProto(true);
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       PjRtBuildStamp::Create(std::move(proto)));

  EXPECT_THAT(
      stamp.GetStringAttribute("bool_attr"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Attribute 'bool_attr' has type bool, not the requested "
                    "type \"string\".")));
}

TEST(PjRtVersionProtoTest, EqualityHelper) {
  PjRtVersionProto base;
  base.set_name("mosaic");
  base.set_major_version(1);
  base.set_minor_version(2);

  PjRtVersionProto same = base;

  PjRtVersionProto diff_name = base;
  diff_name.set_name("not_mosaic");

  PjRtVersionProto diff_major = base;
  diff_major.set_major_version(2);

  PjRtVersionProto diff_minor = base;
  diff_minor.set_minor_version(3);

  EXPECT_TRUE(PjRtVersionProtoEquals(base, same));
  EXPECT_FALSE(PjRtVersionProtoEquals(base, diff_name));
  EXPECT_FALSE(PjRtVersionProtoEquals(base, diff_major));
  EXPECT_FALSE(PjRtVersionProtoEquals(base, diff_minor));
}

TEST(PjRtVersionProtoTest, MatchesProtoFieldCount) {
  EXPECT_EQ(PjRtVersionProto::descriptor()->field_count(), 3)
      << "PjRtVersionProto changed! A field was added or removed to the proto. "
         "Update ValidateVersions(), PjRtVersionProtoEquals(), "
         "full_source_hash calculation in Create(), and this test.";
}

TEST(PjRtComponentProtoTest, MatchesProtoFieldCount) {
  EXPECT_EQ(PjRtComponentBuildStampProto::descriptor()->field_count(), 4)
      << "PjRtComponentBuildStampProto changed! A field was added or removed "
         "to the proto. Update ValidateComponents(), full_source_hash "
         "calculation in Create(), and this test.";
}

}  // namespace
}  // namespace xla
