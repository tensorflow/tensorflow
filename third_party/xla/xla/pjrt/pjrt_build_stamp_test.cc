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
#include <type_traits>
#include <utility>
#include <vector>

#include "google/protobuf/timestamp.pb.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/pjrt/pjrt_build_stamp_writer.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_compiler_variant.h"
#include "xla/pjrt/proto/pjrt_build_stamp.pb.h"
#include "xla/pjrt/proto/pjrt_value_type.pb.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::tsl::proto_testing::EqualsProto;

// Fails if a member is `const`, since PjRtBuildStamp must be assignable e.g.
// into an existing `absl::StatusOr<PjRtBuildStamp>`.
static_assert(std::is_move_assignable_v<PjRtBuildStamp>);
// Fails if moves silently fall back to copies, e.g. a `const` member or a
// user-declared destructor/copy constructor).
static_assert(std::is_nothrow_move_constructible_v<PjRtBuildStamp>);

void AddVersion(PjRtBuildStampProto& proto, absl::string_view name,
                int64_t major_version, int64_t minor_version) {
  PjRtCompatibilityVersionProto* version = proto.add_compatibility_versions();
  version->set_name(name);
  version->set_major_version(major_version);
  version->set_minor_version(minor_version);
}

constexpr PjRtCompilerVariant kCustomVariantId = 12345;
constexpr PjRtCompilerVariant kVariantAId = 67890;
constexpr PjRtCompilerVariant kForgeVariantId = 11111;
constexpr PjRtCompilerVariant kUnknownVariantId = 99999;

TEST(PjRtBuildStampTest, BasicAccessors) {
  PjRtBuildStampProto proto;
  proto.set_build_label("target_foo");
  proto.mutable_build_timestamp()->set_seconds(1700000000);
  proto.set_vcs_revision("git_rev_123");
  proto.set_source_hash("hash_abcdef01");

  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       PjRtBuildStamp::FromProto(std::move(proto)));

  EXPECT_EQ(stamp.build_label(), "target_foo");
  EXPECT_EQ(stamp.build_timestamp(), absl::FromUnixSeconds(1700000000));
  EXPECT_EQ(stamp.vcs_revision(), "git_rev_123");
  EXPECT_EQ(stamp.source_hash(), "hash_abcdef01");
}

TEST(PjRtBuildStampTest, DefaultBuildTimestampIsInfinitePast) {
  PjRtBuildStampProto proto;
  proto.set_source_hash("hash_no_ts");

  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       PjRtBuildStamp::FromProto(std::move(proto)));

  EXPECT_EQ(stamp.build_timestamp(), absl::InfinitePast());
}

TEST(PjRtBuildStampTest, BuildTimestampIncludesNanos) {
  PjRtBuildStampProto proto;
  proto.mutable_build_timestamp()->set_seconds(1700000000);
  proto.mutable_build_timestamp()->set_nanos(123456789);

  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       PjRtBuildStamp::FromProto(std::move(proto)));

  EXPECT_EQ(stamp.build_timestamp(),
            absl::FromUnixSeconds(1700000000) + absl::Nanoseconds(123456789));
}

TEST(PjRtBuildStampTest, ToProtoRoundTrip) {
  PjRtBuildStampProto proto;
  proto.set_build_label("label1");
  proto.mutable_build_timestamp()->set_seconds(1700000000);
  proto.mutable_build_timestamp()->set_nanos(123);
  proto.set_vcs_revision("rev1");
  proto.set_source_hash("hash1");
  auto* comp = proto.add_components();
  comp->set_kind("xla.ffi.handler");
  comp->set_name("custom_op");
  comp->set_owner("pkg:custom_op");
  comp->set_hash("comp_hash_1");
  ASSERT_OK(
      SetAttribute(proto, "vendor.candidate", std::string("candidate_1")));
  AddVersion(proto, "foo", 7, 1);

  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp1,
                       PjRtBuildStamp::FromProto(std::move(proto)));
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp2,
                       PjRtBuildStamp::FromProto(stamp1.ToProto()));

  EXPECT_THAT(stamp2.ToProto(), EqualsProto(stamp1.ToProto()));
  EXPECT_EQ(stamp2.full_source_hash(), stamp1.full_source_hash());
}

TEST(PjRtBuildStampTest, FromProtoFailsWhenMissingBuildIdentity) {
  EXPECT_THAT(PjRtBuildStamp::FromProto(PjRtBuildStampProto{}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("at least one build identity field")));
}

TEST(PjRtBuildStampTest, FromProtoFailsWhenVcsRevisionIsUnknownSentinel) {
  PjRtBuildStampProto proto;
  proto.set_vcs_revision("<unknown>");

  EXPECT_THAT(PjRtBuildStamp::FromProto(std::move(proto)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("vcs_revision cannot be the '<unknown>'")));
}

TEST(PjRtBuildStampTest, FromProtoFailsWhenBuildTimestampNanosOutOfRange) {
  PjRtBuildStampProto neg_nanos;
  neg_nanos.mutable_build_timestamp()->set_seconds(100);
  neg_nanos.mutable_build_timestamp()->set_nanos(-1);
  EXPECT_THAT(PjRtBuildStamp::FromProto(std::move(neg_nanos)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("build_timestamp.nanos must be in")));

  PjRtBuildStampProto overflow_nanos;
  overflow_nanos.mutable_build_timestamp()->set_seconds(100);
  overflow_nanos.mutable_build_timestamp()->set_nanos(1000000000);
  EXPECT_THAT(PjRtBuildStamp::FromProto(std::move(overflow_nanos)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("build_timestamp.nanos must be in")));
}

TEST(PjRtBuildStampTest, FromProtoFailsWhenBuildTimestampSecondsIsNegative) {
  PjRtBuildStampProto neg_seconds;
  neg_seconds.mutable_build_timestamp()->set_seconds(-1);
  neg_seconds.mutable_build_timestamp()->set_nanos(100);
  EXPECT_THAT(
      PjRtBuildStamp::FromProto(std::move(neg_seconds)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("build_timestamp.seconds must be non-negative")));
}

TEST(PjRtBuildStampTest, FromProtoFailsWhenComponentHasEmptyName) {
  PjRtBuildStampProto proto;
  proto.set_source_hash("hash");
  proto.add_components()->set_hash("fp");

  EXPECT_THAT(
      PjRtBuildStamp::FromProto(std::move(proto)),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Component build stamp name and hash must not be empty")));
}

TEST(PjRtBuildStampTest, FromProtoFailsWhenComponentHasEmptyHash) {
  PjRtBuildStampProto proto;
  proto.set_source_hash("hash");
  proto.add_components()->set_name("comp");

  EXPECT_THAT(
      PjRtBuildStamp::FromProto(std::move(proto)),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Component build stamp name and hash must not be empty")));
}

TEST(PjRtBuildStampTest,
     FromProtoFailsWhenDuplicateComponentHasConflictingHashes) {
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
      PjRtBuildStamp::FromProto(std::move(proto)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Duplicate component found with conflicting hashes")));
}

TEST(PjRtBuildStampTest, FromProtoSortsAndDeduplicatesComponents) {
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
                       PjRtBuildStamp::FromProto(std::move(proto)));

  EXPECT_THAT(stamp.ToProto().components(),
              ElementsAre(EqualsProto(c2), EqualsProto(c1)));
}

TEST(PjRtBuildStampTest, FromProtoFailsWhenVersionHasEmptyName) {
  PjRtBuildStampProto proto;
  proto.set_source_hash("base_hash");
  auto* v = proto.add_compatibility_versions();
  v->set_name("");
  v->set_major_version(1);

  EXPECT_THAT(PjRtBuildStamp::FromProto(std::move(proto)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Version name must not be empty")));
}

TEST(PjRtBuildStampTest, FromProtoFailsWhenMajorVersionIsNegative) {
  PjRtBuildStampProto proto;
  proto.set_source_hash("base_hash");
  auto* v = proto.add_compatibility_versions();
  v->set_name("mosaic");
  v->set_major_version(-1);

  EXPECT_THAT(PjRtBuildStamp::FromProto(std::move(proto)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Version numbers must be non-negative")));
}

TEST(PjRtBuildStampTest, FromProtoFailsWhenMinorVersionIsNegative) {
  PjRtBuildStampProto proto;
  proto.set_source_hash("base_hash");
  auto* v = proto.add_compatibility_versions();
  v->set_name("mosaic");
  v->set_major_version(1);
  v->set_minor_version(-1);

  EXPECT_THAT(PjRtBuildStamp::FromProto(std::move(proto)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Version numbers must be non-negative")));
}

TEST(PjRtBuildStampTest,
     FromProtoFailsWhenDuplicateVersionHasConflictingMajorVersion) {
  PjRtBuildStampProto proto;
  proto.set_source_hash("base_hash");
  auto* v1 = proto.add_compatibility_versions();
  v1->set_name("mosaic");
  v1->set_major_version(12);
  auto* v2 = proto.add_compatibility_versions();
  v2->set_name("mosaic");
  v2->set_major_version(13);

  EXPECT_THAT(
      PjRtBuildStamp::FromProto(std::move(proto)),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "Duplicate version found with conflicting values for 'mosaic'")));
}

TEST(PjRtBuildStampTest,
     FromProtoFailsWhenDuplicateVersionHasConflictingMinorVersion) {
  PjRtBuildStampProto proto;
  proto.set_source_hash("base_hash");
  auto* v1 = proto.add_compatibility_versions();
  v1->set_name("mosaic");
  v1->set_major_version(12);
  v1->set_minor_version(0);
  auto* v2 = proto.add_compatibility_versions();
  v2->set_name("mosaic");
  v2->set_major_version(12);
  v2->set_minor_version(1);

  EXPECT_THAT(
      PjRtBuildStamp::FromProto(std::move(proto)),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "Duplicate version found with conflicting values for 'mosaic'")));
}

TEST(PjRtBuildStampTest, FromProtoSortsAndDeduplicatesVersions) {
  PjRtCompatibilityVersionProto v1;
  v1.set_name("mosaic");
  v1.set_major_version(12);
  v1.set_minor_version(1);
  PjRtCompatibilityVersionProto v2;
  v2.set_name("tpu_runtime");
  v2.set_major_version(3);
  v2.set_minor_version(0);

  PjRtBuildStampProto proto;
  proto.set_source_hash("base_hash");
  *proto.add_compatibility_versions() = v2;
  *proto.add_compatibility_versions() = v1;
  *proto.add_compatibility_versions() = v2;

  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       PjRtBuildStamp::FromProto(std::move(proto)));

  EXPECT_THAT(stamp.ToProto().compatibility_versions(),
              ElementsAre(EqualsProto(v1), EqualsProto(v2)));
}

TEST(PjRtBuildStampTest, FromProtoFailsWhenAttributeKeyIsEmpty) {
  PjRtBuildStampProto proto;
  proto.set_source_hash("base_hash");
  (*proto.mutable_attributes())[""] = PjRtValueTypeToProto(true);

  EXPECT_THAT(PjRtBuildStamp::FromProto(std::move(proto)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Attribute key must not be empty")));
}

TEST(PjRtBuildStampTest, FromProtoFailsWhenAttributeValueIsNotSet) {
  PjRtBuildStampProto proto;
  proto.set_source_hash("base_hash");
  (*proto.mutable_attributes())["empty_attr"] = PjRtValueTypeProto{};

  EXPECT_THAT(
      PjRtBuildStamp::FromProto(std::move(proto)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Attribute value for key='empty_attr' must be set")));
}

TEST(PjRtBuildStampTest,
     FullSourceHashEqualsSourceHashWhenNoComponentsOrVersions) {
  PjRtBuildStampProto proto;
  proto.set_source_hash("hash1");

  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       PjRtBuildStamp::FromProto(std::move(proto)));

  EXPECT_EQ(stamp.full_source_hash(), "hash1");
}

TEST(PjRtBuildStampTest, FullSourceHashFallsBackToVcsRevision) {
  PjRtBuildStampProto proto;
  proto.set_vcs_revision("git_rev_456");
  proto.mutable_build_timestamp()->set_seconds(1700000000);

  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       PjRtBuildStamp::FromProto(std::move(proto)));

  EXPECT_EQ(stamp.full_source_hash(), "vcs:git_rev_456");
  EXPECT_THAT(stamp.source_hash(), IsEmpty());
  EXPECT_THAT(stamp.ToProto().source_hash(), IsEmpty());
}

TEST(PjRtBuildStampTest, FullSourceHashFallsBackToTimestamp) {
  PjRtBuildStampProto proto;
  proto.mutable_build_timestamp()->set_seconds(1700000000);
  proto.mutable_build_timestamp()->set_nanos(12345);

  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       PjRtBuildStamp::FromProto(std::move(proto)));

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
                       PjRtBuildStamp::FromProto(std::move(base_proto)));
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp comp_stamp,
                       PjRtBuildStamp::FromProto(std::move(with_comp)));

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
  AddVersion(with_ver, "foo", 1, 0);

  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp base_stamp,
                       PjRtBuildStamp::FromProto(std::move(base_proto)));
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp ver_stamp,
                       PjRtBuildStamp::FromProto(std::move(with_ver)));

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
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp1,
                       PjRtBuildStamp::FromProto(proto1));

  PjRtBuildStampProto proto2 = proto1;
  auto* comp = proto2.add_components();
  comp->set_kind("custom_call");
  comp->set_name("op_a");
  comp->set_owner("pkg:op_a");
  comp->set_hash("hash_a");
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp2,
                       PjRtBuildStamp::FromProto(std::move(proto2)));

  EXPECT_NE(stamp1.full_source_hash(), stamp2.full_source_hash());
}

TEST(PjRtBuildStampTest,
     FullSourceHashDistinguishesSourceAndComponentBoundaries) {
  PjRtBuildStampProto proto1;
  proto1.set_source_hash("ab");
  auto* c1 = proto1.add_components();
  c1->set_name("comp1");
  c1->set_hash("cd");
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp1,
                       PjRtBuildStamp::FromProto(std::move(proto1)));

  PjRtBuildStampProto proto2;
  proto2.set_source_hash("abc");
  auto* c2 = proto2.add_components();
  c2->set_name("comp1");
  c2->set_hash("d");
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp2,
                       PjRtBuildStamp::FromProto(std::move(proto2)));

  EXPECT_NE(stamp1.full_source_hash(), stamp2.full_source_hash());

  PjRtBuildStampProto two_comps;
  two_comps.set_source_hash("base");
  auto* tc1 = two_comps.add_components();
  tc1->set_name("comp1");
  tc1->set_hash("ab");
  auto* tc2 = two_comps.add_components();
  tc2->set_name("comp2");
  tc2->set_hash("cd");
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp two_comps_stamp,
                       PjRtBuildStamp::FromProto(std::move(two_comps)));

  PjRtBuildStampProto one_comp;
  one_comp.set_source_hash("base");
  auto* oc1 = one_comp.add_components();
  oc1->set_name("comp1");
  oc1->set_hash("abcd");
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp one_comp_stamp,
                       PjRtBuildStamp::FromProto(std::move(one_comp)));

  EXPECT_NE(two_comps_stamp.full_source_hash(),
            one_comp_stamp.full_source_hash());
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
                       PjRtBuildStamp::FromProto(std::move(proto1)));

  PjRtBuildStampProto proto2;
  proto2.set_source_hash("base_hash");
  auto* c2 = proto2.add_components();
  c2->set_kind("xla.gpu.native_custom_call");
  c2->set_name("alpha");
  c2->set_owner("pkg:alpha");
  c2->set_hash("2222");
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp2,
                       PjRtBuildStamp::FromProto(std::move(proto2)));

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
                       PjRtBuildStamp::FromProto(std::move(proto1)));

  PjRtBuildStampProto proto2;
  proto2.set_source_hash("base_hash");
  *proto2.add_components() = c1;
  *proto2.add_components() = c2;
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp2,
                       PjRtBuildStamp::FromProto(std::move(proto2)));

  EXPECT_EQ(stamp1.full_source_hash(), stamp2.full_source_hash());
}

TEST(PjRtBuildStampTest, DifferentVersionProducesDifferentFullSourceHash) {
  PjRtBuildStampProto proto1;
  proto1.set_source_hash("base_hash");
  AddVersion(proto1, "foo", 12, 0);
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp1,
                       PjRtBuildStamp::FromProto(std::move(proto1)));

  PjRtBuildStampProto proto2;
  proto2.set_source_hash("base_hash");
  AddVersion(proto2, "foo", 13, 0);
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp2,
                       PjRtBuildStamp::FromProto(std::move(proto2)));

  EXPECT_NE(stamp1.full_source_hash(), stamp2.full_source_hash());
}

TEST(PjRtBuildStampTest, FullSourceHashIsInvariantToVersionOrderAndDuplicates) {
  PjRtCompatibilityVersionProto v1;
  v1.set_name("mosaic");
  v1.set_major_version(12);
  v1.set_minor_version(1);
  PjRtCompatibilityVersionProto v2;
  v2.set_name("tpu_runtime");
  v2.set_major_version(3);
  v2.set_minor_version(0);

  PjRtBuildStampProto proto1;
  proto1.set_source_hash("base_hash");
  *proto1.add_compatibility_versions() = v1;
  *proto1.add_compatibility_versions() = v2;
  *proto1.add_compatibility_versions() = v1;
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp1,
                       PjRtBuildStamp::FromProto(std::move(proto1)));

  PjRtBuildStampProto proto2;
  proto2.set_source_hash("base_hash");
  *proto2.add_compatibility_versions() = v2;
  *proto2.add_compatibility_versions() = v1;
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp2,
                       PjRtBuildStamp::FromProto(std::move(proto2)));

  EXPECT_EQ(stamp1.full_source_hash(), stamp2.full_source_hash());
}

TEST(PjRtBuildStampTest, AttributesDoNotAffectFullSourceHash) {
  PjRtBuildStampProto base_proto;
  base_proto.set_source_hash("hash1");
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp base_stamp,
                       PjRtBuildStamp::FromProto(base_proto));

  PjRtBuildStampProto attr_proto = base_proto;
  ASSERT_OK(SetAttribute(attr_proto, "vendor.candidate",
                         std::string("xla_tpu_candidate_1")));
  ASSERT_OK(SetAttribute(attr_proto, "is_experimental", true));
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp attr_stamp,
                       PjRtBuildStamp::FromProto(std::move(attr_proto)));

  EXPECT_EQ(base_stamp.full_source_hash(), attr_stamp.full_source_hash());
}

TEST(PjRtBuildStampTest, ProvenanceMetadataDoesNotAffectFullSourceHash) {
  PjRtBuildStampProto proto1;
  proto1.set_source_hash("hash1");
  proto1.set_build_label("label_a");
  proto1.set_vcs_revision("rev_a");
  proto1.mutable_build_timestamp()->set_seconds(100);
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp1,
                       PjRtBuildStamp::FromProto(std::move(proto1)));

  PjRtBuildStampProto proto2;
  proto2.set_source_hash("hash1");
  proto2.set_build_label("label_b");
  proto2.set_vcs_revision("rev_b");
  proto2.mutable_build_timestamp()->set_seconds(200);
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp2,
                       PjRtBuildStamp::FromProto(std::move(proto2)));

  EXPECT_EQ(stamp1.full_source_hash(), stamp2.full_source_hash());
}

TEST(PjRtBuildStampTest, GetVersionReturnsMatchingVersion) {
  PjRtCompatibilityVersionProto expected;
  expected.set_name("tpu_runtime");
  expected.set_major_version(3);
  expected.set_minor_version(2);

  PjRtBuildStampProto proto;
  proto.set_source_hash("hash1");
  *proto.add_compatibility_versions() = expected;
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       PjRtBuildStamp::FromProto(std::move(proto)));

  EXPECT_THAT(stamp.GetVersion("tpu_runtime"),
              IsOkAndHolds(EqualsProto(expected)));
}

TEST(PjRtBuildStampTest, GetVersionForMissingNameReturnsNotFound) {
  PjRtBuildStampProto proto;
  proto.set_source_hash("hash1");
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       PjRtBuildStamp::FromProto(std::move(proto)));

  EXPECT_THAT(stamp.GetVersion("non_existent"),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr("Version 'non_existent' not found.")));
}

TEST(PjRtBuildStampTest, ToProtoRetainsAttributes) {
  PjRtBuildStampProto proto;
  proto.set_source_hash("hash1");
  ASSERT_OK(SetAttribute(proto, "is_experimental", true));
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       PjRtBuildStamp::FromProto(std::move(proto)));

  ASSERT_EQ(stamp.ToProto().attributes().size(), 1);
  EXPECT_TRUE(stamp.ToProto().attributes().at("is_experimental").bool_value());
}

TEST(PjRtBuildStampTest, SetAttributeFailsWhenKeyIsEmpty) {
  PjRtBuildStampProto proto;
  EXPECT_THAT(SetAttribute(proto, "", true),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Attribute key must not be empty")));
}

TEST(PjRtBuildStampTest, SetAttributeFailsWhenKeyAlreadyExists) {
  PjRtBuildStampProto proto;
  ASSERT_OK(SetAttribute(proto, "key1", true));
  EXPECT_THAT(SetAttribute(proto, "key1", false),
              StatusIs(absl::StatusCode::kAlreadyExists,
                       HasSubstr("Attribute 'key1' is already set")));
}

TEST(PjRtBuildStampTest, GetAttributeReturnsTypedValues) {
  PjRtBuildStampProto proto;
  proto.set_source_hash("hash1");
  ASSERT_OK(
      SetAttribute(proto, "str_attr", std::string("xla_tpu_candidate_1")));
  ASSERT_OK(SetAttribute(proto, "bool_attr", true));
  ASSERT_OK(SetAttribute(proto, "int_attr", int64_t{42}));
  ASSERT_OK(SetAttribute(proto, "float_attr", float{3.5f}));
  ASSERT_OK(SetAttribute(proto, "int_vec_attr", std::vector<int64_t>{1, 2, 3}));
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       PjRtBuildStamp::FromProto(std::move(proto)));

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
                       PjRtBuildStamp::FromProto(std::move(proto)));

  EXPECT_THAT(stamp.GetAttribute<int64_t>("non_existent"),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr("Attribute 'non_existent' not found.")));
}

TEST(PjRtBuildStampTest, GetAttributeWithWrongTypeReturnsInvalidArgument) {
  PjRtBuildStampProto proto;
  proto.set_source_hash("hash1");
  ASSERT_OK(SetAttribute(proto, "bool_attr", true));
  ASSERT_OK(SetAttribute(proto, "float_attr", float{42.5}));
  ASSERT_OK(SetAttribute(proto, "int_attr", int64_t{42}));
  ASSERT_OK(
      SetAttribute(proto, "int_vector_attr", std::vector<int64_t>{1, 2, 3}));
  ASSERT_OK(
      SetAttribute(proto, "string_attr", std::string("xla_tpu_candidate_1")));
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       PjRtBuildStamp::FromProto(std::move(proto)));

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
  ASSERT_OK(SetAttribute(proto, "vendor.candidate",
                         std::string("xla_tpu_candidate_1")));
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       PjRtBuildStamp::FromProto(std::move(proto)));

  EXPECT_THAT(stamp.GetStringAttribute("vendor.candidate"),
              IsOkAndHolds("xla_tpu_candidate_1"));
}

TEST(PjRtBuildStampTest, GetStringAttributeForMissingKeyReturnsNotFound) {
  PjRtBuildStampProto proto;
  proto.set_source_hash("hash1");
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       PjRtBuildStamp::FromProto(std::move(proto)));

  EXPECT_THAT(stamp.GetStringAttribute("non_existent"),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr("Attribute 'non_existent' not found.")));
}

TEST(PjRtBuildStampTest,
     GetStringAttributeWithWrongTypeReturnsInvalidArgument) {
  PjRtBuildStampProto proto;
  proto.set_source_hash("hash1");
  ASSERT_OK(SetAttribute(proto, "bool_attr", true));
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       PjRtBuildStamp::FromProto(std::move(proto)));

  EXPECT_THAT(
      stamp.GetStringAttribute("bool_attr"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Attribute 'bool_attr' has type bool, not the requested "
                    "type \"string\".")));
}

TEST(PjRtCompatibilityVersionProtoTest, MatchesProtoFieldCount) {
  EXPECT_EQ(PjRtCompatibilityVersionProto::descriptor()->field_count(), 3)
      << "PjRtCompatibilityVersionProto changed! A field was added or removed "
         "to the proto. Update ValidateVersions(), full_source_hash "
         "calculation in FromProto(), and this test.";
}

TEST(PjRtComponentProtoTest, MatchesProtoFieldCount) {
  EXPECT_EQ(PjRtComponentBuildStampProto::descriptor()->field_count(), 4)
      << "PjRtComponentBuildStampProto changed! A field was added or removed "
         "to the proto. Update ValidateComponents(), full_source_hash "
         "calculation in FromProto(), and this test.";
}

TEST(PjRtBuildStampRegistryTest, RegisterAndLookupCompiler) {
  PjRtBuildStampRegistry registry;
  int factory_calls = 0;
  auto factory = [&]() -> absl::StatusOr<PjRtBuildStamp> {
    factory_calls++;
    PjRtBuildStampProto proto;
    proto.set_source_hash("hash1");
    return PjRtBuildStamp::FromProto(std::move(proto));
  };

  ASSERT_OK(
      registry.RegisterBuildStampFactory(CudaId(), kCustomVariantId, factory));

  ASSERT_OK_AND_ASSIGN(const PjRtBuildStamp* absl_nonnull stamp,
                       registry.GetBuildStamp(CudaId(), kCustomVariantId));
  EXPECT_EQ(stamp->source_hash(), "hash1");
  EXPECT_EQ(factory_calls, 1);

  // Subsequent lookups must be memoized without re-invoking the factory.
  ASSERT_OK_AND_ASSIGN(const PjRtBuildStamp* absl_nonnull stamp2,
                       registry.GetBuildStamp(CudaId(), kCustomVariantId));
  EXPECT_EQ(stamp2, stamp);
  EXPECT_EQ(stamp2->source_hash(), "hash1");
  EXPECT_EQ(factory_calls, 1);
}

class PjRtBuildStampGlobalRegistryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    PjRtBuildStampRegistry::Global().ClearCacheForTesting();
  }

  void TearDown() override {
    PjRtBuildStampRegistry::Global().ClearCacheForTesting();
  }
};

TEST_F(PjRtBuildStampGlobalRegistryTest, PjRtRegisterAndLookupCompiler) {
  auto factory = []() -> absl::StatusOr<PjRtBuildStamp> {
    PjRtBuildStampProto proto;
    proto.set_source_hash("hash2");
    return PjRtBuildStamp::FromProto(std::move(proto));
  };

  PjRtRegisterBuildStampFactory(TpuId(), LinkedCompilerVariantId(), factory);
  ASSERT_OK_AND_ASSIGN(const PjRtBuildStamp* absl_nonnull stamp1,
                       PjRtGetBuildStamp(TpuId(), LinkedCompilerVariantId()));
  EXPECT_EQ(stamp1->source_hash(), "hash2");

  ASSERT_OK_AND_ASSIGN(const PjRtBuildStamp* absl_nonnull stamp2,
                       PjRtGetBuildStamp(TpuId()));
  EXPECT_EQ(stamp1, stamp2);
}

TEST_F(PjRtBuildStampGlobalRegistryTest, PjRtGetBuildStampFullSourceHash) {
  auto factory = []() -> absl::StatusOr<PjRtBuildStamp> {
    PjRtBuildStampProto proto;
    proto.set_source_hash("full_hash_test");
    return PjRtBuildStamp::FromProto(std::move(proto));
  };

  PjRtRegisterBuildStampFactory(CudaId(), kCustomVariantId, factory);
  ASSERT_OK_AND_ASSIGN(absl::string_view hash, PjRtGetBuildStampFullSourceHash(
                                                   CudaId(), kCustomVariantId));
  EXPECT_EQ(hash, "full_hash_test");
}

TEST_F(PjRtBuildStampGlobalRegistryTest,
       PjRtRegisterAndLookupDefaultVariantWithHelper) {
  auto factory = []() -> absl::StatusOr<PjRtBuildStamp> {
    PjRtBuildStampProto proto;
    proto.set_source_hash("id_hash");
    return PjRtBuildStamp::FromProto(std::move(proto));
  };

  PjRtRegisterBuildStampFactory(TpuId(), factory);
  ASSERT_OK_AND_ASSIGN(const PjRtBuildStamp* absl_nonnull stamp,
                       PjRtGetBuildStamp(TpuId()));
  EXPECT_EQ(stamp->source_hash(), "id_hash");
}

TEST(PjRtBuildStampRegistryTest, LookupNonExistentFails) {
  PjRtBuildStampRegistry registry;
  EXPECT_THAT(registry.GetBuildStamp(TpuId(), kUnknownVariantId),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(PjRtBuildStampGlobalRegistryTest, LookupNonExistentWithHelperFails) {
  EXPECT_THAT(PjRtGetBuildStamp(TpuId()),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(PjRtBuildStampGlobalRegistryTest,
       LookupNonExistentWithFullSourceHashHelperFails) {
  EXPECT_THAT(PjRtGetBuildStampFullSourceHash(CudaId()),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(PjRtGetBuildStampFullSourceHash(CudaId(), kUnknownVariantId),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(PjRtBuildStampRegistryTest, RegisterAndLookupDefaultVariant) {
  PjRtBuildStampRegistry registry;
  auto linked_factory = []() -> absl::StatusOr<PjRtBuildStamp> {
    PjRtBuildStampProto proto;
    proto.set_source_hash("linked_hash");
    return PjRtBuildStamp::FromProto(std::move(proto));
  };

  ASSERT_OK(registry.RegisterBuildStampFactory(TpuId(), linked_factory));
  ASSERT_OK_AND_ASSIGN(const PjRtBuildStamp* absl_nonnull default_stamp,
                       registry.GetBuildStamp(TpuId()));
  EXPECT_EQ(default_stamp->source_hash(), "linked_hash");

  ASSERT_OK_AND_ASSIGN(
      const PjRtBuildStamp* absl_nonnull linked_stamp,
      registry.GetBuildStamp(TpuId(), LinkedCompilerVariantId()));
  EXPECT_EQ(linked_stamp, default_stamp);
  EXPECT_EQ(linked_stamp->source_hash(), "linked_hash");
}

TEST(PjRtBuildStampRegistryTest, RegisterAndLookupMultipleVariants) {
  PjRtBuildStampRegistry registry;
  auto tpu_linked_factory = []() -> absl::StatusOr<PjRtBuildStamp> {
    PjRtBuildStampProto proto;
    proto.set_source_hash("tpu_linked");
    return PjRtBuildStamp::FromProto(std::move(proto));
  };
  auto tpu_forge_factory = []() -> absl::StatusOr<PjRtBuildStamp> {
    PjRtBuildStampProto proto;
    proto.set_source_hash("tpu_forge");
    return PjRtBuildStamp::FromProto(std::move(proto));
  };

  ASSERT_OK(registry.RegisterBuildStampFactory(
      TpuId(), LinkedCompilerVariantId(), tpu_linked_factory));
  ASSERT_OK(registry.RegisterBuildStampFactory(TpuId(), kForgeVariantId,
                                               tpu_forge_factory));

  ASSERT_OK_AND_ASSIGN(
      const PjRtBuildStamp* absl_nonnull tpu_linked_stamp,
      registry.GetBuildStamp(TpuId(), LinkedCompilerVariantId()));
  EXPECT_EQ(tpu_linked_stamp->source_hash(), "tpu_linked");

  ASSERT_OK_AND_ASSIGN(const PjRtBuildStamp* absl_nonnull tpu_forge_stamp,
                       registry.GetBuildStamp(TpuId(), kForgeVariantId));
  EXPECT_EQ(tpu_forge_stamp->source_hash(), "tpu_forge");
}

TEST(PjRtBuildStampRegistryTest, FactoryErrorIsMemoized) {
  PjRtBuildStampRegistry registry;
  int factory_calls = 0;
  auto failing_factory = [&]() -> absl::StatusOr<PjRtBuildStamp> {
    factory_calls++;
    return absl::InternalError("boom");
  };

  ASSERT_OK(registry.RegisterBuildStampFactory(TpuId(), kCustomVariantId,
                                               failing_factory));
  EXPECT_THAT(registry.GetBuildStamp(TpuId(), kCustomVariantId),
              StatusIs(absl::StatusCode::kInternal));
  EXPECT_EQ(factory_calls, 1);

  EXPECT_THAT(registry.GetBuildStamp(TpuId(), kCustomVariantId),
              StatusIs(absl::StatusCode::kInternal));
  EXPECT_EQ(factory_calls, 1);
}

TEST(PjRtBuildStampRegistryTest, DuplicateRegistrationFails) {
  PjRtBuildStampRegistry registry;
  auto factory = []() -> absl::StatusOr<PjRtBuildStamp> {
    PjRtBuildStampProto proto;
    proto.set_source_hash("h");
    return PjRtBuildStamp::FromProto(std::move(proto));
  };

  ASSERT_OK(registry.RegisterBuildStampFactory(TpuId(), kVariantAId, factory));
  EXPECT_THAT(registry.RegisterBuildStampFactory(TpuId(), kVariantAId, factory),
              StatusIs(absl::StatusCode::kAlreadyExists));
}

}  // namespace
}  // namespace xla
