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

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/nullability.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler_variant.h"
#include "xla/pjrt/proto/pjrt_build_stamp.pb.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::tsl::proto_testing::EqualsProto;
using ::tsl::proto_testing::ParseTextProtoOrDie;

absl::StatusOr<PjRtBuildStamp> FromTextProto(absl::string_view text_proto) {
  return PjRtBuildStamp::FromProto(
      ParseTextProtoOrDie<PjRtBuildStampProto>(text_proto));
}

std::string FullSourceHash(absl::string_view text_proto) {
  absl::StatusOr<PjRtBuildStamp> stamp = FromTextProto(text_proto);
  CHECK_OK(stamp);
  return std::string(stamp->full_source_hash());
}

TEST(PjRtCompatibilityVersionProtoTest, MatchesProtoFieldCount) {
  EXPECT_EQ(PjRtCompatibilityVersionProto::descriptor()->field_count(), 3)
      << "PjRtCompatibilityVersionProto changed! A field was added or removed "
         "to the proto. Update ValidateVersions(), full_source_hash "
         "calculation in FromProto(), and this test.";
}

TEST(PjRtComponentBuildStampProtoTest, MatchesProtoFieldCount) {
  EXPECT_EQ(PjRtComponentBuildStampProto::descriptor()->field_count(), 4)
      << "PjRtComponentBuildStampProto changed! A field was added or removed "
         "to the proto. Update ValidateComponents(), full_source_hash "
         "calculation in FromProto(), and this test.";
}

TEST(PjRtBuildStampFromProtoTest, FailsWhenMissingBuildIdentity) {
  EXPECT_THAT(PjRtBuildStamp::FromProto(PjRtBuildStampProto{}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("at least one build identity field")));
}

TEST(PjRtBuildStampFromProtoTest, FailsWhenVcsRevisionIsUnknownSentinel) {
  EXPECT_THAT(FromTextProto(R"pb(vcs_revision: "<unknown>")pb"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("vcs_revision cannot be the '<unknown>'")));
}

TEST(PjRtBuildStampFromProtoTest, FailsWhenBuildTimestampIsInvalid) {
  EXPECT_THAT(
      FromTextProto(R"pb(build_timestamp { seconds: -1 nanos: 100 })pb"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("build_timestamp.seconds must be non-negative")));
}

TEST(PjRtBuildStampFromProtoTest, FailsWhenComponentHasEmptyName) {
  EXPECT_THAT(
      FromTextProto(R"pb(
        source_hash: "hash"
        components { hash: "fp" }
      )pb"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Component build stamp name and hash must not be empty")));
}

TEST(PjRtBuildStampFromProtoTest, FailsWhenComponentHasEmptyHash) {
  EXPECT_THAT(
      FromTextProto(R"pb(
        source_hash: "hash"
        components { name: "comp" }
      )pb"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Component build stamp name and hash must not be empty")));
}

TEST(PjRtBuildStampFromProtoTest,
     FailsWhenDuplicateComponentHasConflictingHashes) {
  EXPECT_THAT(
      FromTextProto(R"pb(
        source_hash: "base_hash"
        components { kind: "k" name: "n" owner: "o" hash: "hash_1" }
        components { kind: "k" name: "n" owner: "o" hash: "hash_2" }
      )pb"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Duplicate component found with conflicting hashes")));
}

// Only components with the same (kind, name, owner) must agree on `hash`.
TEST(PjRtBuildStampFromProtoTest,
     AcceptsSameNameWithDifferentKindOrOwnerAndHash) {
  ASSERT_OK_AND_ASSIGN(
      PjRtBuildStamp stamp, FromTextProto(R"pb(
        source_hash: "base"
        components { kind: "k1" name: "n" owner: "o1" hash: "1" }
        components { kind: "k1" name: "n" owner: "o2" hash: "2" }
        components { kind: "k2" name: "n" owner: "o1" hash: "3" }
      )pb"));

  EXPECT_THAT(
      stamp.ToProto().components(),
      ElementsAre(
          EqualsProto(R"pb(kind: "k1" name: "n" owner: "o1" hash: "1")pb"),
          EqualsProto(R"pb(kind: "k1" name: "n" owner: "o2" hash: "2")pb"),
          EqualsProto(R"pb(kind: "k2" name: "n" owner: "o1" hash: "3")pb")));
}

TEST(PjRtBuildStampFromProtoTest, SortsAndDeduplicatesComponents) {
  ASSERT_OK_AND_ASSIGN(
      PjRtBuildStamp input, FromTextProto(R"pb(
        source_hash: "base"
        components { kind: "k2" name: "a" owner: "o" hash: "1" }
        components { kind: "k1" name: "b" owner: "o" hash: "2" }
        components { kind: "k2" name: "a" owner: "o" hash: "1" }
        components { kind: "k3" name: "c" owner: "o" hash: "3" }
        components { kind: "k2" name: "a" owner: "o" hash: "1" }
      )pb"));
  ASSERT_OK_AND_ASSIGN(
      PjRtBuildStamp expected, FromTextProto(R"pb(
        source_hash: "base"
        components { kind: "k1" name: "b" owner: "o" hash: "2" }
        components { kind: "k2" name: "a" owner: "o" hash: "1" }
        components { kind: "k3" name: "c" owner: "o" hash: "3" }
      )pb"));
  EXPECT_THAT(input.ToProto(), EqualsProto(expected.ToProto()));
}

// Each invalid case is covered by `ValidateCompatibilityVersionTest`.
TEST(PjRtBuildStampFromProtoTest, FailsWhenVersionIsInvalid) {
  EXPECT_THAT(FromTextProto(R"pb(
                source_hash: "base_hash"
                compatibility_versions { name: "" major_version: 1 }
              )pb"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Version name must not be empty")));
}

TEST(PjRtBuildStampFromProtoTest,
     FailsWhenDuplicateVersionHasConflictingMajorVersion) {
  EXPECT_THAT(
      FromTextProto(R"pb(
        source_hash: "base_hash"
        compatibility_versions { name: "mosaic" major_version: 12 }
        compatibility_versions { name: "mosaic" major_version: 13 }
      )pb"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "Duplicate version found with conflicting values for 'mosaic'")));
}

TEST(PjRtBuildStampFromProtoTest,
     FailsWhenDuplicateVersionHasConflictingMinorVersion) {
  EXPECT_THAT(
      FromTextProto(R"pb(
        source_hash: "base_hash"
        compatibility_versions { name: "mosaic" major_version: 12 }
        compatibility_versions {
          name: "mosaic"
          major_version: 12
          minor_version: 1
        }
      )pb"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "Duplicate version found with conflicting values for 'mosaic'")));
}

TEST(PjRtBuildStampFromProtoTest, SortsAndDeduplicatesVersions) {
  ASSERT_OK_AND_ASSIGN(
      PjRtBuildStamp input, FromTextProto(R"pb(
        source_hash: "base_hash"
        compatibility_versions { name: "v3" major_version: 123 }
        compatibility_versions { name: "v2" major_version: 12 }
        compatibility_versions { name: "v2" major_version: 12 }
        compatibility_versions { name: "v1" major_version: 1 }
        compatibility_versions { name: "v2" major_version: 12 }
        compatibility_versions { name: "v1" major_version: 1 }
      )pb"));

  ASSERT_OK_AND_ASSIGN(
      PjRtBuildStamp expected, FromTextProto(R"pb(
        source_hash: "base_hash"
        compatibility_versions { name: "v1" major_version: 1 }
        compatibility_versions { name: "v2" major_version: 12 }
        compatibility_versions { name: "v3" major_version: 123 }
      )pb"));

  EXPECT_THAT(input.ToProto(), EqualsProto(expected.ToProto()));
  EXPECT_EQ(input.full_source_hash(), expected.full_source_hash());
}

// Each invalid case is covered by `ValidateAttributeTest`.
TEST(PjRtBuildStampFromProtoTest, FailsWhenAttributeIsInvalid) {
  EXPECT_THAT(
      FromTextProto(R"pb(
        source_hash: "base_hash"
        attributes {
          key: "empty_attr"
          value {}
        }
      )pb"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Attribute value for key='empty_attr' must be set")));
}

TEST(PjRtBuildStampAccessorsTest, ReturnsProvenanceFields) {
  ASSERT_OK_AND_ASSIGN(
      PjRtBuildStamp stamp, FromTextProto(R"pb(
        build_label: "target_foo"
        build_timestamp { seconds: 1700000000 nanos: 123456789 }
        vcs_revision: "git_rev_123"
        source_hash: "hash_abcdef01"
      )pb"));

  EXPECT_EQ(stamp.build_label(), "target_foo");
  EXPECT_EQ(stamp.build_timestamp(),
            absl::FromUnixSeconds(1700000000) + absl::Nanoseconds(123456789));
  EXPECT_EQ(stamp.vcs_revision(), "git_rev_123");
  EXPECT_EQ(stamp.source_hash(), "hash_abcdef01");
}

TEST(PjRtBuildStampAccessorsTest, DefaultBuildTimestampIsInfinitePast) {
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       FromTextProto(R"pb(source_hash: "hash_no_ts")pb"));

  EXPECT_EQ(stamp.build_timestamp(), absl::InfinitePast());
}

TEST(PjRtBuildStampAccessorsTest, ToProtoRoundTripsCanonicalProto) {
  // Already canonical: at most one component and version, so no reordering.
  const PjRtBuildStampProto proto = ParseTextProtoOrDie(R"pb(
    build_label: "label1"
    build_timestamp { seconds: 1700000000 nanos: 123 }
    vcs_revision: "rev1"
    source_hash: "hash1"
    components {
      kind: "xla.ffi.handler"
      name: "custom_op"
      owner: "pkg:custom_op"
      hash: "comp_hash_1"
    }
    attributes {
      key: "vendor.candidate"
      value { string_value: "candidate_1" }
    }
    compatibility_versions { name: "foo" major_version: 7 minor_version: 1 }
  )pb");

  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp1, PjRtBuildStamp::FromProto(proto));
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp2,
                       PjRtBuildStamp::FromProto(stamp1.ToProto()));

  EXPECT_THAT(stamp1.ToProto(), EqualsProto(proto));
  EXPECT_EQ(stamp2.full_source_hash(), stamp1.full_source_hash());
}

TEST(PjRtBuildStampFullSourceHashTest,
     EqualsSourceHashWhenNoComponentsOrVersions) {
  EXPECT_EQ(FullSourceHash(R"pb(source_hash: "hash1")pb"), "hash1");
}

TEST(PjRtBuildStampFullSourceHashTest, FallsBackToVcsRevision) {
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp, FromTextProto(R"pb(
                         vcs_revision: "git_rev_456"
                         build_timestamp { seconds: 1700000000 }
                       )pb"));

  EXPECT_EQ(stamp.full_source_hash(), "vcs:git_rev_456");
  EXPECT_THAT(stamp.source_hash(), IsEmpty());
  EXPECT_THAT(stamp.ToProto().source_hash(), IsEmpty());
}

TEST(PjRtBuildStampFullSourceHashTest, FallsBackToTimestamp) {
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp, FromTextProto(R"pb(
                         build_timestamp { seconds: 1700000000 nanos: 12345 }
                       )pb"));

  EXPECT_EQ(stamp.full_source_hash(), "ts:1700000000.000012345");
  EXPECT_THAT(stamp.source_hash(), IsEmpty());
  EXPECT_THAT(stamp.ToProto().source_hash(), IsEmpty());
}

TEST(PjRtBuildStampFullSourceHashTest, IncludesComponentsWithFallback) {
  EXPECT_NE(FullSourceHash(R"pb(
              vcs_revision: "git_rev_456"
              components { name: "op_a" hash: "hash_a" }
            )pb"),
            "vcs:git_rev_456");
}

TEST(PjRtBuildStampFullSourceHashTest, IncludesVersionsWithFallback) {
  EXPECT_NE(FullSourceHash(R"pb(
              build_timestamp { seconds: 1700000000 nanos: 12345 }
              compatibility_versions { name: "foo" major_version: 1 }
            )pb"),
            "ts:1700000000.000012345");
}

TEST(PjRtBuildStampFullSourceHashTest, ChangesWithComponentHash) {
  const std::string no_comp = FullSourceHash(R"pb(source_hash: "base")pb");
  const std::string comp_1111 = FullSourceHash(R"pb(
    source_hash: "base"
    components { kind: "k" name: "n" owner: "o" hash: "1111" }
  )pb");
  const std::string comp_2222 = FullSourceHash(R"pb(
    source_hash: "base"
    components { kind: "k" name: "n" owner: "o" hash: "2222" }
  )pb");

  EXPECT_NE(no_comp, comp_1111);
  EXPECT_NE(comp_1111, comp_2222);
}

TEST(PjRtBuildStampFullSourceHashTest, IgnoresComponentKindNameAndOwner) {
  const std::string base = FullSourceHash(R"pb(
    source_hash: "base"
    components { kind: "k1" name: "n1" owner: "o1" hash: "h" }
  )pb");

  EXPECT_EQ(FullSourceHash(R"pb(
              source_hash: "base"
              components { kind: "k2" name: "n1" owner: "o1" hash: "h" }
            )pb"),
            base);
  EXPECT_EQ(FullSourceHash(R"pb(
              source_hash: "base"
              components { kind: "k1" name: "n2" owner: "o1" hash: "h" }
            )pb"),
            base);
  EXPECT_EQ(FullSourceHash(R"pb(
              source_hash: "base"
              components { kind: "k1" name: "n1" owner: "o2" hash: "h" }
            )pb"),
            base);
}

TEST(PjRtBuildStampFullSourceHashTest,
     DistinguishesSourceAndComponentBoundaries) {
  EXPECT_NE(FullSourceHash(R"pb(
              source_hash: "ab"
              components { name: "comp1" hash: "cd" }
            )pb"),
            FullSourceHash(R"pb(
              source_hash: "abc"
              components { name: "comp1" hash: "d" }
            )pb"));
  EXPECT_NE(FullSourceHash(R"pb(
              source_hash: "base"
              components { name: "comp1" hash: "ab" }
              components { name: "comp2" hash: "cd" }
            )pb"),
            FullSourceHash(R"pb(
              source_hash: "base"
              components { name: "comp1" hash: "abcd" }
            )pb"));
}

TEST(PjRtBuildStampFullSourceHashTest, ChangesWithVersion) {
  EXPECT_NE(FullSourceHash(R"pb(
              source_hash: "base_hash"
              compatibility_versions { name: "foo" major_version: 12 }
            )pb"),
            FullSourceHash(R"pb(
              source_hash: "base_hash"
              compatibility_versions { name: "foo" major_version: 13 }
            )pb"));
}

TEST(PjRtBuildStampFullSourceHashTest, IgnoresAttributes) {
  EXPECT_EQ(FullSourceHash(R"pb(source_hash: "hash1")pb"), FullSourceHash(R"pb(
              source_hash: "hash1"
              attributes {
                key: "vendor.candidate"
                value { string_value: "xla_tpu_candidate_1" }
              }
              attributes {
                key: "is_experimental"
                value { bool_value: true }
              }
            )pb"));
}

TEST(PjRtBuildStampFullSourceHashTest, IgnoresProvenanceMetadata) {
  EXPECT_EQ(FullSourceHash(R"pb(
              source_hash: "hash1"
              build_label: "label_a"
              vcs_revision: "rev_a"
              build_timestamp { seconds: 100 }
            )pb"),
            FullSourceHash(R"pb(
              source_hash: "hash1"
              build_label: "label_b"
              vcs_revision: "rev_b"
              build_timestamp { seconds: 200 }
            )pb"));
}

TEST(PjRtBuildStampGetVersionTest, ReturnsMatchingVersion) {
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp, FromTextProto(R"pb(
                         source_hash: "hash1"
                         compatibility_versions { name: "a" major_version: 1 }
                         compatibility_versions { name: "b" major_version: 2 }
                       )pb"));

  EXPECT_THAT(stamp.GetVersion("b"),
              IsOkAndHolds(EqualsProto(R"pb(name: "b" major_version: 2)pb")));
}

TEST(PjRtBuildStampGetVersionTest, ReturnsNotFoundForMissingName) {
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       FromTextProto(R"pb(source_hash: "hash1")pb"));

  EXPECT_THAT(stamp.GetVersion("non_existent"),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr("Version 'non_existent' not found.")));
}

TEST(PjRtBuildStampGetAttributeTest, ReturnsTypedValues) {
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp, FromTextProto(R"pb(
                         source_hash: "hash1"
                         attributes {
                           key: "str_attr"
                           value { string_value: "xla_tpu_candidate_1" }
                         }
                         attributes {
                           key: "bool_attr"
                           value { bool_value: true }
                         }
                         attributes {
                           key: "int_attr"
                           value { int_value: 42 }
                         }
                         attributes {
                           key: "float_attr"
                           value { float_value: 3.5 }
                         }
                         attributes {
                           key: "int_vec_attr"
                           value { int_vector { values: [ 1, 2, 3 ] } }
                         }
                       )pb"));

  EXPECT_THAT(stamp.GetAttribute<std::string>("str_attr"),
              IsOkAndHolds("xla_tpu_candidate_1"));
  EXPECT_THAT(stamp.GetAttribute<bool>("bool_attr"), IsOkAndHolds(true));
  EXPECT_THAT(stamp.GetAttribute<int64_t>("int_attr"),
              IsOkAndHolds(int64_t{42}));
  EXPECT_THAT(stamp.GetAttribute<float>("float_attr"), IsOkAndHolds(3.5f));
  EXPECT_THAT(stamp.GetAttribute<std::vector<int64_t>>("int_vec_attr"),
              IsOkAndHolds(ElementsAre(1, 2, 3)));
}

TEST(PjRtBuildStampGetAttributeTest, ReturnsNotFoundForMissingKey) {
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       FromTextProto(R"pb(source_hash: "hash1")pb"));

  EXPECT_THAT(stamp.GetAttribute<int64_t>("non_existent"),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr("Attribute 'non_existent' not found.")));
}

// Type names in the error come from `PjRtTypeName` and `PjRtValueTypeCaseName`,
// which are tested in `pjrt_common_test.cc`.
TEST(PjRtBuildStampGetAttributeTest, ReturnsInvalidArgumentForWrongType) {
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp, FromTextProto(R"pb(
                         source_hash: "hash1"
                         attributes {
                           key: "bool_attr"
                           value { bool_value: true }
                         }
                       )pb"));

  EXPECT_THAT(
      stamp.GetAttribute<std::string>("bool_attr"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Attribute 'bool_attr' has type bool, not the requested "
                    "type \"string\".")));
}

TEST(PjRtBuildStampGetStringAttributeTest, ReturnsValue) {
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp, FromTextProto(R"pb(
                         source_hash: "hash1"
                         attributes {
                           key: "vendor.candidate"
                           value { string_value: "xla_tpu_candidate_1" }
                         }
                       )pb"));

  EXPECT_THAT(stamp.GetStringAttribute("vendor.candidate"),
              IsOkAndHolds("xla_tpu_candidate_1"));
}

TEST(PjRtBuildStampGetStringAttributeTest, ReturnsNotFoundForMissingKey) {
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       FromTextProto(R"pb(source_hash: "hash1")pb"));

  EXPECT_THAT(stamp.GetStringAttribute("non_existent"),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr("Attribute 'non_existent' not found.")));
}

TEST(PjRtBuildStampGetStringAttributeTest, ReturnsInvalidArgumentForWrongType) {
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp, FromTextProto(R"pb(
                         source_hash: "hash1"
                         attributes {
                           key: "bool_attr"
                           value { bool_value: true }
                         }
                       )pb"));

  EXPECT_THAT(
      stamp.GetStringAttribute("bool_attr"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Attribute 'bool_attr' has type bool, not the requested "
                    "type \"string\".")));
}

TEST(ValidateVcsRevisionTest, AcceptsEmptyAndValidRevisions) {
  EXPECT_THAT(ValidateVcsRevision(""), IsOk());
  EXPECT_THAT(ValidateVcsRevision("git_rev_123"), IsOk());
  EXPECT_THAT(ValidateVcsRevision("<unknown>"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("vcs_revision cannot be the '<unknown>'")));
}

TEST(ValidateBuildTimestampTest, AcceptsBoundaryValues) {
  // `seconds == 0` is what unstamped builds report.
  EXPECT_THAT(ValidateBuildTimestamp(ParseTextProtoOrDie(R"pb()pb")), IsOk());
  EXPECT_THAT(
      ValidateBuildTimestamp(ParseTextProtoOrDie(R"pb(nanos: 999999999)pb")),
      IsOk());
  EXPECT_THAT(
      ValidateBuildTimestamp(ParseTextProtoOrDie(R"pb(seconds: 1700000000)pb")),
      IsOk());
}

TEST(ValidateBuildTimestampTest, RejectsNanosOutOfRange) {
  EXPECT_THAT(ValidateBuildTimestamp(ParseTextProtoOrDie(R"pb(nanos: -1)pb")),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("build_timestamp.nanos must be in")));
  EXPECT_THAT(
      ValidateBuildTimestamp(ParseTextProtoOrDie(R"pb(nanos: 1000000000)pb")),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("build_timestamp.nanos must be in")));
}

TEST(ValidateBuildTimestampTest, RejectsNegativeSeconds) {
  EXPECT_THAT(
      ValidateBuildTimestamp(ParseTextProtoOrDie(R"pb(seconds: -1)pb")),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("build_timestamp.seconds must be non-negative")));
}

TEST(ValidateCompatibilityVersionTest, AcceptsZeroVersion) {
  EXPECT_THAT(ValidateCompatibilityVersion(ParseTextProtoOrDie(
                  R"pb(name: "mosaic" major_version: 0 minor_version: 0)pb")),
              IsOk());
}

TEST(ValidateCompatibilityVersionTest, RejectsEmptyName) {
  EXPECT_THAT(ValidateCompatibilityVersion(
                  ParseTextProtoOrDie(R"pb(major_version: 1)pb")),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Version name must not be empty")));
}

TEST(ValidateCompatibilityVersionTest, RejectsNegativeVersionNumbers) {
  EXPECT_THAT(ValidateCompatibilityVersion(ParseTextProtoOrDie(
                  R"pb(name: "mosaic" major_version: -1)pb")),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Version numbers must be non-negative")));
  EXPECT_THAT(ValidateCompatibilityVersion(ParseTextProtoOrDie(
                  R"pb(name: "mosaic" major_version: 1 minor_version: -1)pb")),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Version numbers must be non-negative")));
}

TEST(ValidateAttributeTest, AcceptsSetValue) {
  EXPECT_THAT(
      ValidateAttribute("key", ParseTextProtoOrDie(R"pb(bool_value: false)pb")),
      IsOk());
}

TEST(ValidateAttributeTest, RejectsEmptyKey) {
  EXPECT_THAT(
      ValidateAttribute("", ParseTextProtoOrDie(R"pb(bool_value: true)pb")),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Attribute key must not be empty")));
}

TEST(ValidateAttributeTest, RejectsUnsetValue) {
  EXPECT_THAT(
      ValidateAttribute("empty_attr", ParseTextProtoOrDie(R"pb()pb")),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Attribute value for key='empty_attr' must be set")));
}

// Platform IDs and compiler variants are opaque to the registry, so arbitrary
// values are used.
constexpr PjRtPlatformId kPlatformA = 1;
constexpr PjRtCompilerVariant kVariantA = 12345;
constexpr PjRtCompilerVariant kVariantB = 67890;

// The global registry is append-only, so each global test uses its own
// platform IDs to stay independent of test order.
constexpr PjRtPlatformId kGlobalRegisterPlatformA = 0xB5000001;
constexpr PjRtPlatformId kGlobalRegisterPlatformB = 0xB5000002;
constexpr PjRtPlatformId kGlobalFullSourceHashPlatform = 0xB5000003;
constexpr PjRtPlatformId kGlobalUnregisteredPlatform = 0xB5000004;
constexpr PjRtPlatformId kGlobalDuplicatePlatform = 0xB5000005;

PjRtBuildStampRegistry::Factory FactoryWithSourceHash(std::string source_hash) {
  return [source_hash = std::move(source_hash)]() {
    PjRtBuildStampProto proto;
    proto.set_source_hash(source_hash);
    return PjRtBuildStamp::FromProto(std::move(proto));
  };
}

TEST(PjRtBuildStampRegistryTest, MemoizesStamp) {
  const PjRtBuildStampProto proto = ParseTextProtoOrDie(R"pb(
    source_hash: "hash1"
    components { kind: "k" name: "n" owner: "o" hash: "comp_hash" }
    compatibility_versions { name: "v" major_version: 1 }
  )pb");
  ASSERT_OK_AND_ASSIGN(const PjRtBuildStamp expected,
                       PjRtBuildStamp::FromProto(proto));
  PjRtBuildStampRegistry registry;
  std::atomic<int> factory_calls = 0;
  ASSERT_OK(registry.RegisterBuildStampFactory(
      kPlatformA, kVariantA, [&]() -> absl::StatusOr<PjRtBuildStamp> {
        ++factory_calls;
        return PjRtBuildStamp::FromProto(proto);
      }));

  constexpr int kNumThreads = 8;
  std::vector<absl::StatusOr<const PjRtBuildStamp*>> results(kNumThreads);
  {
    std::vector<std::thread> threads;
    threads.reserve(kNumThreads);
    for (int i = 0; i < kNumThreads; ++i) {
      threads.emplace_back([&registry, &results, i] {
        results[i] = registry.GetBuildStamp(kPlatformA, kVariantA);
      });
    }
    for (std::thread& thread : threads) {
      thread.join();
    }
  }

  EXPECT_EQ(factory_calls, 1);
  ASSERT_OK_AND_ASSIGN(const PjRtBuildStamp* absl_nonnull first, results[0]);
  EXPECT_THAT(first->ToProto(), EqualsProto(expected.ToProto()));
  EXPECT_EQ(first->full_source_hash(), expected.full_source_hash());
  // Every thread has the same memoized stamp.
  EXPECT_THAT(results, Each(IsOkAndHolds(first)));
}

TEST(PjRtBuildStampRegistryTest, MemoizeFactoryError) {
  PjRtBuildStampRegistry registry;
  int factory_calls = 0;
  ASSERT_OK(registry.RegisterBuildStampFactory(
      kPlatformA, kVariantA, [&]() -> absl::StatusOr<PjRtBuildStamp> {
        ++factory_calls;
        return absl::InternalError("boom");
      }));

  EXPECT_THAT(registry.GetBuildStamp(kPlatformA, kVariantA),
              StatusIs(absl::StatusCode::kInternal, HasSubstr("boom")));
  EXPECT_THAT(registry.GetBuildStamp(kPlatformA, kVariantA),
              StatusIs(absl::StatusCode::kInternal, HasSubstr("boom")));
  EXPECT_EQ(factory_calls, 1);
}

TEST(PjRtBuildStampRegistryTest, DefaultsToLinkedCompilerVariant) {
  PjRtBuildStampRegistry registry;
  ASSERT_OK(registry.RegisterBuildStampFactory(
      kPlatformA, FactoryWithSourceHash("linked_hash")));

  ASSERT_OK_AND_ASSIGN(const PjRtBuildStamp* absl_nonnull stamp,
                       registry.GetBuildStamp(kPlatformA));
  EXPECT_EQ(stamp->source_hash(), "linked_hash");
  EXPECT_THAT(registry.GetBuildStamp(kPlatformA, LinkedCompilerVariantId()),
              IsOkAndHolds(stamp));
}

TEST(PjRtBuildStampRegistryTest, KeysByPlatformAndVariant) {
  PjRtBuildStampRegistry registry;
  ASSERT_OK(registry.RegisterBuildStampFactory(kPlatformA, kVariantA,
                                               FactoryWithSourceHash("a_a")));
  ASSERT_OK(registry.RegisterBuildStampFactory(kPlatformA, kVariantB,
                                               FactoryWithSourceHash("a_b")));

  ASSERT_OK_AND_ASSIGN(const PjRtBuildStamp* absl_nonnull stamp_a,
                       registry.GetBuildStamp(kPlatformA, kVariantA));
  EXPECT_EQ(stamp_a->source_hash(), "a_a");
  ASSERT_OK_AND_ASSIGN(const PjRtBuildStamp* absl_nonnull stamp_b,
                       registry.GetBuildStamp(kPlatformA, kVariantB));
  EXPECT_EQ(stamp_b->source_hash(), "a_b");
}

TEST(PjRtBuildStampRegistryTest, LookupOfUnregisteredKeyFails) {
  PjRtBuildStampRegistry registry;
  EXPECT_THAT(registry.GetBuildStamp(kPlatformA, kVariantA),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr("No registered build stamp factory")));
}

TEST(PjRtBuildStampRegistryTest, DuplicateRegistrationFails) {
  PjRtBuildStampRegistry registry;
  ASSERT_OK(registry.RegisterBuildStampFactory(kPlatformA, kVariantA,
                                               FactoryWithSourceHash("h1")));
  EXPECT_THAT(registry.RegisterBuildStampFactory(kPlatformA, kVariantA,
                                                 FactoryWithSourceHash("h2")),
              StatusIs(absl::StatusCode::kAlreadyExists));
}

TEST(PjRtBuildStampGlobalRegistryTest, RegisterAndLookup) {
  PjRtRegisterBuildStampFactory(kGlobalRegisterPlatformA, kVariantA,
                                FactoryWithSourceHash("hash_a"));
  PjRtRegisterBuildStampFactory(kGlobalRegisterPlatformB,
                                FactoryWithSourceHash("hash_b"));

  ASSERT_OK_AND_ASSIGN(const PjRtBuildStamp* absl_nonnull stamp_a,
                       PjRtGetBuildStamp(kGlobalRegisterPlatformA, kVariantA));
  EXPECT_EQ(stamp_a->source_hash(), "hash_a");
  ASSERT_OK_AND_ASSIGN(const PjRtBuildStamp* absl_nonnull stamp_b,
                       PjRtGetBuildStamp(kGlobalRegisterPlatformB));
  EXPECT_EQ(stamp_b->source_hash(), "hash_b");
}

TEST(PjRtBuildStampGlobalRegistryTest, GetFullSourceHash) {
  PjRtRegisterBuildStampFactory(kGlobalFullSourceHashPlatform, kVariantA,
                                FactoryWithSourceHash("full_hash"));

  EXPECT_THAT(
      PjRtGetBuildStampFullSourceHash(kGlobalFullSourceHashPlatform, kVariantA),
      IsOkAndHolds("full_hash"));
}

TEST(PjRtBuildStampGlobalRegistryTest, LookupOfUnregisteredKeyFails) {
  EXPECT_THAT(PjRtGetBuildStamp(kGlobalUnregisteredPlatform),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(
      PjRtGetBuildStampFullSourceHash(kGlobalUnregisteredPlatform, kVariantA),
      StatusIs(absl::StatusCode::kNotFound));
}

TEST(PjRtBuildStampGlobalRegistryDeathTest, DuplicateRegistrationCrashes) {
  PjRtRegisterBuildStampFactory(kGlobalDuplicatePlatform,
                                FactoryWithSourceHash("h1"));
  EXPECT_DEATH(PjRtRegisterBuildStampFactory(kGlobalDuplicatePlatform,
                                             FactoryWithSourceHash("h2")),
               "already registered");
}

}  // namespace
}  // namespace xla
