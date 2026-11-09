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
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/pjrt/proto/pjrt_build_stamp.pb.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
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

bool IsLowercaseHex128(absl::string_view s) {
  if (s.size() != 32) {
    return false;
  }
  for (const char c : s) {
    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) {
      return false;
    }
  }
  return true;
}

TEST(PjRtCompatibilityVersionProtoTest, MatchesProtoFieldCount) {
  EXPECT_EQ(PjRtCompatibilityVersionProto::descriptor()->field_count(), 3)
      << "PjRtCompatibilityVersionProto changed! A field was added or removed "
         "to the proto. Update ValidateVersions(), ComputeFullSourceHash(), "
         "and this test.";
}

TEST(PjRtComponentBuildStampProtoTest, MatchesProtoFieldCount) {
  EXPECT_EQ(PjRtComponentBuildStampProto::descriptor()->field_count(), 4)
      << "PjRtComponentBuildStampProto changed! A field was added or removed "
         "to the proto. Update ValidateComponents(), ComputeFullSourceHash(), "
         "and this test.";
}

TEST(PjRtBuildStampProtoTest, MatchesProtoFieldCount) {
  EXPECT_EQ(PjRtBuildStampProto::descriptor()->field_count(), 7)
      << "PjRtBuildStampProto changed! A field was added or removed "
         "to the proto. Update FromProto(), ComputeFullSourceHash(), and this "
         "test.";
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
  EXPECT_THAT(FromTextProto(R"pb(vcs_revision: "unknown")pb"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("vcs_revision cannot be the 'unknown'")));
}

TEST(PjRtBuildStampFromProtoTest, FailsWhenBuildTimestampIsInvalid) {
  // Unstamped builds report `seconds == 0`, which must be rejected so
  // `PjRtBuildStampProtoFromBuildData()` leaves `build_timestamp` unset.
  EXPECT_THAT(
      FromTextProto(R"pb(build_timestamp { seconds: -1 nanos: 100 })pb"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("build_timestamp.seconds must be positive")));
  EXPECT_THAT(FromTextProto(R"pb(build_timestamp { seconds: 0 })pb"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("build_timestamp.seconds must be positive")));
  // One second past 9999-12-31T23:59:59Z, the `google.protobuf.Timestamp` max.
  EXPECT_THAT(FromTextProto(R"pb(build_timestamp { seconds: 253402300800 })pb"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("build_timestamp.seconds must be at most "
                                 "253402300799")));
}
TEST(PjRtBuildStampFromProtoTest,
     FailsWhenSecondaryIdentityIsInvalidWithSourceHash) {
  EXPECT_THAT(FromTextProto(R"pb(
                source_hash: "hash" vcs_revision: "unknown"
              )pb"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("vcs_revision cannot be the 'unknown'")));
  EXPECT_THAT(FromTextProto(R"pb(
                source_hash: "hash"
                build_timestamp { seconds: 0 }
              )pb"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("build_timestamp.seconds must be positive")));
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
        components { kind: "k" name: "other" owner: "o" hash: "hash_3" }
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
  EXPECT_EQ(input.full_source_hash(), expected.full_source_hash());
}

// Other invalid cases are covered by `ValidateCompatibilityVersionTest`.
TEST(PjRtBuildStampFromProtoTest, FailsWhenVersionNameIsEmpty) {
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
        compatibility_versions { name: "other" major_version: 1 }
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

// Other invalid cases are covered by `ValidateAttributeTest`.
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
  // Catch any changes in how the full source hash is computed.
  EXPECT_EQ(stamp1.full_source_hash(), "b5bfc36e1fc61b6cd0d92731917c0320");
}

TEST(PjRtBuildStampFullSourceHashTest,
     FormatsAsHex128AndChangesWithSourceHash) {
  const std::string hash1 = FullSourceHash(R"pb(source_hash: "hash1")pb");
  const std::string hash2 = FullSourceHash(R"pb(source_hash: "hash2")pb");

  EXPECT_TRUE(IsLowercaseHex128(hash1)) << hash1;
  EXPECT_TRUE(IsLowercaseHex128(hash2)) << hash2;
  EXPECT_EQ(FullSourceHash(R"pb(source_hash: "hash1")pb"), hash1);
  EXPECT_NE(hash1, hash2);
}

TEST(PjRtBuildStampFullSourceHashTest, FallsBackToVcsRevision) {
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp, FromTextProto(R"pb(
                         vcs_revision: "git_rev_456"
                         build_timestamp { seconds: 1700000000 }
                       )pb"));

  EXPECT_TRUE(IsLowercaseHex128(stamp.full_source_hash()))
      << stamp.full_source_hash();
  EXPECT_EQ(stamp.full_source_hash(), FullSourceHash(R"pb(
              vcs_revision: "git_rev_456"
              build_timestamp { seconds: 1800000000 }
            )pb"));
  EXPECT_NE(stamp.full_source_hash(),
            FullSourceHash(R"pb(vcs_revision: "git_rev_789")pb"));
  EXPECT_THAT(stamp.source_hash(), IsEmpty());
  EXPECT_THAT(stamp.ToProto().source_hash(), IsEmpty());
}

TEST(PjRtBuildStampFullSourceHashTest, FallsBackToTimestamp) {
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp, FromTextProto(R"pb(
                         build_timestamp { seconds: 1700000000 nanos: 12345 }
                       )pb"));

  EXPECT_TRUE(IsLowercaseHex128(stamp.full_source_hash()))
      << stamp.full_source_hash();
  EXPECT_NE(stamp.full_source_hash(), FullSourceHash(R"pb(
              build_timestamp { seconds: 1700000000 nanos: 12346 }
            )pb"));
  EXPECT_NE(stamp.full_source_hash(), FullSourceHash(R"pb(
              build_timestamp { seconds: 1700000001 nanos: 12345 }
            )pb"));
  EXPECT_THAT(stamp.source_hash(), IsEmpty());
  EXPECT_THAT(stamp.ToProto().source_hash(), IsEmpty());
}

TEST(PjRtBuildStampFullSourceHashTest, SameHashWhenNanosIsZeroOrUnset) {
  // `nanos` has implicit presence, so an explicit zero equals an unset value.
  EXPECT_EQ(FullSourceHash(R"pb(build_timestamp { seconds: 1700000000 })pb"),
            FullSourceHash(
                R"pb(build_timestamp { seconds: 1700000000 nanos: 0 })pb"));
}

TEST(PjRtBuildStampFullSourceHashTest, IncludesComponentsWithFallback) {
  EXPECT_NE(FullSourceHash(R"pb(
              vcs_revision: "git_rev_456"
              components { name: "op_a" hash: "hash_a" }
            )pb"),
            FullSourceHash(R"pb(vcs_revision: "git_rev_456")pb"));
}

TEST(PjRtBuildStampFullSourceHashTest, IncludesVersionsWithFallback) {
  EXPECT_NE(FullSourceHash(R"pb(
              build_timestamp { seconds: 1700000000 nanos: 12345 }
              compatibility_versions { name: "foo" major_version: 1 }
            )pb"),
            FullSourceHash(
                R"pb(build_timestamp { seconds: 1700000000 nanos: 12345 })pb"));
}

TEST(PjRtBuildStampFullSourceHashTest, ChangesWithAllComponentFields) {
  const std::string no_comp = FullSourceHash(R"pb(source_hash: "base")pb");
  const std::string base = FullSourceHash(R"pb(
    source_hash: "base"
    components { kind: "k1" name: "n1" owner: "o1" hash: "h1" }
  )pb");

  EXPECT_NE(no_comp, base);
  EXPECT_NE(FullSourceHash(R"pb(
              source_hash: "base"
              components { kind: "k2" name: "n1" owner: "o1" hash: "h1" }
            )pb"),
            base);
  EXPECT_NE(FullSourceHash(R"pb(
              source_hash: "base"
              components { kind: "k1" name: "n2" owner: "o1" hash: "h1" }
            )pb"),
            base);
  EXPECT_NE(FullSourceHash(R"pb(
              source_hash: "base"
              components { kind: "k1" name: "n1" owner: "o2" hash: "h1" }
            )pb"),
            base);
  EXPECT_NE(FullSourceHash(R"pb(
              source_hash: "base"
              components { kind: "k1" name: "n1" owner: "o1" hash: "h2" }
            )pb"),
            base);

  // Swapping hashes across two component names must change the hash.
  EXPECT_NE(FullSourceHash(R"pb(
              source_hash: "base"
              components { name: "op_a" hash: "h1" }
              components { name: "op_b" hash: "h2" }
            )pb"),
            FullSourceHash(R"pb(
              source_hash: "base"
              components { name: "op_a" hash: "h2" }
              components { name: "op_b" hash: "h1" }
            )pb"));
}

// The `Distinguishes*` tests below pin each safeguard of the
// `full_source_hash` encoding (section tags, length prefixes, element counts,
// and delimiters). Every pair collides under a plausible weaker encoding, noted
// above it, and must stay distinct under the real one. The weaker encodings
// are:
//   - Raw field: one string field appended without its tag or length prefix.
//   - Join: all fields joined with `|`, without tags, length prefixes, or
//     counts (e.g. `absl::StrJoin(fields, "|")`).
//   - Concat: all fields concatenated with no delimiter (e.g. `absl::StrCat`).

TEST(PjRtBuildStampFullSourceHashTest, DistinguishesIdentityFallbacks) {
  // Untagged fallbacks: the same value in different identity fields.
  EXPECT_NE(FullSourceHash(R"pb(source_hash: "abc123")pb"),
            FullSourceHash(R"pb(vcs_revision: "abc123")pb"));
  // Join: an untagged `vcs_revision` vs. an untagged timestamp.
  EXPECT_NE(FullSourceHash(R"pb(vcs_revision: "1700000000|0")pb"),
            FullSourceHash(R"pb(build_timestamp { seconds: 1700000000 })pb"));

  // Raw field: one identity field spelling out another branch's tagged
  // encoding.
  EXPECT_NE(FullSourceHash(R"pb(source_hash: "R|6:abc123")pb"),
            FullSourceHash(R"pb(vcs_revision: "abc123")pb"));
  EXPECT_NE(FullSourceHash(R"pb(vcs_revision: "S|6:abc123")pb"),
            FullSourceHash(R"pb(source_hash: "abc123")pb"));
  EXPECT_NE(FullSourceHash(R"pb(source_hash: "T|1700000000|0")pb"),
            FullSourceHash(R"pb(build_timestamp { seconds: 1700000000 })pb"));
  EXPECT_NE(FullSourceHash(R"pb(vcs_revision: "T|1700000000|0")pb"),
            FullSourceHash(R"pb(build_timestamp { seconds: 1700000000 })pb"));
}

TEST(PjRtBuildStampFullSourceHashTest, DistinguishesShiftedFieldBoundaries) {
  // Join, or component strings without length prefixes: a `|` inside a field
  // shifts the boundary between `kind` and `name`.
  EXPECT_NE(
      FullSourceHash(R"pb(
        source_hash: "base"
        components { kind: "xla.ffi|handler" name: "op" owner: "pkg" hash: "h" }
      )pb"),
      FullSourceHash(R"pb(
        source_hash: "base"
        components { kind: "xla.ffi" name: "handler|op" owner: "pkg" hash: "h" }
      )pb"));

  // Join, or version names without length prefixes: a `|` inside a name shifts
  // the boundary between adjacent versions, even with matching counts.
  EXPECT_NE(
      FullSourceHash(R"pb(
        source_hash: "base"
        compatibility_versions { name: "a" major_version: 1 minor_version: 2 }
        compatibility_versions {
          name: "b|3|4|c"
          major_version: 5
          minor_version: 6
        }
      )pb"),
      FullSourceHash(R"pb(
        source_hash: "base"
        compatibility_versions {
          name: "a|1|2|b"
          major_version: 3
          minor_version: 4
        }
        compatibility_versions { name: "c" major_version: 5 minor_version: 6 }
      )pb"));

  // Concat: empty strings shift which field holds a value.
  EXPECT_NE(FullSourceHash(R"pb(
              source_hash: "base"
              components { kind: "" name: "x" owner: "y" hash: "h" }
            )pb"),
            FullSourceHash(R"pb(
              source_hash: "base"
              components { kind: "x" name: "y" owner: "" hash: "h" }
            )pb"));

  // Integers without delimiters: digits shift between adjacent integers.
  EXPECT_NE(FullSourceHash(R"pb(
              build_timestamp { seconds: 1700000001 nanos: 23 }
            )pb"),
            FullSourceHash(R"pb(
              build_timestamp { seconds: 170000000 nanos: 123 }
            )pb"));
  EXPECT_NE(FullSourceHash(R"pb(
              source_hash: "base"
              compatibility_versions {
                name: "foo"
                major_version: 12
                minor_version: 3
              }
            )pb"),
            FullSourceHash(R"pb(
              source_hash: "base"
              compatibility_versions {
                name: "foo"
                major_version: 1
                minor_version: 23
              }
            )pb"));
}

TEST(PjRtBuildStampFullSourceHashTest, DistinguishesSectionBoundaries) {
  // Join, or components without counts and length prefixes: one component's
  // `hash` spells out a second component.
  EXPECT_NE(
      FullSourceHash(R"pb(
        source_hash: "base"
        components { kind: "k" name: "op_a" owner: "o" hash: "h1|k|op_b|o|h2" }
      )pb"),
      FullSourceHash(R"pb(
        source_hash: "base"
        components { kind: "k" name: "op_a" owner: "o" hash: "h1" }
        components { kind: "k" name: "op_b" owner: "o" hash: "h2" }
      )pb"));

  // Join, or versions without counts and length prefixes: one version's name
  // spells out a second version.
  EXPECT_NE(FullSourceHash(R"pb(
              source_hash: "base"
              compatibility_versions { name: "a|1|0|b" major_version: 2 }
            )pb"),
            FullSourceHash(R"pb(
              source_hash: "base"
              compatibility_versions { name: "a" major_version: 1 }
              compatibility_versions { name: "b" major_version: 2 }
            )pb"));

  // Join: an entry moves between the components and versions sections.
  EXPECT_NE(FullSourceHash(R"pb(
              source_hash: "base"
              components { kind: "k" name: "mosaic" owner: "1" hash: "2" }
            )pb"),
            FullSourceHash(R"pb(
              source_hash: "base"
              compatibility_versions {
                name: "k|mosaic"
                major_version: 1
                minor_version: 2
              }
            )pb"));

  // Join: `source_hash` absorbs a component.
  EXPECT_NE(FullSourceHash(R"pb(source_hash: "abc|k|n|o|h")pb"),
            FullSourceHash(R"pb(
              source_hash: "abc"
              components { kind: "k" name: "n" owner: "o" hash: "h" }
            )pb"));

  // `source_hash` without a length prefix (everything else unchanged):
  // `source_hash` absorbs the start of the sections, and a version name
  // absorbs the rest. Both encode as
  // `S|abc|C|0|V|1|15:x|C|0|V|1|3:foo|1|0|`.
  EXPECT_NE(
      FullSourceHash(R"pb(
        source_hash: "abc|C|0|V|1|15:x"
        compatibility_versions { name: "foo" major_version: 1 }
      )pb"),
      FullSourceHash(R"pb(
        source_hash: "abc"
        compatibility_versions { name: "x|C|0|V|1|3:foo" major_version: 1 }
      )pb"));
}

TEST(PjRtBuildStampFullSourceHashTest, ChangesWithVersion) {
  EXPECT_NE(FullSourceHash(R"pb(
              source_hash: "base_hash"
              compatibility_versions { name: "foo" major_version: 12 }
            )pb"),
            FullSourceHash(R"pb(
              source_hash: "base_hash"
              compatibility_versions { name: "bar" major_version: 12 }
            )pb"));
  EXPECT_NE(FullSourceHash(R"pb(
              source_hash: "base_hash"
              compatibility_versions { name: "foo" major_version: 12 }
            )pb"),
            FullSourceHash(R"pb(
              source_hash: "base_hash"
              compatibility_versions { name: "foo" major_version: 13 }
            )pb"));
  EXPECT_NE(FullSourceHash(R"pb(
              source_hash: "base_hash"
              compatibility_versions {
                name: "foo"
                major_version: 12
                minor_version: 0
              }
            )pb"),
            FullSourceHash(R"pb(
              source_hash: "base_hash"
              compatibility_versions {
                name: "foo"
                major_version: 12
                minor_version: 1
              }
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

// `FromProto()` skips validation of an empty `vcs_revision`, so the empty case
// is only reachable directly. `PjRtBuildStampProtoFromBuildData()` relies on it
// to leave `vcs_revision` unset.
TEST(ValidateVcsRevisionTest, RejectsEmpty) {
  EXPECT_THAT(ValidateVcsRevision(""),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("vcs_revision must not be empty")));
}

TEST(ValidateBuildTimestampTest, AcceptsBoundaryValues) {
  EXPECT_THAT(ValidateBuildTimestamp(ParseTextProtoOrDie(R"pb(seconds: 1)pb")),
              IsOk());
  EXPECT_THAT(
      ValidateBuildTimestamp(ParseTextProtoOrDie(R"pb(seconds: 1
                                                      nanos: 999999999)pb")),
      IsOk());
  EXPECT_THAT(
      ValidateBuildTimestamp(ParseTextProtoOrDie(R"pb(seconds: 1700000000)pb")),
      IsOk());
  EXPECT_THAT(
      ValidateBuildTimestamp(ParseTextProtoOrDie(R"pb(seconds: 253402300799
                                                      nanos: 999999999)pb")),
      IsOk());
}

TEST(ValidateBuildTimestampTest, RejectsNanosOutOfRange) {
  EXPECT_THAT(ValidateBuildTimestamp(ParseTextProtoOrDie(R"pb(seconds: 1
                                                              nanos: -1)pb")),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("build_timestamp.nanos must be in")));
  EXPECT_THAT(
      ValidateBuildTimestamp(ParseTextProtoOrDie(R"pb(seconds: 1
                                                      nanos: 1000000000)pb")),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("build_timestamp.nanos must be in")));
}

TEST(ValidateCompatibilityVersionTest, AcceptsZeroVersion) {
  EXPECT_THAT(ValidateCompatibilityVersion(ParseTextProtoOrDie(
                  R"pb(name: "mosaic" major_version: 0 minor_version: 0)pb")),
              IsOk());
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

}  // namespace
}  // namespace xla
