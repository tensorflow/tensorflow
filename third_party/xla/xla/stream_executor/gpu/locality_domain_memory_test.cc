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

#include "xla/stream_executor/gpu/locality_domain_memory.h"

#include <cstdint>
#include <limits>
#include <string_view>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/fingerprint.h"

namespace stream_executor::gpu {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;

TEST(LocalityDomainMemoryTest, EncodesAndDecodesIds) {
  EXPECT_THAT(EncodeLocalityDomainMemorySpace(0), IsOkAndHolds(16));
  EXPECT_THAT(EncodeLocalityDomainMemorySpace(1), IsOkAndHolds(17));
  EXPECT_THAT(EncodeLocalityDomainMemorySpace(2), IsOkAndHolds(18));

  EXPECT_THAT(DecodeLocalityDomainMemorySpace(16), IsOkAndHolds(0));
  EXPECT_THAT(DecodeLocalityDomainMemorySpace(17), IsOkAndHolds(1));
  EXPECT_THAT(DecodeLocalityDomainMemorySpace(18), IsOkAndHolds(2));
}

TEST(LocalityDomainMemoryTest, EncodesAndDecodesHighestReservedId) {
  EXPECT_THAT(EncodeLocalityDomainMemorySpace(31), IsOkAndHolds(47));
  EXPECT_THAT(DecodeLocalityDomainMemorySpace(47), IsOkAndHolds(31));
}

TEST(LocalityDomainMemoryTest, RejectsOutOfRangeIds) {
  for (int64_t locality_domain_id :
       {int64_t{-1}, kMaxLocalityDomains, int64_t{256},
        std::numeric_limits<int64_t>::max()}) {
    EXPECT_THAT(EncodeLocalityDomainMemorySpace(locality_domain_id),
                StatusIs(absl::StatusCode::kInvalidArgument))
        << locality_domain_id;
  }
}

TEST(LocalityDomainMemoryTest, RejectsNonDomainMemorySpaces) {
  for (int64_t memory_space :
       {int64_t{-1}, int64_t{0}, int64_t{1}, int64_t{2}, int64_t{3}, int64_t{5},
        int64_t{15}, int64_t{48}, int64_t{272}}) {
    EXPECT_FALSE(IsLocalityDomainMemorySpace(memory_space));
    EXPECT_THAT(DecodeLocalityDomainMemorySpace(memory_space),
                StatusIs(absl::StatusCode::kInvalidArgument))
        << memory_space;
  }
  EXPECT_TRUE(IsLocalityDomainMemorySpace(16));
  EXPECT_TRUE(IsLocalityDomainMemorySpace(47));
}

TEST(LocalityDomainMemoryTest, BuildsAndParsesCanonicalKinds) {
  EXPECT_THAT(LocalityDomainMemoryKind(0), IsOkAndHolds("locality_domain:0"));
  EXPECT_THAT(LocalityDomainMemoryKind(31), IsOkAndHolds("locality_domain:31"));
  EXPECT_THAT(ParseLocalityDomainMemoryKind("locality_domain:0"),
              IsOkAndHolds(0));
  EXPECT_THAT(ParseLocalityDomainMemoryKind("locality_domain:31"),
              IsOkAndHolds(31));
}

TEST(LocalityDomainMemoryTest, KindIdsUseCanonicalKindFingerprint) {
  EXPECT_THAT(
      LocalityDomainMemoryKindId(0),
      IsOkAndHolds(static_cast<int>(tsl::Fingerprint32("locality_domain:0"))));
  EXPECT_THAT(
      LocalityDomainMemoryKindId(31),
      IsOkAndHolds(static_cast<int>(tsl::Fingerprint32("locality_domain:31"))));
  ASSERT_OK_AND_ASSIGN(absl::string_view first, LocalityDomainMemoryKind(1));
  ASSERT_OK_AND_ASSIGN(absl::string_view second, LocalityDomainMemoryKind(1));
  EXPECT_EQ(first.data(), second.data());
}

TEST(LocalityDomainMemoryTest, RejectsOutOfRangeKindIds) {
  EXPECT_THAT(LocalityDomainMemoryKind(-1),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(LocalityDomainMemoryKind(kMaxLocalityDomains),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(LocalityDomainMemoryKind(std::numeric_limits<int64_t>::max()),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(LocalityDomainMemoryKindId(-1),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(LocalityDomainMemoryKindId(kMaxLocalityDomains),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(LocalityDomainMemoryKindId(std::numeric_limits<int64_t>::max()),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(LocalityDomainMemoryTest, RejectsInvalidOrNonCanonicalKinds) {
  for (const char* memory_kind :
       {"", "device", "locality_domain:", "locality_domain:-1",
        "locality_domain:+1", "locality_domain:01", "locality_domain: 1",
        "locality_domain:1 ", "locality_domain:32",
        "locality_domain:9223372036854775808"}) {
    EXPECT_THAT(ParseLocalityDomainMemoryKind(memory_kind),
                StatusIs(absl::StatusCode::kInvalidArgument))
        << memory_kind;
  }
}

}  // namespace
}  // namespace stream_executor::gpu
