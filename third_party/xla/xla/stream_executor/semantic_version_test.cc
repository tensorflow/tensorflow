/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/stream_executor/semantic_version.h"

#include <algorithm>
#include <array>
#include <sstream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/hash/hash_testing.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/test.h"

namespace stream_executor {
namespace {

TEST(SemanticVersion, Construction) {
  SemanticVersion version{1, 2, 3};
  EXPECT_EQ(version.major(), 1);
  EXPECT_EQ(version.minor(), 2);
  EXPECT_EQ(version.patch(), 3);
}

TEST(SemanticVersion, ConstructionFromArray) {
  SemanticVersion version{std::array<unsigned, 3>{1, 2, 3}};
  EXPECT_EQ(version.major(), 1);
  EXPECT_EQ(version.minor(), 2);
  EXPECT_EQ(version.patch(), 3);
}

TEST(SemanticVersion, Mutation) {
  SemanticVersion version{0, 0, 0};
  version.major() = 1;
  version.minor() = 2;
  version.patch() = 3;

  EXPECT_EQ(version.major(), 1);
  EXPECT_EQ(version.minor(), 2);
  EXPECT_EQ(version.patch(), 3);
}

TEST(SemanticVersion, ParseFromStringSuccess) {
  absl::StatusOr<SemanticVersion> version =
      SemanticVersion::ParseFromString("1.2.3");
  ASSERT_THAT(version, tsl::testing::IsOk());
  EXPECT_EQ(version->major(), 1);
  EXPECT_EQ(version->minor(), 2);
  EXPECT_EQ(version->patch(), 3);
}

TEST(SemanticVersion, ParseFromStringInvalid) {
  auto test = [](absl::string_view str) {
    absl::StatusOr<SemanticVersion> version =
        SemanticVersion::ParseFromString(str);
    EXPECT_THAT(version,
                tsl::testing::StatusIs(absl::StatusCode::kInvalidArgument));
  };

  test("1.2");
  test("1.2.3dev5");
}

TEST(SemanticVersion, ToString) {
  SemanticVersion version{1, 2, 3};
  EXPECT_EQ(version.ToString(), "1.2.3");
}

TEST(SemanticVersion, AbslStringify) {
  SemanticVersion version{1, 2, 3};
  EXPECT_EQ(absl::StrCat(version), version.ToString());
}

TEST(SemanticVersion, OStream) {
  SemanticVersion version{1, 2, 3};

  std::ostringstream os;
  os << version;
  EXPECT_EQ(os.str(), version.ToString());
}

TEST(SemanticVersion, Equality) {
  SemanticVersion version{1, 2, 3};
  SemanticVersion other{1, 2, 4};

  EXPECT_EQ(version, version);
  EXPECT_FALSE(version != version);

  EXPECT_NE(version, other);
  EXPECT_FALSE(version == other);
}

TEST(SemanticVersion, Ordering) {
  std::array<SemanticVersion, 5> versions = {
      SemanticVersion{3, 3, 3}, SemanticVersion{0, 0, 0},
      SemanticVersion{1, 2, 3}, SemanticVersion{1, 2, 4},
      SemanticVersion{1, 3, 0}};
  std::sort(versions.begin(), versions.end());
  EXPECT_THAT(versions, testing::ElementsAre(
                            SemanticVersion{0, 0, 0}, SemanticVersion{1, 2, 3},
                            SemanticVersion{1, 2, 4}, SemanticVersion{1, 3, 0},
                            SemanticVersion{3, 3, 3}));
}

TEST(SemanticVersion, Hash) {
  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly({
      SemanticVersion{0, 0, 0},
      SemanticVersion{1, 2, 3},
      SemanticVersion{1, 2, 4},
      SemanticVersion{1, 3, 0},
      SemanticVersion{3, 3, 3},
  }));
}

}  // namespace
}  // namespace stream_executor
