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

#include <algorithm>
#include <cstdint>
#include <ctime>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/builddata/builddata.h"
#include "xla/tsl/builddata/utils.h"
#include "xla/tsl/platform/logging.h"

namespace {
namespace builddata = ::tsl::builddata;
using ::testing::HasSubstr;
using ::tsl::builddata::ParseChangelist;
using ::tsl::builddata::ParseMintStatus;

TEST(BuildDataUtilsTest, ParseChangelist) {
  EXPECT_EQ(ParseChangelist("123456"), 123456);
  EXPECT_EQ(ParseChangelist("unknown"), 0);
  EXPECT_EQ(ParseChangelist(""), -1);
  EXPECT_EQ(ParseChangelist(nullptr), -1);
  EXPECT_EQ(ParseChangelist("not_a_number"), -2);
}

TEST(BuildDataUtilsTest, ParseMintStatus) {
  EXPECT_EQ(ParseMintStatus("mint"), 1);
  EXPECT_EQ(ParseMintStatus("modified"), 0);
  EXPECT_EQ(ParseMintStatus("unknown"), -1);
}

TEST(BuildDataTest, ExerciseAPI) {
  // Log all values to exercise the API and show them in test logs
  LOG(INFO) << "BuildInfo: " << builddata::BuildInfo();
  LOG(INFO) << "BuildId: " << builddata::BuildId();
  LOG(INFO) << "BuildDir: " << builddata::BuildDir();
  LOG(INFO) << "SourceUri: " << builddata::SourceUri();
  LOG(INFO) << "BuildHost: " << builddata::BuildHost();
  LOG(INFO) << "BuildTarget: " << builddata::BuildTarget();
  LOG(INFO) << "TargetName: " << builddata::TargetName();
  LOG(INFO) << "BuildLabel: " << builddata::BuildLabel();
  LOG(INFO) << "BuildClient: " << builddata::BuildClient();
  LOG(INFO) << "Timestamp: " << builddata::Timestamp();
  LOG(INFO) << "TimestampAsInt: " << builddata::TimestampAsInt();
  LOG(INFO) << "SourceRevision: " << builddata::SourceRevision();
  LOG(INFO) << "SourceRevisionAsInt: " << builddata::SourceRevisionAsInt();
  LOG(INFO) << "BaselineSourceRevision: "
            << builddata::BaselineSourceRevision();
  LOG(INFO) << "BaselineSourceRevisionAsInt: "
            << builddata::BaselineSourceRevisionAsInt();
  LOG(INFO) << "ClientStatus: " << static_cast<int>(builddata::ClientStatus());
  LOG(INFO) << "ClientStatusAsString: " << builddata::ClientStatusAsString();
  LOG(INFO) << "CompilerTarget: " << builddata::CompilerTarget();

  // Basic assertions
  EXPECT_FALSE(builddata::BuildInfo().empty());
  EXPECT_FALSE(builddata::BuildHost().empty());
  EXPECT_FALSE(builddata::BuildDir().empty());
  EXPECT_FALSE(builddata::BuildTarget().empty());
  EXPECT_FALSE(builddata::TargetName().empty());
  EXPECT_FALSE(builddata::CompilerTarget().empty());

  // Check consistency of SourceRevision
  absl::string_view cl = builddata::SourceRevision();
  int64_t cl_int = builddata::SourceRevisionAsInt();
  if (cl.empty()) {
    EXPECT_EQ(cl_int, -1);
  } else if (cl == "<unknown>") {
    EXPECT_EQ(cl_int, 0);
  } else {
    bool is_numeric =
        !cl.empty() && std::all_of(cl.begin(), cl.end(), absl::ascii_isdigit);
    if (is_numeric) {
      EXPECT_GT(cl_int, 0);
    } else {
      EXPECT_EQ(cl_int, -2);
    }
  }

  // Check consistency of BaselineSourceRevision
  absl::string_view bcl = builddata::BaselineSourceRevision();
  int64_t bcl_int = builddata::BaselineSourceRevisionAsInt();
  if (bcl.empty()) {
    EXPECT_EQ(bcl_int, -1);
  } else if (bcl == "<unknown>") {
    EXPECT_EQ(bcl_int, 0);
  } else {
    bool is_numeric = !bcl.empty() &&
                      std::all_of(bcl.begin(), bcl.end(), absl::ascii_isdigit);
    if (is_numeric) {
      EXPECT_GT(bcl_int, 0);
    } else {
      EXPECT_EQ(bcl_int, -2);
    }
  }

  // Check ClientStatus consistency
  builddata::ClientStatusType status = builddata::ClientStatus();
  absl::string_view status_str = builddata::ClientStatusAsString();
  if (status == builddata::MINT) {
    EXPECT_EQ(status_str, "mint");
  } else if (status == builddata::MODIFIED) {
    EXPECT_EQ(status_str, "modified");
  } else {
    EXPECT_EQ(status, builddata::UNKNOWN);
    EXPECT_EQ(status_str, "unknown");
  }

  // Check Timestamp consistency
  time_t ts_int = builddata::TimestampAsInt();
  std::string ts_str = builddata::Timestamp();
  if (ts_int > 0) {
    EXPECT_THAT(ts_str, HasSubstr(std::to_string(ts_int)));
  }
  if (absl::StartsWith(ts_str, "Built on ")) {
    EXPECT_THAT(ts_str, HasSubstr("[TZ=America/Los_Angeles]"));
  }
}

}  // namespace
