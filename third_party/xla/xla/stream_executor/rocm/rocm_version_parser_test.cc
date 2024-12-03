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

#include "xla/stream_executor/rocm/rocm_version_parser.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "rocm/include/hip/hip_version.h"
#include "xla/stream_executor/semantic_version.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/test.h"

namespace stream_executor {

namespace {
using tsl::testing::IsOkAndHolds;
using tsl::testing::StatusIs;

TEST(ParseRocmVersionTest, Simple) {
  EXPECT_THAT(stream_executor::ParseRocmVersion(60'100'002),
              IsOkAndHolds(SemanticVersion(6, 1, 2)));
}

TEST(RocmVersionParserTest, NegativeIntegerIsNotAValidVersion) {
  EXPECT_THAT(ParseRocmVersion(-42),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(RocmVersionParserTest, AlignsWithHIPVersion) {
  EXPECT_THAT(ParseRocmVersion(HIP_VERSION),
              IsOkAndHolds(SemanticVersion{HIP_VERSION_MAJOR, HIP_VERSION_MINOR,
                                           HIP_VERSION_PATCH}));
}

}  // namespace
}  // namespace stream_executor
