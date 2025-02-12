// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/litert/core/build_stamp.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/test/matchers.h"

namespace litert::internal {

namespace {

using ::testing::litert::IsError;

static constexpr absl::string_view kSocModel = "TestSocModel";
static constexpr absl::string_view kSocMan = "TestSocMan";

TEST(TestBuildStamp, MakeBuildStampInputsTooLarge) {
  // NOLINTNEXTLINE
  std::string long_manufacturer(256, 'a');
  auto res = MakeBuildStamp(long_manufacturer, kSocModel);
  EXPECT_THAT(res, IsError(kLiteRtStatusErrorInvalidArgument));
}

TEST(TestBuildStamp, MakeBuildStamp) {
  auto stamp = MakeBuildStamp(kSocMan, kSocModel);
  auto pstamp = ParseBuildStamp(*stamp);
  auto [man, model] = *pstamp;
  EXPECT_EQ(man, kSocMan);
  EXPECT_EQ(model, kSocModel);
}

}  // namespace

}  // namespace litert::internal
