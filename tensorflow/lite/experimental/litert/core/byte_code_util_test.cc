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

#include "tensorflow/lite/experimental/litert/core/byte_code_util.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/test/common.h"

namespace litert::internal {

namespace {

using ::testing::StartsWith;

static constexpr absl::string_view kSocModel = "TestSocModel";
static constexpr absl::string_view kSocMan = "TestSocMan";
static constexpr Serialization kSerialization = Serialization::kAppend;

TEST(TestBuildStamp, MakeBuildStampInputsTooLarge) {
  // NOLINTNEXTLINE
  std::string long_manufacturer(256, 'a');
  auto res = MakeBuildStamp(long_manufacturer, kSocModel, kSerialization);
  EXPECT_EQ(res.Error().Status(), kLiteRtStatusErrorInvalidArgument);
}

TEST(TestBuildStamp, MakeBuildStamp) {
  auto stamp = MakeBuildStamp(kSocMan, kSocModel, kSerialization);
  auto pstamp = ParseBuildStamp(*stamp);
  auto [man, model, serial] = *pstamp;
  EXPECT_EQ(man, kSocMan);
  EXPECT_EQ(model, kSocModel);
  EXPECT_EQ(serial, kSerialization);
}

TEST(TestByteCodePlaceholder, ParseBadPlaceholder) {
  OwningBufferRef<uint8_t> placeholder;
  auto res = ParseByteCodePlaceholder(placeholder);
  EXPECT_EQ(res.Error().Status(), kLiteRtStatusErrorInvalidArgument);
}

TEST(TestByteCodePlaceholder, BuildAndParseEmptyInvalid) {
  auto placeholder = MakeByteCodePlaceholder();
  ASSERT_THAT(placeholder.StrView(), StartsWith(kByteCodePrefix));
  auto res = ParseByteCodePlaceholder(placeholder);
  EXPECT_EQ(res.Error().Status(), kLiteRtStatusErrorInvalidArgument);
}

TEST(TestByteCodePlaceholder, BuildAndFinishByteCodePlaceholder) {
  auto placeholder = MakeByteCodePlaceholder();

  static constexpr size_t kByteCodeSize = 200;
  LITERT_ASSERT_STATUS_OK(
      FinishByteCodePlaceholders(placeholder, kByteCodeSize));

  auto p_placeholder = ParseByteCodePlaceholder(placeholder);
  auto [offset, size] = *p_placeholder;
  EXPECT_EQ(offset, placeholder.Size());
  EXPECT_EQ(size, kByteCodeSize);
}

TEST(TestByteCodePlaceholder, BuildAndFinishByteCodePlaceholderTooLarge) {
  auto placeholder = MakeByteCodePlaceholder();

  static constexpr size_t kByteCodeSize = std::numeric_limits<size_t>::max();
  LITERT_ASSERT_STATUS_HAS_CODE(
      FinishByteCodePlaceholders(placeholder, kByteCodeSize),
      kLiteRtStatusErrorInvalidArgument);
}

TEST(TestExecInfo, ExecInfo) {
  auto exec_info = MakeExecInfo("entry_point", "key");
  auto p_exec_info = ParseExecInfo(*exec_info);
  auto [entry_point, key] = *p_exec_info;
  EXPECT_EQ(entry_point, "entry_point");
  EXPECT_EQ(key, "key");
}

TEST(TestExecInfo, ExecInfoTooLarge) {
  // NOLINTNEXTLINE
  std::string long_entry_point(256, 'a');
  auto res = MakeExecInfo(long_entry_point, "key");
  EXPECT_EQ(res.Error().Status(), kLiteRtStatusErrorInvalidArgument);
}

}  // namespace

}  // namespace litert::internal
