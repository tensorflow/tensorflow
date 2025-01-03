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

#include "tensorflow/lite/experimental/litert/core/byte_code/schema.h"

#include <cstdint>
#include <string>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"

namespace litert::internal {
namespace {

static constexpr absl::string_view kBackendId = "foo";
static constexpr absl::string_view kEntryPointName = "bar";
static constexpr absl::string_view kMetadataKey = "key";

TEST(ExecInfoTest, SerializeAndParse) {
  auto serialized =
      SerializeExecInfo({kBackendId, kEntryPointName, kMetadataKey});
  ASSERT_TRUE(serialized);
  auto parsed = ExecInfoFromBuf(*serialized);
  ASSERT_TRUE(parsed);
  EXPECT_EQ(parsed->backend_id, kBackendId);
  EXPECT_EQ(parsed->entrypoint_name, kEntryPointName);
  EXPECT_EQ(parsed->metadata_key, kMetadataKey);
}

TEST(ExecInfoTest, SerializeTooLarge) {
  // NOLINTNEXTLINE
  std::string long_backend_id(256, 'a');
  auto res =
      SerializeExecInfo({long_backend_id, kEntryPointName, kMetadataKey});
  EXPECT_EQ(res.Error().Status(), kLiteRtStatusErrorInvalidArgument);
}

TEST(ExecInfoTest, ParseInvalid) {
  OwningBufferRef<uint8_t> buf(absl::string_view("not_exec_info"));
  auto res = ExecInfoFromBuf(buf);
  EXPECT_EQ(res.Error().Status(), kLiteRtStatusErrorInvalidArgument);
}

}  // namespace
}  // namespace litert::internal
