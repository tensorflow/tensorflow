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

#include "tensorflow/lite/experimental/litert/core/util/flatbuffer_tools.h"

#include <cstdint>

#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/test/common.h"

namespace litert::internal {
namespace {

FlatbufferWrapper::Ptr TestFlatbuffer() {
  const auto tfl_path = testing::GetTestFilePath("one_mul.tflite");
  return *FlatbufferWrapper::CreateFromTflFile(tfl_path);
}

static const absl::string_view kKey = "MyKey";
static const absl::string_view kData = "MyData";

TEST(FlatbufferToolsTest, Metadata) {
  auto flatbuffer = TestFlatbuffer();
  ASSERT_NE(flatbuffer, nullptr);

  LITERT_ASSERT_STATUS_OK(
      PushMetadata(kKey, flatbuffer->UnpackedModel(),
                   BufferRef<uint8_t>(kData.data(), kData.size())));

  auto metadata = GetMetadata(kKey, flatbuffer->UnpackedModel());
  ASSERT_TRUE(metadata);
  EXPECT_EQ(metadata->StrView(), kData);
}

TEST(FlatbufferToolsTest, GetMetadataNotFound) {
  auto flatbuffer = TestFlatbuffer();
  ASSERT_NE(flatbuffer, nullptr);
  EXPECT_FALSE(GetMetadata(kKey, flatbuffer->UnpackedModel()));
}

TEST(FlatbufferToolsTest, TflBuffer) {
  auto flatbuffer = TestFlatbuffer();
  ASSERT_NE(flatbuffer, nullptr);

  auto ind = PushTflBuffer(flatbuffer->UnpackedModel(),
                           BufferRef<uint8_t>(kData.data(), kData.size()));
  ASSERT_TRUE(ind);

  auto buf = GetTflBuffer(flatbuffer->UnpackedModel(), *ind);
  ASSERT_TRUE(buf);
  ASSERT_EQ(buf->StrView(), kData);
}

TEST(FlatbufferToolsTest, GetTflBufferNotFound) {
  auto flatbuffer = TestFlatbuffer();
  ASSERT_NE(flatbuffer, nullptr);

  auto buf = GetTflBuffer(flatbuffer->UnpackedModel(), 100);
  ASSERT_FALSE(buf);
}

TEST(FlatbufferToolsTest, GetTflOpCode) {
  auto flatbuffer = TestFlatbuffer();
  ASSERT_NE(flatbuffer, nullptr);

  auto op_code = GetTflOpCode(flatbuffer->UnpackedModel(), 0);
  ASSERT_TRUE(op_code);
}

TEST(FlatbufferToolsTest, GetTflOpCodeNotFound) {
  auto flatbuffer = TestFlatbuffer();
  ASSERT_NE(flatbuffer, nullptr);

  auto op_code = GetTflOpCode(flatbuffer->UnpackedModel(), 100);
  ASSERT_FALSE(op_code);
}

}  // namespace
}  // namespace litert::internal
