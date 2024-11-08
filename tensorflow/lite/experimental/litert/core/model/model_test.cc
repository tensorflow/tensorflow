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

#include "tensorflow/lite/experimental/litert/core/model/model.h"

#include <cstdint>
#include <memory>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/test/common.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace litert::internal {

namespace {

TEST(ModelTest, GetMetadata) {
  LiteRtModelT model;
  model.flatbuffer_model = std::make_unique<tflite::ModelT>();

  static constexpr absl::string_view kMetadata = "VALUE";
  static constexpr absl::string_view kKey = "KEY";

  ASSERT_STATUS_OK(
      model.PushMetadata(kKey, OwningBufferRef<uint8_t>(kMetadata)));
  ASSERT_RESULT_OK_ASSIGN(auto found_metadata, model.FindMetadata(kKey));

  EXPECT_EQ(found_metadata.StrView(), kMetadata);
}

TEST(ModelTest, MetadataDNE) {
  LiteRtModelT model;
  model.flatbuffer_model = std::make_unique<tflite::ModelT>();

  auto res = model.FindMetadata("FOO");
  ASSERT_FALSE(res.HasValue());
}

}  // namespace
}  // namespace litert::internal
