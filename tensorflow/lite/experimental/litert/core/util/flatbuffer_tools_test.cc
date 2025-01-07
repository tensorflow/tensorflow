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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/test/common.h"

namespace litert::internal {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::Lt;

FlatbufferWrapper::Ptr TestFlatbuffer(
    absl::string_view filename = "one_mul.tflite") {
  const auto tfl_path = testing::GetTestFilePath(filename);
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

TEST(FlatbufferToolsTest, StaticTensorTypeTest) {
  auto flatbuffer = TestFlatbuffer();
  auto& tensor = flatbuffer->UnpackedModel().subgraphs.front()->tensors.front();

  TflShapeInfo shape(*tensor);

  ASSERT_TRUE(IsRankedTensorType(shape));
  ASSERT_TRUE(IsStaticTensorType(shape));

  auto static_shape = AsStaticShape(shape);

  ASSERT_TRUE(static_shape);
  ASSERT_THAT(*static_shape, ElementsAreArray({2, 2}));
}

TEST(FlatbufferToolsTest, UnrankedTensorTypeTest) {
  auto flatbuffer = TestFlatbuffer("unranked_tensor.tflite");
  auto& tensor = flatbuffer->UnpackedModel().subgraphs.front()->tensors.front();

  TflShapeInfo shape(*tensor);

  ASSERT_FALSE(IsRankedTensorType(shape));
}

TEST(FlatbufferToolsTest, RankedDynamicTensorTypeTest) {
  auto flatbuffer = TestFlatbuffer("dynamic_shape_tensor.tflite");
  auto& tensor = flatbuffer->UnpackedModel().subgraphs.front()->tensors.front();

  TflShapeInfo shape(*tensor);

  ASSERT_TRUE(IsRankedTensorType(shape));
  ASSERT_FALSE(IsStaticTensorType(shape));

  auto dyn_shape = AsDynamicShape(shape);

  ASSERT_TRUE(dyn_shape);
  ASSERT_THAT(*dyn_shape, ElementsAre(Lt(0), 2));
}

TEST(FlatbufferToolsTest, PerTensorQuantizedTest) {
  auto flatbuffer =
      TestFlatbuffer("single_add_default_a16w8_recipe_quantized.tflite");
  auto& tensor = flatbuffer->UnpackedModel().subgraphs.front()->tensors.front();

  const auto* const q_parms = tensor->quantization.get();

  ASSERT_TRUE(IsQuantized(q_parms));
  EXPECT_TRUE(IsPerTensorQuantized(q_parms));

  auto per_tensor = AsPerTensorQparams(q_parms);
  ASSERT_TRUE(per_tensor);
}

}  // namespace
}  // namespace litert::internal
