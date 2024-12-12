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
#include "tensorflow/lite/experimental/litert/test/test_macros.h"

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
  auto tfl_model = flatbuffer->Unpack();

  LITERT_ASSERT_STATUS_OK(PushMetadata(
      kKey, *tfl_model, BufferRef<uint8_t>(kData.data(), kData.size())));

  auto metadata = GetMetadata(kKey, *tfl_model);
  ASSERT_TRUE(metadata);
  EXPECT_EQ(metadata->StrView(), kData);
}

TEST(FlatbufferToolsTest, GetMetadataNotFound) {
  auto flatbuffer = TestFlatbuffer();
  auto tfl_model = flatbuffer->Unpack();
  ASSERT_NE(flatbuffer, nullptr);
  EXPECT_FALSE(GetMetadata(kKey, *tfl_model));
}

TEST(FlatbufferToolsTest, TflBuffer) {
  auto flatbuffer = TestFlatbuffer();
  ASSERT_NE(flatbuffer, nullptr);
  auto tfl_model = flatbuffer->Unpack();

  auto ind = PushTflBuffer((*tfl_model),
                           BufferRef<uint8_t>(kData.data(), kData.size()));
  ASSERT_TRUE(ind);

  auto buf = GetTflBuffer((*tfl_model), *ind);
  ASSERT_TRUE(buf);
  ASSERT_EQ(buf->StrView(), kData);
}

TEST(FlatbufferToolsTest, GetTflBufferNotFound) {
  auto flatbuffer = TestFlatbuffer();
  ASSERT_NE(flatbuffer, nullptr);
  auto tfl_model = flatbuffer->Unpack();

  auto buf = GetTflBuffer((*tfl_model), 100);
  ASSERT_FALSE(buf);
}

TEST(FlatbufferToolsTest, GetTflOpCode) {
  auto flatbuffer = TestFlatbuffer();
  ASSERT_NE(flatbuffer, nullptr);
  auto tfl_model = flatbuffer->Unpack();

  auto op_code = GetTflOpCode((*tfl_model), 0);
  ASSERT_TRUE(op_code);
}

TEST(FlatbufferToolsTest, GetTflOpCodeNotFound) {
  auto flatbuffer = TestFlatbuffer();
  ASSERT_NE(flatbuffer, nullptr);
  auto tfl_model = flatbuffer->Unpack();

  auto op_code = GetTflOpCode((*tfl_model), 100);
  ASSERT_FALSE(op_code);
}

TEST(FlatbufferToolsTest, StaticTensorTypeTest) {
  auto flatbuffer = TestFlatbuffer();
  auto tfl_model = flatbuffer->Unpack();
  auto& tensor = tfl_model->subgraphs.front()->tensors.front();

  TflShapeInfo shape(*tensor);

  ASSERT_TRUE(IsRankedTensorType(shape));
  ASSERT_TRUE(IsStaticTensorType(shape));

  auto static_shape = AsStaticShape(shape);

  ASSERT_TRUE(static_shape);
  ASSERT_THAT(*static_shape, ElementsAreArray({2, 2}));
}

TEST(FlatbufferToolsTest, UnrankedTensorTypeTest) {
  auto flatbuffer = TestFlatbuffer("unranked_tensor.tflite");
  auto tfl_model = flatbuffer->Unpack();
  auto& tensor = tfl_model->subgraphs.front()->tensors.front();

  TflShapeInfo shape(*tensor);

  ASSERT_FALSE(IsRankedTensorType(shape));
}

TEST(FlatbufferToolsTest, RankedDynamicTensorTypeTest) {
  auto flatbuffer = TestFlatbuffer("dynamic_shape_tensor.tflite");
  auto tfl_model = flatbuffer->Unpack();
  auto& tensor = tfl_model->subgraphs.front()->tensors.front();

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
  auto tfl_model = flatbuffer->Unpack();
  auto& tensor = tfl_model->subgraphs.front()->tensors.front();

  const auto* const q_parms = tensor->quantization.get();

  ASSERT_TRUE(IsQuantized(q_parms));
  EXPECT_TRUE(IsPerTensorQuantized(q_parms));

  auto per_tensor = AsPerTensorQparams(q_parms);
  ASSERT_TRUE(per_tensor);
}

TEST(FlatbufferToolsTest, PerChannelQuantizedTest) {
  auto flatbuffer = TestFlatbuffer("static_w8_a16_quantized_k_einsum.tflite");
  auto tfl_model = flatbuffer->Unpack();
  auto& tensor = tfl_model->subgraphs.front()->tensors[1];

  const auto* const q_parms = tensor->quantization.get();

  ASSERT_TRUE(IsQuantized(q_parms));
  EXPECT_TRUE(IsPerChannelQuantized(q_parms));

  auto per_channel = AsPerChannelQparams(q_parms);
  ASSERT_TRUE(per_channel);
}

}  // namespace
}  // namespace litert::internal
