/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/contrib/lite/toco/tflite/types.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace toco {

namespace tflite {
namespace {

using flatbuffers::FlatBufferBuilder;
using flatbuffers::Offset;
using flatbuffers::Vector;

// These are types that exist in TF Mini but don't have a correspondence
// in TF Lite.
static const ArrayDataType kUnsupportedTocoTypes[] = {ArrayDataType::kNone,
                                                      ArrayDataType::kBool};

// These are TF Lite types for which there is no correspondence in TF Mini.
static const ::tflite::TensorType kUnsupportedTfLiteTypes[] = {
    ::tflite::TensorType_FLOAT16};

// A little helper to match flatbuffer offsets.
MATCHER_P(HasOffset, value, "") { return arg.o == value; }

// Helper function that creates an array, writes it into a flatbuffer, and then
// reads it back in.
template <ArrayDataType T>
Array ToFlatBufferAndBack(std::initializer_list<::toco::DataType<T>> items) {
  // NOTE: This test does not construct the full buffers list. Since
  // Deserialize normally takes a buffer, we need to synthesize one and provide
  // an index that is non-zero so the buffer is not assumed to be empty.
  Array src;
  src.data_type = T;
  src.GetMutableBuffer<T>().data = items;

  Array result;
  flatbuffers::FlatBufferBuilder builder;
  builder.Finish(CreateTensor(builder, 0, DataType::Serialize(T),
                              /*buffer*/ 1));  // Can't use 0 which means empty.
  flatbuffers::FlatBufferBuilder buffer_builder;
  Offset<Vector<uint8_t>> data_buffer =
      DataBuffer::Serialize(src, &buffer_builder);
  buffer_builder.Finish(::tflite::CreateBuffer(buffer_builder, data_buffer));

  auto* tensor =
      flatbuffers::GetRoot<::tflite::Tensor>(builder.GetBufferPointer());
  auto* buffer =
      flatbuffers::GetRoot<::tflite::Buffer>(buffer_builder.GetBufferPointer());
  DataBuffer::Deserialize(*tensor, *buffer, &result);
  return result;
}

TEST(DataType, SupportedTypes) {
  std::vector<std::pair<ArrayDataType, ::tflite::TensorType>> testdata = {
      {ArrayDataType::kUint8, ::tflite::TensorType_UINT8},
      {ArrayDataType::kInt32, ::tflite::TensorType_INT32},
      {ArrayDataType::kInt64, ::tflite::TensorType_INT64},
      {ArrayDataType::kFloat, ::tflite::TensorType_FLOAT32}};
  for (auto x : testdata) {
    EXPECT_EQ(x.second, DataType::Serialize(x.first));
    EXPECT_EQ(x.first, DataType::Deserialize(x.second));
  }
}

TEST(DataType, UnsupportedTypes) {
  for (::tflite::TensorType t : kUnsupportedTfLiteTypes) {
    EXPECT_DEATH(DataType::Deserialize(t), "Unhandled tensor type.");
  }

  // Unsupported types are all serialized as FLOAT32 currently.
  for (ArrayDataType t : kUnsupportedTocoTypes) {
    EXPECT_EQ(::tflite::TensorType_FLOAT32, DataType::Serialize(t));
  }
}

TEST(DataBuffer, EmptyBuffers) {
  flatbuffers::FlatBufferBuilder builder;
  Array array;
  EXPECT_THAT(DataBuffer::Serialize(array, &builder), HasOffset(0));

  builder.Finish(::tflite::CreateTensor(builder));
  auto* tensor =
      flatbuffers::GetRoot<::tflite::Tensor>(builder.GetBufferPointer());
  flatbuffers::FlatBufferBuilder buffer_builder;
  Offset<Vector<uint8_t>> v = buffer_builder.CreateVector<uint8_t>({});
  buffer_builder.Finish(::tflite::CreateBuffer(buffer_builder, v));
  auto* buffer =
      flatbuffers::GetRoot<::tflite::Buffer>(buffer_builder.GetBufferPointer());

  DataBuffer::Deserialize(*tensor, *buffer, &array);
  EXPECT_EQ(nullptr, array.buffer);
}

TEST(DataBuffer, UnsupportedTypes) {
  for (ArrayDataType t : kUnsupportedTocoTypes) {
    flatbuffers::FlatBufferBuilder builder;
    Array array;
    array.data_type = t;
    array.GetMutableBuffer<ArrayDataType::kFloat>();  // This is OK.
    EXPECT_DEATH(DataBuffer::Serialize(array, &builder),
                 "Unhandled array data type.");
  }

  for (::tflite::TensorType t : kUnsupportedTfLiteTypes) {
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(::tflite::CreateTensor(builder, 0, t, /*buffer*/ 1));
    flatbuffers::FlatBufferBuilder buffer_builder;
    Offset<Vector<uint8_t>> v = buffer_builder.CreateVector<uint8_t>({1});
    buffer_builder.Finish(::tflite::CreateBuffer(buffer_builder, v));
    auto* buffer = flatbuffers::GetRoot<::tflite::Buffer>(
        buffer_builder.GetBufferPointer());
    auto* tensor =
        flatbuffers::GetRoot<::tflite::Tensor>(builder.GetBufferPointer());
    Array array;
    EXPECT_DEATH(DataBuffer::Deserialize(*tensor, *buffer, &array),
                 "Unhandled tensor type.");
  }
}

TEST(DataBuffer, Float) {
  Array recovered = ToFlatBufferAndBack<ArrayDataType::kFloat>({1.0f, 2.0f});
  EXPECT_THAT(recovered.GetBuffer<ArrayDataType::kFloat>().data,
              ::testing::ElementsAre(1.0f, 2.0f));
}

TEST(DataBuffer, Uint8) {
  Array recovered = ToFlatBufferAndBack<ArrayDataType::kUint8>({127, 244});
  EXPECT_THAT(recovered.GetBuffer<ArrayDataType::kUint8>().data,
              ::testing::ElementsAre(127, 244));
}

TEST(DataBuffer, Int32) {
  Array recovered = ToFlatBufferAndBack<ArrayDataType::kInt32>({1, 1 << 30});
  EXPECT_THAT(recovered.GetBuffer<ArrayDataType::kInt32>().data,
              ::testing::ElementsAre(1, 1 << 30));
}

TEST(DataBuffer, String) {
  Array recovered = ToFlatBufferAndBack<ArrayDataType::kString>(
      {"AA", "BBB", "Best. String. Ever."});
  EXPECT_THAT(recovered.GetBuffer<ArrayDataType::kString>().data,
              ::testing::ElementsAre("AA", "BBB", "Best. String. Ever."));
}

TEST(Padding, All) {
  EXPECT_EQ(::tflite::Padding_SAME, Padding::Serialize(PaddingType::kSame));
  EXPECT_EQ(PaddingType::kSame, Padding::Deserialize(::tflite::Padding_SAME));

  EXPECT_EQ(::tflite::Padding_VALID, Padding::Serialize(PaddingType::kValid));
  EXPECT_EQ(PaddingType::kValid, Padding::Deserialize(::tflite::Padding_VALID));

  EXPECT_DEATH(Padding::Serialize(static_cast<PaddingType>(10000)),
               "Unhandled padding type.");
  EXPECT_DEATH(Padding::Deserialize(10000), "Unhandled padding.");
}

TEST(ActivationFunction, All) {
  std::vector<
      std::pair<FusedActivationFunctionType, ::tflite::ActivationFunctionType>>
      testdata = {{FusedActivationFunctionType::kNone,
                   ::tflite::ActivationFunctionType_NONE},
                  {FusedActivationFunctionType::kRelu,
                   ::tflite::ActivationFunctionType_RELU},
                  {FusedActivationFunctionType::kRelu6,
                   ::tflite::ActivationFunctionType_RELU6},
                  {FusedActivationFunctionType::kRelu1,
                   ::tflite::ActivationFunctionType_RELU_N1_TO_1}};
  for (auto x : testdata) {
    EXPECT_EQ(x.second, ActivationFunction::Serialize(x.first));
    EXPECT_EQ(x.first, ActivationFunction::Deserialize(x.second));
  }

  EXPECT_DEATH(ActivationFunction::Serialize(
                   static_cast<FusedActivationFunctionType>(10000)),
               "Unhandled fused activation function type.");
  EXPECT_DEATH(ActivationFunction::Deserialize(10000),
               "Unhandled fused activation function type.");
}

}  // namespace
}  // namespace tflite

}  // namespace toco
