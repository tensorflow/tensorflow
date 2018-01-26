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
#include "tensorflow/contrib/lite/tools/verifier.h"
#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/util.h"
#include <gtest/gtest.h>
#include "tensorflow/contrib/lite/allocation.h"
#include "tensorflow/contrib/lite/error_reporter.h"
#include "tensorflow/contrib/lite/schema/schema_generated.h"
#include "tensorflow/contrib/lite/testing/util.h"
#include "tensorflow/contrib/lite/version.h"

namespace tflite {

using flatbuffers::FlatBufferBuilder;
using flatbuffers::Offset;
using flatbuffers::Vector;

// Class that abstracts the list of buffers at the end of the TF Lite structure
class DeferredBufferWriter {
 public:
  DeferredBufferWriter() {
    data_.push_back({});  // sentinel empty buffer.
  }

  Offset<Vector<Offset<Buffer>>> BuildBuffers(FlatBufferBuilder *builder) {
    std::vector<Offset<Buffer>> buffer_vector;
    for (const auto &vec : data_) {
      auto data_buffer = builder->CreateVector(vec.data(), vec.size());
      buffer_vector.push_back(tflite::CreateBuffer(*builder, data_buffer));
    }
    return builder->CreateVector(buffer_vector);
  }

  // Registers a buffer index and takes ownership of the data to write to it.
  int Record(std::vector<uint8_t> data) {
    int buffer_index = data_.size();
    data_.emplace_back(std::move(data));
    return buffer_index;
  }

 private:
  std::vector<std::vector<unsigned char>> data_;
};

TEST(VerifyModel, TestEmptyModel) {
  FlatBufferBuilder builder;
  auto model = CreateModel(builder, /*version=*/TFLITE_SCHEMA_VERSION,
                           /*operator_codes=*/0, /*subgraphs=*/0,
                           /*description=*/0, /*buffers=*/0);
  ::tflite::FinishModelBuffer(builder, model);

  ASSERT_TRUE(Verify(builder.GetBufferPointer(), builder.GetSize()));
}

TEST(VerifyModel, TestSimpleModel) {
  FlatBufferBuilder builder;
  auto inputs = builder.CreateVector<int32_t>({0});
  auto outputs = builder.CreateVector<int32_t>({1});
  auto operator_codes = builder.CreateVector(std::vector<Offset<OperatorCode>>{
      CreateOperatorCodeDirect(builder, BuiltinOperator_CUSTOM, "test")});
  auto operators =
      builder.CreateVector(std::vector<Offset<Operator>>{CreateOperator(
          builder, /*opcode_index=*/0,
          /*inputs=*/builder.CreateVector<int32_t>({0}),
          /*outputs=*/builder.CreateVector<int32_t>({1}), BuiltinOptions_NONE,
          /*builtin_options=*/0,
          /*custom_options=*/0, ::tflite::CustomOptionsFormat_FLEXBUFFERS)});
  std::vector<int> shape;
  auto tensors = builder.CreateVector(std::vector<Offset<Tensor>>{
      CreateTensorDirect(builder, &shape, TensorType_INT32, /*buffer=*/0,
                         "input", /*quantization=*/0),
      CreateTensorDirect(builder, &shape, TensorType_INT32, /*buffer=*/0,
                         "output", /*quantization=*/0)});
  auto subgraph = std::vector<Offset<SubGraph>>(
      {CreateSubGraph(builder, tensors, inputs, outputs, operators,
                      builder.CreateString("Main"))});

  auto model = CreateModel(builder, TFLITE_SCHEMA_VERSION, operator_codes,
                           builder.CreateVector(subgraph),
                           builder.CreateString("SmartReply"), /*buffers=*/0);

  ::tflite::FinishModelBuffer(builder, model);
  ASSERT_TRUE(Verify(builder.GetBufferPointer(), builder.GetSize()));
}

TEST(VerifyModel, TestCorruptedData) {
  string model = "123";
  ASSERT_FALSE(Verify(model.data(), model.size()));
}

TEST(VerifyModel, TestUnsupportedVersion) {
  FlatBufferBuilder builder;
  auto model = CreateModel(builder, /*version=*/1, /*operator_codes=*/0,
                           /*subgraphs=*/0, /*description=*/0, /*buffers=*/0);
  ::tflite::FinishModelBuffer(builder, model);
  ASSERT_FALSE(Verify(builder.GetBufferPointer(), builder.GetSize()));
}

TEST(VerifyModel, TestRandomModificationIsNotAllowed) {
  FlatBufferBuilder builder;
  auto model = CreateModel(builder, /*version=*/TFLITE_SCHEMA_VERSION,
                           /*operator_codes=*/0,
                           /*subgraphs=*/0, /*description=*/0, /*buffers=*/0);
  ::tflite::FinishModelBuffer(builder, model);

  string model_content(reinterpret_cast<char *>(builder.GetBufferPointer()),
                       builder.GetSize());
  for (int i = 0; i < model_content.size(); i++) {
    model_content[i] = (model_content[i] + 137) % 255;
    EXPECT_FALSE(Verify(model_content.data(), model_content.size()))
        << "Fail at position: " << i;
  }
}

// TODO(yichengfan): make up malicious files to test with.

}  // namespace tflite

int main(int argc, char **argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
