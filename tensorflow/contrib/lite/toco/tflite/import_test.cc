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
#include "tensorflow/contrib/lite/toco/tflite/import.h"

#include "flatbuffers/flexbuffers.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/contrib/lite/schema/schema_generated.h"
#include "tensorflow/contrib/lite/version.h"

namespace toco {

namespace tflite {
namespace {

using ::testing::ElementsAre;

class ImportTest : public ::testing::Test {
 protected:
  template <typename T>
  flatbuffers::Offset<flatbuffers::Vector<unsigned char>> CreateDataVector(
      const std::vector<T>& data) {
    return builder_.CreateVector(reinterpret_cast<const uint8_t*>(data.data()),
                                 sizeof(T) * data.size());
  }
  // This is a very simplistic model. We are not interested in testing all the
  // details here, since tf.mini's testing framework will be exercising all the
  // conversions multiple times, and the conversion of operators is tested by
  // separate unittests.
  void BuildTestModel() {
    // The tensors
    auto q = ::tflite::CreateQuantizationParameters(
        builder_,
        /*min=*/builder_.CreateVector<float>({0.1f}),
        /*max=*/builder_.CreateVector<float>({0.2f}),
        /*scale=*/builder_.CreateVector<float>({0.3f}),
        /*zero_point=*/builder_.CreateVector<int64_t>({100ll}));
    auto buf0 = ::tflite::CreateBuffer(builder_, CreateDataVector<float>({}));
    auto buf1 =
        ::tflite::CreateBuffer(builder_, CreateDataVector<float>({1.0f, 2.0f}));
    auto buf2 =
        ::tflite::CreateBuffer(builder_, CreateDataVector<float>({3.0f}));
    auto buffers = builder_.CreateVector(
        std::vector<flatbuffers::Offset<::tflite::Buffer>>({buf0, buf1, buf2}));
    auto t1 = ::tflite::CreateTensor(builder_,
                                     builder_.CreateVector<int>({1, 2, 3, 4}),
                                     ::tflite::TensorType_FLOAT32, 1,
                                     builder_.CreateString("tensor_one"), q);
    auto t2 =
        ::tflite::CreateTensor(builder_, builder_.CreateVector<int>({2, 1}),
                               ::tflite::TensorType_FLOAT32, 2,
                               builder_.CreateString("tensor_two"), q);
    auto tensors = builder_.CreateVector(
        std::vector<flatbuffers::Offset<::tflite::Tensor>>({t1, t2}));

    // The operator codes.
    auto c1 =
        ::tflite::CreateOperatorCode(builder_, ::tflite::BuiltinOperator_CUSTOM,
                                     builder_.CreateString("custom_op_one"));
    auto c2 = ::tflite::CreateOperatorCode(
        builder_, ::tflite::BuiltinOperator_CONV_2D, 0);
    auto opcodes = builder_.CreateVector(
        std::vector<flatbuffers::Offset<::tflite::OperatorCode>>({c1, c2}));

    auto subgraph = ::tflite::CreateSubGraph(builder_, tensors, 0, 0, 0);
    std::vector<flatbuffers::Offset<::tflite::SubGraph>> subgraph_vector(
        {subgraph});
    auto subgraphs = builder_.CreateVector(subgraph_vector);
    auto s = builder_.CreateString("");
    builder_.Finish(::tflite::CreateModel(builder_, TFLITE_SCHEMA_VERSION,
                                          opcodes, subgraphs, s, buffers));

    input_model_ = ::tflite::GetModel(builder_.GetBufferPointer());
  }
  string InputModelAsString() {
    return string(reinterpret_cast<char*>(builder_.GetBufferPointer()),
                  builder_.GetSize());
  }
  flatbuffers::FlatBufferBuilder builder_;
  // const uint8_t* buffer_ = nullptr;
  const ::tflite::Model* input_model_ = nullptr;
};

TEST_F(ImportTest, LoadTensorsTable) {
  BuildTestModel();

  details::TensorsTable tensors;
  details::LoadTensorsTable(*input_model_, &tensors);
  EXPECT_THAT(tensors, ElementsAre("tensor_one", "tensor_two"));
}

TEST_F(ImportTest, LoadOperatorsTable) {
  BuildTestModel();

  details::OperatorsTable operators;
  details::LoadOperatorsTable(*input_model_, &operators);
  EXPECT_THAT(operators, ElementsAre("custom_op_one", "CONV_2D"));
}

TEST_F(ImportTest, Tensors) {
  BuildTestModel();

  auto model = Import(ModelFlags(), InputModelAsString());

  ASSERT_GT(model->arrays.count("tensor_one"), 0);
  Array& a1 = model->GetArray("tensor_one");
  EXPECT_EQ(ArrayDataType::kFloat, a1.data_type);
  EXPECT_THAT(a1.GetBuffer<ArrayDataType::kFloat>().data,
              ElementsAre(1.0f, 2.0f));
  ASSERT_TRUE(a1.has_shape());
  EXPECT_THAT(a1.shape().dims(), ElementsAre(1, 2, 3, 4));

  const auto& mm = a1.minmax;
  ASSERT_TRUE(mm.get());
  EXPECT_FLOAT_EQ(0.1, mm->min);
  EXPECT_FLOAT_EQ(0.2, mm->max);

  const auto& q = a1.quantization_params;
  ASSERT_TRUE(q.get());
  EXPECT_FLOAT_EQ(0.3, q->scale);
  EXPECT_EQ(100, q->zero_point);
}

// TODO(ahentz): still need tests for Operators and IOTensors.

}  // namespace
}  // namespace tflite

}  // namespace toco
