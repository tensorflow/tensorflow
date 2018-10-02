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

using flatbuffers::Offset;
using flatbuffers::Vector;
class ImportTest : public ::testing::Test {
 protected:
  template <typename T>
  Offset<Vector<unsigned char>> CreateDataVector(const std::vector<T>& data) {
    return builder_.CreateVector(reinterpret_cast<const uint8_t*>(data.data()),
                                 sizeof(T) * data.size());
  }

  Offset<Vector<Offset<::tflite::Buffer>>> BuildBuffers() {
    auto buf0 = ::tflite::CreateBuffer(builder_, CreateDataVector<float>({}));
    auto buf1 = ::tflite::CreateBuffer(
        builder_, CreateDataVector<float>({1.0f, 2.0f, 3.0f, 4.0f}));
    auto buf2 =
        ::tflite::CreateBuffer(builder_, CreateDataVector<float>({3.0f, 4.0f}));
    return builder_.CreateVector(
        std::vector<Offset<::tflite::Buffer>>({buf0, buf1, buf2}));
  }

  Offset<Vector<Offset<::tflite::Tensor>>> BuildTensors() {
    auto q = ::tflite::CreateQuantizationParameters(
        builder_,
        /*min=*/builder_.CreateVector<float>({0.1f}),
        /*max=*/builder_.CreateVector<float>({0.2f}),
        /*scale=*/builder_.CreateVector<float>({0.3f}),
        /*zero_point=*/builder_.CreateVector<int64_t>({100ll}));
    auto t1 =
        ::tflite::CreateTensor(builder_, builder_.CreateVector<int>({1, 2, 2}),
                               ::tflite::TensorType_FLOAT32, 1,
                               builder_.CreateString("tensor_one"), q);
    auto t2 =
        ::tflite::CreateTensor(builder_, builder_.CreateVector<int>({2, 1}),
                               ::tflite::TensorType_FLOAT32, 2,
                               builder_.CreateString("tensor_two"), q);
    return builder_.CreateVector(
        std::vector<Offset<::tflite::Tensor>>({t1, t2}));
  }

  Offset<Vector<Offset<::tflite::OperatorCode>>> BuildOpCodes(
      std::initializer_list<::tflite::BuiltinOperator> op_codes) {
    std::vector<Offset<::tflite::OperatorCode>> op_codes_vector;
    for (auto op : op_codes) {
      op_codes_vector.push_back(::tflite::CreateOperatorCode(builder_, op, 0));
    }
    return builder_.CreateVector(op_codes_vector);
  }

  Offset<Vector<Offset<::tflite::OperatorCode>>> BuildOpCodes() {
    return BuildOpCodes({::tflite::BuiltinOperator_MAX_POOL_2D,
                         ::tflite::BuiltinOperator_CONV_2D});
  }

  Offset<Vector<Offset<::tflite::Operator>>> BuildOperators(
      std::initializer_list<int> inputs, std::initializer_list<int> outputs) {
    auto is = builder_.CreateVector<int>(inputs);
    if (inputs.size() == 0) is = 0;
    auto os = builder_.CreateVector<int>(outputs);
    if (outputs.size() == 0) os = 0;
    auto op = ::tflite::CreateOperator(
        builder_, 0, is, os, ::tflite::BuiltinOptions_Conv2DOptions,
        ::tflite::CreateConv2DOptions(builder_, ::tflite::Padding_VALID, 1, 1,
                                      ::tflite::ActivationFunctionType_NONE)
            .Union(),
        /*custom_options=*/0, ::tflite::CustomOptionsFormat_FLEXBUFFERS);

    return builder_.CreateVector(std::vector<Offset<::tflite::Operator>>({op}));
  }

  Offset<Vector<Offset<::tflite::Operator>>> BuildOperators() {
    return BuildOperators({0}, {1});
  }

  Offset<Vector<Offset<::tflite::SubGraph>>> BuildSubGraphs(
      Offset<Vector<Offset<::tflite::Tensor>>> tensors,
      Offset<Vector<Offset<::tflite::Operator>>> operators,
      int num_sub_graphs = 1) {
    std::vector<int32_t> inputs = {0};
    std::vector<int32_t> outputs = {1};
    std::vector<Offset<::tflite::SubGraph>> v;
    for (int i = 0; i < num_sub_graphs; ++i) {
      v.push_back(::tflite::CreateSubGraph(
          builder_, tensors, builder_.CreateVector(inputs),
          builder_.CreateVector(outputs), operators,
          builder_.CreateString("subgraph")));
    }
    return builder_.CreateVector(v);
  }

  // This is a very simplistic model. We are not interested in testing all the
  // details here, since tf.mini's testing framework will be exercising all the
  // conversions multiple times, and the conversion of operators is tested by
  // separate unittests.
  void BuildTestModel() {
    auto buffers = BuildBuffers();
    auto tensors = BuildTensors();
    auto opcodes = BuildOpCodes();
    auto operators = BuildOperators();
    auto subgraphs = BuildSubGraphs(tensors, operators);
    auto s = builder_.CreateString("");

    ::tflite::FinishModelBuffer(
        builder_, ::tflite::CreateModel(builder_, TFLITE_SCHEMA_VERSION,
                                        opcodes, subgraphs, s, buffers));

    input_model_ = ::tflite::GetModel(builder_.GetBufferPointer());
  }
  string InputModelAsString() {
    return string(reinterpret_cast<char*>(builder_.GetBufferPointer()),
                  builder_.GetSize());
  }
  flatbuffers::FlatBufferBuilder builder_;
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
  EXPECT_THAT(operators, ElementsAre("MAX_POOL_2D", "CONV_2D"));
}

TEST_F(ImportTest, Tensors) {
  BuildTestModel();

  auto model = Import(ModelFlags(), InputModelAsString());

  ASSERT_GT(model->HasArray("tensor_one"), 0);
  Array& a1 = model->GetArray("tensor_one");
  EXPECT_EQ(ArrayDataType::kFloat, a1.data_type);
  EXPECT_THAT(a1.GetBuffer<ArrayDataType::kFloat>().data,
              ElementsAre(1.0f, 2.0f, 3.0f, 4.0f));
  ASSERT_TRUE(a1.has_shape());
  EXPECT_THAT(a1.shape().dims(), ElementsAre(1, 2, 2));

  const auto& mm = a1.minmax;
  ASSERT_TRUE(mm.get());
  EXPECT_FLOAT_EQ(0.1, mm->min);
  EXPECT_FLOAT_EQ(0.2, mm->max);

  const auto& q = a1.quantization_params;
  ASSERT_TRUE(q.get());
  EXPECT_FLOAT_EQ(0.3, q->scale);
  EXPECT_EQ(100, q->zero_point);
}

TEST_F(ImportTest, NoBuffers) {
  auto buffers = 0;
  auto tensors = BuildTensors();
  auto opcodes = BuildOpCodes();
  auto operators = BuildOperators();
  auto subgraphs = BuildSubGraphs(tensors, operators);
  auto comment = builder_.CreateString("");
  ::tflite::FinishModelBuffer(
      builder_, ::tflite::CreateModel(builder_, TFLITE_SCHEMA_VERSION, opcodes,
                                      subgraphs, comment, buffers));
  EXPECT_DEATH(Import(ModelFlags(), InputModelAsString()),
               "Missing 'buffers' section.");
}

TEST_F(ImportTest, NoInputs) {
  auto buffers = BuildBuffers();
  auto tensors = BuildTensors();
  auto opcodes = BuildOpCodes();
  auto operators = BuildOperators({}, {1});
  auto subgraphs = BuildSubGraphs(tensors, operators);
  auto comment = builder_.CreateString("");
  ::tflite::FinishModelBuffer(
      builder_, ::tflite::CreateModel(builder_, TFLITE_SCHEMA_VERSION, opcodes,
                                      subgraphs, comment, buffers));
  EXPECT_DEATH(Import(ModelFlags(), InputModelAsString()),
               "Missing 'inputs' for operator.");
}

TEST_F(ImportTest, NoOutputs) {
  auto buffers = BuildBuffers();
  auto tensors = BuildTensors();
  auto opcodes = BuildOpCodes();
  auto operators = BuildOperators({0}, {});
  auto subgraphs = BuildSubGraphs(tensors, operators);
  auto comment = builder_.CreateString("");
  ::tflite::FinishModelBuffer(
      builder_, ::tflite::CreateModel(builder_, TFLITE_SCHEMA_VERSION, opcodes,
                                      subgraphs, comment, buffers));
  EXPECT_DEATH(Import(ModelFlags(), InputModelAsString()),
               "Missing 'outputs' for operator.");
}

TEST_F(ImportTest, InvalidOpCode) {
  auto buffers = BuildBuffers();
  auto tensors = BuildTensors();
  auto opcodes = BuildOpCodes({static_cast<::tflite::BuiltinOperator>(-1),
                               ::tflite::BuiltinOperator_CONV_2D});
  auto operators = BuildOperators();
  auto subgraphs = BuildSubGraphs(tensors, operators);
  auto comment = builder_.CreateString("");
  ::tflite::FinishModelBuffer(
      builder_, ::tflite::CreateModel(builder_, TFLITE_SCHEMA_VERSION, opcodes,
                                      subgraphs, comment, buffers));
  EXPECT_DEATH(Import(ModelFlags(), InputModelAsString()),
               "Operator id '-1' is out of range.");
}

TEST_F(ImportTest, MultipleSubGraphs) {
  auto buffers = BuildBuffers();
  auto tensors = BuildTensors();
  auto opcodes = BuildOpCodes();
  auto operators = BuildOperators();
  auto subgraphs = BuildSubGraphs(tensors, operators, 2);
  auto comment = builder_.CreateString("");
  ::tflite::FinishModelBuffer(
      builder_, ::tflite::CreateModel(builder_, TFLITE_SCHEMA_VERSION, opcodes,
                                      subgraphs, comment, buffers));

  input_model_ = ::tflite::GetModel(builder_.GetBufferPointer());

  EXPECT_DEATH(Import(ModelFlags(), InputModelAsString()),
               "Number of subgraphs in tflite should be exactly 1.");
}

// TODO(ahentz): still need tests for Operators and IOTensors.

}  // namespace
}  // namespace tflite

}  // namespace toco
