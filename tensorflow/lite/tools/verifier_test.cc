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
#include "tensorflow/lite/tools/verifier.h"

#include <memory>
#include <string>
#include <vector>

#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/util.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/lite/allocation.h"
#include "tensorflow/lite/core/api/flatbuffer_conversions.h"
#include "tensorflow/lite/error_reporter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/mutable_op_resolver.h"
#include "tensorflow/lite/op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/testing/util.h"
#include "tensorflow/lite/util.h"
#include "tensorflow/lite/version.h"

namespace tflite {

namespace {
static const char* kSparseTensorTestModel =
    "tensorflow/lite/testdata/sparse_tensor.bin";
}  // namespace

class MockErrorReporter : public ErrorReporter {
 public:
  MockErrorReporter() : buffer_size_(0) {}
  int Report(const char* format, va_list args) override {
    buffer_size_ = vsnprintf(buffer_, kBufferSize, format, args);
    return buffer_size_;
  }
  int GetBufferSize() { return buffer_size_; }

  string GetAsString() const { return string(buffer_, buffer_size_); }

 private:
  static constexpr int kBufferSize = 256;
  char buffer_[kBufferSize];
  int buffer_size_;
};

// Build single subgraph model.
class TfLiteFlatbufferModelBuilder {
 public:
  TfLiteFlatbufferModelBuilder() {
    buffers_.push_back(
        CreateBuffer(builder_, builder_.CreateVector(std::vector<uint8_t>{})));
  }

  TfLiteFlatbufferModelBuilder(const std::vector<BuiltinOperator>& builtin_ops,
                               const std::vector<std::string>& custom_ops) {
    buffers_.push_back(
        CreateBuffer(builder_, builder_.CreateVector(std::vector<uint8_t>{})));

    for (const auto& iter : builtin_ops) {
      resolver_.AddBuiltin(iter, &fake_op_);
    }
    for (const auto& iter : custom_ops) {
      resolver_.AddCustom(iter.data(), &fake_op_);
    }
  }

  void AddTensor(const std::vector<int>& shape, tflite::TensorType type,
                 const std::vector<uint8_t>& buffer, const char* name,
                 const bool is_variable = false) {
    int buffer_index = 0;
    if (!buffer.empty()) {
      buffer_index = buffers_.size();
      buffers_.push_back(CreateBuffer(builder_, builder_.CreateVector(buffer)));
    }
    if (shape.empty()) {
      tensors_.push_back(CreateTensorDirect(builder_, /*shape=*/nullptr, type,
                                            buffer_index, name,
                                            /*quantization=*/0, is_variable));
      return;
    }
    tensors_.push_back(CreateTensorDirect(builder_, &shape, type, buffer_index,
                                          name, /*quantization=*/0,
                                          is_variable));
  }

  void AddOperator(const std::vector<int32_t>& inputs,
                   const std::vector<int32_t>& outputs,
                   tflite::BuiltinOperator builtin_op, const char* custom_op) {
    operator_codes_.push_back(
        CreateOperatorCodeDirect(builder_, builtin_op, custom_op));
    operators_.push_back(CreateOperator(
        builder_, operator_codes_.size() - 1, builder_.CreateVector(inputs),
        builder_.CreateVector(outputs), BuiltinOptions_NONE,
        /*builtin_options=*/0,
        /*custom_options=*/0, tflite::CustomOptionsFormat_FLEXBUFFERS));
  }

  enum BuilderMode {
    kBuilderModeEmptyVectorIsEmpty,
    kBuilderModeEmptyVectorIsNull,
    kBuilderModeDefault = kBuilderModeEmptyVectorIsEmpty,
  };
  void FinishModel(const std::vector<int32_t>& inputs,
                   const std::vector<int32_t>& outputs,
                   BuilderMode mode = kBuilderModeDefault) {
    auto subgraph = std::vector<flatbuffers::Offset<SubGraph>>({CreateSubGraph(
        builder_, CreateVector(tensors_, mode), CreateVector(inputs, mode),
        CreateVector(outputs, mode), CreateVector(operators_, mode),
        builder_.CreateString("test_subgraph"))});
    auto result = CreateModel(
        builder_, TFLITE_SCHEMA_VERSION, CreateVector(operator_codes_, mode),
        CreateVector(subgraph, mode), builder_.CreateString("test_model"),
        CreateVector(buffers_, mode));
    tflite::FinishModelBuffer(builder_, result);
  }

  bool Verify() {
    return tflite::Verify(builder_.GetBufferPointer(), builder_.GetSize(),
                          resolver_, &mock_reporter_);
  }

  string GetErrorString() { return mock_reporter_.GetAsString(); }

 private:
  template <typename T>
  flatbuffers::Offset<flatbuffers::Vector<T>> CreateVector(
      const std::vector<T>& v, BuilderMode mode) {
    if (mode == kBuilderModeEmptyVectorIsNull && v.empty()) {
      return 0;
    }
    return builder_.CreateVector(v);
  }

  flatbuffers::FlatBufferBuilder builder_;
  MutableOpResolver resolver_;
  TfLiteRegistration fake_op_;
  MockErrorReporter mock_reporter_;
  std::vector<flatbuffers::Offset<Operator>> operators_;
  std::vector<flatbuffers::Offset<OperatorCode>> operator_codes_;
  std::vector<flatbuffers::Offset<Tensor>> tensors_;
  std::vector<flatbuffers::Offset<Buffer>> buffers_;
};

TEST(VerifyModel, TestEmptyModel) {
  flatbuffers::FlatBufferBuilder builder;
  auto model = CreateModel(builder, /*version=*/TFLITE_SCHEMA_VERSION,
                           /*operator_codes=*/0, /*subgraphs=*/0,
                           /*description=*/0, /*buffers=*/0);
  ::tflite::FinishModelBuffer(builder, model);

  MockErrorReporter mock_reporter;
  ASSERT_FALSE(Verify(builder.GetBufferPointer(), builder.GetSize(),
                      MutableOpResolver{}, &mock_reporter));
  EXPECT_THAT(mock_reporter.GetAsString(),
              ::testing::ContainsRegex("Missing 'subgraphs' section."));
}

TEST(VerifyModel, TestEmptyVector) {
  TfLiteFlatbufferModelBuilder builder({}, {"test"});
  builder.AddOperator({0, 1}, {3}, BuiltinOperator_CUSTOM, "test");
  builder.AddTensor({2, 3}, TensorType_UINT8, {1, 2, 3, 4, 5, 6}, "input");
  builder.AddTensor({}, TensorType_UINT8, {}, "empty_vector");
  builder.AddTensor(
      {2}, TensorType_STRING,
      {2, 0, 0, 0, 16, 0, 0, 0, 17, 0, 0, 0, 19, 0, 0, 0, 'A', 'B', 'C'},
      "data");
  builder.AddTensor({2, 3}, TensorType_INT32, {}, "output");
  builder.FinishModel({0, 1}, {3});
  ASSERT_TRUE(builder.Verify());
}

TEST(VerifyModel, TestSimpleModel) {
  TfLiteFlatbufferModelBuilder builder({}, {"test"});
  builder.AddOperator({0, 1}, {2}, BuiltinOperator_CUSTOM, "test");
  builder.AddTensor({2, 3}, TensorType_UINT8, {1, 2, 3, 4, 5, 6}, "input");
  builder.AddTensor(
      {2}, TensorType_STRING,
      {2, 0, 0, 0, 16, 0, 0, 0, 17, 0, 0, 0, 19, 0, 0, 0, 'A', 'B', 'C'},
      "data");
  builder.AddTensor({2, 3}, TensorType_INT32, {}, "output");
  builder.FinishModel({0, 1}, {2});
  ASSERT_TRUE(builder.Verify());
  EXPECT_EQ("", builder.GetErrorString());
}

TEST(VerifyModel, TestNullTensors) {
  TfLiteFlatbufferModelBuilder builder({}, {"test"});
  builder.AddOperator({0, 1}, {2}, BuiltinOperator_CUSTOM, "test");
  builder.FinishModel(
      {}, {2}, TfLiteFlatbufferModelBuilder::kBuilderModeEmptyVectorIsNull);
  ASSERT_FALSE(builder.Verify());
  EXPECT_EQ(builder.GetErrorString(),
            "Input tensor 0 to op 0 (CUSTOM) is not produced");
}

TEST(VerifyModel, TestNullOperators) {
  TfLiteFlatbufferModelBuilder builder({}, {"test"});
  builder.FinishModel(
      {0, 1}, {2}, TfLiteFlatbufferModelBuilder::kBuilderModeEmptyVectorIsNull);
  ASSERT_FALSE(builder.Verify());
  EXPECT_THAT(
      builder.GetErrorString(),
      ::testing::ContainsRegex("Missing 'operators' section in subgraph"));
}

TEST(VerifyModel, TestNullInputs) {
  TfLiteFlatbufferModelBuilder builder({}, {"test"});
  builder.AddOperator({0, 1}, {2}, BuiltinOperator_CUSTOM, "test");
  builder.AddTensor({2, 3}, TensorType_UINT8, {1, 2, 3, 4, 5, 6}, "input");
  builder.AddTensor(
      {2}, TensorType_STRING,
      {2, 0, 0, 0, 16, 0, 0, 0, 17, 0, 0, 0, 19, 0, 0, 0, 'A', 'B', 'C'},
      "data");
  builder.AddTensor({2, 3}, TensorType_INT32, {}, "output");
  builder.FinishModel(
      {}, {2}, TfLiteFlatbufferModelBuilder::kBuilderModeEmptyVectorIsNull);
  ASSERT_TRUE(builder.Verify());
  EXPECT_EQ("", builder.GetErrorString());
}

TEST(VerifyModel, TestCorruptedData) {
  std::string model = "123";
  MockErrorReporter mock_reporter;
  ASSERT_FALSE(
      Verify(model.data(), model.size(), MutableOpResolver{}, &mock_reporter));
  EXPECT_THAT(mock_reporter.GetAsString(),
              ::testing::ContainsRegex("Invalid flatbuffer format"));
}

TEST(VerifyModel, TestUnsupportedVersion) {
  flatbuffers::FlatBufferBuilder builder;
  auto model = CreateModel(builder, /*version=*/1, /*operator_codes=*/0,
                           /*subgraphs=*/0, /*description=*/0, /*buffers=*/0);
  ::tflite::FinishModelBuffer(builder, model);
  MockErrorReporter mock_reporter;
  ASSERT_FALSE(Verify(builder.GetBufferPointer(), builder.GetSize(),
                      MutableOpResolver{}, &mock_reporter));
  EXPECT_THAT(mock_reporter.GetAsString(),
              ::testing::ContainsRegex("Invalid model version 1"));
}

TEST(VerifyModel, TestRandomModificationIsNotAllowed) {
  flatbuffers::FlatBufferBuilder builder;
  auto model = CreateModel(builder, /*version=*/TFLITE_SCHEMA_VERSION,
                           /*operator_codes=*/0,
                           /*subgraphs=*/0, /*description=*/0, /*buffers=*/0);
  ::tflite::FinishModelBuffer(builder, model);

  std::string model_content(reinterpret_cast<char*>(builder.GetBufferPointer()),
                            builder.GetSize());
  for (size_t i = 0; i < model_content.size(); i++) {
    model_content[i] = (model_content[i] + 137) % 255;
    EXPECT_FALSE(Verify(model_content.data(), model_content.size(),
                        MutableOpResolver{}, DefaultErrorReporter()))
        << "Fail at position: " << i;
  }
}

TEST(VerifyModel, TestIntTensorShapeIsGreaterThanBuffer) {
  TfLiteFlatbufferModelBuilder builder;
  builder.AddTensor({2, 3}, TensorType_UINT8, {1, 2, 3, 4}, "input");
  builder.FinishModel({}, {});
  ASSERT_FALSE(builder.Verify());
  EXPECT_THAT(builder.GetErrorString(),
              ::testing::ContainsRegex("Tensor input requires 6 bytes, but is "
                                       "allocated with 4 bytes buffer"));
}

TEST(VerifyModel, TestIntTensorShapeIsSmallerThanBuffer) {
  TfLiteFlatbufferModelBuilder builder;
  builder.AddTensor({2, 1}, TensorType_UINT8, {1, 2, 3, 4}, "input");
  builder.FinishModel({}, {});
  ASSERT_FALSE(builder.Verify());
  EXPECT_THAT(builder.GetErrorString(),
              ::testing::ContainsRegex("Tensor input requires 2 bytes, but is "
                                       "allocated with 4 bytes buffer"));
}

TEST(VerifyModel, TestIntTensorShapeOverflow) {
  TfLiteFlatbufferModelBuilder builder;
  builder.AddTensor({1024, 2048, 4096}, TensorType_UINT8, {1, 2, 3, 4},
                    "input");
  builder.FinishModel({}, {});
  ASSERT_FALSE(builder.Verify());
  EXPECT_THAT(builder.GetErrorString(),
              ::testing::ContainsRegex("Tensor input dimension overflow"));
}

TEST(VerifyModel, TensorBufferIsNotValid) {
  flatbuffers::FlatBufferBuilder builder;
  std::vector<int> shape = {2, 3};
  auto tensors = builder.CreateVector(std::vector<flatbuffers::Offset<Tensor>>{
      CreateTensorDirect(builder, &shape, TensorType_INT32, /*buffer=*/2,
                         "input", /*quantization=*/0)});
  auto subgraph = std::vector<flatbuffers::Offset<SubGraph>>(
      {CreateSubGraph(builder, tensors, /*inputs=*/0, /*outputs=*/0,
                      /*operators=*/0, builder.CreateString("Main"))});

  auto buffers = builder.CreateVector(std::vector<flatbuffers::Offset<Buffer>>{
      CreateBuffer(builder, builder.CreateVector(
                                std::vector<uint8_t>{1, 2, 3, 4, 5, 6})),
  });

  auto model = CreateModel(builder, TFLITE_SCHEMA_VERSION, /*operator_codes=*/0,
                           builder.CreateVector(subgraph),
                           builder.CreateString("SmartReply"), buffers);

  ::tflite::FinishModelBuffer(builder, model);
  MockErrorReporter mock_reporter;
  ASSERT_FALSE(Verify(builder.GetBufferPointer(), builder.GetSize(),
                      MutableOpResolver{}, &mock_reporter));
  EXPECT_THAT(
      mock_reporter.GetAsString(),
      ::testing::ContainsRegex("Missing 'operators' section in subgraph."));
}

TEST(VerifyModel, StringTensorIsEmpty) {
  TfLiteFlatbufferModelBuilder builder;
  builder.AddTensor({2}, TensorType_STRING, {0x00}, "input");
  builder.FinishModel({}, {});
  ASSERT_FALSE(builder.Verify());
  EXPECT_EQ(builder.GetErrorString(), "String tensor input is invalid (empty)");
}

TEST(VerifyModel, StringTensorHasInvalidNumString) {
  TfLiteFlatbufferModelBuilder builder;
  builder.AddTensor(
      {2}, TensorType_STRING,
      {0x00, 0x00, 0x00, 0x20, 16, 0, 0, 0, 17, 0, 0, 0, 18, 0, 0, 0, 'A', 'B'},
      "input");
  builder.FinishModel({}, {});
  ASSERT_FALSE(builder.Verify());
  EXPECT_THAT(
      builder.GetErrorString(),
      ::testing::ContainsRegex(
          "String tensor input buffer requires at least -2147483640 bytes, "
          "but is allocated with 18 bytes"));
}

TEST(VerifyModel, StringTensorOffsetTooSmall) {
  TfLiteFlatbufferModelBuilder builder;
  builder.AddTensor(
      {2}, TensorType_STRING,
      {2, 0, 0, 0, 12, 0, 0, 0, 17, 0, 0, 0, 18, 0, 0, 0, 'A', 'B'}, "input");
  builder.FinishModel({}, {});
  ASSERT_FALSE(builder.Verify());
  EXPECT_THAT(builder.GetErrorString(),
              ::testing::ContainsRegex(
                  "String tensor input buffer initial offset must be: 16"));
}

TEST(VerifyModel, StringTensorOffsetOutOfRange) {
  TfLiteFlatbufferModelBuilder builder;
  builder.AddTensor(
      {2}, TensorType_STRING,
      {2, 0, 0, 0, 16, 0, 0, 0, 17, 0, 0, 0, 22, 0, 0, 0, 'A', 'B'}, "input");
  builder.FinishModel({}, {});
  ASSERT_FALSE(builder.Verify());
  EXPECT_THAT(builder.GetErrorString(),
              ::testing::ContainsRegex(
                  "String tensor input buffer is invalid: index 2"));
}

TEST(VerifyModel, StringTensorIsLargerThanRequired) {
  TfLiteFlatbufferModelBuilder builder;
  builder.AddTensor(
      {2}, TensorType_STRING,
      {2, 0, 0, 0, 16, 0, 0, 0, 17, 0, 0, 0, 18, 0, 0, 0, 'A', 'B', 'C'},
      "input");
  builder.FinishModel({}, {});
  ASSERT_FALSE(builder.Verify());
  EXPECT_THAT(builder.GetErrorString(),
              ::testing::ContainsRegex(
                  "String tensor input buffer last offset must be 19"));
}

TEST(VerifyModel, AllOpsAreSupported) {
  TfLiteFlatbufferModelBuilder builder({BuiltinOperator_ADD}, {"CustomOp"});
  builder.AddTensor({2, 2}, TensorType_UINT8, {1, 2, 3, 4}, "input1");
  builder.AddTensor({2, 2}, TensorType_UINT8, {1, 2, 3, 4}, "input2");
  builder.AddTensor({2, 2}, TensorType_UINT8, {}, "output1");
  builder.AddTensor({2, 2}, TensorType_UINT8, {}, "output2");
  builder.AddOperator({0, 1}, {2}, BuiltinOperator_ADD, nullptr);
  builder.AddOperator({0, 1}, {3}, BuiltinOperator_CUSTOM, "CustomOp");
  builder.FinishModel({}, {});
  ASSERT_TRUE(builder.Verify());
  EXPECT_EQ("", builder.GetErrorString());
}

TEST(VerifyModel, UseUnsupportedBuiltinOps) {
  TfLiteFlatbufferModelBuilder builder({BuiltinOperator_SUB}, {"CustomOp"});
  builder.AddTensor({2, 2}, TensorType_UINT8, {1, 2, 3, 4}, "input1");
  builder.AddTensor({2, 2}, TensorType_UINT8, {1, 2, 3, 4}, "input2");
  builder.AddTensor({2, 2}, TensorType_UINT8, {}, "output");
  builder.AddOperator({0, 1}, {2}, BuiltinOperator_ADD, nullptr);
  builder.FinishModel({}, {});
  ASSERT_FALSE(builder.Verify());
  EXPECT_THAT(
      builder.GetErrorString(),
      ::testing::ContainsRegex("Unsupported builtin op: ADD, version: 1"));
}

TEST(VerifyModel, UseUnsupportedCustomOps) {
  TfLiteFlatbufferModelBuilder builder({BuiltinOperator_ADD}, {"NewOp"});
  builder.AddTensor({2, 2}, TensorType_UINT8, {1, 2, 3, 4}, "input1");
  builder.AddTensor({2, 2}, TensorType_UINT8, {1, 2, 3, 4}, "input2");
  builder.AddTensor({2, 2}, TensorType_UINT8, {}, "output");
  builder.AddOperator({0, 1}, {2}, BuiltinOperator_CUSTOM, "Not supported");
  builder.FinishModel({}, {});
  ASSERT_FALSE(builder.Verify());
  EXPECT_THAT(builder.GetErrorString(),
              ::testing::ContainsRegex(
                  "Unsupported custom op: Not supported, version: 1"));
}

TEST(VerifyModel, UnpopulatedInputToOp) {
  TfLiteFlatbufferModelBuilder builder({}, {"test"});
  builder.AddOperator({1, 2}, {3}, BuiltinOperator_CUSTOM, "test");
  builder.AddTensor({2, 3}, TensorType_UINT8, {1, 2, 3, 4, 5, 6}, "input");
  // This tensor will never be populated.
  builder.AddTensor({2, 3}, TensorType_UINT8, {}, "invalid_input");
  builder.AddTensor(
      {2}, TensorType_STRING,
      {2, 0, 0, 0, 16, 0, 0, 0, 17, 0, 0, 0, 19, 0, 0, 0, 'A', 'B', 'C'},
      "data");
  builder.AddTensor({2, 3}, TensorType_INT32, {}, "output");
  builder.FinishModel({0, 2}, {3});
  ASSERT_FALSE(builder.Verify());
  EXPECT_EQ("Input tensor 1 to op 0 (CUSTOM) is not produced",
            builder.GetErrorString());
}

TEST(VerifyModel, MultipleOpsOutputToSameTensor) {
  TfLiteFlatbufferModelBuilder builder({BuiltinOperator_ADD}, {"CustomOp"});
  builder.AddTensor({2, 2}, TensorType_UINT8, {1, 2, 3, 4}, "input1");
  builder.AddTensor({2, 2}, TensorType_UINT8, {1, 2, 3, 4}, "input2");
  builder.AddTensor({2, 2}, TensorType_UINT8, {}, "output1");
  builder.AddOperator({0, 1}, {2}, BuiltinOperator_ADD, nullptr);
  // This can't output to "output1", since the first operator does that.
  builder.AddOperator({0, 1}, {2}, BuiltinOperator_CUSTOM, "CustomOp");
  builder.FinishModel({}, {});
  ASSERT_FALSE(builder.Verify());
  EXPECT_EQ(
      "Output tensor 2 to op 1 (CUSTOM) is an output from another op. "
      "There is a cycle in the graph",
      builder.GetErrorString());
}

TEST(VerifyModel, OutputIsAConstantTensor) {
  TfLiteFlatbufferModelBuilder builder({}, {"test"});
  builder.AddOperator({0, 1}, {2}, BuiltinOperator_CUSTOM, "test");
  builder.AddTensor({2, 3}, TensorType_UINT8, {1, 2, 3, 4, 5, 6}, "input");
  builder.AddTensor(
      {2}, TensorType_STRING,
      {2, 0, 0, 0, 16, 0, 0, 0, 17, 0, 0, 0, 19, 0, 0, 0, 'A', 'B', 'C'},
      "data");
  // Output shouldn't be populated with constant value.
  builder.AddTensor({2, 3}, TensorType_INT32, {1, 2, 3, 4, 5, 6}, "output");
  builder.FinishModel({0, 1}, {2});
  ASSERT_FALSE(builder.Verify());
  EXPECT_EQ("Output tensor 2 to op 0 (CUSTOM) is a constant",
            builder.GetErrorString());
}

TEST(VerifyModel, OutputIsSubgraphInput) {
  TfLiteFlatbufferModelBuilder builder({}, {"test"});
  builder.AddOperator({0, 1}, {2}, BuiltinOperator_CUSTOM, "test");
  builder.AddTensor({2, 3}, TensorType_UINT8, {1, 2, 3, 4, 5, 6}, "input");
  builder.AddTensor(
      {2}, TensorType_STRING,
      {2, 0, 0, 0, 16, 0, 0, 0, 17, 0, 0, 0, 19, 0, 0, 0, 'A', 'B', 'C'},
      "data");
  builder.AddTensor({2, 3}, TensorType_INT32, {}, "output");
  // Output shouldn't be a subgraph input.
  builder.FinishModel({0, 1, 2}, {2});
  ASSERT_FALSE(builder.Verify());
  EXPECT_EQ("Output tensor 2 to op 0 (CUSTOM) is a subgraph input",
            builder.GetErrorString());
}

TEST(VerifyModel, OutputIsAVariable) {
  TfLiteFlatbufferModelBuilder builder({}, {"test"});
  builder.AddOperator({0, 1}, {2}, BuiltinOperator_CUSTOM, "test");
  builder.AddTensor({2, 3}, TensorType_UINT8, {1, 2, 3, 4, 5, 6}, "input");
  builder.AddTensor(
      {2}, TensorType_STRING,
      {2, 0, 0, 0, 16, 0, 0, 0, 17, 0, 0, 0, 19, 0, 0, 0, 'A', 'B', 'C'},
      "data");
  // Output shouldn't be a variable.
  builder.AddTensor({2, 3}, TensorType_INT32, {}, "output", /*variable*/ true);
  builder.FinishModel({0, 1}, {2});
  ASSERT_FALSE(builder.Verify());
  EXPECT_EQ("Output tensor 2 to op 0 (CUSTOM) is a variable",
            builder.GetErrorString());
}

TEST(VerifyModel, OpWithOptionalTensor) {
  TfLiteFlatbufferModelBuilder builder({}, {"test"});
  builder.AddOperator({kTfLiteOptionalTensor, 0, 1}, {2},
                      BuiltinOperator_CUSTOM, "test");
  builder.AddTensor({2, 3}, TensorType_UINT8, {1, 2, 3, 4, 5, 6}, "input");
  builder.AddTensor(
      {2}, TensorType_STRING,
      {2, 0, 0, 0, 16, 0, 0, 0, 17, 0, 0, 0, 19, 0, 0, 0, 'A', 'B', 'C'},
      "data");
  builder.AddTensor({2, 3}, TensorType_INT32, {}, "output");
  builder.FinishModel({0, 1}, {2});
  ASSERT_TRUE(builder.Verify());
  EXPECT_EQ("", builder.GetErrorString());
}

TEST(VerifyModel, TypedTensorShapeMismatchWithTensorBufferSize) {
  TfLiteFlatbufferModelBuilder builder;
  for (int tensor_type = TensorType_MIN; tensor_type <= TensorType_MAX;
       ++tensor_type) {
    if (tensor_type == TensorType_STRING) continue;
    builder.AddTensor({2, 3}, static_cast<TensorType>(tensor_type),
                      {1, 2, 3, 4}, "input");
    builder.FinishModel({}, {});
    ASSERT_FALSE(builder.Verify());
    EXPECT_THAT(
        builder.GetErrorString(),
        ::testing::ContainsRegex("Tensor input requires .* bytes, but is "
                                 "allocated with 4 bytes buffer"));
  }
}

TEST(VerifyModel, TypedTensorShapeMatchesTensorBufferSize) {
  TfLiteFlatbufferModelBuilder builder;
  for (int tensor_type = TensorType_MIN; tensor_type <= TensorType_MAX;
       ++tensor_type) {
    if (tensor_type == TensorType_STRING) continue;
    TfLiteType lite_type = kTfLiteNoType;
    ASSERT_EQ(ConvertTensorType(static_cast<TensorType>(tensor_type),
                                &lite_type, /*error_reporter=*/nullptr),
              kTfLiteOk);
    size_t size_bytes = 0;
    ASSERT_EQ(GetSizeOfType(/*context=*/nullptr, lite_type, &size_bytes),
              kTfLiteOk);
    std::vector<uint8_t> buffer(size_bytes);
    builder.AddTensor({1}, static_cast<TensorType>(tensor_type), buffer,
                      "input");
    builder.FinishModel({}, {});
    ASSERT_TRUE(builder.Verify());
  }
}

TEST(VerifyModel, SimpleValidSparseTensor) {
  const auto model = FlatBufferModel::BuildFromFile(kSparseTensorTestModel);
  ASSERT_TRUE(model);

  std::unique_ptr<ModelT> scoped_model;
  scoped_model.reset(model->GetModel()->UnPack());

  flatbuffers::FlatBufferBuilder builder;
  auto model_ = Model::Pack(builder, scoped_model.get());

  ::tflite::FinishModelBuffer(builder, model_);
  MockErrorReporter mock_reporter;
  MutableOpResolver resolver;
  TfLiteRegistration fake_op;
  resolver.AddCustom("FakeOp", &fake_op);
  Verify(builder.GetBufferPointer(), builder.GetSize(), resolver,
         &mock_reporter);
  ASSERT_TRUE(Verify(builder.GetBufferPointer(), builder.GetSize(), resolver,
                     &mock_reporter));
}

TEST(VerifyModel, InvalidSparseTensorMissingBlockMap) {
  const auto model = FlatBufferModel::BuildFromFile(kSparseTensorTestModel);
  ASSERT_TRUE(model);

  std::unique_ptr<ModelT> scoped_model;
  scoped_model.reset(model->GetModel()->UnPack());

  auto* tensor = scoped_model->subgraphs[0]->tensors[0].get();
  tensor->sparsity->block_map = {};

  flatbuffers::FlatBufferBuilder builder;
  auto model_ = Model::Pack(builder, scoped_model.get());

  ::tflite::FinishModelBuffer(builder, model_);
  MockErrorReporter mock_reporter;
  MutableOpResolver resolver;
  TfLiteRegistration fake_op;
  resolver.AddCustom("FakeOp", &fake_op);
  ASSERT_FALSE(Verify(builder.GetBufferPointer(), builder.GetSize(), resolver,
                      &mock_reporter));
  EXPECT_THAT(mock_reporter.GetAsString(),
              ::testing::ContainsRegex("invalid sparsity parameters"));
}

TEST(VerifyModel, InvalidSparseTensorIndexOutOfBound) {
  const auto model = FlatBufferModel::BuildFromFile(kSparseTensorTestModel);
  ASSERT_TRUE(model);

  std::unique_ptr<ModelT> scoped_model;
  scoped_model.reset(model->GetModel()->UnPack());

  auto* tensor = scoped_model->subgraphs[0]->tensors[0].get();
  tensor->sparsity->dim_metadata[1]->array_indices[1] = 5;

  flatbuffers::FlatBufferBuilder builder;
  auto model_ = Model::Pack(builder, scoped_model.get());

  ::tflite::FinishModelBuffer(builder, model_);
  MockErrorReporter mock_reporter;
  MutableOpResolver resolver;
  TfLiteRegistration fake_op;
  resolver.AddCustom("FakeOp", &fake_op);
  ASSERT_FALSE(Verify(builder.GetBufferPointer(), builder.GetSize(), resolver,
                      &mock_reporter));
  EXPECT_THAT(mock_reporter.GetAsString(),
              ::testing::ContainsRegex("invalid sparsity parameters"));
}

TEST(VerifyModel, InvalidSparseTensorInvalidBuffer) {
  const auto model = FlatBufferModel::BuildFromFile(kSparseTensorTestModel);
  ASSERT_TRUE(model);

  std::unique_ptr<ModelT> scoped_model;
  scoped_model.reset(model->GetModel()->UnPack());

  // Expected to have 12 numbers in buffer.
  scoped_model->buffers[1]->data = {0, 1, 2, 3, 4, 5, 6, 7};

  flatbuffers::FlatBufferBuilder builder;
  auto model_ = Model::Pack(builder, scoped_model.get());

  ::tflite::FinishModelBuffer(builder, model_);
  MockErrorReporter mock_reporter;
  MutableOpResolver resolver;
  TfLiteRegistration fake_op;
  resolver.AddCustom("FakeOp", &fake_op);
  ASSERT_FALSE(Verify(builder.GetBufferPointer(), builder.GetSize(), resolver,
                      &mock_reporter));
  EXPECT_THAT(mock_reporter.GetAsString(),
              ::testing::ContainsRegex(
                  "requires 12 bytes, but is allocated with 8 bytes buffer"));
}

TEST(VerifyModel, InvalidSparseTensorInvalidTraversalOrder) {
  const auto model = FlatBufferModel::BuildFromFile(kSparseTensorTestModel);
  ASSERT_TRUE(model);

  std::unique_ptr<ModelT> scoped_model;
  scoped_model.reset(model->GetModel()->UnPack());

  auto* tensor = scoped_model->subgraphs[0]->tensors[0].get();
  // Valid dimensions are (0, 1, 2, 3) in this test model.
  tensor->sparsity->traversal_order[0] = 10;

  flatbuffers::FlatBufferBuilder builder;
  auto model_ = Model::Pack(builder, scoped_model.get());

  ::tflite::FinishModelBuffer(builder, model_);
  MockErrorReporter mock_reporter;
  MutableOpResolver resolver;
  TfLiteRegistration fake_op;
  resolver.AddCustom("FakeOp", &fake_op);
  ASSERT_FALSE(Verify(builder.GetBufferPointer(), builder.GetSize(), resolver,
                      &mock_reporter));
  EXPECT_THAT(mock_reporter.GetAsString(),
              ::testing::ContainsRegex("invalid sparsity parameters"));
}

TEST(VerifyModel, ValidSparseTensorBCSC) {
  const auto model = FlatBufferModel::BuildFromFile(kSparseTensorTestModel);
  ASSERT_TRUE(model);

  std::unique_ptr<ModelT> scoped_model;
  scoped_model.reset(model->GetModel()->UnPack());

  auto* tensor = scoped_model->subgraphs[0]->tensors[0].get();
  tensor->sparsity->traversal_order = {1, 0, 3, 2};
  tensor->sparsity->block_map = {0, 1};
  tensor->sparsity->dim_metadata[0]->format = DimensionType_DENSE;
  tensor->sparsity->dim_metadata[0]->dense_size = 2;

  tensor->sparsity->dim_metadata[1]->format = DimensionType_SPARSE_CSR;
  tensor->sparsity->dim_metadata[1]->array_segments = {0, 1, 3};
  tensor->sparsity->dim_metadata[1]->array_indices = {0, 0, 1};

  tensor->sparsity->dim_metadata[2]->format = DimensionType_DENSE;
  tensor->sparsity->dim_metadata[2]->dense_size = 2;
  tensor->sparsity->dim_metadata[3]->format = DimensionType_DENSE;
  tensor->sparsity->dim_metadata[3]->dense_size = 2;

  flatbuffers::FlatBufferBuilder builder;
  auto model_ = Model::Pack(builder, scoped_model.get());

  ::tflite::FinishModelBuffer(builder, model_);
  MockErrorReporter mock_reporter;
  MutableOpResolver resolver;
  TfLiteRegistration fake_op;
  resolver.AddCustom("FakeOp", &fake_op);
  ASSERT_TRUE(Verify(builder.GetBufferPointer(), builder.GetSize(), resolver,
                     &mock_reporter));
}

// TODO(b/145614687): Add more tricky test cases for sparse tensor verification.
// TODO(yichengfan): make up malicious files to test with.

}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
