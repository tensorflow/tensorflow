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
#include <string>
#include <vector>

#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/util.h"
#include <gtest/gtest.h>
#include "tensorflow/contrib/lite/allocation.h"
#include "tensorflow/contrib/lite/error_reporter.h"
#include "tensorflow/contrib/lite/schema/schema_generated.h"
#include "tensorflow/contrib/lite/testing/util.h"
#include "tensorflow/contrib/lite/tools/mutable_op_resolver.h"
#include "tensorflow/contrib/lite/tools/verifier.h"
#include "tensorflow/contrib/lite/version.h"
#include "tensorflow/core/framework/numeric_types.h"

namespace tflite {

using flatbuffers::FlatBufferBuilder;
using flatbuffers::Offset;

// Build single subgraph model.
class TfLiteFlatbufferModelBuilder {
 public:
  TfLiteFlatbufferModelBuilder() {
    buffers_.push_back(
        CreateBuffer(builder_, builder_.CreateVector(std::vector<uint8_t>{})));
  }

  TfLiteFlatbufferModelBuilder(const std::vector<BuiltinOperator>& builtin_ops,
                               const std::vector<string>& custom_ops) {
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
                 const std::vector<uint8_t>& buffer, const char* name) {
    int buffer_index = 0;
    if (!buffer.empty()) {
      buffer_index = buffers_.size();
      buffers_.push_back(CreateBuffer(builder_, builder_.CreateVector(buffer)));
    }
    tensors_.push_back(CreateTensorDirect(builder_, &shape, type, buffer_index,
                                          name, /*quantization=*/0));
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

  void FinishModel(const std::vector<int32_t>& inputs,
                   const std::vector<int32_t>& outputs) {
    auto subgraph = std::vector<Offset<SubGraph>>({CreateSubGraph(
        builder_, builder_.CreateVector(tensors_),
        builder_.CreateVector(inputs), builder_.CreateVector(outputs),
        builder_.CreateVector(operators_),
        builder_.CreateString("test_subgraph"))});
    auto result = CreateModel(
        builder_, TFLITE_SCHEMA_VERSION, builder_.CreateVector(operator_codes_),
        builder_.CreateVector(subgraph), builder_.CreateString("test_model"),
        builder_.CreateVector(buffers_));
    tflite::FinishModelBuffer(builder_, result);
  }

  bool Verify() {
    return tflite::Verify(builder_.GetBufferPointer(), builder_.GetSize(),
                          resolver_, DefaultErrorReporter());
  }

 private:
  FlatBufferBuilder builder_;
  MutableOpResolver resolver_;
  TfLiteRegistration fake_op_;
  std::vector<Offset<Operator>> operators_;
  std::vector<Offset<OperatorCode>> operator_codes_;
  std::vector<Offset<Tensor>> tensors_;
  std::vector<Offset<Buffer>> buffers_;
};

TEST(VerifyModel, TestEmptyModel) {
  FlatBufferBuilder builder;
  auto model = CreateModel(builder, /*version=*/TFLITE_SCHEMA_VERSION,
                           /*operator_codes=*/0, /*subgraphs=*/0,
                           /*description=*/0, /*buffers=*/0);
  ::tflite::FinishModelBuffer(builder, model);

  ASSERT_FALSE(Verify(builder.GetBufferPointer(), builder.GetSize(),
                      MutableOpResolver{}, DefaultErrorReporter()));
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
}

TEST(VerifyModel, TestCorruptedData) {
  std::string model = "123";
  ASSERT_FALSE(Verify(model.data(), model.size(), MutableOpResolver{},
                      /*error_reporter=*/nullptr));
}

TEST(VerifyModel, TestUnsupportedVersion) {
  FlatBufferBuilder builder;
  auto model = CreateModel(builder, /*version=*/1, /*operator_codes=*/0,
                           /*subgraphs=*/0, /*description=*/0, /*buffers=*/0);
  ::tflite::FinishModelBuffer(builder, model);
  ASSERT_FALSE(Verify(builder.GetBufferPointer(), builder.GetSize(),
                      MutableOpResolver{}, DefaultErrorReporter()));
}

TEST(VerifyModel, TestRandomModificationIsNotAllowed) {
  FlatBufferBuilder builder;
  auto model = CreateModel(builder, /*version=*/TFLITE_SCHEMA_VERSION,
                           /*operator_codes=*/0,
                           /*subgraphs=*/0, /*description=*/0, /*buffers=*/0);
  ::tflite::FinishModelBuffer(builder, model);

  std::string model_content(reinterpret_cast<char*>(builder.GetBufferPointer()),
                            builder.GetSize());
  for (int i = 0; i < model_content.size(); i++) {
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
}

TEST(VerifyModel, TestIntTensorShapeIsSmallerThanBuffer) {
  TfLiteFlatbufferModelBuilder builder;
  builder.AddTensor({2, 1}, TensorType_UINT8, {1, 2, 3, 4}, "input");
  builder.FinishModel({}, {});
  ASSERT_FALSE(builder.Verify());
}

TEST(VerifyModel, TestIntTensorShapeOverflow) {
  TfLiteFlatbufferModelBuilder builder;
  builder.AddTensor({1024, 2048, 4096}, TensorType_UINT8, {1, 2, 3, 4},
                    "input");
  builder.FinishModel({}, {});
  ASSERT_FALSE(builder.Verify());
}

TEST(VerifyModel, TensorBufferIsNotValid) {
  FlatBufferBuilder builder;
  std::vector<int> shape = {2, 3};
  auto tensors = builder.CreateVector(std::vector<Offset<Tensor>>{
      CreateTensorDirect(builder, &shape, TensorType_INT32, /*buffer=*/2,
                         "input", /*quantization=*/0)});
  auto subgraph = std::vector<Offset<SubGraph>>(
      {CreateSubGraph(builder, tensors, /*inputs=*/0, /*outputs=*/0,
                      /*operators=*/0, builder.CreateString("Main"))});

  auto buffers = builder.CreateVector(std::vector<Offset<Buffer>>{
      CreateBuffer(builder,
                   builder.CreateVector(std::vector<uint8>{1, 2, 3, 4, 5, 6})),
  });

  auto model = CreateModel(builder, TFLITE_SCHEMA_VERSION, /*operator_codes=*/0,
                           builder.CreateVector(subgraph),
                           builder.CreateString("SmartReply"), buffers);

  ::tflite::FinishModelBuffer(builder, model);
  ASSERT_FALSE(Verify(builder.GetBufferPointer(), builder.GetSize(),
                      MutableOpResolver{}, DefaultErrorReporter()));
}

TEST(VerifyModel, StringTensorHasInvalidNumString) {
  TfLiteFlatbufferModelBuilder builder;
  builder.AddTensor(
      {2}, TensorType_STRING,
      {0x00, 0x00, 0x00, 0x20, 16, 0, 0, 0, 17, 0, 0, 0, 18, 0, 0, 0, 'A', 'B'},
      "input");
  builder.FinishModel({}, {});
  ASSERT_FALSE(builder.Verify());
}

TEST(VerifyModel, StringTensorOffsetTooSmall) {
  TfLiteFlatbufferModelBuilder builder;
  builder.AddTensor(
      {2}, TensorType_STRING,
      {2, 0, 0, 0, 12, 0, 0, 0, 17, 0, 0, 0, 18, 0, 0, 0, 'A', 'B'}, "input");
  builder.FinishModel({}, {});
  ASSERT_FALSE(builder.Verify());
}

TEST(VerifyModel, StringTensorOffsetOutOfRange) {
  TfLiteFlatbufferModelBuilder builder;
  builder.AddTensor(
      {2}, TensorType_STRING,
      {2, 0, 0, 0, 16, 0, 0, 0, 17, 0, 0, 0, 22, 0, 0, 0, 'A', 'B'}, "input");
  builder.FinishModel({}, {});
  ASSERT_FALSE(builder.Verify());
}

TEST(VerifyModel, StringTensorIsLargerThanRequired) {
  TfLiteFlatbufferModelBuilder builder;
  builder.AddTensor(
      {2}, TensorType_STRING,
      {2, 0, 0, 0, 16, 0, 0, 0, 17, 0, 0, 0, 18, 0, 0, 0, 'A', 'B', 'C'},
      "input");
  builder.FinishModel({}, {});
  ASSERT_FALSE(builder.Verify());
}

TEST(VerifyModel, AllOpsAreSupported) {
  TfLiteFlatbufferModelBuilder builder({BuiltinOperator_ADD}, {"CustomOp"});
  builder.AddTensor({2, 3}, TensorType_UINT8, {1, 2, 3, 4}, "input1");
  builder.AddTensor({2, 3}, TensorType_UINT8, {1, 2, 3, 4}, "input2");
  builder.AddTensor({2, 3}, TensorType_UINT8, {}, "output");
  builder.AddOperator({0, 1}, {2}, BuiltinOperator_ADD, nullptr);
  builder.AddOperator({0, 1}, {2}, BuiltinOperator_CUSTOM, "CustomOp");
  builder.FinishModel({}, {});
  ASSERT_FALSE(builder.Verify());
}

TEST(VerifyModel, UseUnsupportedBuiltinOps) {
  TfLiteFlatbufferModelBuilder builder({BuiltinOperator_SUB}, {"CustomOp"});
  builder.AddTensor({2, 3}, TensorType_UINT8, {1, 2, 3, 4}, "input1");
  builder.AddTensor({2, 3}, TensorType_UINT8, {1, 2, 3, 4}, "input2");
  builder.AddTensor({2, 3}, TensorType_UINT8, {}, "output");
  builder.AddOperator({0, 1}, {2}, BuiltinOperator_ADD, nullptr);
  builder.FinishModel({}, {});
  ASSERT_FALSE(builder.Verify());
}

TEST(VerifyModel, UseUnsupportedCustomOps) {
  TfLiteFlatbufferModelBuilder builder({BuiltinOperator_ADD}, {"NewOp"});
  builder.AddTensor({2, 3}, TensorType_UINT8, {1, 2, 3, 4}, "input1");
  builder.AddTensor({2, 3}, TensorType_UINT8, {1, 2, 3, 4}, "input2");
  builder.AddTensor({2, 3}, TensorType_UINT8, {}, "output");
  builder.AddOperator({0, 1}, {2}, BuiltinOperator_CUSTOM, "Not supported");
  builder.FinishModel({}, {});
  ASSERT_FALSE(builder.Verify());
}

// TODO(yichengfan): make up malicious files to test with.

}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
