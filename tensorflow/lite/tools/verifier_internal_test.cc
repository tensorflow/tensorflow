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
#include "tensorflow/lite/tools/verifier_internal.h"

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
#include "tensorflow/lite/schema/schema_conversion_utils.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/testing/util.h"
#include "tensorflow/lite/util.h"
#include "tensorflow/lite/version.h"

namespace tflite {

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

  bool Verify(const void* buf, size_t length) {
    return tflite::internal::VerifyFlatBufferAndGetModel(buf, length);
  }

  bool Verify() {
    return Verify(builder_.GetBufferPointer(), builder_.GetSize());
  }

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

  ASSERT_TRUE(::tflite::internal::VerifyFlatBufferAndGetModel(
      builder.GetBufferPointer(), builder.GetSize()));
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
  ASSERT_FALSE(::tflite::internal::VerifyFlatBufferAndGetModel(model.data(),
                                                               model.size()));
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
    EXPECT_FALSE(tflite::internal::VerifyFlatBufferAndGetModel(
        model_content.data(), model_content.size()))
        << "Fail at position: " << i;
  }
}

}  // namespace tflite
