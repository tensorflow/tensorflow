/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/core/api/flatbuffer_conversions.h"

#include <cstring>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/string_type.h"

namespace tflite {
namespace {

class MockErrorReporter : public ErrorReporter {
 public:
  MockErrorReporter() : buffer_size_(0) {}
  int Report(const char* format, va_list args) override {
    buffer_size_ = vsnprintf(buffer_, kBufferSize, format, args);
    return buffer_size_;
  }
  char* GetBuffer() { return buffer_; }
  int GetBufferSize() { return buffer_size_; }

  string GetAsString() const { return string(buffer_, buffer_size_); }

 private:
  static constexpr int kBufferSize = 256;
  char buffer_[kBufferSize];
  int buffer_size_;
};

// Used to determine how the op data parsing function creates its working space.
class MockDataAllocator : public BuiltinDataAllocator {
 public:
  MockDataAllocator() : is_allocated_(false) {}
  void* Allocate(size_t size, size_t alignment_hint) override {
    EXPECT_FALSE(is_allocated_);
    const int max_size = kBufferSize;
    EXPECT_LE(size, max_size);
    is_allocated_ = true;
    return buffer_;
  }
  void Deallocate(void* data) override { is_allocated_ = false; }

 private:
  static constexpr int kBufferSize = 1024;
  char buffer_[kBufferSize];
  bool is_allocated_;
};

}  // namespace

class FlatbufferConversionsTest : public ::testing::Test {
 public:
  const Operator* BuildTestOperator(BuiltinOptions op_type,
                                    flatbuffers::Offset<void> options) {
    flatbuffers::Offset<Operator> offset =
        CreateOperatorDirect(builder_, 0, nullptr, nullptr, op_type, options,
                             nullptr, CustomOptionsFormat_FLEXBUFFERS, nullptr);
    builder_.Finish(offset);
    void* pointer = builder_.GetBufferPointer();
    return flatbuffers::GetRoot<Operator>(pointer);
  }

 protected:
  MockErrorReporter mock_reporter_;
  MockDataAllocator mock_allocator_;
  flatbuffers::FlatBufferBuilder builder_;
};

TEST_F(FlatbufferConversionsTest, ParseSqueezeAll) {
  const Operator* op = BuildTestOperator(
      BuiltinOptions_SqueezeOptions, CreateSqueezeOptions(builder_).Union());
  void* output_data = nullptr;
  EXPECT_EQ(kTfLiteOk, ParseOpData(op, BuiltinOperator_SQUEEZE, &mock_reporter_,
                                   &mock_allocator_, &output_data));
}

TEST_F(FlatbufferConversionsTest, ParseDynamicReshape) {
  const Operator* op = BuildTestOperator(
      BuiltinOptions_ReshapeOptions, CreateReshapeOptions(builder_).Union());
  void* output_data = nullptr;
  EXPECT_EQ(kTfLiteOk, ParseOpData(op, BuiltinOperator_RESHAPE, &mock_reporter_,
                                   &mock_allocator_, &output_data));
}

TEST_F(FlatbufferConversionsTest, TestParseOpDataConv) {
  const Operator* conv_op =
      BuildTestOperator(BuiltinOptions_Conv2DOptions,
                        CreateConv2DOptions(builder_, Padding_SAME, 1, 2,
                                            ActivationFunctionType_RELU, 3, 4)
                            .Union());
  void* output_data = nullptr;
  EXPECT_EQ(kTfLiteOk,
            ParseOpData(conv_op, BuiltinOperator_CONV_2D, &mock_reporter_,
                        &mock_allocator_, &output_data));
  EXPECT_NE(nullptr, output_data);
  TfLiteConvParams* params = reinterpret_cast<TfLiteConvParams*>(output_data);
  EXPECT_EQ(kTfLitePaddingSame, params->padding);
  EXPECT_EQ(1, params->stride_width);
  EXPECT_EQ(2, params->stride_height);
  EXPECT_EQ(kTfLiteActRelu, params->activation);
  EXPECT_EQ(3, params->dilation_width_factor);
  EXPECT_EQ(4, params->dilation_height_factor);
}

TEST_F(FlatbufferConversionsTest, ParseBadFullyConnected) {
  const Operator* conv_op = BuildTestOperator(
      BuiltinOptions_FullyConnectedOptions,
      CreateFullyConnectedOptions(
          builder_, ActivationFunctionType_RELU,
          static_cast<FullyConnectedOptionsWeightsFormat>(-1), true)
          .Union());
  void* output_data = nullptr;
  EXPECT_EQ(kTfLiteError,
            ParseOpData(conv_op, BuiltinOperator_FULLY_CONNECTED,
                        &mock_reporter_, &mock_allocator_, &output_data));
}

TEST_F(FlatbufferConversionsTest, TestParseOpDataCustom) {
  const Operator* custom_op =
      BuildTestOperator(BuiltinOptions_NONE, flatbuffers::Offset<void>());
  void* output_data = nullptr;
  EXPECT_EQ(kTfLiteOk,
            ParseOpData(custom_op, BuiltinOperator_CUSTOM, &mock_reporter_,
                        &mock_allocator_, &output_data));
  EXPECT_EQ(nullptr, output_data);
}

TEST_F(FlatbufferConversionsTest, TestConvertTensorType) {
  TfLiteType type;
  EXPECT_EQ(kTfLiteOk,
            ConvertTensorType(TensorType_FLOAT32, &type, &mock_reporter_));
  EXPECT_EQ(kTfLiteFloat32, type);
}

TEST_F(FlatbufferConversionsTest, TestConvertTensorTypeFloat16) {
  TfLiteType type;
  EXPECT_EQ(kTfLiteOk,
            ConvertTensorType(TensorType_FLOAT16, &type, &mock_reporter_));
  EXPECT_EQ(kTfLiteFloat16, type);
}

}  // namespace tflite
