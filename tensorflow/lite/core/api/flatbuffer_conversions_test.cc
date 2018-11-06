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

#include <gtest/gtest.h>
#include "tensorflow/lite/c/builtin_op_data.h"

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

 private:
  static constexpr int kBufferSize = 256;
  char buffer_[kBufferSize];
  int buffer_size_;
};

// Used to determine how the op data parsing function creates its working space.
class MockDataAllocator : public BuiltinDataAllocator {
 public:
  MockDataAllocator() : is_allocated_(false) {}
  void* Allocate(size_t size) override {
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

TEST(FlatbufferConversions, TestParseOpDataConv) {
  MockErrorReporter mock_reporter;
  ErrorReporter* reporter = &mock_reporter;
  MockDataAllocator mock_allocator;

  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<void> conv_options =
      CreateConv2DOptions(builder, Padding_SAME, 1, 2,
                          ActivationFunctionType_RELU, 3, 4)
          .Union();
  flatbuffers::Offset<Operator> conv_offset = CreateOperatorDirect(
      builder, 0, nullptr, nullptr, BuiltinOptions_Conv2DOptions, conv_options,
      nullptr, CustomOptionsFormat_FLEXBUFFERS, nullptr);
  builder.Finish(conv_offset);
  void* conv_pointer = builder.GetBufferPointer();
  const Operator* conv_op = flatbuffers::GetRoot<Operator>(conv_pointer);
  void* output_data = nullptr;
  EXPECT_EQ(kTfLiteOk, ParseOpData(conv_op, BuiltinOperator_CONV_2D, reporter,
                                   &mock_allocator, &output_data));
  EXPECT_NE(nullptr, output_data);
  TfLiteConvParams* params = reinterpret_cast<TfLiteConvParams*>(output_data);
  EXPECT_EQ(kTfLitePaddingSame, params->padding);
  EXPECT_EQ(1, params->stride_width);
  EXPECT_EQ(2, params->stride_height);
  EXPECT_EQ(kTfLiteActRelu, params->activation);
  EXPECT_EQ(3, params->dilation_width_factor);
  EXPECT_EQ(4, params->dilation_height_factor);
}

TEST(FlatbufferConversions, TestParseOpDataCustom) {
  MockErrorReporter mock_reporter;
  ErrorReporter* reporter = &mock_reporter;
  MockDataAllocator mock_allocator;

  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<void> null_options;
  flatbuffers::Offset<Operator> custom_offset = CreateOperatorDirect(
      builder, 0, nullptr, nullptr, BuiltinOptions_NONE, null_options, nullptr,
      CustomOptionsFormat_FLEXBUFFERS, nullptr);
  builder.Finish(custom_offset);
  void* custom_pointer = builder.GetBufferPointer();
  const Operator* custom_op = flatbuffers::GetRoot<Operator>(custom_pointer);
  void* output_data = nullptr;
  EXPECT_EQ(kTfLiteOk, ParseOpData(custom_op, BuiltinOperator_CUSTOM, reporter,
                                   &mock_allocator, &output_data));
  EXPECT_EQ(nullptr, output_data);
}

TEST(FlatbufferConversions, TestConvertTensorType) {
  MockErrorReporter mock_reporter;
  ErrorReporter* reporter = &mock_reporter;
  TfLiteType type;
  EXPECT_EQ(kTfLiteOk, ConvertTensorType(TensorType_FLOAT32, &type, reporter));
  EXPECT_EQ(kTfLiteFloat32, type);
}

}  // namespace tflite

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
