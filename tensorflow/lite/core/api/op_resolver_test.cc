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

#include "tensorflow/lite/core/api/op_resolver.h"

#include <cstring>

#include <gtest/gtest.h>

namespace tflite {
namespace {
void* MockInit(TfLiteContext* context, const char* buffer, size_t length) {
  // Do nothing.
  return nullptr;
}

void MockFree(TfLiteContext* context, void* buffer) {
  // Do nothing.
}

TfLiteStatus MockPrepare(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteOk;
}

TfLiteStatus MockInvoke(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteOk;
}

class MockOpResolver : public OpResolver {
 public:
  const TfLiteRegistration* FindOp(BuiltinOperator op,
                                   int version) const override {
    if (op == BuiltinOperator_CONV_2D) {
      static TfLiteRegistration r = {MockInit, MockFree, MockPrepare,
                                     MockInvoke};
      return &r;
    } else {
      return nullptr;
    }
  }
  const TfLiteRegistration* FindOp(const char* op, int version) const override {
    if (strcmp(op, "mock_custom") == 0) {
      static TfLiteRegistration r = {MockInit, MockFree, MockPrepare,
                                     MockInvoke};
      return &r;
    } else {
      return nullptr;
    }
  }
};

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

}  // namespace

TEST(OpResolver, TestResolver) {
  MockOpResolver mock_resolver;
  OpResolver* resolver = &mock_resolver;

  const TfLiteRegistration* registration =
      resolver->FindOp(BuiltinOperator_CONV_2D, 0);
  EXPECT_NE(nullptr, registration);
  EXPECT_EQ(nullptr, registration->init(nullptr, nullptr, 0));
  EXPECT_EQ(kTfLiteOk, registration->prepare(nullptr, nullptr));
  EXPECT_EQ(kTfLiteOk, registration->invoke(nullptr, nullptr));

  registration = resolver->FindOp(BuiltinOperator_CAST, 0);
  EXPECT_EQ(nullptr, registration);

  registration = resolver->FindOp("mock_custom", 0);
  EXPECT_NE(nullptr, registration);
  EXPECT_EQ(nullptr, registration->init(nullptr, nullptr, 0));
  EXPECT_EQ(kTfLiteOk, registration->prepare(nullptr, nullptr));
  EXPECT_EQ(kTfLiteOk, registration->invoke(nullptr, nullptr));

  registration = resolver->FindOp("nonexistent_custom", 0);
  EXPECT_EQ(nullptr, registration);
}

TEST(OpResolver, TestGetRegistrationFromOpCodeConv) {
  MockOpResolver mock_resolver;
  OpResolver* resolver = &mock_resolver;
  MockErrorReporter mock_reporter;
  ErrorReporter* reporter = &mock_reporter;

  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<OperatorCode> conv_offset =
      CreateOperatorCodeDirect(builder, BuiltinOperator_CONV_2D, nullptr, 0);
  builder.Finish(conv_offset);
  void* conv_pointer = builder.GetBufferPointer();
  const OperatorCode* conv_code =
      flatbuffers::GetRoot<OperatorCode>(conv_pointer);
  const TfLiteRegistration* registration = nullptr;
  EXPECT_EQ(kTfLiteOk, GetRegistrationFromOpCode(conv_code, *resolver, reporter,
                                                 &registration));
  EXPECT_NE(nullptr, registration);
  EXPECT_EQ(nullptr, registration->init(nullptr, nullptr, 0));
  EXPECT_EQ(kTfLiteOk, registration->prepare(nullptr, nullptr));
  EXPECT_EQ(kTfLiteOk, registration->invoke(nullptr, nullptr));
  EXPECT_EQ(0, mock_reporter.GetBufferSize());
}

TEST(OpResolver, TestGetRegistrationFromOpCodeCast) {
  MockOpResolver mock_resolver;
  OpResolver* resolver = &mock_resolver;
  MockErrorReporter mock_reporter;
  ErrorReporter* reporter = &mock_reporter;

  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<OperatorCode> conv_offset =
      CreateOperatorCodeDirect(builder, BuiltinOperator_CAST, nullptr, 0);
  builder.Finish(conv_offset);
  void* conv_pointer = builder.GetBufferPointer();
  const OperatorCode* conv_code =
      flatbuffers::GetRoot<OperatorCode>(conv_pointer);
  const TfLiteRegistration* registration = nullptr;
  EXPECT_EQ(kTfLiteError, GetRegistrationFromOpCode(conv_code, *resolver,
                                                    reporter, &registration));
  EXPECT_EQ(nullptr, registration);
  EXPECT_NE(0, mock_reporter.GetBufferSize());
}

TEST(OpResolver, TestGetRegistrationFromOpCodeCustom) {
  MockOpResolver mock_resolver;
  OpResolver* resolver = &mock_resolver;
  MockErrorReporter mock_reporter;
  ErrorReporter* reporter = &mock_reporter;

  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<OperatorCode> conv_offset = CreateOperatorCodeDirect(
      builder, BuiltinOperator_CUSTOM, "mock_custom", 0);
  builder.Finish(conv_offset);
  void* conv_pointer = builder.GetBufferPointer();
  const OperatorCode* conv_code =
      flatbuffers::GetRoot<OperatorCode>(conv_pointer);
  const TfLiteRegistration* registration = nullptr;
  EXPECT_EQ(kTfLiteOk, GetRegistrationFromOpCode(conv_code, *resolver, reporter,
                                                 &registration));
  EXPECT_NE(nullptr, registration);
  EXPECT_EQ(nullptr, registration->init(nullptr, nullptr, 0));
  EXPECT_EQ(kTfLiteOk, registration->prepare(nullptr, nullptr));
  EXPECT_EQ(kTfLiteOk, registration->invoke(nullptr, nullptr));
  EXPECT_EQ(0, mock_reporter.GetBufferSize());
}

TEST(OpResolver, TestGetRegistrationFromOpCodeNonexistentCustom) {
  MockOpResolver mock_resolver;
  OpResolver* resolver = &mock_resolver;
  MockErrorReporter mock_reporter;
  ErrorReporter* reporter = &mock_reporter;

  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<OperatorCode> conv_offset = CreateOperatorCodeDirect(
      builder, BuiltinOperator_CUSTOM, "nonexistent_custom", 0);
  builder.Finish(conv_offset);
  void* conv_pointer = builder.GetBufferPointer();
  const OperatorCode* conv_code =
      flatbuffers::GetRoot<OperatorCode>(conv_pointer);
  const TfLiteRegistration* registration = nullptr;
  EXPECT_EQ(kTfLiteError, GetRegistrationFromOpCode(conv_code, *resolver,
                                                    reporter, &registration));
  EXPECT_EQ(nullptr, registration);
  // There is no error, since unresolved custom ops are checked while preparing
  // nodes.
  EXPECT_EQ(0, mock_reporter.GetBufferSize());
}

}  // namespace tflite

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
