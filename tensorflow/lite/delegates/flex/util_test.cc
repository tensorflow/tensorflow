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
#include "tensorflow/lite/delegates/flex/util.h"

#include <cstdarg>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/string_type.h"
#include "tensorflow/lite/testing/util.h"

namespace tflite {
namespace flex {
namespace {

using tensorflow::DT_FLOAT;
using tensorflow::DT_INT32;
using tensorflow::Tensor;
using ::testing::ElementsAre;

struct TestContext : public TfLiteContext {
  string error;
  std::vector<int> new_size;
};

void ReportError(TfLiteContext* context, const char* format, ...) {
  TestContext* c = static_cast<TestContext*>(context);
  const size_t kBufferSize = 1024;
  char temp_buffer[kBufferSize];

  va_list args;
  va_start(args, format);
  vsnprintf(temp_buffer, kBufferSize, format, args);
  va_end(args);

  c->error = temp_buffer;
}

TfLiteStatus ResizeTensor(TfLiteContext* context, TfLiteTensor* tensor,
                          TfLiteIntArray* new_size) {
  TestContext* c = static_cast<TestContext*>(context);
  c->new_size.clear();
  for (int i = 0; i < new_size->size; ++i) {
    c->new_size.push_back(new_size->data[i]);
  }
  TfLiteIntArrayFree(new_size);
  return kTfLiteOk;
}

TEST(UtilTest, ConvertStatus) {
  TestContext context;
  context.ReportError = ReportError;

  EXPECT_EQ(ConvertStatus(&context, tensorflow::errors::Internal("Some Error")),
            kTfLiteError);
  EXPECT_EQ(context.error, "Some Error");

  context.error.clear();
  EXPECT_EQ(ConvertStatus(&context, tensorflow::Status()), kTfLiteOk);
  EXPECT_TRUE(context.error.empty());
}

TEST(UtilTest, CopyShapeAndType) {
  TestContext context;
  context.ReportError = ReportError;
  context.ResizeTensor = ResizeTensor;

  TfLiteTensor dst;

  EXPECT_EQ(CopyShapeAndType(&context, Tensor(), &dst), kTfLiteOk);
  EXPECT_THAT(context.new_size, ElementsAre(0));
  EXPECT_EQ(dst.type, kTfLiteFloat32);

  EXPECT_EQ(CopyShapeAndType(&context, Tensor(DT_FLOAT, {1, 2}), &dst),
            kTfLiteOk);
  EXPECT_THAT(context.new_size, ElementsAre(1, 2));
  EXPECT_EQ(dst.type, kTfLiteFloat32);

  EXPECT_EQ(CopyShapeAndType(&context, Tensor(DT_INT32, {1, 2}), &dst),
            kTfLiteOk);
  EXPECT_THAT(context.new_size, ElementsAre(1, 2));
  EXPECT_EQ(dst.type, kTfLiteInt32);

  EXPECT_EQ(CopyShapeAndType(&context, Tensor(DT_FLOAT, {1LL << 44, 2}), &dst),
            kTfLiteError);
  EXPECT_EQ(context.error,
            "Dimension value in TensorFlow shape is larger than supported by "
            "TF Lite");

  EXPECT_EQ(
      CopyShapeAndType(&context, Tensor(tensorflow::DT_HALF, {1, 2}), &dst),
      kTfLiteOk);
  EXPECT_THAT(context.new_size, ElementsAre(1, 2));
  EXPECT_EQ(dst.type, kTfLiteFloat16);
}

TEST(UtilTest, TypeConversionsFromTFLite) {
  EXPECT_EQ(TF_FLOAT, GetTensorFlowDataType(kTfLiteNoType));
  EXPECT_EQ(TF_FLOAT, GetTensorFlowDataType(kTfLiteFloat32));
  EXPECT_EQ(TF_HALF, GetTensorFlowDataType(kTfLiteFloat16));
  EXPECT_EQ(TF_DOUBLE, GetTensorFlowDataType(kTfLiteFloat64));
  EXPECT_EQ(TF_INT16, GetTensorFlowDataType(kTfLiteInt16));
  EXPECT_EQ(TF_INT32, GetTensorFlowDataType(kTfLiteInt32));
  EXPECT_EQ(TF_UINT8, GetTensorFlowDataType(kTfLiteUInt8));
  EXPECT_EQ(TF_INT64, GetTensorFlowDataType(kTfLiteInt64));
  EXPECT_EQ(TF_UINT64, GetTensorFlowDataType(kTfLiteUInt64));
  EXPECT_EQ(TF_COMPLEX64, GetTensorFlowDataType(kTfLiteComplex64));
  EXPECT_EQ(TF_COMPLEX128, GetTensorFlowDataType(kTfLiteComplex128));
  EXPECT_EQ(TF_STRING, GetTensorFlowDataType(kTfLiteString));
  EXPECT_EQ(TF_BOOL, GetTensorFlowDataType(kTfLiteBool));
  EXPECT_EQ(TF_RESOURCE, GetTensorFlowDataType(kTfLiteResource));
  EXPECT_EQ(TF_VARIANT, GetTensorFlowDataType(kTfLiteVariant));
}

TEST(UtilTest, TypeConversionsFromTensorFlow) {
  EXPECT_EQ(kTfLiteFloat16, GetTensorFlowLiteType(TF_HALF));
  EXPECT_EQ(kTfLiteFloat32, GetTensorFlowLiteType(TF_FLOAT));
  EXPECT_EQ(kTfLiteFloat64, GetTensorFlowLiteType(TF_DOUBLE));
  EXPECT_EQ(kTfLiteInt16, GetTensorFlowLiteType(TF_INT16));
  EXPECT_EQ(kTfLiteInt32, GetTensorFlowLiteType(TF_INT32));
  EXPECT_EQ(kTfLiteUInt8, GetTensorFlowLiteType(TF_UINT8));
  EXPECT_EQ(kTfLiteInt64, GetTensorFlowLiteType(TF_INT64));
  EXPECT_EQ(kTfLiteUInt64, GetTensorFlowLiteType(TF_UINT64));
  EXPECT_EQ(kTfLiteComplex64, GetTensorFlowLiteType(TF_COMPLEX64));
  EXPECT_EQ(kTfLiteComplex128, GetTensorFlowLiteType(TF_COMPLEX128));
  EXPECT_EQ(kTfLiteString, GetTensorFlowLiteType(TF_STRING));
  EXPECT_EQ(kTfLiteBool, GetTensorFlowLiteType(TF_BOOL));
  EXPECT_EQ(kTfLiteResource, GetTensorFlowLiteType(TF_RESOURCE));
  EXPECT_EQ(kTfLiteVariant, GetTensorFlowLiteType(TF_VARIANT));
}

}  // namespace
}  // namespace flex
}  // namespace tflite
