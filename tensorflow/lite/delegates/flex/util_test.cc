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
#include <iterator>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/framework/resource_handle.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/string_type.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/testing/util.h"
#include "tensorflow/lite/util.h"

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
  // TODO(b/246806634): Tensorflow DT_INT4 type doesn't exist yet
  EXPECT_EQ(TF_INT8, GetTensorFlowDataType(kTfLiteInt4));
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

TEST(UtilTest, GetTfLiteResourceIdentifier) {
  // Constructs a fake resource tensor.
  TfLiteTensor tensor;
  tensor.allocation_type = kTfLiteDynamic;
  tensor.type = kTfLiteResource;
  std::vector<int> dims = {1};
  tensor.dims = ConvertVectorToTfLiteIntArray(dims);
  tensor.data.raw = nullptr;
  TfLiteTensorRealloc(sizeof(int32_t), &tensor);
  tensor.delegate = nullptr;
  tensor.data.i32[0] = 1;

  EXPECT_EQ(TfLiteResourceIdentifier(&tensor), "tflite_resource_variable:1");
  TfLiteIntArrayFree(tensor.dims);
  TfLiteTensorDataFree(&tensor);
}

TEST(UtilTest, GetTfLiteResourceTensorFromResourceHandle) {
  tensorflow::ResourceHandle handle;
  handle.set_name("tflite_resource_variable:1");

  TfLiteTensor tensor;
  tensor.allocation_type = kTfLiteDynamic;
  tensor.type = kTfLiteResource;
  tensor.data.raw = nullptr;
  std::vector<int> dims = {1};
  tensor.dims = ConvertVectorToTfLiteIntArray(dims);
  EXPECT_TRUE(GetTfLiteResourceTensorFromResourceHandle(handle, &tensor));
  EXPECT_EQ(tensor.data.i32[0], 1);

  TfLiteIntArrayFree(tensor.dims);
  TfLiteTensorDataFree(&tensor);
}

TEST(UtilTest, CreateTfTensorFromTfLiteTensorResourceOrVariant) {
  TfLiteTensor tensor;
  tensor.type = kTfLiteResource;
  EXPECT_EQ(CreateTfTensorFromTfLiteTensor(&tensor).status().code(),
            tensorflow::error::INVALID_ARGUMENT);
  tensor.type = kTfLiteVariant;
  EXPECT_EQ(CreateTfTensorFromTfLiteTensor(&tensor).status().code(),
            tensorflow::error::INVALID_ARGUMENT);
}

TEST(UtilTest, CreateTfTensorFromTfLiteTensorFloat) {
  TfLiteTensor tflite_tensor;
  tflite_tensor.type = kTfLiteFloat32;
  tflite_tensor.allocation_type = kTfLiteDynamic;
  tflite_tensor.sparsity = nullptr;
  tflite_tensor.dims_signature = nullptr;

  TfLiteQuantization quant;
  quant.type = kTfLiteNoQuantization;
  quant.params = nullptr;
  tflite_tensor.quantization = quant;

  TfLiteIntArray* dims = TfLiteIntArrayCreate(2);
  dims->data[0] = 1;
  dims->data[1] = 3;
  tflite_tensor.dims = dims;
  float data_arr[] = {1.1, 0.456, 0.322};
  std::vector<float> data(std::begin(data_arr), std::end(data_arr));
  size_t num_bytes = data.size() * sizeof(float);
  tflite_tensor.data.raw = static_cast<char*>(malloc(num_bytes));
  memcpy(tflite_tensor.data.raw, data.data(), num_bytes);
  tflite_tensor.bytes = num_bytes;

  auto tf_tensor_or = CreateTfTensorFromTfLiteTensor(&tflite_tensor);
  EXPECT_TRUE(tf_tensor_or.ok());
  tensorflow::Tensor tf_tensor = tf_tensor_or.value();
  EXPECT_EQ(tf_tensor.NumElements(), 3);
  auto* tf_data = static_cast<float*>(tf_tensor.data());
  for (float weight : data_arr) {
    EXPECT_EQ(*tf_data, weight);
    tf_data++;
  }

  TfLiteTensorFree(&tflite_tensor);
}

TEST(UtilTest, CreateTfTensorFromTfLiteTensorString) {
  TfLiteTensor tflite_tensor{};
  tflite_tensor.type = kTfLiteString;
  tflite_tensor.is_variable = false;
  tflite_tensor.sparsity = nullptr;
  tflite_tensor.data.raw = nullptr;
  tflite_tensor.dims_signature = nullptr;
  tflite_tensor.allocation_type = kTfLiteArenaRw;

  TfLiteQuantization quant;
  quant.type = kTfLiteNoQuantization;
  quant.params = nullptr;
  tflite_tensor.quantization = quant;

  TfLiteIntArray* dims = TfLiteIntArrayCreate(2);
  dims->data[0] = 1;
  dims->data[1] = 2;
  tflite_tensor.dims = dims;
  std::string data_arr[] = {std::string("a_str\0ing", 9), "b_string"};
  tflite::DynamicBuffer buf;
  for (const auto& value : data_arr) {
    buf.AddString(value.data(), value.length());
  }
  buf.WriteToTensor(&tflite_tensor, nullptr);

  auto tf_tensor_or = CreateTfTensorFromTfLiteTensor(&tflite_tensor);
  EXPECT_TRUE(tf_tensor_or.ok());
  tensorflow::Tensor tf_tensor = tf_tensor_or.value();
  EXPECT_EQ(tf_tensor.NumElements(), 2);
  auto* tf_data = static_cast<tensorflow::tstring*>(tf_tensor.data());
  for (const auto& str : data_arr) {
    EXPECT_EQ(*tf_data, str);
    tf_data++;
  }
  TfLiteTensorFree(&tflite_tensor);
}

}  // namespace
}  // namespace flex
}  // namespace tflite
