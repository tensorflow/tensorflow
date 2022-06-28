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
#include "tensorflow/lite/delegates/flex/buffer_map.h"

#include <sys/types.h>

#include <functional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/delegates/flex/buffer_map_util.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/testing/util.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace flex {
namespace {

using ::testing::ElementsAre;

// A bit of RAII to simplify handling of TfLiteTensors in the tests.
using UniqueTfLiteTensor =
    std::unique_ptr<TfLiteTensor, std::function<void(TfLiteTensor*)>>;

template <typename T>
UniqueTfLiteTensor MakeLiteTensor(const std::vector<int>& shape,
                                  const std::vector<T>& data) {
  auto tensor = UniqueTfLiteTensor(new TfLiteTensor(), [](TfLiteTensor* t) {
    TfLiteTensorDataFree(t);
    TfLiteIntArrayFree(t->dims);
    delete t;
  });
  tensor->allocation_type = kTfLiteDynamic;
  tensor->type = typeToTfLiteType<T>();
  tensor->dims = ConvertVectorToTfLiteIntArray(shape);
  TfLiteTensorRealloc(data.size() * sizeof(T), tensor.get());
  memcpy(tensor->data.raw, data.data(), data.size() * sizeof(T));
  return tensor;
}

template <>
UniqueTfLiteTensor MakeLiteTensor<string>(const std::vector<int>& shape,
                                          const std::vector<string>& data) {
  auto tensor = UniqueTfLiteTensor(new TfLiteTensor(), [](TfLiteTensor* t) {
    TfLiteTensorDataFree(t);
    TfLiteIntArrayFree(t->dims);
    delete t;
  });
  tensor->allocation_type = kTfLiteDynamic;
  tensor->type = typeToTfLiteType<string>();
  tensor->dims = ConvertVectorToTfLiteIntArray(shape);
  TfLiteTensorRealloc(data.size() * sizeof(string), tensor.get());

  DynamicBuffer b;
  for (const string& s : data) {
    b.AddString(s.data(), s.size());
  }
  b.WriteToTensor(tensor.get(), ConvertVectorToTfLiteIntArray(shape));
  return tensor;
}

template <typename T>
tensorflow::Tensor MakeTensor(const std::vector<int64_t>& shape,
                              const std::vector<T>& data,
                              tensorflow::DataType dtype) {
  tensorflow::Tensor tensor(dtype, tensorflow::TensorShape(shape));
  memcpy(tensor.data(), data.data(), data.size() * sizeof(T));
  return tensor;
}

std::vector<int64_t> GetTensorShape(const tensorflow::Tensor& t) {
  std::vector<int64_t> shape(t.dims());
  for (int i = 0; i < t.dims(); ++i) {
    shape[i] = t.dim_size(i);
  }
  return shape;
}

template <typename T>
std::vector<T> GetTensorData(const tensorflow::Tensor& t) {
  const T* data = t.flat<T>().data();
  return std::vector<T>(data, data + t.NumElements());
}

TEST(BufferMapTest, EmptyBuffer) {
  BufferMap buffer_map;
  EXPECT_FALSE(buffer_map.HasTensor(0));
}

TEST(BufferMapTest, SetFromTfLite) {
  BufferMap buffer_map;

  UniqueTfLiteTensor t =
      MakeLiteTensor<float>({1, 2, 1, 3}, {0, 0, 0, 0.123f, 0, 0});
  buffer_map.SetFromTfLite(0, t.get());
  ASSERT_TRUE(buffer_map.HasTensor(0));

  EXPECT_THAT(GetTensorData<float>(buffer_map.GetTensor(0)),
              ElementsAre(0, 0, 0, 0.123f, 0, 0));

  // Also check details of the tensor.
  tensorflow::Tensor out_tensor = buffer_map.GetTensor(0);
  ASSERT_EQ(out_tensor.dtype(), tensorflow::DT_FLOAT);
  ASSERT_EQ(out_tensor.NumElements(), 6);
  ASSERT_THAT(GetTensorShape(out_tensor), ElementsAre(1, 2, 1, 3));
}

TEST(BufferMapTest, SetFromTfLiteString) {
  BufferMap buffer_map;

  UniqueTfLiteTensor t =
      MakeLiteTensor<string>({1, 2, 1, 3}, {"", "", "", "str1", "", ""});
  buffer_map.SetFromTfLite(0, t.get());
  ASSERT_TRUE(buffer_map.HasTensor(0));

  EXPECT_THAT(GetTensorData<tensorflow::tstring>(buffer_map.GetTensor(0)),
              ElementsAre("", "", "", "str1", "", ""));

  // Also check details of the tensor.
  tensorflow::Tensor out_tensor = buffer_map.GetTensor(0);
  ASSERT_EQ(out_tensor.dtype(), tensorflow::DT_STRING);
  ASSERT_EQ(out_tensor.NumElements(), 6);
  ASSERT_THAT(GetTensorShape(out_tensor), ElementsAre(1, 2, 1, 3));
}

TEST(BufferMapTest, SetFromTfLiteTwice) {
  UniqueTfLiteTensor t1 =
      MakeLiteTensor<float>({1, 2, 1, 3}, {0, 0, 0, 0.123f, 0, 0});
  UniqueTfLiteTensor t2 =
      MakeLiteTensor<int>({1, 2, 4}, {0, 0, 0, 3, 0, 0, 1, 2});

  BufferMap buffer_map;
  buffer_map.SetFromTfLite(0, t1.get());
  buffer_map.SetFromTfLite(0, t2.get());

  EXPECT_THAT(GetTensorData<int>(buffer_map.GetTensor(0)),
              ElementsAre(0, 0, 0, 3, 0, 0, 1, 2));
}

TEST(BufferMapTest, SetFromTfLiteStringTwice) {
  UniqueTfLiteTensor t1 =
      MakeLiteTensor<float>({1, 2, 1, 3}, {0, 0, 0, 0.123f, 0, 0});
  UniqueTfLiteTensor t2 =
      MakeLiteTensor<string>({1, 2, 4}, {"", "", "", "s3", "", "", "s1", "s2"});

  BufferMap buffer_map;
  buffer_map.SetFromTfLite(0, t1.get());
  buffer_map.SetFromTfLite(0, t2.get());

  EXPECT_THAT(GetTensorData<tensorflow::tstring>(buffer_map.GetTensor(0)),
              ElementsAre("", "", "", "s3", "", "", "s1", "s2"));
}

TEST(BufferMapTest, SetFromTfLiteBuiltinResource) {
  BufferMap buffer_map;

  // Constructs a fake resource tensor.
  auto tensor = UniqueTfLiteTensor(new TfLiteTensor(), [](TfLiteTensor* t) {
    TfLiteTensorDataFree(t);
    TfLiteIntArrayFree(t->dims);
    delete t;
  });
  tensor->allocation_type = kTfLiteDynamic;
  tensor->type = kTfLiteResource;
  tensor->dims = ConvertVectorToTfLiteIntArray({1});
  TfLiteTensorRealloc(sizeof(int32_t), tensor.get());
  tensor->delegate = nullptr;
  tensor->data.i32[0] = 1;

  buffer_map.SetFromTfLite(0, tensor.get());
  // Also check details of the tensor.
  tensorflow::Tensor out_tensor = buffer_map.GetTensor(0);
  ASSERT_EQ(out_tensor.dtype(), tensorflow::DT_RESOURCE);
  ASSERT_EQ(out_tensor.NumElements(), 1);
  tensorflow::ResourceHandle handle =
      out_tensor.flat<tensorflow::ResourceHandle>()(0);
  EXPECT_EQ(handle.name(), "tflite_resource_variable:1");
}

TEST(BufferMapTest, SetFromTensorFlow) {
  tensorflow::Tensor t1 = MakeTensor<float>(
      {1, 2, 1, 3}, {0, 0, 0, 0.123f, 0, 0}, tensorflow::DT_FLOAT);

  BufferMap buffer_map;
  buffer_map.SetFromTensorFlow(0, t1);

  EXPECT_THAT(GetTensorData<float>(buffer_map.GetTensor(0)),
              ElementsAre(0, 0, 0, 0.123f, 0, 0));

  // Also check details of the tensor.
  tensorflow::Tensor out_tensor = buffer_map.GetTensor(0);
  ASSERT_EQ(out_tensor.dtype(), tensorflow::DT_FLOAT);
  ASSERT_EQ(out_tensor.NumElements(), 6);
  ASSERT_THAT(GetTensorShape(out_tensor), ElementsAre(1, 2, 1, 3));
}

TEST(BufferMapTest, SetFromTensorFlowTwice) {
  tensorflow::Tensor t1 = MakeTensor<float>(
      {1, 2, 1, 3}, {0, 0, 0, 0.123f, 0, 0}, tensorflow::DT_FLOAT);
  tensorflow::Tensor t2 = MakeTensor<int>({1, 2, 4}, {0, 0, 0, 3, 0, 0, 1, 2},
                                          tensorflow::DT_INT32);
  BufferMap buffer_map;
  buffer_map.SetFromTensorFlow(0, t1);
  buffer_map.SetFromTensorFlow(0, t2);

  EXPECT_THAT(GetTensorData<int>(buffer_map.GetTensor(0)),
              ElementsAre(0, 0, 0, 3, 0, 0, 1, 2));
}

TEST(BufferMapTest, TfLiteOverwritesTensorFlow) {
  tensorflow::Tensor t1 = MakeTensor<float>(
      {1, 2, 1, 3}, {0, 0, 0, 0.123f, 0, 0}, tensorflow::DT_FLOAT);
  UniqueTfLiteTensor t2 =
      MakeLiteTensor<int>({1, 2, 4}, {0, 0, 0, 3, 0, 0, 1, 2});

  BufferMap buffer_map;
  buffer_map.SetFromTensorFlow(0, t1);
  buffer_map.SetFromTfLite(0, t2.get());

  EXPECT_FALSE(buffer_map.IsTensorFlowTensor(0));
  EXPECT_THAT(GetTensorData<int>(buffer_map.GetTensor(0)),
              ElementsAre(0, 0, 0, 3, 0, 0, 1, 2));
}

TEST(BufferMapTest, TensorFlowOverwritesTfLite) {
  tensorflow::Tensor t1 = MakeTensor<float>(
      {1, 2, 1, 3}, {0, 0, 0, 0.123f, 0, 0}, tensorflow::DT_FLOAT);
  UniqueTfLiteTensor t2 =
      MakeLiteTensor<int>({1, 2, 4}, {0, 0, 0, 3, 0, 0, 1, 2});
  BufferMap buffer_map;
  buffer_map.SetFromTfLite(0, t2.get());
  buffer_map.SetFromTensorFlow(0, t1);

  EXPECT_TRUE(buffer_map.IsTensorFlowTensor(0));
  EXPECT_THAT(GetTensorData<float>(buffer_map.GetTensor(0)),
              ElementsAre(0, 0, 0, 0.123f, 0, 0));
}

TEST(BufferMapTest, TensorflowBufferReuse) {
  TfLiteTensor tensor;
  tensor.allocation_type = kTfLiteDynamic;
  tensor.data.raw = nullptr;
  TfLiteTensorRealloc(10, &tensor);
  CHECK(tensor.data.raw);
  EXPECT_EQ(tensor.bytes, 10);

  TfLiteTensorBuffer* tensor_buffer_reused = new TfLiteTensorBuffer(&tensor);
  // Checks that the underlying buffer is reused.
  EXPECT_TRUE(tensor_buffer_reused->BufferReusedFromTfLiteTensor());
  EXPECT_EQ(tensor_buffer_reused->data(), tensor.data.raw);
  tensor_buffer_reused->Unref();

  TfLiteTensorDataFree(&tensor);
}

}  // namespace
}  // namespace flex
}  // namespace tflite
