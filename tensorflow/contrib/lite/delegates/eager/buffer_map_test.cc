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
#include "tensorflow/contrib/lite/delegates/eager/buffer_map.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/testing/util.h"
#include "tensorflow/contrib/lite/util.h"

namespace tflite {
namespace eager {
namespace {

using ::testing::ElementsAre;

// A bit of RAII to simplify handling of TfLiteTensors in the tests.
using UniqueTfLiteTensor =
    std::unique_ptr<TfLiteTensor, std::function<void(TfLiteTensor*)>>;

template <typename T>
UniqueTfLiteTensor MakeLiteTensor(const std::vector<int>& shape,
                                  const std::vector<T>& data) {
  auto tensor = UniqueTfLiteTensor(new TfLiteTensor, [](TfLiteTensor* t) {
    TfLiteTensorDataFree(t);
    TfLiteIntArrayFree(t->dims);
    delete t;
  });
  tensor->allocation_type = kTfLiteDynamic;
  tensor->type = typeToTfLiteType<T>();
  tensor->dims = ConvertVectorToTfLiteIntArray(shape);
  tensor->data.raw = nullptr;
  TfLiteTensorRealloc(data.size() * sizeof(T), tensor.get());
  memcpy(tensor->data.raw, data.data(), data.size() * sizeof(T));
  return tensor;
}

template <typename T>
tensorflow::Tensor MakeTensor(const std::vector<int>& shape,
                              const std::vector<T>& data) {
  BufferMap buffer_map;  // BufferMap is the easiest way to build the tensor.
  UniqueTfLiteTensor t1 = MakeLiteTensor<T>(shape, data);
  buffer_map.SetFromTfLite(0, t1.get());
  return buffer_map.GetTensor(0);
}

std::vector<tensorflow::int64> GetTensorShape(const tensorflow::Tensor& t) {
  std::vector<tensorflow::int64> shape(t.dims());
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

TEST(BufferMapTest, SetFromTensorFlow) {
  tensorflow::Tensor t1 =
      MakeTensor<float>({1, 2, 1, 3}, {0, 0, 0, 0.123f, 0, 0});

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
  tensorflow::Tensor t1 =
      MakeTensor<float>({1, 2, 1, 3}, {0, 0, 0, 0.123f, 0, 0});
  tensorflow::Tensor t2 = MakeTensor<int>({1, 2, 4}, {0, 0, 0, 3, 0, 0, 1, 2});
  BufferMap buffer_map;
  buffer_map.SetFromTensorFlow(0, t1);
  buffer_map.SetFromTensorFlow(0, t2);

  EXPECT_THAT(GetTensorData<int>(buffer_map.GetTensor(0)),
              ElementsAre(0, 0, 0, 3, 0, 0, 1, 2));
}

TEST(BufferMapTest, TfLiteOverwritesTensorFlow) {
  tensorflow::Tensor t1 =
      MakeTensor<float>({1, 2, 1, 3}, {0, 0, 0, 0.123f, 0, 0});
  UniqueTfLiteTensor t2 =
      MakeLiteTensor<int>({1, 2, 4}, {0, 0, 0, 3, 0, 0, 1, 2});

  BufferMap buffer_map;
  buffer_map.SetFromTensorFlow(0, t1);
  buffer_map.SetFromTfLite(0, t2.get());

  EXPECT_THAT(GetTensorData<int>(buffer_map.GetTensor(0)),
              ElementsAre(0, 0, 0, 3, 0, 0, 1, 2));
}

TEST(BufferMapTest, TensorFlowOverwritesTfLite) {
  tensorflow::Tensor t1 =
      MakeTensor<float>({1, 2, 1, 3}, {0, 0, 0, 0.123f, 0, 0});
  UniqueTfLiteTensor t2 =
      MakeLiteTensor<int>({1, 2, 4}, {0, 0, 0, 3, 0, 0, 1, 2});
  BufferMap buffer_map;
  buffer_map.SetFromTfLite(0, t2.get());
  buffer_map.SetFromTensorFlow(0, t1);

  EXPECT_THAT(GetTensorData<float>(buffer_map.GetTensor(0)),
              ElementsAre(0, 0, 0, 0.123f, 0, 0));
}

}  // namespace
}  // namespace eager
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
