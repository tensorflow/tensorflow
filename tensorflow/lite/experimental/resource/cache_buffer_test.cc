/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/experimental/resource/cache_buffer.h"

#include <gtest/gtest.h>
#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace resource {

TEST(CacheBufferTest, Initialize) {
  TfLiteIntArray* shape = TfLiteIntArrayCreate(4);
  shape->data[0] = 1;
  shape->data[1] = 3;
  shape->data[2] = 5;
  shape->data[3] = 7;

  TfLiteType type = kTfLiteFloat32;
  CacheBuffer cache_buffer;
  cache_buffer.Initialize(*shape, type);

  EXPECT_EQ(cache_buffer.GetTensor()->type, type);
  EXPECT_EQ(cache_buffer.GetTensor()->dims->size, 4);
  EXPECT_EQ(cache_buffer.GetTensor()->dims->data[0], 1);
  EXPECT_EQ(cache_buffer.GetTensor()->dims->data[1], 3);
  EXPECT_EQ(cache_buffer.GetTensor()->bytes, 420);
  ASSERT_NE(cache_buffer.GetTensor()->data.raw, nullptr);
  EXPECT_EQ(cache_buffer.GetNumEntries(), 0);
  cache_buffer.SetNumEntries(3);
  EXPECT_EQ(cache_buffer.GetNumEntries(), 3);
  TfLiteIntArrayFree(shape);
}

}  // namespace resource
}  // namespace tflite
