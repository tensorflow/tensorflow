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

  CacheBuffer cache_buffer;
  cache_buffer.Initialize(*shape);

  EXPECT_EQ(cache_buffer.GetSize(), 420);
  ASSERT_NE(cache_buffer.GetBuffer(), nullptr);
  EXPECT_EQ(cache_buffer.GetNumEntries(0), 0);
  EXPECT_EQ(cache_buffer.GetNumEntries(1), 0);
  EXPECT_EQ(cache_buffer.GetNumEntries(2), 0);
  cache_buffer.SetNumEntries(0, 3);
  EXPECT_EQ(cache_buffer.GetNumEntries(0), 3);
  TfLiteIntArrayFree(shape);
}

}  // namespace resource
}  // namespace tflite
