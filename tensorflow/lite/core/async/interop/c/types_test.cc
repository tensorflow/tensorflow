/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/core/async/interop/c/types.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace {

TEST(TypesTest, TfLiteBackendBuffer) {
  auto* tflite_buf = TfLiteBackendBufferCreate();

  EXPECT_EQ(nullptr, TfLiteBackendBufferGetPtr(tflite_buf));

  char buf[42];
  TfLiteBackendBufferSetPtr(tflite_buf, buf);
  EXPECT_EQ(buf, TfLiteBackendBufferGetPtr(tflite_buf));

  char another_buf[7];
  TfLiteBackendBufferSetPtr(tflite_buf, another_buf);
  EXPECT_EQ(another_buf, TfLiteBackendBufferGetPtr(tflite_buf));

  TfLiteBackendBufferDelete(tflite_buf);
}

TEST(TypesTest, TfLiteSynchronization) {
  auto* tflite_sync = TfLiteSynchronizationCreate();

  EXPECT_EQ(nullptr, TfLiteSynchronizationGetPtr(tflite_sync));

  int fd = 42;
  TfLiteSynchronizationSetPtr(tflite_sync, &fd);
  EXPECT_EQ(&fd, TfLiteSynchronizationGetPtr(tflite_sync));

  double sync = 7.11;
  TfLiteSynchronizationSetPtr(tflite_sync, &sync);
  EXPECT_EQ(&sync, TfLiteSynchronizationGetPtr(tflite_sync));

  TfLiteSynchronizationDelete(tflite_sync);
}

}  // namespace
