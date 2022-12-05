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
#include "tensorflow/lite/core/async/task_internal.h"

#include <string>

#include <gtest/gtest.h>
#include "tensorflow/lite/core/async/async_kernel_internal.h"
#include "tensorflow/lite/core/async/common.h"
#include "tensorflow/lite/core/async/interop/c/types.h"

namespace tflite::async {

TEST(TfLiteExecutionTaskTest, BasicTest) {
  tflite::async::ExecutionTask task;
  tflite::async::ExecutionTask::TensorNameMapT input_names;
  input_names["x"] = 1;
  input_names["y"] = 2;
  tflite::async::ExecutionTask::TensorNameMapT output_names;
  output_names["a"] = 3;
  task.SetInputNameMap(&input_names);
  task.SetOutputNameMap(&output_names);

  auto* sync = TfLiteSynchronizationCreate();

  EXPECT_EQ(kTfLiteOk, task.SetBufferHandle(kTfLiteIoInput, "x", 42));
  EXPECT_EQ(kTfLiteOk, task.SetBufferHandle(kTfLiteIoInput, "y", 43));
  EXPECT_EQ(kTfLiteOk, task.SetBufferHandle(kTfLiteIoOutput, "a", 44));
  EXPECT_EQ(kTfLiteOk, task.SetSynchronization(kTfLiteIoInput, "x", sync));

  EXPECT_EQ(42, task.GetBufferHandle(kTfLiteIoInput, "x"));
  EXPECT_EQ(43, task.GetBufferHandle(kTfLiteIoInput, "y"));
  EXPECT_EQ(44, task.GetBufferHandle(kTfLiteIoOutput, "a"));
  EXPECT_EQ(sync, task.GetSynchronization(kTfLiteIoInput, "x"));
  EXPECT_EQ(nullptr, task.GetSynchronization(kTfLiteIoInput, "y"));
  EXPECT_EQ(nullptr, task.GetSynchronization(kTfLiteIoOutput, "a"));

  TfLiteSynchronizationDelete(sync);
}

TEST(TfLiteExecutionTaskTest, NameMapUninitialized) {
  tflite::async::ExecutionTask task;

  EXPECT_EQ(kTfLiteNullBufferHandle,
            task.GetBufferHandle(kTfLiteIoInput, "foo"));
  EXPECT_EQ(kTfLiteNullBufferHandle,
            task.GetBufferHandle(kTfLiteIoOutput, "foo"));
  EXPECT_EQ(nullptr, task.GetSynchronization(kTfLiteIoOutput, "foo"));
  EXPECT_EQ(nullptr, task.GetSynchronization(kTfLiteIoOutput, "foo"));
}

TEST(TfLiteExecutionTaskTest, NoMatchingName) {
  tflite::async::ExecutionTask task;
  tflite::async::ExecutionTask::TensorNameMapT input_names;
  input_names["x"] = 1;
  input_names["y"] = 2;
  tflite::async::ExecutionTask::TensorNameMapT output_names;
  output_names["a"] = 3;
  task.SetInputNameMap(&input_names);
  task.SetOutputNameMap(&output_names);

  auto* sync = TfLiteSynchronizationCreate();

  EXPECT_EQ(kTfLiteError, task.SetBufferHandle(kTfLiteIoInput, "xx", 42));
  EXPECT_EQ(kTfLiteError, task.SetBufferHandle(kTfLiteIoOutput, "aa", 44));
  EXPECT_EQ(kTfLiteError, task.SetSynchronization(kTfLiteIoInput, "xx", sync));
  EXPECT_EQ(kTfLiteError, task.SetSynchronization(kTfLiteIoOutput, "aa", sync));

  EXPECT_EQ(kTfLiteNullBufferHandle,
            task.GetBufferHandle(kTfLiteIoInput, "xx"));
  EXPECT_EQ(kTfLiteNullBufferHandle,
            task.GetBufferHandle(kTfLiteIoOutput, "aa"));
  EXPECT_EQ(nullptr, task.GetSynchronization(kTfLiteIoInput, "xx"));
  EXPECT_EQ(nullptr, task.GetSynchronization(kTfLiteIoOutput, "aa"));

  TfLiteSynchronizationDelete(sync);
}

TEST(TfLiteExecutionTaskTest, DelegateData) {
  TfLiteAsyncKernel kernel{};
  int data = 0;
  tflite::async::ExecutionTask task;

  EXPECT_EQ(nullptr, task.GetDelegateExecutionData(&kernel));

  task.SetDelegateExecutionData(&kernel, &data);
  EXPECT_EQ(&data, task.GetDelegateExecutionData(&kernel));
}

}  // namespace tflite::async
