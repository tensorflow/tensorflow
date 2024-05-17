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
#include "tensorflow/lite/core/async/c/types.h"
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

  EXPECT_EQ(kTfLiteOk, task.SetBufferHandle(kTfLiteIoTypeInput, "x", 42));
  EXPECT_EQ(kTfLiteOk, task.SetBufferHandle(kTfLiteIoTypeInput, "y", 43));
  EXPECT_EQ(kTfLiteOk, task.SetBufferHandle(kTfLiteIoTypeOutput, "a", 44));
  EXPECT_EQ(kTfLiteOk, task.SetSynchronization(kTfLiteIoTypeInput, "x", sync));

  EXPECT_EQ(42, task.GetBufferHandle(kTfLiteIoTypeInput, "x"));
  EXPECT_EQ(43, task.GetBufferHandle(kTfLiteIoTypeInput, "y"));
  EXPECT_EQ(44, task.GetBufferHandle(kTfLiteIoTypeOutput, "a"));
  EXPECT_EQ(sync, task.GetSynchronization(kTfLiteIoTypeInput, "x"));
  EXPECT_EQ(nullptr, task.GetSynchronization(kTfLiteIoTypeInput, "y"));
  EXPECT_EQ(nullptr, task.GetSynchronization(kTfLiteIoTypeOutput, "a"));

  TfLiteSynchronizationDelete(sync);
}

TEST(TfLiteExecutionTaskTest, NameMapUninitialized) {
  tflite::async::ExecutionTask task;

  EXPECT_EQ(kTfLiteNullBufferHandle,
            task.GetBufferHandle(kTfLiteIoTypeInput, "foo"));
  EXPECT_EQ(kTfLiteNullBufferHandle,
            task.GetBufferHandle(kTfLiteIoTypeOutput, "foo"));
  EXPECT_EQ(nullptr, task.GetSynchronization(kTfLiteIoTypeOutput, "foo"));
  EXPECT_EQ(nullptr, task.GetSynchronization(kTfLiteIoTypeOutput, "foo"));
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

  EXPECT_EQ(kTfLiteError, task.SetBufferHandle(kTfLiteIoTypeInput, "xx", 42));
  EXPECT_EQ(kTfLiteError, task.SetBufferHandle(kTfLiteIoTypeOutput, "aa", 44));
  EXPECT_EQ(kTfLiteError,
            task.SetSynchronization(kTfLiteIoTypeInput, "xx", sync));
  EXPECT_EQ(kTfLiteError,
            task.SetSynchronization(kTfLiteIoTypeOutput, "aa", sync));

  EXPECT_EQ(kTfLiteNullBufferHandle,
            task.GetBufferHandle(kTfLiteIoTypeInput, "xx"));
  EXPECT_EQ(kTfLiteNullBufferHandle,
            task.GetBufferHandle(kTfLiteIoTypeOutput, "aa"));
  EXPECT_EQ(nullptr, task.GetSynchronization(kTfLiteIoTypeInput, "xx"));
  EXPECT_EQ(nullptr, task.GetSynchronization(kTfLiteIoTypeOutput, "aa"));

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
