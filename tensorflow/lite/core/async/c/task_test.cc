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
#include "tensorflow/lite/core/async/c/task.h"

#include <gtest/gtest.h>
#include "tensorflow/lite/core/async/c/types.h"
#include "tensorflow/lite/core/async/interop/c/types.h"
#include "tensorflow/lite/core/async/task_internal.h"
#include "tensorflow/lite/core/c/common.h"

namespace {

class TfLiteExecutionTaskTest : public ::testing::Test {
 protected:
  void SetUp() override {
    input_names_["x"] = 1;
    input_names_["y"] = 2;
    output_names_["a"] = 3;
    task_.task->SetInputNameMap(&input_names_);
    task_.task->SetOutputNameMap(&output_names_);
  }

  TfLiteExecutionTask* task() { return &task_; }

 protected:
  tflite::async::ExecutionTask::TensorNameMapT input_names_;
  tflite::async::ExecutionTask::TensorNameMapT output_names_;
  TfLiteExecutionTask task_;
};

TEST_F(TfLiteExecutionTaskTest, BasicTest) {
  auto* sync = TfLiteSynchronizationCreate();

  EXPECT_EQ(kTfLiteOk,
            TfLiteExecutionTaskSetBuffer(task(), kTfLiteIoTypeInput, "x", 42));
  EXPECT_EQ(kTfLiteOk,
            TfLiteExecutionTaskSetBuffer(task(), kTfLiteIoTypeInput, "y", 43));
  EXPECT_EQ(kTfLiteOk,
            TfLiteExecutionTaskSetBuffer(task(), kTfLiteIoTypeOutput, "a", 44));
  EXPECT_EQ(kTfLiteOk,
            TfLiteExecutionTaskSetSync(task(), kTfLiteIoTypeInput, "x", sync));

  EXPECT_EQ(
      42, TfLiteExecutionTaskGetBufferByName(task(), kTfLiteIoTypeInput, "x"));
  EXPECT_EQ(
      43, TfLiteExecutionTaskGetBufferByName(task(), kTfLiteIoTypeInput, "y"));
  EXPECT_EQ(
      44, TfLiteExecutionTaskGetBufferByName(task(), kTfLiteIoTypeOutput, "a"));
  EXPECT_EQ(sync,
            TfLiteExecutionTaskGetSyncByName(task(), kTfLiteIoTypeInput, "x"));
  EXPECT_EQ(nullptr,
            TfLiteExecutionTaskGetSyncByName(task(), kTfLiteIoTypeInput, "y"));
  EXPECT_EQ(nullptr,
            TfLiteExecutionTaskGetSyncByName(task(), kTfLiteIoTypeOutput, "a"));

  TfLiteSynchronizationDelete(sync);
}

TEST_F(TfLiteExecutionTaskTest, BasicTestByTensorIndex) {
  auto* sync = TfLiteSynchronizationCreate();

  EXPECT_EQ(kTfLiteOk,
            TfLiteExecutionTaskSetBuffer(task(), kTfLiteIoTypeInput, "x", 42));
  EXPECT_EQ(kTfLiteOk,
            TfLiteExecutionTaskSetBuffer(task(), kTfLiteIoTypeInput, "y", 43));
  EXPECT_EQ(kTfLiteOk,
            TfLiteExecutionTaskSetBuffer(task(), kTfLiteIoTypeOutput, "a", 44));
  EXPECT_EQ(kTfLiteOk,
            TfLiteExecutionTaskSetSync(task(), kTfLiteIoTypeInput, "x", sync));

  EXPECT_EQ(42, TfLiteExecutionTaskGetBufferByIndex(task(), 1));
  EXPECT_EQ(43, TfLiteExecutionTaskGetBufferByIndex(task(), 2));
  EXPECT_EQ(44, TfLiteExecutionTaskGetBufferByIndex(task(), 3));
  EXPECT_EQ(sync, TfLiteExecutionTaskGetSyncByIndex(task(), 1));
  EXPECT_EQ(nullptr, TfLiteExecutionTaskGetSyncByIndex(task(), 2));
  EXPECT_EQ(nullptr, TfLiteExecutionTaskGetSyncByIndex(task(), 3));

  TfLiteSynchronizationDelete(sync);
}

TEST_F(TfLiteExecutionTaskTest, NullTest) {
  EXPECT_EQ(kTfLiteError,
            TfLiteExecutionTaskSetBuffer(nullptr, kTfLiteIoTypeInput, "x", 42));
  EXPECT_EQ(kTfLiteError, TfLiteExecutionTaskSetSync(
                              nullptr, kTfLiteIoTypeInput, "x", nullptr));
  EXPECT_EQ(kTfLiteNullBufferHandle, TfLiteExecutionTaskGetBufferByName(
                                         nullptr, kTfLiteIoTypeOutput, "a"));
  EXPECT_EQ(nullptr,
            TfLiteExecutionTaskGetSyncByName(nullptr, kTfLiteIoTypeInput, "x"));
  EXPECT_EQ(kTfLiteNullBufferHandle,
            TfLiteExecutionTaskGetBufferByIndex(nullptr, 3));
  EXPECT_EQ(nullptr, TfLiteExecutionTaskGetSyncByIndex(nullptr, 3));
  EXPECT_EQ(kTfLiteError, TfLiteExecutionTaskGetStatus(nullptr));
  TfLiteExecutionTaskSetStatus(nullptr, kTfLiteOk);
}

TEST_F(TfLiteExecutionTaskTest, StatusTest) {
  EXPECT_EQ(kTfLiteOk, TfLiteExecutionTaskGetStatus(task()));
  TfLiteExecutionTaskSetStatus(task(), kTfLiteError);
  EXPECT_EQ(kTfLiteError, TfLiteExecutionTaskGetStatus(task()));
}

}  // namespace
