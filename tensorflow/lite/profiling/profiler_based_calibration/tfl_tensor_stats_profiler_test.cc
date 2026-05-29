/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
you may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/profiling/profiler_based_calibration/tfl_tensor_stats_profiler.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/core/api/profiler.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/interpreter.h"

namespace odml {
namespace {

TEST(TensorStatsProfilerTest, IgnoreNotOperatorInvokeEvents) {
  tflite::Interpreter interpreter;
  TensorStatsProfiler profiler(interpreter, [](const TfLiteTensor*) {});

  // Any event type other than OPERATOR_INVOKE_EVENT should be ignored.
  EXPECT_EQ(
      profiler.BeginEvent("Invoke", tflite::Profiler::EventType::DEFAULT, 0, 0),
      0);
  EXPECT_EQ(
      profiler.BeginEvent(
          "Invoke", tflite::Profiler::EventType::DELEGATE_OPERATOR_INVOKE_EVENT,
          0, 0),
      0);
}

TEST(TensorStatsProfilerTest, HandleOperatorInvokeEvent) {
  // 1. Build a graph with 2 inputs, 1 intermediate, and 1 output tensor.
  tflite::Interpreter interpreter;
  interpreter.AddTensors(4);
  interpreter.SetInputs({0, 1});
  interpreter.SetOutputs({3});

  // 2. Define dummy operator lifecycle callbacks.
  TfLiteRegistration reg = {
      .init = [](TfLiteContext* context, const char* buffer,
                 size_t length) -> void* { return nullptr; },
      .free = [](TfLiteContext* context, void* buffer) {},
      .prepare = [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
        return kTfLiteOk;
      },
      .invoke = [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
        return kTfLiteOk;
      },
  };

  // 3. Attach Ops to sequence the tensor dependencies.
  // Op 0: {0, 1} -> 2.
  ASSERT_EQ(interpreter.AddNodeWithParameters(
                /*inputs=*/{0, 1}, /*outputs=*/{2}, /*init_data=*/nullptr,
                /*init_data_size=*/0, /*builtin_data=*/nullptr,
                /*registration=*/&reg),
            kTfLiteOk);
  // Op 1: 2 -> 3.
  ASSERT_EQ(interpreter.AddNodeWithParameters(
                /*inputs=*/{2}, /*outputs=*/{3}, /*init_data=*/nullptr,
                /*init_data_size=*/0, /*builtin_data=*/nullptr,
                /*registration=*/&reg),
            kTfLiteOk);

  // 4. Allocate memory and prepare profiler to accumulate tensors.
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
  std::vector<const TfLiteTensor*> captured_tensors;
  TensorStatsProfiler profiler(interpreter, [&](const TfLiteTensor* t) {
    captured_tensors.push_back(t);
  });

  // 5. Trigger sequential execution events across the operators.
  // Simulate execution of subgraph invoke boundary.
  const uint32_t h_subgraph =
      profiler.BeginEvent("Invoke", tflite::Profiler::EventType::DEFAULT, 0, 0);
  profiler.EndEvent(h_subgraph);

  // Simulate execution of Op 0 (node 0).
  const uint32_t h0 = profiler.BeginEvent(
      "OP1", tflite::Profiler::EventType::OPERATOR_INVOKE_EVENT, 0, 0);
  profiler.EndEvent(h0);

  // Simulate execution of Op 1 (node 1).
  const uint32_t h1 = profiler.BeginEvent(
      "OP2", tflite::Profiler::EventType::OPERATOR_INVOKE_EVENT, 1, 0);
  profiler.EndEvent(h1);

  // 6. Verify that all input, intermediate, and output tensors are captured
  // correctly.
  EXPECT_EQ(captured_tensors.size(), 4);
  EXPECT_EQ(captured_tensors[0], interpreter.tensor(0));
  EXPECT_EQ(captured_tensors[1], interpreter.tensor(1));
  EXPECT_EQ(captured_tensors[2], interpreter.tensor(2));
  EXPECT_EQ(captured_tensors[3], interpreter.tensor(3));
}

}  // namespace
}  // namespace odml
