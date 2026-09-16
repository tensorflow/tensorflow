/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/profiling/profiler_based_calibration/tfl_calibration_utils.h"

#include <cstdint>
#include <cstring>
#include <memory>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/profiling/memory_info.h"

namespace odml {
namespace {

class TflCalibrationUtilsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    interpreter_ = std::make_unique<tflite::Interpreter>();
  }

  std::unique_ptr<tflite::Interpreter> interpreter_;
};

TEST_F(TflCalibrationUtilsTest, InvokeWithCalibrationFloat) {
  // 1. Add and configure 2 float32 tensors.
  interpreter_->AddTensors(2);
  interpreter_->SetInputs({0});
  interpreter_->SetOutputs({1});
  interpreter_->SetTensorParametersReadWrite(
      /*tensor_index=*/0, /*type=*/kTfLiteFloat32, /*name=*/"input",
      /*dims=*/{1, 4}, /*quantization=*/TfLiteQuantization());
  interpreter_->SetTensorParametersReadWrite(
      /*tensor_index=*/1, /*type=*/kTfLiteFloat32, /*name=*/"output",
      /*dims=*/{1, 4}, /*quantization=*/TfLiteQuantization());

  // 2. Prepare input data values.
  const float in[] = {1.0, 2.0, -1.0, 0.0};

  // 3. Register op with a dummy invoker.
  // Counter must be static because TfLiteRegistration requires a C-style
  // function pointer (stateless lambda), which cannot capture local variables.
  static int invoke_count;
  invoke_count = 0;  // Reset counter before each test.
  TfLiteRegistration reg = {
      .invoke = [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
        invoke_count++;
        return kTfLiteOk;
      }};
  ASSERT_EQ(interpreter_->AddNodeWithParameters(
                /*inputs=*/{0}, /*outputs=*/{1}, /*init_data=*/nullptr,
                /*init_data_size=*/0, /*builtin_data=*/nullptr,
                /*registration=*/&reg),
            kTfLiteOk);

  // 4. Allocate the input tensors and populate data before calibration.
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  memcpy(interpreter_->tensor(0)->data.raw, in, sizeof(in));

  // 5. Run calibration step and check min/max limits.
  auto status_or_stats = InvokeWithCalibration(interpreter_.get(), 0);
  ASSERT_TRUE(status_or_stats.ok());
  auto stats = *status_or_stats;
  EXPECT_EQ(invoke_count, 1);
  EXPECT_TRUE(stats.count("input"));
  EXPECT_EQ(stats["input"].min, -1.0f);
  EXPECT_EQ(stats["input"].max, 2.0f);
}

TEST_F(TflCalibrationUtilsTest, InvokeWithCalibrationInt32) {
  // 1. Add and configure 1 int32 tensor.
  interpreter_->AddTensors(1);
  interpreter_->SetInputs({0});
  interpreter_->SetTensorParametersReadWrite(
      /*tensor_index=*/0, /*type=*/kTfLiteInt32, /*name=*/"input_i32",
      /*dims=*/{1, 4}, /*quantization=*/TfLiteQuantization());

  // 2. Prepare int32 input data values.
  const int32_t in[] = {10, 20, -10, 0};

  // 3. Register op with a dummy invoker.
  // Counter must be static because TfLiteRegistration requires a C-style
  // function pointer (stateless lambda), which cannot capture local variables.
  static int invoke_count;
  invoke_count = 0;  // Reset counter before each test.
  TfLiteRegistration reg = {
      .invoke = [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
        invoke_count++;
        return kTfLiteOk;
      }};
  ASSERT_EQ(interpreter_->AddNodeWithParameters(
                /*inputs=*/{0}, /*outputs=*/{}, /*init_data=*/nullptr,
                /*init_data_size=*/0, /*builtin_data=*/nullptr,
                /*registration=*/&reg),
            kTfLiteOk);

  // 4. Allocate the int32 tensors and populate data before calibration.
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  memcpy(interpreter_->tensor(0)->data.raw, in, sizeof(in));

  // 5. Run calibration step and check min/max limits.
  auto status_or_stats = InvokeWithCalibration(interpreter_.get(), 0);
  ASSERT_TRUE(status_or_stats.ok());
  auto stats = *status_or_stats;
  EXPECT_EQ(invoke_count, 1);
  EXPECT_TRUE(stats.count("input_i32"));
  EXPECT_EQ(stats["input_i32"].min, -10.0f);
  EXPECT_EQ(stats["input_i32"].max, 20.0f);
}

TEST_F(TflCalibrationUtilsTest, GetMemoryUsage) {
  tflite::profiling::memory::MemoryUsage usage = GetMemoryUsage();
  // We can't check exact values, but they should be filled somehow.
  EXPECT_GT(usage.total_allocated_bytes, 0);
}

}  // namespace
}  // namespace odml
