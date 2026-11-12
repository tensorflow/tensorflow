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

#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/profiling/memory_info.h"
#include "tensorflow/lite/stateful_error_reporter.h"

namespace odml {
namespace {

using ::testing::HasSubstr;

// Stub error reporter returning a fixed message for error-propagation tests.
class FakeErrorReporter : public tflite::StatefulErrorReporter {
 public:
  int Report(const char*, va_list) override { return 0; }
  std::string message() override { return "custom error message"; }
};

class TflCalibrationUtilsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    interpreter_ = std::make_unique<tflite::Interpreter>();
  }

  // Adds a read-write tensor to `interpreter_` and returns its tensor index.
  int AddTensor(const char* name, TfLiteType type,
                const std::vector<int>& dims) {
    int index = 0;
    ABSL_CHECK_EQ(interpreter_->AddTensors(1, &index), kTfLiteOk);
    ABSL_CHECK_EQ(interpreter_->SetTensorParametersReadWrite(
                      index, type, name, dims, TfLiteQuantization()),
                  kTfLiteOk);
    return index;
  }

  // Copies `data` into the allocated buffer of tensor `index`.
  template <typename T, size_t N>
  void PopulateTensor(int index, const T (&data)[N]) {
    std::memcpy(interpreter_->tensor(index)->data.raw, data, sizeof(data));
  }

  // Appends a no-op node consuming `inputs` with no outputs.
  void AddNoOpNode(const std::vector<int>& inputs) {
    static TfLiteRegistration reg = {
        .invoke = [](TfLiteContext*, TfLiteNode*) { return kTfLiteOk; }};
    ABSL_CHECK_EQ(interpreter_->AddNodeWithParameters(
                      inputs, /*outputs=*/{}, /*init_data=*/nullptr,
                      /*init_data_size=*/0, /*builtin_data=*/nullptr, &reg),
                  kTfLiteOk);
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

TEST_F(TflCalibrationUtilsTest, SkipsConstantTensors) {
  const int input_idx = AddTensor("input", kTfLiteFloat32, {1, 2});
  int const_idx = 0;
  ASSERT_EQ(interpreter_->AddTensors(1, &const_idx), kTfLiteOk);
  const float weights[] = {-10.0f, 10.0f};
  ASSERT_EQ(interpreter_->SetTensorParametersReadOnly(
                const_idx, kTfLiteFloat32, "const_weights", {1, 2},
                TfLiteQuantization(), reinterpret_cast<const char*>(weights),
                sizeof(weights)),
            kTfLiteOk);
  // Include `const_idx` in subgraph inputs so `TensorStatsProfiler` visits it.
  interpreter_->SetInputs({input_idx, const_idx});
  AddNoOpNode({input_idx, const_idx});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  const float in[] = {-1.0f, 3.0f};
  PopulateTensor(input_idx, in);

  auto status_or_stats = InvokeWithCalibration(interpreter_.get(), 0);
  ASSERT_TRUE(status_or_stats.ok());
  EXPECT_EQ(status_or_stats->count("const_weights"), 0);
  ASSERT_EQ(status_or_stats->count("input"), 1);
  EXPECT_FLOAT_EQ(status_or_stats->at("input").min, -1.0f);
  EXPECT_FLOAT_EQ(status_or_stats->at("input").max, 3.0f);
}

TEST_F(TflCalibrationUtilsTest, ReportsInvokeError) {
  const int input_idx = AddTensor("input", kTfLiteFloat32, {1, 2});
  interpreter_->SetInputs({input_idx});
  static TfLiteRegistration fail_reg = {
      .invoke = [](TfLiteContext*, TfLiteNode*) { return kTfLiteError; }};
  ASSERT_EQ(interpreter_->AddNodeWithParameters(
                /*inputs=*/{input_idx}, /*outputs=*/{}, /*init_data=*/nullptr,
                /*init_data_size=*/0, /*builtin_data=*/nullptr, &fail_reg),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  const float in[] = {0.0f, 0.0f};
  PopulateTensor(input_idx, in);

  auto status_without_reporter = InvokeWithCalibration(interpreter_.get(), 0);
  ASSERT_FALSE(status_without_reporter.ok());
  EXPECT_EQ(status_without_reporter.status().code(),
            absl::StatusCode::kInternal);
  EXPECT_EQ(status_without_reporter.status().message(),
            "InvokeWithCalibration failed");

  FakeErrorReporter reporter;
  auto status_with_reporter =
      InvokeWithCalibration(interpreter_.get(), 0, &reporter);
  ASSERT_FALSE(status_with_reporter.ok());
  EXPECT_EQ(status_with_reporter.status().code(), absl::StatusCode::kInternal);
  EXPECT_THAT(std::string(status_with_reporter.status().message()),
              HasSubstr("InvokeWithCalibration failed: custom error message"));
}

TEST_F(TflCalibrationUtilsTest, GetMemoryUsage) {
  tflite::profiling::memory::MemoryUsage usage = GetMemoryUsage();
  // We can't check exact values, but they should be filled somehow.
  EXPECT_GT(usage.total_allocated_bytes, 0);
}

}  // namespace
}  // namespace odml
