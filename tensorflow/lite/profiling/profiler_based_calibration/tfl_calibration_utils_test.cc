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
#include <limits>
#include <memory>
#include <numeric>
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

int64_t SumCounts(const TensorHistogram& hist) {
  return std::accumulate(hist.counts.begin(), hist.counts.end(), int64_t{0});
}

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

TEST_F(TflCalibrationUtilsTest, FiltersNaNAndOutOfRangeFloats) {
  const int input_idx = AddTensor("input", kTfLiteFloat32, {1, 7});
  const int all_nan_idx = AddTensor("all_nan", kTfLiteFloat32, {1, 2});
  const int all_below_idx = AddTensor("all_below", kTfLiteFloat32, {1, 2});
  const int all_above_idx = AddTensor("all_above", kTfLiteFloat32, {1, 2});
  const int both_out_idx = AddTensor("both_out", kTfLiteFloat32, {1, 2});
  interpreter_->SetInputs(
      {input_idx, all_nan_idx, all_below_idx, all_above_idx, both_out_idx});
  AddNoOpNode(
      {input_idx, all_nan_idx, all_below_idx, all_above_idx, both_out_idx});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  constexpr float kInf = std::numeric_limits<float>::infinity();
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  // Include exact boundary values `-3e38f` and `3e38f` to verify the valid
  // range `(-3e38, 3e38)` is exclusive.
  const float in[] = {-3e38f, 3e38f, -kInf, kInf, kNaN, -2.5f, 3.5f};
  const float nans[] = {kNaN, kNaN};
  const float below[] = {-kInf, -3.1e38f};
  const float above[] = {kInf, 3.1e38f};
  const float both_out[] = {-3.1e38f, 3.1e38f};
  PopulateTensor(input_idx, in);
  PopulateTensor(all_nan_idx, nans);
  PopulateTensor(all_below_idx, below);
  PopulateTensor(all_above_idx, above);
  PopulateTensor(both_out_idx, both_out);

  auto status_or_stats = InvokeWithCalibration(interpreter_.get(), 0);
  ASSERT_TRUE(status_or_stats.ok());
  const auto& stats = *status_or_stats;
  ASSERT_EQ(stats.count("input"), 1);
  EXPECT_FLOAT_EQ(stats.at("input").min, -2.5f);
  EXPECT_FLOAT_EQ(stats.at("input").max, 3.5f);
  EXPECT_FALSE(stats.at("input").histogram.has_value());
  EXPECT_EQ(stats.count("all_nan"), 0);

  // When every non-NaN value falls outside `(kValidFloatMin, kValidFloatMax)`,
  // min/max falls back to the raw non-NaN min/max.
  ASSERT_EQ(stats.count("all_below"), 1);
  EXPECT_EQ(stats.at("all_below").min, -kInf);
  EXPECT_FLOAT_EQ(stats.at("all_below").max, -3.1e38f);

  ASSERT_EQ(stats.count("all_above"), 1);
  EXPECT_FLOAT_EQ(stats.at("all_above").min, 3.1e38f);
  EXPECT_EQ(stats.at("all_above").max, kInf);

  ASSERT_EQ(stats.count("both_out"), 1);
  EXPECT_FLOAT_EQ(stats.at("both_out").min, -3.1e38f);
  EXPECT_FLOAT_EQ(stats.at("both_out").max, 3.1e38f);
}

TEST_F(TflCalibrationUtilsTest, SkipsEmptyAndUnnamedTensors) {
  const int valid_idx = AddTensor("valid", kTfLiteFloat32, {1, 2});
  const int unallocated_empty_idx = AddTensor("empty", kTfLiteFloat32, {1, 0});
  const int zero_bytes_idx = AddTensor("zero_bytes_i32", kTfLiteInt32, {1, 0});
  const int partial_bytes_idx =
      AddTensor("partial_bytes_f32", kTfLiteFloat32, {1, 2});
  const int unnamed_idx = AddTensor(nullptr, kTfLiteFloat32, {1, 2});
  interpreter_->SetInputs({valid_idx, unallocated_empty_idx, zero_bytes_idx,
                           partial_bytes_idx, unnamed_idx});
  AddNoOpNode({valid_idx, unallocated_empty_idx, zero_bytes_idx,
               partial_bytes_idx, unnamed_idx});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  const float valid_data[] = {-1.5f, 2.5f};
  const float partial_data[] = {-7.0f, 7.0f};
  const float unnamed_data[] = {-4.0f, 4.0f};
  PopulateTensor(valid_idx, valid_data);
  PopulateTensor(partial_bytes_idx, partial_data);
  PopulateTensor(unnamed_idx, unnamed_data);
  // Point `zero_bytes_idx` at a non-null buffer while `bytes` remains 0, and
  // shrink `partial_bytes_idx` below one float, so tensors without a complete
  // element are skipped independently of `data.raw`.
  int32_t dummy_i32 = 42;
  interpreter_->tensor(zero_bytes_idx)->data.raw =
      reinterpret_cast<char*>(&dummy_i32);
  interpreter_->tensor(partial_bytes_idx)->bytes = sizeof(float) - 1;

  auto status_or_stats = InvokeWithCalibration(interpreter_.get(), 0);
  ASSERT_TRUE(status_or_stats.ok());
  const auto& stats = *status_or_stats;
  EXPECT_EQ(stats.size(), 1);
  ASSERT_EQ(stats.count("valid"), 1);
  EXPECT_FLOAT_EQ(stats.at("valid").min, -1.5f);
  EXPECT_FLOAT_EQ(stats.at("valid").max, 2.5f);
}

TEST_F(TflCalibrationUtilsTest, MergesRangesAcrossSameNamedTensors) {
  const int f0_idx = AddTensor("shared_f32", kTfLiteFloat32, {1, 2});
  const int f1_idx = AddTensor("shared_f32", kTfLiteFloat32, {1, 2});
  const int f2_idx = AddTensor("shared_f32", kTfLiteFloat32, {1, 2});
  const int i0_idx = AddTensor("shared_i32", kTfLiteInt32, {1, 2});
  const int i1_idx = AddTensor("shared_i32", kTfLiteInt32, {1, 2});
  interpreter_->SetInputs({f0_idx, f1_idx, f2_idx, i0_idx, i1_idx});
  AddNoOpNode({f0_idx, f1_idx, f2_idx, i0_idx, i1_idx});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  const float f0[] = {-2.0f, 4.0f};
  const float f1[] = {-5.0f, 1.0f};
  const float f2[] = {0.0f, 8.0f};
  const int32_t i0[] = {-3, 10};
  const int32_t i1[] = {-12, 4};
  PopulateTensor(f0_idx, f0);
  PopulateTensor(f1_idx, f1);
  PopulateTensor(f2_idx, f2);
  PopulateTensor(i0_idx, i0);
  PopulateTensor(i1_idx, i1);

  auto status_or_stats = InvokeWithCalibration(interpreter_.get(), 0);
  ASSERT_TRUE(status_or_stats.ok());
  const auto& stats = *status_or_stats;
  ASSERT_EQ(stats.count("shared_f32"), 1);
  EXPECT_FLOAT_EQ(stats.at("shared_f32").min, -5.0f);
  EXPECT_FLOAT_EQ(stats.at("shared_f32").max, 8.0f);
  ASSERT_EQ(stats.count("shared_i32"), 1);
  EXPECT_FLOAT_EQ(stats.at("shared_i32").min, -12.0f);
  EXPECT_FLOAT_EQ(stats.at("shared_i32").max, 10.0f);
}

TEST_F(TflCalibrationUtilsTest, CollectsPerInvocationFloatHistogramsOnly) {
  const int f32_idx = AddTensor("input_f32", kTfLiteFloat32, {1, 6});
  const int i32_idx = AddTensor("input_i32", kTfLiteInt32, {1, 4});
  const int out_of_range_idx =
      AddTensor("out_of_range_f32", kTfLiteFloat32, {1, 2});
  const int all_nan_idx = AddTensor("all_nan_f32", kTfLiteFloat32, {1, 2});
  interpreter_->SetInputs({f32_idx, i32_idx, out_of_range_idx, all_nan_idx});
  AddNoOpNode({f32_idx, i32_idx, out_of_range_idx, all_nan_idx});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  constexpr float kInf = std::numeric_limits<float>::infinity();
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  const float in1[] = {-kInf, -2.0f, -1.0f, 1.0f, 2.0f, kInf};
  const int32_t in_i32[] = {-10, 0, 10, 20};
  const float out_of_range[] = {-kInf, 3.1e38f};
  const float all_nan[] = {kNaN, kNaN};
  PopulateTensor(f32_idx, in1);
  PopulateTensor(i32_idx, in_i32);
  PopulateTensor(out_of_range_idx, out_of_range);
  PopulateTensor(all_nan_idx, all_nan);

  auto status_or_stats1 = InvokeWithCalibration(
      interpreter_.get(), 0, /*reporter=*/nullptr, /*collect_histogram=*/true);
  ASSERT_TRUE(status_or_stats1.ok());
  const auto& stats1 = *status_or_stats1;
  ASSERT_TRUE(stats1.at("input_f32").histogram.has_value());
  const TensorHistogram& hist1 = *stats1.at("input_f32").histogram;
  EXPECT_EQ(hist1.counts.size(), 2048);
  EXPECT_EQ(SumCounts(hist1), 4);
  // Valid range [-2, 2] padded by 10% on each side gives [-2.4, 2.4] and bin
  // width 4.8 / 2048 = 0.00234375.
  EXPECT_NEAR(hist1.lower_bound, -2.4, 1e-6);
  EXPECT_NEAR(hist1.bin_width, 0.00234375, 1e-9);
  EXPECT_FALSE(stats1.at("input_i32").histogram.has_value());
  EXPECT_FALSE(stats1.at("out_of_range_f32").histogram.has_value());
  EXPECT_EQ(stats1.count("all_nan_f32"), 0);

  // A second invocation returns a fresh histogram for its own values only.
  const float in2[] = {-1.5f, 0.0f, 0.5f, 1.5f, kInf, -kInf};
  PopulateTensor(f32_idx, in2);
  auto status_or_stats2 = InvokeWithCalibration(
      interpreter_.get(), 0, /*reporter=*/nullptr, /*collect_histogram=*/true);
  ASSERT_TRUE(status_or_stats2.ok());
  const auto& stats2 = *status_or_stats2;
  EXPECT_FLOAT_EQ(stats2.at("input_f32").min, -1.5f);
  EXPECT_FLOAT_EQ(stats2.at("input_f32").max, 1.5f);
  ASSERT_TRUE(stats2.at("input_f32").histogram.has_value());
  EXPECT_EQ(SumCounts(*stats2.at("input_f32").histogram), 4);
  EXPECT_NEAR(stats2.at("input_f32").histogram->lower_bound, -1.8, 1e-6);
}

TEST_F(TflCalibrationUtilsTest, ExpandsHistogramAcrossSameNamedTensors) {
  const int f0_idx = AddTensor("shared_f32", kTfLiteFloat32, {1, 2});
  const int f1_idx = AddTensor("shared_f32", kTfLiteFloat32, {1, 2});
  interpreter_->SetInputs({f0_idx, f1_idx});
  AddNoOpNode({f0_idx, f1_idx});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  const float f0[] = {-2.0f, 2.0f};
  const float f1[] = {-5.0f, 8.0f};
  PopulateTensor(f0_idx, f0);
  PopulateTensor(f1_idx, f1);

  auto status_or_stats = InvokeWithCalibration(
      interpreter_.get(), 0, /*reporter=*/nullptr, /*collect_histogram=*/true);
  ASSERT_TRUE(status_or_stats.ok());
  const auto& stats = *status_or_stats;
  ASSERT_EQ(stats.count("shared_f32"), 1);
  EXPECT_FLOAT_EQ(stats.at("shared_f32").min, -5.0f);
  EXPECT_FLOAT_EQ(stats.at("shared_f32").max, 8.0f);
  ASSERT_TRUE(stats.at("shared_f32").histogram.has_value());
  const TensorHistogram& hist = *stats.at("shared_f32").histogram;
  EXPECT_EQ(SumCounts(hist), 4);
  EXPECT_LE(hist.lower_bound, -5.0);
  EXPECT_GE(hist.lower_bound + hist.counts.size() * hist.bin_width, 8.0);
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
