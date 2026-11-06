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
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/profiling/memory_info.h"
#include "tensorflow/lite/stateful_error_reporter.h"

namespace odml {
namespace {

using ::testing::HasSubstr;
using ::testing::status::StatusIs;

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

  int AddTensor(const char* name, TfLiteType type,
                const std::vector<int>& dims) {
    int index = 0;
    ABSL_CHECK_EQ(interpreter_->AddTensors(1, &index), kTfLiteOk);
    ABSL_CHECK_EQ(interpreter_->SetTensorParametersReadWrite(
                      index, type, name, dims, TfLiteQuantization()),
                  kTfLiteOk);
    return index;
  }

  template <typename T, size_t N>
  void PopulateTensor(int index, const T (&data)[N]) {
    std::memcpy(interpreter_->tensor(index)->data.raw, data, sizeof(data));
  }

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
  AddTensor("input", kTfLiteFloat32, {1, 4});
  AddTensor("output", kTfLiteFloat32, {1, 4});
  interpreter_->SetInputs({0});
  interpreter_->SetOutputs({1});

  static int invoke_count = 0;
  invoke_count = 0;
  TfLiteRegistration reg = {
      .invoke = [](TfLiteContext*, TfLiteNode*) -> TfLiteStatus {
        ++invoke_count;
        return kTfLiteOk;
      }};
  ASSERT_EQ(interpreter_->AddNodeWithParameters(
                /*inputs=*/{0}, /*outputs=*/{1}, /*init_data=*/nullptr,
                /*init_data_size=*/0, /*builtin_data=*/nullptr,
                /*registration=*/&reg),
            kTfLiteOk);

  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  const float in[] = {1.0f, 2.0f, -1.0f, 0.0f};
  PopulateTensor(0, in);

  ASSERT_OK_AND_ASSIGN(const auto stats,
                       InvokeWithCalibration(interpreter_.get(), 0));
  EXPECT_EQ(invoke_count, 1);
  ASSERT_EQ(stats.count("input"), 1);
  EXPECT_FLOAT_EQ(stats.at("input").min, -1.0f);
  EXPECT_FLOAT_EQ(stats.at("input").max, 2.0f);
  EXPECT_FALSE(stats.at("input").histogram.has_value());
}

TEST_F(TflCalibrationUtilsTest, FiltersNaNAndOutOfRangeFloats) {
  AddTensor("input", kTfLiteFloat32, {1, 6});
  AddTensor("all_nan", kTfLiteFloat32, {1, 2});
  AddTensor("all_inf", kTfLiteFloat32, {1, 2});
  interpreter_->SetInputs({0, 1, 2});
  AddNoOpNode({0, 1, 2});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  constexpr float kInf = std::numeric_limits<float>::infinity();
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  const float in[] = {-kInf, kInf, kNaN, -2.5f, 3.5f, -3.1e38f};
  const float nans[] = {kNaN, kNaN};
  const float infs[] = {-kInf, -kInf};
  PopulateTensor(0, in);
  PopulateTensor(1, nans);
  PopulateTensor(2, infs);

  ASSERT_OK_AND_ASSIGN(
      const auto stats,
      InvokeWithCalibration(interpreter_.get(), 0, /*reporter=*/nullptr,
                            /*collect_histogram=*/true));
  ASSERT_EQ(stats.count("input"), 1);
  EXPECT_FLOAT_EQ(stats.at("input").min, -2.5f);
  EXPECT_FLOAT_EQ(stats.at("input").max, 3.5f);
  ASSERT_TRUE(stats.at("input").histogram.has_value());
  EXPECT_EQ(SumCounts(*stats.at("input").histogram), 2);
  EXPECT_EQ(stats.count("all_nan"), 0);
  // When every value is out of range, min/max falls back to the raw values
  // and no histogram is produced.
  ASSERT_EQ(stats.count("all_inf"), 1);
  EXPECT_EQ(stats.at("all_inf").min, -kInf);
  EXPECT_EQ(stats.at("all_inf").max, -kInf);
  EXPECT_FALSE(stats.at("all_inf").histogram.has_value());
}

TEST_F(TflCalibrationUtilsTest, InvokeWithCalibrationInt32AndMultiNodeMerge) {
  AddTensor("input_i32", kTfLiteInt32, {1, 4});
  AddTensor("input_f32", kTfLiteFloat32, {1, 2});
  interpreter_->SetInputs({0, 1});
  // Two nodes consume the same tensors, so stats are merged across callbacks.
  AddNoOpNode({0, 1});
  AddNoOpNode({0, 1});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  const int32_t in_i32[] = {10, 20, -10, 0};
  const float in_f32[] = {-3.0f, 5.0f};
  PopulateTensor(0, in_i32);
  PopulateTensor(1, in_f32);

  ASSERT_OK_AND_ASSIGN(const auto stats,
                       InvokeWithCalibration(interpreter_.get(), 0));
  ASSERT_EQ(stats.count("input_i32"), 1);
  EXPECT_FLOAT_EQ(stats.at("input_i32").min, -10.0f);
  EXPECT_FLOAT_EQ(stats.at("input_i32").max, 20.0f);
  ASSERT_EQ(stats.count("input_f32"), 1);
  EXPECT_FLOAT_EQ(stats.at("input_f32").min, -3.0f);
  EXPECT_FLOAT_EQ(stats.at("input_f32").max, 5.0f);
}

TEST_F(TflCalibrationUtilsTest, CollectsPerInvocationHistograms) {
  AddTensor("input_f32", kTfLiteFloat32, {1, 4});
  AddTensor("input_i32", kTfLiteInt32, {1, 4});
  interpreter_->SetInputs({0, 1});
  AddNoOpNode({0, 1});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  const float in1[] = {-2.0f, -1.0f, 1.0f, 2.0f};
  const int32_t in_i32[] = {-10, 0, 10, 20};
  PopulateTensor(0, in1);
  PopulateTensor(1, in_i32);
  ASSERT_OK_AND_ASSIGN(
      const auto stats1,
      InvokeWithCalibration(interpreter_.get(), 0, /*reporter=*/nullptr,
                            /*collect_histogram=*/true));
  ASSERT_TRUE(stats1.at("input_f32").histogram.has_value());
  const TensorHistogram& hist1 = *stats1.at("input_f32").histogram;
  EXPECT_EQ(hist1.counts.size(), 2048);
  EXPECT_EQ(SumCounts(hist1), 4);
  // The first observed range [-2, 2] is padded by 10% on each side.
  EXPECT_NEAR(hist1.lower_bound, -2.4, 1e-6);
  EXPECT_NEAR(hist1.bin_width, 4.8 / 2048, 1e-9);
  ASSERT_TRUE(stats1.at("input_i32").histogram.has_value());
  EXPECT_EQ(SumCounts(*stats1.at("input_i32").histogram), 4);

  // A second invocation returns a fresh histogram for its own values only.
  const float in2[] = {-1.5f, 0.0f, 0.5f, 1.5f};
  PopulateTensor(0, in2);
  ASSERT_OK_AND_ASSIGN(
      const auto stats2,
      InvokeWithCalibration(interpreter_.get(), 0, /*reporter=*/nullptr,
                            /*collect_histogram=*/true));
  EXPECT_FLOAT_EQ(stats2.at("input_f32").min, -1.5f);
  EXPECT_FLOAT_EQ(stats2.at("input_f32").max, 1.5f);
  EXPECT_EQ(SumCounts(*stats2.at("input_f32").histogram), 4);
  EXPECT_NEAR(stats2.at("input_f32").histogram->lower_bound, -1.8, 1e-6);
}

TEST_F(TflCalibrationUtilsTest, SkipsConstantTensorsAndReportsInvokeError) {
  int const_idx = 0;
  ASSERT_EQ(interpreter_->AddTensors(1, &const_idx), kTfLiteOk);
  const float weights[] = {-10.0f, 10.0f};
  ASSERT_EQ(interpreter_->SetTensorParametersReadOnly(
                const_idx, kTfLiteFloat32, "const_weights", {1, 2},
                TfLiteQuantization(), reinterpret_cast<const char*>(weights),
                sizeof(weights)),
            kTfLiteOk);
  AddNoOpNode({const_idx});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  ASSERT_OK_AND_ASSIGN(const auto stats,
                       InvokeWithCalibration(interpreter_.get(), 0));
  EXPECT_EQ(stats.count("const_weights"), 0);

  // Create a fresh interpreter with a failing node and verify error propagation
  // with and without a reporter.
  interpreter_ = std::make_unique<tflite::Interpreter>();
  AddTensor("input", kTfLiteFloat32, {1, 2});
  interpreter_->SetInputs({0});
  static TfLiteRegistration fail_reg = {
      .invoke = [](TfLiteContext*, TfLiteNode*) { return kTfLiteError; }};
  ASSERT_EQ(interpreter_->AddNodeWithParameters({0}, {}, nullptr, 0, nullptr,
                                                &fail_reg),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  EXPECT_THAT(InvokeWithCalibration(interpreter_.get(), 0),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("InvokeWithCalibration failed")));

  FakeErrorReporter reporter;
  EXPECT_THAT(
      InvokeWithCalibration(interpreter_.get(), 0, &reporter),
      StatusIs(absl::StatusCode::kInternal, HasSubstr("custom error message")));
}

TEST_F(TflCalibrationUtilsTest, GetMemoryUsage) {
  tflite::profiling::memory::MemoryUsage usage = GetMemoryUsage();
  EXPECT_GT(usage.total_allocated_bytes, 0);
}

}  // namespace
}  // namespace odml
