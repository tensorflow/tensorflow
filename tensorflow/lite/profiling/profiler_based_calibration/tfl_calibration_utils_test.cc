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
#include <limits>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "Eigen/Core"  // from @eigen_archive
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/profiling/memory_info.h"

namespace odml {
namespace {

int64_t SumCounts(const TensorHistogram& hist) {
  int64_t total = 0;
  for (int64_t c : hist.counts) {
    total += c;
  }
  return total;
}

class TflCalibrationUtilsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    interpreter_ = std::make_unique<tflite::Interpreter>();
  }

  // Adds a single no-op node that consumes `inputs`.
  void AddNoOpNode(const std::vector<int>& inputs) {
    static TfLiteRegistration reg = {
        .invoke = [](TfLiteContext*, TfLiteNode*) { return kTfLiteOk; }};
    ASSERT_EQ(interpreter_->AddNodeWithParameters(
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
  if (!status_or_stats.ok()) return;
  auto stats = *status_or_stats;
  EXPECT_EQ(invoke_count, 1);
  EXPECT_TRUE(stats.count("input"));
  EXPECT_EQ(stats["input"].min, -1.0f);
  EXPECT_EQ(stats["input"].max, 2.0f);
}

TEST_F(TflCalibrationUtilsTest, InvokeWithCalibrationFiltersInfAndNaN) {
  interpreter_->AddTensors(1);
  interpreter_->SetInputs({0});
  interpreter_->SetTensorParametersReadWrite(
      /*tensor_index=*/0, /*type=*/kTfLiteFloat32, /*name=*/"input",
      /*dims=*/{1, 6}, /*quantization=*/TfLiteQuantization());

  constexpr float kInf = std::numeric_limits<float>::infinity();
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  const float in[] = {-kInf, kInf, kNaN, -2.5f, 3.5f, -3.1e38f};

  AddNoOpNode({0});

  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  memcpy(interpreter_->tensor(0)->data.raw, in, sizeof(in));

  auto status_or_stats = InvokeWithCalibration(interpreter_.get(), 0);
  ASSERT_TRUE(status_or_stats.ok());
  if (!status_or_stats.ok()) return;
  auto stats = *status_or_stats;
  ASSERT_TRUE(stats.count("input"));
  EXPECT_EQ(stats["input"].min, -2.5f);
  EXPECT_EQ(stats["input"].max, 3.5f);
}

TEST_F(TflCalibrationUtilsTest, InvokeWithCalibrationAllInfFallback) {
  interpreter_->AddTensors(1);
  interpreter_->SetInputs({0});
  interpreter_->SetTensorParametersReadWrite(
      /*tensor_index=*/0, /*type=*/kTfLiteFloat32, /*name=*/"input",
      /*dims=*/{1, 2}, /*quantization=*/TfLiteQuantization());

  constexpr float kInf = std::numeric_limits<float>::infinity();
  const float in[] = {-kInf, -kInf};

  AddNoOpNode({0});

  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  memcpy(interpreter_->tensor(0)->data.raw, in, sizeof(in));

  auto status_or_stats = InvokeWithCalibration(interpreter_.get(), 0);
  ASSERT_TRUE(status_or_stats.ok());
  if (!status_or_stats.ok()) return;
  auto stats = *status_or_stats;
  ASSERT_TRUE(stats.count("input"));
  EXPECT_EQ(stats["input"].min, -kInf);
  EXPECT_EQ(stats["input"].max, -kInf);
}

TEST_F(TflCalibrationUtilsTest, InvokeWithCalibrationFloat16AndBFloat16) {
  interpreter_->AddTensors(2);
  interpreter_->SetInputs({0, 1});
  interpreter_->SetTensorParametersReadWrite(
      /*tensor_index=*/0, /*type=*/kTfLiteFloat16, /*name=*/"input_f16",
      /*dims=*/{1, 4}, /*quantization=*/TfLiteQuantization());
  interpreter_->SetTensorParametersReadWrite(
      /*tensor_index=*/1, /*type=*/kTfLiteBFloat16, /*name=*/"input_bf16",
      /*dims=*/{1, 4}, /*quantization=*/TfLiteQuantization());

  const Eigen::half in_f16[] = {
      Eigen::half(-1.5f), Eigen::half(0.0f), Eigen::half(4.0f),
      Eigen::half(-std::numeric_limits<float>::infinity())};
  const Eigen::bfloat16 in_bf16[] = {
      Eigen::bfloat16(-2.0f), Eigen::bfloat16(1.0f), Eigen::bfloat16(8.0f),
      Eigen::bfloat16(-std::numeric_limits<float>::infinity())};

  AddNoOpNode({0, 1});

  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  memcpy(interpreter_->tensor(0)->data.raw, in_f16, sizeof(in_f16));
  memcpy(interpreter_->tensor(1)->data.raw, in_bf16, sizeof(in_bf16));

  auto status_or_stats = InvokeWithCalibration(interpreter_.get(), 0);
  ASSERT_TRUE(status_or_stats.ok());
  if (!status_or_stats.ok()) return;
  auto stats = *status_or_stats;
  ASSERT_TRUE(stats.count("input_f16"));
  EXPECT_EQ(stats["input_f16"].min, -1.5f);
  EXPECT_EQ(stats["input_f16"].max, 4.0f);
  ASSERT_TRUE(stats.count("input_bf16"));
  EXPECT_EQ(stats["input_bf16"].min, -2.0f);
  EXPECT_EQ(stats["input_bf16"].max, 8.0f);
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
  if (!status_or_stats.ok()) return;
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

TEST_F(TflCalibrationUtilsTest,
       CalibrationSessionAccumulatesHistogramsAcrossInvokes) {
  interpreter_->AddTensors(1);
  interpreter_->SetInputs({0});
  interpreter_->SetTensorParametersReadWrite(
      /*tensor_index=*/0, /*type=*/kTfLiteFloat32, /*name=*/"input",
      /*dims=*/{1, 4}, /*quantization=*/TfLiteQuantization());

  AddNoOpNode({0});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  CalibrationSession session(CalibrationOptions{
      .collect_histogram = true,
      .num_bins = 16,
  });

  const float in1[] = {-2.0f, -1.0f, 1.0f, 2.0f};
  memcpy(interpreter_->tensor(0)->data.raw, in1, sizeof(in1));
  auto stats1 = session.Invoke(interpreter_.get(), 0);
  ASSERT_TRUE(stats1.ok());
  if (!stats1.ok()) return;
  EXPECT_EQ(stats1->at("input").min, -2.0f);
  EXPECT_EQ(stats1->at("input").max, 2.0f);

  // Second batch lies strictly inside the first batch's range.
  const float in2[] = {-1.5f, 0.0f, 0.5f, 1.5f};
  memcpy(interpreter_->tensor(0)->data.raw, in2, sizeof(in2));
  auto stats2 = session.Invoke(interpreter_.get(), 0);
  ASSERT_TRUE(stats2.ok());
  if (!stats2.ok()) return;
  // Min/max is per-invocation, not accumulated.
  EXPECT_EQ(stats2->at("input").min, -1.5f);
  EXPECT_EQ(stats2->at("input").max, 1.5f);

  // Histograms accumulate across invocations.
  auto hists = session.GetHistograms();
  ASSERT_EQ(hists.count("input"), 1);
  const auto& hist = hists.at("input");
  EXPECT_FLOAT_EQ(hist.min, -2.0f);
  EXPECT_FLOAT_EQ(hist.max, 2.0f);
  EXPECT_EQ(hist.counts.size(), 16);
  EXPECT_EQ(SumCounts(hist), 8);

  session.Reset();
  EXPECT_TRUE(session.GetHistograms().empty());
}

TEST_F(TflCalibrationUtilsTest,
       CalibrationSessionRespectsHistogramTensorNamesFilter) {
  interpreter_->AddTensors(2);
  interpreter_->SetInputs({0, 1});
  interpreter_->SetTensorParametersReadWrite(
      /*tensor_index=*/0, /*type=*/kTfLiteFloat32, /*name=*/"input_a",
      /*dims=*/{1, 2}, /*quantization=*/TfLiteQuantization());
  interpreter_->SetTensorParametersReadWrite(
      /*tensor_index=*/1, /*type=*/kTfLiteFloat32, /*name=*/"input_b",
      /*dims=*/{1, 2}, /*quantization=*/TfLiteQuantization());

  AddNoOpNode({0, 1});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  const float in_a[] = {-1.0f, 1.0f};
  const float in_b[] = {-2.0f, 2.0f};
  memcpy(interpreter_->tensor(0)->data.raw, in_a, sizeof(in_a));
  memcpy(interpreter_->tensor(1)->data.raw, in_b, sizeof(in_b));

  CalibrationSession session(CalibrationOptions{
      .collect_histogram = true,
      .num_bins = 16,
      .histogram_tensor_names = {{"input_a"}},
  });

  auto stats = session.Invoke(interpreter_.get(), 0);
  ASSERT_TRUE(stats.ok());
  if (!stats.ok()) return;
  // Both tensors still get per-invoke min/max.
  EXPECT_EQ(stats->count("input_a"), 1);
  EXPECT_EQ(stats->count("input_b"), 1);

  // Only input_a gets a histogram.
  auto hists = session.GetHistograms();
  EXPECT_EQ(hists.count("input_a"), 1);
  EXPECT_EQ(hists.count("input_b"), 0);
}

TEST_F(TflCalibrationUtilsTest,
       CalibrationSessionWithoutCollectHistogramReturnsNoHistograms) {
  interpreter_->AddTensors(1);
  interpreter_->SetInputs({0});
  interpreter_->SetTensorParametersReadWrite(
      /*tensor_index=*/0, /*type=*/kTfLiteFloat32, /*name=*/"input",
      /*dims=*/{1, 2}, /*quantization=*/TfLiteQuantization());
  AddNoOpNode({0});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  const float in[] = {-1.0f, 1.0f};
  memcpy(interpreter_->tensor(0)->data.raw, in, sizeof(in));

  CalibrationSession session;
  auto stats = session.Invoke(interpreter_.get(), 0);
  ASSERT_TRUE(stats.ok());
  if (!stats.ok()) return;
  EXPECT_EQ(stats->count("input"), 1);
  EXPECT_TRUE(session.GetHistograms().empty());
}

TEST_F(TflCalibrationUtilsTest, CalibrationSessionCustomValidRange) {
  interpreter_->AddTensors(1);
  interpreter_->SetInputs({0});
  interpreter_->SetTensorParametersReadWrite(
      /*tensor_index=*/0, /*type=*/kTfLiteFloat32, /*name=*/"input",
      /*dims=*/{1, 6}, /*quantization=*/TfLiteQuantization());
  AddNoOpNode({0});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  // Bounds are exclusive: -1.0f and 1.0f are filtered out.
  const float in[] = {-5.0f, -1.0f, -0.5f, 0.5f, 1.0f, 5.0f};
  memcpy(interpreter_->tensor(0)->data.raw, in, sizeof(in));

  CalibrationSession session(CalibrationOptions{
      .valid_min = -1.0f,
      .valid_max = 1.0f,
      .collect_histogram = true,
      .num_bins = 16,
  });
  auto stats = session.Invoke(interpreter_.get(), 0);
  ASSERT_TRUE(stats.ok());
  if (!stats.ok()) return;
  EXPECT_EQ(stats->at("input").min, -0.5f);
  EXPECT_EQ(stats->at("input").max, 0.5f);

  auto hists = session.GetHistograms();
  ASSERT_EQ(hists.count("input"), 1);
  EXPECT_FLOAT_EQ(hists.at("input").min, -0.5f);
  EXPECT_FLOAT_EQ(hists.at("input").max, 0.5f);
  EXPECT_EQ(SumCounts(hists.at("input")), 2);
}

TEST_F(TflCalibrationUtilsTest,
       CalibrationSessionCollectsHistogramsForF16BF16AndInt32) {
  interpreter_->AddTensors(3);
  interpreter_->SetInputs({0, 1, 2});
  interpreter_->SetTensorParametersReadWrite(
      /*tensor_index=*/0, /*type=*/kTfLiteFloat16, /*name=*/"input_f16",
      /*dims=*/{1, 4}, /*quantization=*/TfLiteQuantization());
  interpreter_->SetTensorParametersReadWrite(
      /*tensor_index=*/1, /*type=*/kTfLiteBFloat16, /*name=*/"input_bf16",
      /*dims=*/{1, 4}, /*quantization=*/TfLiteQuantization());
  interpreter_->SetTensorParametersReadWrite(
      /*tensor_index=*/2, /*type=*/kTfLiteInt32, /*name=*/"input_i32",
      /*dims=*/{1, 4}, /*quantization=*/TfLiteQuantization());
  AddNoOpNode({0, 1, 2});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  constexpr float kInf = std::numeric_limits<float>::infinity();
  const Eigen::half in_f16[] = {Eigen::half(-1.5f), Eigen::half(0.0f),
                                Eigen::half(4.0f), Eigen::half(kInf)};
  const Eigen::bfloat16 in_bf16[] = {
      Eigen::bfloat16(-2.0f), Eigen::bfloat16(1.0f), Eigen::bfloat16(8.0f),
      Eigen::bfloat16(-kInf)};
  const int32_t in_i32[] = {-10, 0, 10, 20};
  memcpy(interpreter_->tensor(0)->data.raw, in_f16, sizeof(in_f16));
  memcpy(interpreter_->tensor(1)->data.raw, in_bf16, sizeof(in_bf16));
  memcpy(interpreter_->tensor(2)->data.raw, in_i32, sizeof(in_i32));

  CalibrationSession session(CalibrationOptions{
      .collect_histogram = true,
      .num_bins = 16,
  });
  ASSERT_TRUE(session.Invoke(interpreter_.get(), 0).ok());

  auto hists = session.GetHistograms();
  ASSERT_EQ(hists.size(), 3);
  // Infinities are excluded from float histograms.
  EXPECT_FLOAT_EQ(hists.at("input_f16").min, -1.5f);
  EXPECT_FLOAT_EQ(hists.at("input_f16").max, 4.0f);
  EXPECT_EQ(SumCounts(hists.at("input_f16")), 3);
  EXPECT_FLOAT_EQ(hists.at("input_bf16").min, -2.0f);
  EXPECT_FLOAT_EQ(hists.at("input_bf16").max, 8.0f);
  EXPECT_EQ(SumCounts(hists.at("input_bf16")), 3);
  EXPECT_FLOAT_EQ(hists.at("input_i32").min, -10.0f);
  EXPECT_FLOAT_EQ(hists.at("input_i32").max, 20.0f);
  EXPECT_EQ(SumCounts(hists.at("input_i32")), 4);
}

TEST_F(TflCalibrationUtilsTest,
       CalibrationSessionMasksOutZeroMaskTokenPositions) {
  interpreter_->AddTensors(2);
  interpreter_->SetInputs({0, 1});
  interpreter_->SetTensorParametersReadWrite(
      /*tensor_index=*/0, /*type=*/kTfLiteFloat32,
      /*name=*/"encoder_1x4_embeddings:0",
      /*dims=*/{1, 4, 2}, /*quantization=*/TfLiteQuantization());
  interpreter_->SetTensorParametersReadWrite(
      /*tensor_index=*/1, /*type=*/kTfLiteFloat32,
      /*name=*/"encoder_1x4_mask:0",
      /*dims=*/{1, 4}, /*quantization=*/TfLiteQuantization());
  AddNoOpNode({0, 1});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  // Only the first 2 token positions (4 elements) are active (mask == 1);
  // the last 2 token positions (mask == 0) contain large padding outliers
  // (-100.0f, 100.0f) that should be excluded from stats and histograms.
  const float emb[] = {
      -1.0f, 0.5f, -0.5f, 2.0f, -100.0f, 100.0f, -100.0f, 100.0f,
  };
  const float mask[] = {1.0f, 1.0f, 0.0f, 0.0f};
  memcpy(interpreter_->tensor(0)->data.raw, emb, sizeof(emb));
  memcpy(interpreter_->tensor(1)->data.raw, mask, sizeof(mask));

  CalibrationSession session(CalibrationOptions{
      .collect_histogram = true,
      .num_bins = 16,
  });
  auto stats = session.Invoke(interpreter_.get(), 0);
  ASSERT_TRUE(stats.ok());
  if (!stats.ok()) return;
  EXPECT_FLOAT_EQ(stats->at("encoder_1x4_embeddings:0").min, -1.0f);
  EXPECT_FLOAT_EQ(stats->at("encoder_1x4_embeddings:0").max, 2.0f);

  auto hists = session.GetHistograms();
  ASSERT_EQ(hists.count("encoder_1x4_embeddings:0"), 1);
  EXPECT_FLOAT_EQ(hists.at("encoder_1x4_embeddings:0").min, -1.0f);
  EXPECT_FLOAT_EQ(hists.at("encoder_1x4_embeddings:0").max, 2.0f);
  EXPECT_EQ(SumCounts(hists.at("encoder_1x4_embeddings:0")), 4);
}

}  // namespace
}  // namespace odml
