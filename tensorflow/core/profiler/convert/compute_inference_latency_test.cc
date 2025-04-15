/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/profiler/convert/compute_inference_latency.h"

#include <gtest/gtest.h>
#include "tensorflow/core/profiler/protobuf/inference_stats.pb.h"

namespace tensorflow::profiler {
namespace {

constexpr double kMaxError = 0.0001;

TEST(ComputeInferenceLatencyResult, InferenceLatencyTest) {
  InferenceStats inference_stats;
  auto& model = (*inference_stats.mutable_inference_stats_per_model())[0];

  // Generates requests for testing.
  for (int i = 0; i < 100; i++) {
    RequestDetail request_detail;
    request_detail.set_start_time_ps(0);
    request_detail.set_end_time_ps(i * 10000);
    request_detail.set_device_time_ps(i * 1000);
    request_detail.set_write_to_device_time_ps(i * 1000);
    model.add_request_details()->Swap(&request_detail);
  }

  auto result = ComputeInferenceLatencyResult(inference_stats);

  // 5 percentiles and 1 average, so 6 results in total.
  ASSERT_EQ(result.latency_breakdowns_size(), 6);

  // Verify 50 percentile result.
  EXPECT_NEAR(result.latency_breakdowns(0).total_latency_us(), 0.5, kMaxError);
  EXPECT_NEAR(result.latency_breakdowns(0).host_latency_us(), 0.4, kMaxError);
  EXPECT_NEAR(result.latency_breakdowns(0).device_latency_us(), 0.05,
              kMaxError);
  EXPECT_NEAR(result.latency_breakdowns(0).communication_latency_us(), 0.05,
              kMaxError);

  // Verify 99.9 percentile result.
  EXPECT_NEAR(result.latency_breakdowns(4).total_latency_us(), 0.99, kMaxError);
  EXPECT_NEAR(result.latency_breakdowns(4).host_latency_us(), 0.792, kMaxError);
  EXPECT_NEAR(result.latency_breakdowns(4).device_latency_us(), 0.099,
              kMaxError);
  EXPECT_NEAR(result.latency_breakdowns(4).communication_latency_us(), 0.099,
              kMaxError);

  // Verify average result.
  EXPECT_NEAR(result.latency_breakdowns(5).total_latency_us(), 0.495,
              kMaxError);
  EXPECT_NEAR(result.latency_breakdowns(5).host_latency_us(), 0.396, kMaxError);
  EXPECT_NEAR(result.latency_breakdowns(5).device_latency_us(), 0.0495,
              kMaxError);
  EXPECT_NEAR(result.latency_breakdowns(5).communication_latency_us(), 0.0495,
              kMaxError);
}

}  // namespace
}  // namespace tensorflow::profiler
