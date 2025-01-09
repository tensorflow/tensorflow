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
#include "tensorflow/core/profiler/convert/inference_stats_sampler.h"

#include "absl/status/statusor.h"
#include "xla/tests/test_utils.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/profiler/protobuf/inference_stats.pb.h"

namespace tensorflow::profiler {
namespace {
using ::tensorflow::profiler::InferenceStats;
using xla::ParseTextProto;

TEST(ConvertInferenceStatsToInferenceProfileTest, TestSort) {
  // Generate an inference stats for test.
  // Requests and batches are ordered by latency (end_time_ps - start_time_ps),
  // this is guaranteed by inference_stats.cc
  InferenceStats inference_stats = ParseTextProto<InferenceStats>(
                                       R"pb(
                                         inference_stats_per_model {
                                           key: 1
                                           value {
                                             request_details {
                                               request_id: 0
                                               start_time_ps: 0
                                               end_time_ps: 10000
                                               batching_request_delay_ps: 2000
                                               batching_request_size: 200
                                             }
                                             request_details {
                                               request_id: 1
                                               start_time_ps: 0
                                               end_time_ps: 20000
                                               batching_request_delay_ps: 1000
                                               batching_request_size: 100
                                             }
                                             request_details {
                                               request_id: 2
                                               start_time_ps: 0
                                               end_time_ps: 30000
                                               batching_request_delay_ps: 3000
                                               batching_request_size: 300
                                             }
                                             batch_details {
                                               batch_id: 3
                                               start_time_ps: 0
                                               end_time_ps: 10000
                                               batch_delay_ps: 2000
                                               padding_amount: 20
                                               batch_size_after_padding: 200
                                             }
                                             batch_details {
                                               batch_id: 4
                                               start_time_ps: 0
                                               end_time_ps: 20000
                                               batch_delay_ps: 1000
                                               padding_amount: 10
                                               batch_size_after_padding: 100
                                             }
                                             batch_details {
                                               batch_id: 5
                                               start_time_ps: 0
                                               end_time_ps: 30000
                                               batch_delay_ps: 3000
                                               padding_amount: 30
                                               batch_size_after_padding: 300
                                             }
                                           }
                                         }
                                       )pb")
                                       .value();

  // Sort by latency, the result does not change.
  auto result_1 = SampleInferenceStats("Latency", "Latency", inference_stats);
  const auto& per_model_1 = result_1.at(1);
  EXPECT_EQ(per_model_1.sampled_requests.at(0).first->request_id(), 0);
  EXPECT_EQ(per_model_1.sampled_requests.at(1).first->request_id(), 1);
  EXPECT_EQ(per_model_1.sampled_requests.at(2).first->request_id(), 2);
  EXPECT_EQ(per_model_1.sampled_batches.at(0).first->batch_id(), 3);
  EXPECT_EQ(per_model_1.sampled_batches.at(1).first->batch_id(), 4);
  EXPECT_EQ(per_model_1.sampled_batches.at(2).first->batch_id(), 5);

  // Sort requests by Request size, sort batches by Padding amount.
  // Verifies the values are in increasing order.
  auto result_2 =
      SampleInferenceStats("Request size", "Padding amount", inference_stats);
  const auto& per_model_2 = result_2.at(1);
  EXPECT_EQ(per_model_2.sampled_requests.at(0).first->batching_request_size(),
            100);
  EXPECT_EQ(per_model_2.sampled_requests.at(1).first->batching_request_size(),
            200);
  EXPECT_EQ(per_model_2.sampled_requests.at(2).first->batching_request_size(),
            300);
  EXPECT_EQ(per_model_2.sampled_batches.at(0).first->padding_amount(), 10);
  EXPECT_EQ(per_model_2.sampled_batches.at(1).first->padding_amount(), 20);
  EXPECT_EQ(per_model_2.sampled_batches.at(2).first->padding_amount(), 30);

  // Sort requests by Request delay for batching, sort batches by
  // Batching delay. Verifies the values are in increasing order.
  auto result_3 = SampleInferenceStats("Request delay for batching",
                                       "Batching delay", inference_stats);
  const auto& per_model_3 = result_3.at(1);
  EXPECT_EQ(
      per_model_3.sampled_requests.at(0).first->batching_request_delay_ps(),
      1000);
  EXPECT_EQ(
      per_model_3.sampled_requests.at(1).first->batching_request_delay_ps(),
      2000);
  EXPECT_EQ(
      per_model_3.sampled_requests.at(2).first->batching_request_delay_ps(),
      3000);
  EXPECT_EQ(per_model_3.sampled_batches.at(0).first->batch_delay_ps(), 1000);
  EXPECT_EQ(per_model_3.sampled_batches.at(1).first->batch_delay_ps(), 2000);
  EXPECT_EQ(per_model_3.sampled_batches.at(2).first->batch_delay_ps(), 3000);
}

}  // namespace
}  // namespace tensorflow::profiler
