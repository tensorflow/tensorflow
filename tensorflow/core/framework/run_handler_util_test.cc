/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/run_handler_util.h"

#include <vector>

#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
namespace tensorflow {
namespace {

void VerifySchedulingRanges(int num_active_requests, int num_threads,
                            int min_threads_per_request,
                            bool print_stats = false) {
  if (print_stats) {
    LOG(INFO) << "Test case# num_active_requests: " << num_active_requests
              << " num_threads: " << num_threads
              << " min_threads: " << min_threads_per_request;
  }
  std::vector<std::uint_fast32_t> start(num_active_requests);
  std::vector<std::uint_fast32_t> end(num_active_requests);

  ComputeInterOpSchedulingRanges(num_active_requests, num_threads,
                                 min_threads_per_request, &start, &end);
  string range_str = "";
  for (int i = 0; i < num_active_requests; ++i) {
    if (i > 0) range_str += " ";
    range_str += strings::StrCat("[", start[i], ", ", end[i], ")");

    ASSERT_GE(start[i], 0) << range_str;
    ASSERT_LE(end[i], num_threads) << range_str;
    if (i > 0) {
      // Due to linearly decreasing demand, #threads(i - 1) >= #threads(i)
      ASSERT_GE(end[i - 1] - start[i - 1], end[i] - start[i]) << range_str;
      // No missing threads.
      ASSERT_GE(end[i - 1], start[i]) << range_str;
    }
    // Each interval is at least of size 'min_threads_per_request'.
    ASSERT_GE((end[i] - start[i]), min_threads_per_request) << range_str;
    // Verify that assigned (quantized) threads is not overly estimated
    // from real demand, when the demand is high (>=
    // min_threads_per_request).
    float entry_weight = num_active_requests - i;
    float total_weight = 0.5f * num_active_requests * (num_active_requests + 1);
    float thread_demand = (entry_weight * num_threads) / total_weight;
    if (thread_demand > min_threads_per_request) {
      // We expect some over-estimation of threads due to quantization,
      // but we hope it's not more than 1 extra thread.
      ASSERT_NEAR(end[i] - start[i], thread_demand, 1.0)
          << "Ranges: " << range_str << " thread_demand: " << thread_demand
          << " i: " << i;
    }
  }
  ASSERT_EQ(end[num_active_requests - 1], num_threads);
  ASSERT_EQ(start[0], 0);
  if (print_stats) {
    LOG(INFO) << "Assigned ranges: " << range_str;
  }
}

TEST(RunHandlerUtilTest, TestComputeInterOpSchedulingRanges) {
  const int kMinThreadsPerRequestBound = 12;
  const int kMaxActiveRequests = 128;
  const int kMaxThreads = 128;

  for (int min_threads_per_request = 1;
       min_threads_per_request <= kMinThreadsPerRequestBound;
       ++min_threads_per_request) {
    for (int num_active_requests = 1; num_active_requests <= kMaxActiveRequests;
         ++num_active_requests) {
      for (int num_threads = min_threads_per_request;
           num_threads <= kMaxThreads; ++num_threads) {
        VerifySchedulingRanges(num_active_requests, num_threads,
                               min_threads_per_request);
      }
    }
  }
}

TEST(RunHandlerUtilTest, TestComputeInterOpStealingRanges) {
  int num_inter_op_threads = 9;
  std::vector<std::uint_fast32_t> start_vec(num_inter_op_threads);
  std::vector<std::uint_fast32_t> end_vec(num_inter_op_threads);

  // When there is 9 threads, there should be two thread groups.
  // The first group has threads [0, 6) with stealing range [0, 6)
  // The second group has threads [6, 9) with stealing range [3, 9)

  ComputeInterOpStealingRanges(num_inter_op_threads, 6, &start_vec, &end_vec);
  int stealing_ranges[2][2] = {{0, 6}, {3, 9}};

  for (int i = 0; i < num_inter_op_threads; ++i) {
    int expected_start = stealing_ranges[i / 6][0];
    int expected_end = stealing_ranges[i / 6][1];
    string message =
        strings::StrCat("Stealing range of thread ", i, " should be [",
                        expected_start, ", ", expected_end, "]");
    ASSERT_EQ(start_vec[i], expected_start) << message;
    ASSERT_EQ(end_vec[i], expected_end) << message;
  }
}

TEST(RunHandlerUtilTest, TestExponentialRequestDistribution) {
  int num_active_requests = 3;
  int num_threads = 10;
  std::vector<int> actual_distribution =
      ChooseRequestsWithExponentialDistribution(num_active_requests,
                                                num_threads);

  std::vector<int> expected_distribution{0, 0, 0, 0, 0, 1, 1, 1, 2, 2};
  ASSERT_EQ(actual_distribution, expected_distribution);
}

TEST(RunHandlerUtilTest, TestParamFromEnvWithDefault) {
  std::vector<double> result = ParamFromEnvWithDefault(
      "RUN_HANDLER_TEST_ENV", std::vector<double>{0, 0, 0});
  EXPECT_EQ(result.size(), 3);
  EXPECT_EQ(result[0], 0);
  EXPECT_EQ(result[1], 0);
  EXPECT_EQ(result[2], 0);

  std::vector<int> result2 = ParamFromEnvWithDefault("RUN_HANDLER_TEST_ENV",
                                                     std::vector<int>{0, 0, 0});
  EXPECT_EQ(result2.size(), 3);
  EXPECT_EQ(result2[0], 0);
  EXPECT_EQ(result2[1], 0);
  EXPECT_EQ(result2[2], 0);

  bool result3 =
      ParamFromEnvBoolWithDefault("RUN_HANDLER_TEST_ENV_BOOL", false);
  EXPECT_EQ(result3, false);

  // Set environment variable.
  EXPECT_EQ(setenv("RUN_HANDLER_TEST_ENV", "1,2,3", true), 0);
  result = ParamFromEnvWithDefault("RUN_HANDLER_TEST_ENV",
                                   std::vector<double>{0, 0, 0});
  EXPECT_EQ(result.size(), 3);
  EXPECT_EQ(result[0], 1);
  EXPECT_EQ(result[1], 2);
  EXPECT_EQ(result[2], 3);
  result2 = ParamFromEnvWithDefault("RUN_HANDLER_TEST_ENV",
                                    std::vector<int>{0, 0, 0});
  EXPECT_EQ(result.size(), 3);
  EXPECT_EQ(result2[0], 1);
  EXPECT_EQ(result2[1], 2);
  EXPECT_EQ(result2[2], 3);

  EXPECT_EQ(setenv("RUN_HANDLER_TEST_ENV_BOOL", "true", true), 0);
  result3 = ParamFromEnvBoolWithDefault("RUN_HANDLER_TEST_ENV_BOOL", false);
  EXPECT_EQ(result3, true);
}

}  // namespace
}  // namespace tensorflow
