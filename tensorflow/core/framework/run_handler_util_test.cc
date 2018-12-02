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
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
namespace tensorflow {
namespace {

void VerifyFunction(int num_active_requests, int num_threads,
                    int min_threads_per_request, bool print_stats = false) {
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
        VerifyFunction(num_active_requests, num_threads,
                       min_threads_per_request);
      }
    }
  }
}

}  // namespace
}  // namespace tensorflow
