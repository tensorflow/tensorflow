/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/platform/cloud/retrying_utils.h"
#include <fstream>
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(RetryingUtilsTest, RetryDelays) {
  std::vector<double> requested_delays;  // requested delays in seconds
  std::function<void(int64)> sleep = [&requested_delays](int64 delay) {
    requested_delays.emplace_back(delay / 1000000.0);
  };
  std::function<Status()> f = []() { return errors::Unavailable("Failed."); };

  const auto& status = RetryingUtils::CallWithRetries(f, 500000L, sleep);
  EXPECT_EQ(errors::Code::ABORTED, status.code());
  EXPECT_TRUE(StringPiece(status.error_message())
                  .contains("All 10 retry attempts "
                            "failed. The last failure: "
                            "Unavailable: Failed."))
      << status;

  EXPECT_EQ(10, requested_delays.size());
  EXPECT_NEAR(0.5, requested_delays[0], 1.0);
  EXPECT_NEAR(1.0, requested_delays[1], 1.0);
  EXPECT_NEAR(2.0, requested_delays[2], 1.0);
  EXPECT_NEAR(4.0, requested_delays[3], 1.0);
  EXPECT_NEAR(8.0, requested_delays[4], 1.0);
  EXPECT_NEAR(16.0, requested_delays[5], 1.0);

  // All subsequent delays are capped at 32 seconds (plus jitter).
  EXPECT_NEAR(32.0, requested_delays[6], 1.0);
  EXPECT_NEAR(32.0, requested_delays[7], 1.0);
  EXPECT_NEAR(32.0, requested_delays[8], 1.0);
  EXPECT_NEAR(32.0, requested_delays[9], 1.0);
}

TEST(RetryingUtilsTest, NotFoundIsNotRetried) {
  std::vector<Status> results(
      {errors::Unavailable("Failed."), errors::NotFound("Not found.")});
  std::function<Status()> f = [&results]() {
    auto result = results[0];
    results.erase(results.begin());
    return result;
  };
  EXPECT_EQ(errors::Code::NOT_FOUND,
            RetryingUtils::CallWithRetries(f, 0).code());
}

TEST(RetryingUtilsTest, EventualSuccess) {
  std::vector<Status> results({errors::Unavailable("Failed."),
                               errors::Unavailable("Failed again."),
                               Status::OK()});
  std::function<Status()> f = [&results]() {
    auto result = results[0];
    results.erase(results.begin());
    return result;
  };
  TF_EXPECT_OK(RetryingUtils::CallWithRetries(f, 0));
}

}  // namespace
}  // namespace tensorflow
