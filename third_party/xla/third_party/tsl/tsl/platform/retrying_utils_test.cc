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
#include "tsl/platform/retrying_utils.h"

#include <cmath>
#include <fstream>

#include "absl/time/time.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/str_util.h"
#include "tsl/platform/test.h"

namespace tsl {
namespace {

TEST(RetryingUtilsTest, CallWithRetries_RetryDelays) {
  std::vector<double> requested_delays;  // requested delays in seconds
  std::function<void(int64_t)> sleep = [&requested_delays](int64_t delay) {
    requested_delays.emplace_back(delay / 1000000.0);
  };
  std::function<absl::Status()> f = []() {
    return errors::Unavailable("Failed.");
  };

  const auto& status = RetryingUtils::CallWithRetries(
      f, sleep, RetryConfig(500000 /* init_delay_time_us */));
  EXPECT_TRUE(errors::IsAborted(status));
  EXPECT_TRUE(absl::StrContains(
      status.message(),
      "All 10 retry attempts failed. The last failure: Failed."))
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

TEST(RetryingUtilsTest, CallWithRetries_NotFoundIsNotRetried) {
  std::vector<absl::Status> results(
      {errors::Unavailable("Failed."), errors::NotFound("Not found.")});
  std::function<absl::Status()> f = [&results]() {
    auto result = results[0];
    results.erase(results.begin());
    return result;
  };
  EXPECT_TRUE(errors::IsNotFound(RetryingUtils::CallWithRetries(
      f, RetryConfig(0 /* init_delay_time_us */))));
}

TEST(RetryingUtilsTest, CallWithRetries_ImmediateSuccess) {
  std::vector<absl::Status> results({absl::OkStatus()});
  std::function<void(int64_t)> sleep = [](int64_t delay) {
    ADD_FAILURE() << "Unexpected call to sleep.";
  };
  std::function<absl::Status()> f = [&results]() {
    auto result = results[0];
    results.erase(results.begin());
    return result;
  };
  TF_EXPECT_OK(RetryingUtils::CallWithRetries(
      f, sleep, RetryConfig(1L /* init_delay_time_us */)));
}

TEST(RetryingUtilsTest, CallWithRetries_EventualSuccess) {
  std::vector<absl::Status> results({errors::Unavailable("Failed."),
                                     errors::Unavailable("Failed again."),
                                     absl::OkStatus()});
  std::function<absl::Status()> f = [&results]() {
    auto result = results[0];
    results.erase(results.begin());
    return result;
  };
  TF_EXPECT_OK(RetryingUtils::CallWithRetries(
      f, RetryConfig(0 /* init_delay_time_us */)));
}

TEST(RetryingUtilsTest, DeleteWithRetries_ImmediateSuccess) {
  std::vector<absl::Status> delete_results({absl::OkStatus()});
  const auto delete_func = [&delete_results]() {
    auto result = delete_results[0];
    delete_results.erase(delete_results.begin());
    return result;
  };
  TF_EXPECT_OK(RetryingUtils::DeleteWithRetries(
      delete_func, RetryConfig(0 /* init_delay_time_us */)));
}

TEST(RetryingUtilsTest, DeleteWithRetries_EventualSuccess) {
  std::vector<absl::Status> delete_results(
      {errors::Unavailable(""), absl::OkStatus()});
  const auto delete_func = [&delete_results]() {
    auto result = delete_results[0];
    delete_results.erase(delete_results.begin());
    return result;
  };
  TF_EXPECT_OK(RetryingUtils::DeleteWithRetries(
      delete_func, RetryConfig(0 /* init_delay_time_us */)));
}

TEST(RetryingUtilsTest, DeleteWithRetries_PermissionDeniedNotRetried) {
  std::vector<absl::Status> delete_results(
      {errors::Unavailable(""), errors::PermissionDenied("")});
  const auto delete_func = [&delete_results]() {
    auto result = delete_results[0];
    delete_results.erase(delete_results.begin());
    return result;
  };
  EXPECT_TRUE(errors::IsPermissionDenied(RetryingUtils::DeleteWithRetries(
      delete_func, RetryConfig(0 /* init_delay_time_us */))));
}

TEST(RetryingUtilsTest, DeleteWithRetries_SuccessThroughFileNotFound) {
  std::vector<absl::Status> delete_results(
      {errors::Unavailable(""), errors::NotFound("")});
  const auto delete_func = [&delete_results]() {
    auto result = delete_results[0];
    delete_results.erase(delete_results.begin());
    return result;
  };
  TF_EXPECT_OK(RetryingUtils::DeleteWithRetries(
      delete_func, RetryConfig(0 /* init_delay_time_us */)));
}

TEST(RetryingUtilsTest, DeleteWithRetries_FirstNotFoundReturnedAsIs) {
  std::vector<absl::Status> delete_results({errors::NotFound("")});
  const auto delete_func = [&delete_results]() {
    auto result = delete_results[0];
    delete_results.erase(delete_results.begin());
    return result;
  };
  EXPECT_EQ(error::NOT_FOUND,
            RetryingUtils::DeleteWithRetries(
                delete_func, RetryConfig(0 /* init_delay_time_us */))
                .code());
}

TEST(RetryingUtilsTest, ComputeRetryBackoff) {
  for (int i = 0; i < 30; ++i) {
    EXPECT_LE(0.4 * absl::Milliseconds(1) +
                  0.6 * absl::Milliseconds(1) * std::pow(1.3, i),
              ComputeRetryBackoff(/*current_retry_attempt=*/i));
    EXPECT_LE(
        ComputeRetryBackoff(/*current_retry_attempt=*/i),
        0.4 * absl::Milliseconds(1) + absl::Milliseconds(1) * std::pow(1.3, i));
  }
}

TEST(RetryingUtilsTest, ComputeRetryBackoff_MinMaxDelays) {
  for (int i = 0; i < 30; ++i) {
    EXPECT_EQ(ComputeRetryBackoff(/*current_retry_attempt=*/i,
                                  /*min_delay=*/absl::Seconds(10)),
              absl::Seconds(10));
    EXPECT_EQ(ComputeRetryBackoff(/*current_retry_attempt=*/i,
                                  /*min_delay=*/absl::Microseconds(1),
                                  /*max_delay=*/absl::Microseconds(1)),
              absl::Microseconds(1));
  }
}

}  // namespace
}  // namespace tsl
