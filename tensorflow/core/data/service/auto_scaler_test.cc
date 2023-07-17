/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/data/service/auto_scaler.h"

#include <optional>

#include "absl/time/time.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/lib/monitoring/cell_reader.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/platform/status_matchers.h"

namespace tensorflow {
namespace data {
namespace {

using ::tsl::testing::StatusIs;

TEST(AutoScalerTest, UpdateOptimalNumberOfWorkersMetricNoReportedTimes) {
  AutoScaler auto_scaler;
  tsl::Status status = auto_scaler.UpdateOptimalNumberOfWorkersMetric();
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kUnavailable));
}

TEST(AutoScalerTest, UpdateOptimalNumberOfWorkersMetricNoReportedPTs) {
  AutoScaler auto_scaler;
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(0, absl::Microseconds(5)));
  tsl::Status status = auto_scaler.UpdateOptimalNumberOfWorkersMetric();
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kUnavailable));
}

TEST(AutoScalerTest, UpdateOptimalNumberOfWorkersMetricNoReportedTPTs) {
  AutoScaler auto_scaler;
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime("/worker/task/0:20000",
                                                absl::Microseconds(10)));
  tsl::Status status = auto_scaler.UpdateOptimalNumberOfWorkersMetric();
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kUnavailable));
}

TEST(AutoScalerTest, UpdateOptimalNumberOfWorkersMetricWithReportedTimes) {
  AutoScaler auto_scaler;
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime("/worker/task/0:20000",
                                                absl::Microseconds(10)));
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(0, absl::Microseconds(5)));
  TF_ASSERT_OK(auto_scaler.UpdateOptimalNumberOfWorkersMetric());
  monitoring::testing::CellReader<int64_t> cell_reader(
      "/tensorflow/data/service/optimal_number_of_workers");
  EXPECT_GT(cell_reader.Read(), 0);
  metrics::RecordTFDataServiceOptimalNumberOfWorkers(0);
}

TEST(AutoScalerTest, GetOptimalNumberOfWorkersInitialState) {
  AutoScaler auto_scaler;
  EXPECT_EQ(auto_scaler.GetOptimalNumberOfWorkers(), std::nullopt);
}

TEST(AutoScalerTest, GetOptimalNumberOfWorkersNoRegisteredWorkers) {
  AutoScaler auto_scaler;
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(0, absl::Microseconds(10)));
  EXPECT_EQ(auto_scaler.GetOptimalNumberOfWorkers(), std::nullopt);
}

TEST(AutoScalerTest, GetOptimalNumberOfWorkersNoRegisteredConsumers) {
  AutoScaler auto_scaler;
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime("/worker/task/0:20000",
                                                absl::Microseconds(10)));
  EXPECT_EQ(auto_scaler.GetOptimalNumberOfWorkers(), std::nullopt);
}

// Worker 0:
//   - Processing time = 0.2 [s] -> Throughput = 5 [elements/s]
// Consumer 0:
//   - Target processing time = 0.025 [s] -> Consumption rate = 40 [elements/s]
//
// Average throughput = 5 [elements/s]
// Sum of consumption rates = 40 [elements/s]
// Estimated number of workers = 40 / 5 = 8
TEST(AutoScalerTest, GetOptimalNumberOfWorkersExpectedEstimate1) {
  AutoScaler auto_scaler;
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime("/worker/task/0:20000",
                                                absl::Seconds(0.2)));
  TF_ASSERT_OK(auto_scaler.ReportTargetProcessingTime(0, absl::Seconds(0.025)));
  EXPECT_EQ(auto_scaler.GetOptimalNumberOfWorkers(), 8);
}

// Worker 0:
//   - Processing time = 0.2 [s] -> Throughput = 5 [elements/s]
// Worker 1:
//   - Processing time = 0.15 [s] -> Throughput = 6.6666 [elements/s]
// Consumer 0:
//   - Target processing time = 0.025 [s] -> Consumption rate = 40 [elements/s]
// Consumer 1:
//   - Target processing time = 0.05 [s] -> Consumption rate = 20 [elements/s]
//
// Average throughput = 5.833 [elements/s]
// Sum of consumption rates = 60 [elements/s]
// Estimated number of workers = 60 / 5.833 = ⌈10.28⌉ = 11
TEST(AutoScalerTest, GetOptimalNumberOfWorkersExpectedEstimate2) {
  AutoScaler auto_scaler;
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime("/worker/task/0:20000",
                                                absl::Seconds(0.2)));
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime("/worker/task/1:20000",
                                                absl::Seconds(0.15)));
  TF_ASSERT_OK(auto_scaler.ReportTargetProcessingTime(0, absl::Seconds(0.025)));
  TF_ASSERT_OK(auto_scaler.ReportTargetProcessingTime(1, absl::Seconds(0.05)));
  EXPECT_EQ(auto_scaler.GetOptimalNumberOfWorkers(), 11);
}

// Worker 0:
//   - Processing time = 0.1 [s] -> Throughput = 10 [elements/s]
// Worker 1:
//   - Processing time = 0.2 [s] -> Throughput = 5 [elements/s]
// Consumer 0:
//   - Target processing time = 0.01 [s] -> Consumption rate = 100 [elements/s]
// Consumer 1:
//   - Target processing time = 0.02 [s] -> Consumption rate = 50 [elements/s]
//
// Average throughput = 7.5 [elements/s]
// Sum of consumption rates = 150 [elements/s]
// Estimated number of workers = 150 / 7.5 = 20
TEST(AutoScalerTest, GetOptimalNumberOfWorkersExpectedEstimate3) {
  AutoScaler auto_scaler;
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime("/worker/task/0:20000",
                                                absl::Seconds(0.1)));
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime("/worker/task/1:20000",
                                                absl::Seconds(0.2)));
  TF_ASSERT_OK(auto_scaler.ReportTargetProcessingTime(0, absl::Seconds(0.01)));
  TF_ASSERT_OK(auto_scaler.ReportTargetProcessingTime(1, absl::Seconds(0.02)));
  EXPECT_EQ(auto_scaler.GetOptimalNumberOfWorkers(), 20);
}

// TODO(armandouv): Delete when we ensure reported time values are correct.

// If outliers are not discarded, the number of workers will be
// unrealistically high (e.g. ~100k workers).
// Worker 0:
//   - Processing time = 0.08 [s] -> Throughput = 12.5 [elements/s]
// Consumer 0:
//   - Target processing time = 0.0000005 [s] -> Consumption rate = 2000000
//   [elements/s] (DISCARDED, replaced by median = 500)
// Consumer 1:
//   - Target processing time = 0.003 [s] -> Consumption rate = 333.333
//   [elements/s]
// Consumer 2:
//   - Target processing time = 0.002 [s] -> Consumption rate = 500 [elements/s]
//
// Average throughput = 12.5 [elements/s]
// Sum of consumption rates = 1333.33 [elements/s]
// Estimated number of workers = 1333.33 / 12.5 = ⌈106.66⌉ = 107
TEST(AutoScalerTest, GetOptimalNumberOfWorkersRemoveOutliersTPT) {
  AutoScaler auto_scaler;
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime("/worker/task/0:20000",
                                                absl::Nanoseconds(80000000)));
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(0, absl::Nanoseconds(500)));
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(1, absl::Nanoseconds(3000000)));
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(2, absl::Nanoseconds(2000000)));
  EXPECT_EQ(auto_scaler.GetOptimalNumberOfWorkers(), 107);
}

// For workers, there can be very small PTs that cause the estimation to be very
// low.
// Worker 0:
//   - Processing time = 0.08 [s] -> Throughput = 12.5 [elements/s]
// Worker 1:
//   - Processing time = 0.07 [s] -> Throughput = 14.285 [elements/s]
// Worker 3:
//   - Processing time = 0.000001 [s] -> Throughput = 1000000 [elements/s]
//   (DISCARDED, replaced by median = 14.285)
// Consumer 0:
//   - Target processing time = 0.0003 [s] -> Consumption rate = 3333.33
//   [elements/s]
//
// Average throughput = 13.69 [elements/s]
// Sum of consumption rates = 3333.33 [elements/s]
// Estimated number of workers = 3333.33 / 13.69 = ⌈243.48⌉ = 244
TEST(AutoScalerTest, GetOptimalNumberOfWorkersRemoveOutliersPT) {
  AutoScaler auto_scaler;
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime("/worker/task/0:20000",
                                                absl::Nanoseconds(80000000)));
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime("/worker/task/1:20000",
                                                absl::Nanoseconds(70000000)));
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime("/worker/task/2:20000",
                                                absl::Nanoseconds(1000)));
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(0, absl::Nanoseconds(300000)));
  EXPECT_EQ(auto_scaler.GetOptimalNumberOfWorkers(), 244);
}

TEST(AutoScalerTest, ReportProcessingTimeNewWorker) {
  AutoScaler auto_scaler;
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime("/worker/task/0:20000",
                                                absl::Microseconds(10)));
}

TEST(AutoScalerTest, ReportProcessingTimeExistingWorker) {
  AutoScaler auto_scaler;
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime("/worker/task/0:20000",
                                                absl::Microseconds(10)));
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime("/worker/task/0:20000",
                                                absl::Microseconds(20)));
}

TEST(AutoScalerTest, ReportProcessingTimeNewAndExisting) {
  AutoScaler auto_scaler;
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime("/worker/task/0:20000",
                                                absl::Microseconds(10)));
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime("/worker/task/1:20000",
                                                absl::Microseconds(20)));
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime("/worker/task/2:20000",
                                                absl::Microseconds(30)));
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime("/worker/task/0:20000",
                                                absl::Microseconds(30)));
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime("/worker/task/1:20000",
                                                absl::Microseconds(20)));
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime("/worker/task/2:20000",
                                                absl::Microseconds(10)));
}

TEST(AutoScalerTest, ReportProcessingTimeZeroDuration) {
  AutoScaler auto_scaler;
  tsl::Status result = auto_scaler.ReportProcessingTime("/worker/task/0:20000",
                                                        absl::ZeroDuration());
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(AutoScalerTest, ReportProcessingTimeNegativeDuration) {
  AutoScaler auto_scaler;
  tsl::Status result = auto_scaler.ReportProcessingTime(
      "/worker/task/0:20000", absl::Microseconds(-10));
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(AutoScalerTest, ReportTargetProcessingTimeNewConsumer) {
  AutoScaler auto_scaler;
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(0, absl::Microseconds(10)));
}

TEST(AutoScalerTest, ReportTargetProcessingTimeExistingConsumer) {
  AutoScaler auto_scaler;
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(0, absl::Microseconds(10)));
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(0, absl::Microseconds(20)));
}

TEST(AutoScalerTest, ReportTargetProcessingTimeNewAndExisting) {
  AutoScaler auto_scaler;
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(0, absl::Microseconds(10)));
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(1, absl::Microseconds(20)));
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(2, absl::Microseconds(30)));
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(0, absl::Microseconds(30)));
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(1, absl::Microseconds(20)));
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(2, absl::Microseconds(10)));
}

TEST(AutoScalerTest, ReportTargetProcessingTimeZeroDuration) {
  AutoScaler auto_scaler;
  tsl::Status result =
      auto_scaler.ReportTargetProcessingTime(0, absl::ZeroDuration());
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(AutoScalerTest, ReportTargetProcessingTimeNegativeDuration) {
  AutoScaler auto_scaler;
  tsl::Status result =
      auto_scaler.ReportTargetProcessingTime(0, absl::Microseconds(-10));
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(AutoScalerTest, RemoveWorkerSuccessful) {
  AutoScaler auto_scaler;
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime("/worker/task/0:20000",
                                                absl::Microseconds(10)));
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime("/worker/task/1:20000",
                                                absl::Microseconds(20)));
  TF_ASSERT_OK(auto_scaler.RemoveWorker("/worker/task/0:20000"));
  TF_ASSERT_OK(auto_scaler.RemoveWorker("/worker/task/1:20000"));
}

TEST(AutoScalerTest, RemoveNonexistentWorker) {
  AutoScaler auto_scaler;
  EXPECT_THAT(auto_scaler.RemoveWorker("/worker/task/0:20000"),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(AutoScalerTest, RemoveWorkerAfterNewPTReported) {
  AutoScaler auto_scaler;
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime("/worker/task/0:20000",
                                                absl::Microseconds(10)));
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime("/worker/task/0:20000",
                                                absl::Microseconds(20)));
  TF_ASSERT_OK(auto_scaler.RemoveWorker("/worker/task/0:20000"));
}

TEST(AutoScalerTest, RemoveConsumerSuccessful) {
  AutoScaler auto_scaler;
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(0, absl::Microseconds(30)));
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(1, absl::Microseconds(30)));
  TF_ASSERT_OK(auto_scaler.RemoveConsumer(0));
  TF_ASSERT_OK(auto_scaler.RemoveConsumer(1));
}

TEST(AutoScalerTest, RemoveNonexistentConsumer) {
  AutoScaler auto_scaler;
  EXPECT_THAT(auto_scaler.RemoveConsumer(0),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(AutoScalerTest, RemoveConsumerAfterNewTPTReported) {
  AutoScaler auto_scaler;
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(0, absl::Microseconds(30)));
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(0, absl::Microseconds(20)));
  TF_ASSERT_OK(auto_scaler.RemoveConsumer(0));
}

}  // namespace

}  // namespace data
}  // namespace tensorflow
