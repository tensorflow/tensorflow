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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/time/time.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/status_matchers.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/lib/monitoring/cell_reader.h"

namespace tensorflow {
namespace data {
namespace {

using ::tsl::testing::StatusIs;

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
  absl::Status result = auto_scaler.ReportProcessingTime("/worker/task/0:20000",
                                                         absl::ZeroDuration());
  EXPECT_THAT(result,
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(AutoScalerTest, ReportProcessingTimeNegativeDuration) {
  AutoScaler auto_scaler;
  absl::Status result = auto_scaler.ReportProcessingTime(
      "/worker/task/0:20000", absl::Microseconds(-10));
  EXPECT_THAT(result,
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
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
  absl::Status result =
      auto_scaler.ReportTargetProcessingTime(0, absl::ZeroDuration());
  EXPECT_THAT(result,
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(AutoScalerTest, ReportTargetProcessingTimeNegativeDuration) {
  AutoScaler auto_scaler;
  absl::Status result =
      auto_scaler.ReportTargetProcessingTime(0, absl::Microseconds(-10));
  EXPECT_THAT(result,
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
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
              absl_testing::StatusIs(absl::StatusCode::kNotFound));
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
              absl_testing::StatusIs(absl::StatusCode::kNotFound));
}

TEST(AutoScalerTest, RemoveConsumerAfterNewTPTReported) {
  AutoScaler auto_scaler;
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(0, absl::Microseconds(30)));
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(0, absl::Microseconds(20)));
  TF_ASSERT_OK(auto_scaler.RemoveConsumer(0));
}

TEST(MultipleIterationsAutoScalerTest, UnregisterExistingIteration) {
  MultipleIterationsAutoScaler auto_scaler;
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(0, 0, absl::Microseconds(5)));
  TF_ASSERT_OK(auto_scaler.UnregisterIteration(0));
}

TEST(MultipleIterationsAutoScalerTest, UnregisterNonexistentIteration) {
  MultipleIterationsAutoScaler auto_scaler;
  EXPECT_THAT(auto_scaler.UnregisterIteration(0),
              absl_testing::StatusIs(absl::StatusCode::kNotFound));
}

TEST(MultipleIterationsAutoScalerTest,
     UpdateOptimalNumberOfWorkersMetricInvalidCurrentWorkers) {
  MultipleIterationsAutoScaler auto_scaler;
  absl::Status status = auto_scaler.UpdateOptimalNumberOfWorkersMetric(0);
  EXPECT_THAT(status,
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
  status = auto_scaler.UpdateOptimalNumberOfWorkersMetric(-1);
  EXPECT_THAT(status,
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(MultipleIterationsAutoScalerTest,
     UpdateOptimalNumberOfWorkersMetricNoReportedTimes) {
  MultipleIterationsAutoScaler auto_scaler;
  absl::Status status = auto_scaler.UpdateOptimalNumberOfWorkersMetric(1);
  EXPECT_THAT(status, absl_testing::StatusIs(absl::StatusCode::kUnavailable));
}

TEST(MultipleIterationsAutoScalerTest,
     UpdateOptimalNumberOfWorkersMetricNoReportedPTs) {
  MultipleIterationsAutoScaler auto_scaler;

  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(0, 0, absl::Microseconds(5)));
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(1, 0, absl::Microseconds(5)));
  absl::Status status = auto_scaler.UpdateOptimalNumberOfWorkersMetric(1);
  EXPECT_THAT(status, absl_testing::StatusIs(absl::StatusCode::kUnavailable));
}

TEST(MultipleIterationsAutoScalerTest,
     UpdateOptimalNumberOfWorkersMetricNoReportedTPTs) {
  MultipleIterationsAutoScaler auto_scaler;

  TF_ASSERT_OK(auto_scaler.ReportProcessingTime(0, "/worker/task/0:20000",
                                                absl::Microseconds(10)));
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime(1, "/worker/task/0:20000",
                                                absl::Microseconds(10)));
  absl::Status status = auto_scaler.UpdateOptimalNumberOfWorkersMetric(1);
  EXPECT_THAT(status, absl_testing::StatusIs(absl::StatusCode::kUnavailable));
}

TEST(MultipleIterationsAutoScalerTest,
     UpdateOptimalNumberOfWorkersMetricWithReportedTimes) {
  MultipleIterationsAutoScaler auto_scaler;

  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(0, 0, absl::Microseconds(5)));
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime(0, "/worker/task/0:20000",
                                                absl::Microseconds(10)));
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(1, 0, absl::Microseconds(5)));
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime(1, "/worker/task/0:20000",
                                                absl::Microseconds(10)));
  TF_ASSERT_OK(auto_scaler.UpdateOptimalNumberOfWorkersMetric(1));
  monitoring::testing::CellReader<int64_t> cell_reader(
      "/tensorflow/data/service/optimal_number_of_workers");
  EXPECT_GT(cell_reader.Read(), 0);
  metrics::RecordTFDataServiceOptimalNumberOfWorkers(0);
}

TEST(MultipleIterationsAutoScalerTest,
     UpdateOptimalNumberOfWorkersMetricIncreaseWithinLimit) {
  MultipleIterationsAutoScaler auto_scaler;

  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(0, 0, absl::Microseconds(10)));
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime(0, "/worker/task/0:20000",
                                                absl::Microseconds(500)));
  // Estimated workers = 50. Current workers = 15.
  // 50 <= 15 * 4 = 60, so the estimate is not modified.
  TF_ASSERT_OK(auto_scaler.UpdateOptimalNumberOfWorkersMetric(15));
  monitoring::testing::CellReader<int64_t> cell_reader(
      "/tensorflow/data/service/optimal_number_of_workers");
  EXPECT_EQ(cell_reader.Read(), 50);
  metrics::RecordTFDataServiceOptimalNumberOfWorkers(0);
}

TEST(MultipleIterationsAutoScalerTest,
     UpdateOptimalNumberOfWorkersMetric4xIncreaseLimit) {
  MultipleIterationsAutoScaler auto_scaler;

  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(0, 0, absl::Microseconds(1)));
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime(0, "/worker/task/0:20000",
                                                absl::Microseconds(10)));
  // Estimated workers = 10. Current workers = 2.
  // 10 > 4 * 2 = 8, so the estimate is limited to 8.
  TF_ASSERT_OK(auto_scaler.UpdateOptimalNumberOfWorkersMetric(2));
  monitoring::testing::CellReader<int64_t> cell_reader(
      "/tensorflow/data/service/optimal_number_of_workers");
  EXPECT_EQ(cell_reader.Read(), 8);
  metrics::RecordTFDataServiceOptimalNumberOfWorkers(0);
}

TEST(MultipleIterationsAutoScalerTest,
     UpdateOptimalNumberOfWorkersMetric500IncreaseLimit) {
  MultipleIterationsAutoScaler auto_scaler;

  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(0, 0, absl::Microseconds(1)));
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime(0, "/worker/task/0:20000",
                                                absl::Microseconds(10000)));
  // Estimated workers = 10000. Current workers = 1000.
  // 10000 > 1000 * 4 = 4000, and 10000 > 1000 + 500 = 1500, so the estimate is
  // limited to min(4000, 1500) = 1500.
  TF_ASSERT_OK(auto_scaler.UpdateOptimalNumberOfWorkersMetric(1000));
  monitoring::testing::CellReader<int64_t> cell_reader(
      "/tensorflow/data/service/optimal_number_of_workers");
  EXPECT_EQ(cell_reader.Read(), 1500);
  metrics::RecordTFDataServiceOptimalNumberOfWorkers(0);
}

TEST(MultipleIterationsAutoScalerTest,
     UpdateOptimalNumberOfWorkersMetricMaxLimit) {
  MultipleIterationsAutoScaler auto_scaler;

  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(0, 0, absl::Microseconds(1)));
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime(0, "/worker/task/0:20000",
                                                absl::Microseconds(200000)));
  // Estimated workers = 200000. Current workers = 99700.
  // The estimate is limited to 100k workers.
  TF_ASSERT_OK(auto_scaler.UpdateOptimalNumberOfWorkersMetric(99700));
  monitoring::testing::CellReader<int64_t> cell_reader(
      "/tensorflow/data/service/optimal_number_of_workers");
  EXPECT_EQ(cell_reader.Read(), 100000);
  metrics::RecordTFDataServiceOptimalNumberOfWorkers(0);
}

TEST(MultipleIterationsAutoScalerTest, GetOptimalNumberOfWorkersInitialState) {
  MultipleIterationsAutoScaler auto_scaler;
  EXPECT_EQ(auto_scaler.GetOptimalNumberOfWorkers(), std::nullopt);
}

TEST(MultipleIterationsAutoScalerTest,
     GetOptimalNumberOfWorkersNoRegisteredWorkers) {
  MultipleIterationsAutoScaler auto_scaler;

  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(0, 0, absl::Microseconds(5)));
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(1, 0, absl::Microseconds(5)));
  EXPECT_EQ(auto_scaler.GetOptimalNumberOfWorkers(), std::nullopt);
}

TEST(MultipleIterationsAutoScalerTest,
     GetOptimalNumberOfWorkersNoRegisteredConsumers) {
  MultipleIterationsAutoScaler auto_scaler;

  TF_ASSERT_OK(auto_scaler.ReportProcessingTime(0, "/worker/task/0:20000",
                                                absl::Microseconds(10)));
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime(1, "/worker/task/0:20000",
                                                absl::Microseconds(10)));
  EXPECT_EQ(auto_scaler.GetOptimalNumberOfWorkers(), std::nullopt);
}

TEST(MultipleIterationsAutoScalerTest,
     GetOptimalNumberOfWorkersExpectedEstimate1) {
  MultipleIterationsAutoScaler auto_scaler;

  // Estimated number of workers for iteration 0 = 8
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime(0, "/worker/task/0:20000",
                                                absl::Seconds(0.2)));
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(0, 0, absl::Seconds(0.025)));

  // Estimated number of workers for iteration 1 = 11
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime(1, "/worker/task/0:20000",
                                                absl::Seconds(0.2)));
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime(1, "/worker/task/1:20000",
                                                absl::Seconds(0.15)));
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(1, 0, absl::Seconds(0.025)));
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(1, 1, absl::Seconds(0.05)));

  // max(8, 11) = 11 workers
  EXPECT_EQ(auto_scaler.GetOptimalNumberOfWorkers(), 11);
}

TEST(MultipleIterationsAutoScalerTest,
     GetOptimalNumberOfWorkersExpectedEstimate2) {
  MultipleIterationsAutoScaler auto_scaler;

  // Estimated number of workers for iteration 0 = 8
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime(0, "/worker/task/0:20000",
                                                absl::Seconds(0.2)));
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(0, 0, absl::Seconds(0.025)));

  // Estimated number of workers for iteration 1 = 11
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime(1, "/worker/task/0:20000",
                                                absl::Seconds(0.2)));
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime(1, "/worker/task/1:20000",
                                                absl::Seconds(0.15)));
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(1, 0, absl::Seconds(0.025)));
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(1, 1, absl::Seconds(0.05)));

  // Estimated number of workers for iteration 2 = 20
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime(2, "/worker/task/0:20000",
                                                absl::Seconds(0.1)));
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime(2, "/worker/task/1:20000",
                                                absl::Seconds(0.2)));
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(2, 0, absl::Seconds(0.01)));
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(2, 1, absl::Seconds(0.02)));

  // max(8, 11, 20) = 20 workers
  EXPECT_EQ(auto_scaler.GetOptimalNumberOfWorkers(), 20);
}

TEST(MultipleIterationsAutoScalerTest, ReportProcessingTimeNewIteration) {
  MultipleIterationsAutoScaler auto_scaler;
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime(0, "/worker/task/0:20000",
                                                absl::Microseconds(10)));
}

TEST(MultipleIterationsAutoScalerTest, ReportProcessingTimeNewWorker) {
  MultipleIterationsAutoScaler auto_scaler;

  TF_ASSERT_OK(auto_scaler.ReportProcessingTime(0, "/worker/task/0:20000",
                                                absl::Microseconds(10)));
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime(0, "/worker/task/1:20000",
                                                absl::Microseconds(10)));
}

TEST(MultipleIterationsAutoScalerTest, ReportProcessingTimeExistingWorker) {
  MultipleIterationsAutoScaler auto_scaler;

  TF_ASSERT_OK(auto_scaler.ReportProcessingTime(0, "/worker/task/0:20000",
                                                absl::Microseconds(10)));
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime(0, "/worker/task/0:20000",
                                                absl::Microseconds(10)));
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime(1, "/worker/task/0:20000",
                                                absl::Microseconds(10)));
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime(1, "/worker/task/0:20000",
                                                absl::Microseconds(10)));
}

TEST(MultipleIterationsAutoScalerTest, ReportProcessingTimeNewAndExisting) {
  MultipleIterationsAutoScaler auto_scaler;

  TF_ASSERT_OK(auto_scaler.ReportProcessingTime(0, "/worker/task/0:20000",
                                                absl::Microseconds(10)));
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime(0, "/worker/task/1:20000",
                                                absl::Microseconds(10)));
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime(1, "/worker/task/0:20000",
                                                absl::Microseconds(10)));
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime(1, "/worker/task/1:20000",
                                                absl::Microseconds(10)));

  TF_ASSERT_OK(auto_scaler.ReportProcessingTime(0, "/worker/task/0:20000",
                                                absl::Microseconds(20)));
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime(0, "/worker/task/1:20000",
                                                absl::Microseconds(30)));
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime(1, "/worker/task/0:20000",
                                                absl::Microseconds(20)));
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime(1, "/worker/task/1:20000",
                                                absl::Microseconds(30)));
}

TEST(MultipleIterationsAutoScalerTest, ReportProcessingTimeZeroDuration) {
  MultipleIterationsAutoScaler auto_scaler;

  absl::Status result = auto_scaler.ReportProcessingTime(
      0, "/worker/task/0:20000", absl::ZeroDuration());
  EXPECT_THAT(result,
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(MultipleIterationsAutoScalerTest, ReportProcessingTimeNegativeDuration) {
  MultipleIterationsAutoScaler auto_scaler;

  absl::Status result = auto_scaler.ReportProcessingTime(
      0, "/worker/task/0:20000", absl::Microseconds(-10));
  EXPECT_THAT(result,
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(MultipleIterationsAutoScalerTest, ReportTargetProcessingTimeNewIteration) {
  MultipleIterationsAutoScaler auto_scaler;
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(0, 0, absl::Microseconds(10)));
}

TEST(MultipleIterationsAutoScalerTest, ReportTargetProcessingTimeNewConsumer) {
  MultipleIterationsAutoScaler auto_scaler;

  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(0, 0, absl::Microseconds(10)));
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(0, 1, absl::Microseconds(10)));
}

TEST(MultipleIterationsAutoScalerTest,
     ReportTargetProcessingTimeExistingWorker) {
  MultipleIterationsAutoScaler auto_scaler;

  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(0, 0, absl::Microseconds(10)));
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(0, 0, absl::Microseconds(10)));
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(1, 0, absl::Microseconds(10)));
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(1, 0, absl::Microseconds(10)));
}

TEST(MultipleIterationsAutoScalerTest,
     ReportTargetProcessingTimeNewAndExisting) {
  MultipleIterationsAutoScaler auto_scaler;

  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(0, 0, absl::Microseconds(10)));
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(0, 1, absl::Microseconds(10)));
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(1, 0, absl::Microseconds(10)));
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(1, 1, absl::Microseconds(10)));

  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(0, 0, absl::Microseconds(20)));
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(0, 1, absl::Microseconds(30)));
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(1, 0, absl::Microseconds(20)));
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(1, 1, absl::Microseconds(30)));
}

TEST(MultipleIterationsAutoScalerTest, ReportTargetProcessingTimeZeroDuration) {
  MultipleIterationsAutoScaler auto_scaler;

  absl::Status result =
      auto_scaler.ReportTargetProcessingTime(0, 0, absl::ZeroDuration());
  EXPECT_THAT(result,
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(MultipleIterationsAutoScalerTest,
     ReportTargetProcessingTimeNegativeDuration) {
  MultipleIterationsAutoScaler auto_scaler;

  absl::Status result =
      auto_scaler.ReportTargetProcessingTime(0, 0, absl::Microseconds(-10));
  EXPECT_THAT(result,
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(MultipleIterationsAutoScalerTest, RemoveWorkerUnregisteredIteration) {
  MultipleIterationsAutoScaler auto_scaler;
  EXPECT_THAT(auto_scaler.RemoveWorker(0, "/worker/task/1:20000"),
              absl_testing::StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(auto_scaler.RemoveWorker(1, "/worker/task/1:20000"),
              absl_testing::StatusIs(absl::StatusCode::kNotFound));
}

TEST(MultipleIterationsAutoScalerTest, RemoveWorkerSuccessful) {
  MultipleIterationsAutoScaler auto_scaler;

  TF_ASSERT_OK(auto_scaler.ReportProcessingTime(0, "/worker/task/0:20000",
                                                absl::Microseconds(10)));
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime(1, "/worker/task/0:20000",
                                                absl::Microseconds(20)));
  TF_ASSERT_OK(auto_scaler.RemoveWorker(0, "/worker/task/0:20000"));
  TF_ASSERT_OK(auto_scaler.RemoveWorker(1, "/worker/task/0:20000"));
}

TEST(MultipleIterationsAutoScalerTest, RemoveNonexistentWorker) {
  MultipleIterationsAutoScaler auto_scaler;
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime(0, "/worker/task/0:20000",
                                                absl::Microseconds(10)));
  EXPECT_THAT(auto_scaler.RemoveWorker(0, "/worker/task/1:20000"),
              absl_testing::StatusIs(absl::StatusCode::kNotFound));
}

TEST(MultipleIterationsAutoScalerTest, RemoveWorkerAfterNewPTReported) {
  MultipleIterationsAutoScaler auto_scaler;

  TF_ASSERT_OK(auto_scaler.ReportProcessingTime(0, "/worker/task/0:20000",
                                                absl::Microseconds(10)));
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime(0, "/worker/task/0:20000",
                                                absl::Microseconds(20)));
  TF_ASSERT_OK(auto_scaler.RemoveWorker(0, "/worker/task/0:20000"));
}

TEST(MultipleIterationsAutoScalerTest, RemoveConsumerUnregisteredIteration) {
  MultipleIterationsAutoScaler auto_scaler;
  EXPECT_THAT(auto_scaler.RemoveConsumer(0, 0),
              absl_testing::StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(auto_scaler.RemoveConsumer(1, 0),
              absl_testing::StatusIs(absl::StatusCode::kNotFound));
}

TEST(MultipleIterationsAutoScalerTest, RemoveConsumerSuccessful) {
  MultipleIterationsAutoScaler auto_scaler;

  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(0, 0, absl::Microseconds(10)));
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(1, 0, absl::Microseconds(20)));
  TF_ASSERT_OK(auto_scaler.RemoveConsumer(0, 0));
  TF_ASSERT_OK(auto_scaler.RemoveConsumer(1, 0));
}

TEST(MultipleIterationsAutoScalerTest, RemoveNonexistentConsumer) {
  MultipleIterationsAutoScaler auto_scaler;
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(0, 0, absl::Microseconds(10)));
  EXPECT_THAT(auto_scaler.RemoveConsumer(0, 1),
              absl_testing::StatusIs(absl::StatusCode::kNotFound));
}

TEST(MultipleIterationsAutoScalerTest, RemoveConsumerAfterNewTPTReported) {
  MultipleIterationsAutoScaler auto_scaler;

  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(0, 0, absl::Microseconds(10)));
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(0, 0, absl::Microseconds(20)));
  TF_ASSERT_OK(auto_scaler.RemoveConsumer(0, 0));
}

}  // namespace

}  // namespace data
}  // namespace tensorflow
