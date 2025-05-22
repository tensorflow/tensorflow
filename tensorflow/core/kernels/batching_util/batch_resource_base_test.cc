/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/batching_util/batch_resource_base.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/criticality.h"
#include "tensorflow/core/common_runtime/cost_constants.h"
#include "tensorflow/core/common_runtime/cost_measurement.h"
#include "tensorflow/core/common_runtime/cost_measurement_registry.h"
#include "tensorflow/core/common_runtime/request_cost.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/device_factory.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/batching_util/batch_scheduler.h"
#include "tensorflow/core/kernels/batching_util/batch_scheduler_utils.h"
#include "tensorflow/core/kernels/batching_util/batch_stats.h"
#include "tensorflow/core/kernels/batching_util/shared_batch_scheduler.h"
#include "tensorflow/core/lib/monitoring/cell_reader.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"
#include "tsl/platform/status.h"

namespace tensorflow {
namespace serving {
namespace {

using ::testing::Pair;
using ::testing::UnorderedElementsAre;

TEST(BatchTaskCriticalityTest, CriticalityDefaultsToCritical) {
  BatchResourceBase::BatchTask batch_task;
  EXPECT_EQ(batch_task.criticality(), tsl::criticality::Criticality::kCritical);
}

#if defined(PLATFORM_GOOGLE)
TEST(BatchTaskCriticalityTest, CriticalitySuccessfullyPropagated) {
  std::vector<BatchResourceBase::BatchTask> batch_tasks;
  // Tasks created with the scoped criticalities must have proper criticalities
  // set.
  {
    tsl::criticality::ScopedCriticality scoped_criticality(
        tsl::criticality::Criticality::kCriticalPlus);
    ASSERT_EQ(tsl::criticality::GetCriticality(),
              tsl::criticality::Criticality::kCriticalPlus);
    batch_tasks.push_back(BatchResourceBase::BatchTask());
  }
  {
    tsl::criticality::ScopedCriticality scoped_criticality(
        tsl::criticality::Criticality::kCritical);
    ASSERT_EQ(tsl::criticality::GetCriticality(),
              tsl::criticality::Criticality::kCritical);
    batch_tasks.push_back(BatchResourceBase::BatchTask());
  }
  {
    tsl::criticality::ScopedCriticality scoped_criticality(
        tsl::criticality::Criticality::kSheddablePlus);
    ASSERT_EQ(tsl::criticality::GetCriticality(),
              tsl::criticality::Criticality::kSheddablePlus);
    batch_tasks.push_back(BatchResourceBase::BatchTask());
  }
  {
    tsl::criticality::ScopedCriticality scoped_criticality(
        tsl::criticality::Criticality::kSheddable);
    ASSERT_EQ(tsl::criticality::GetCriticality(),
              tsl::criticality::Criticality::kSheddable);
    batch_tasks.push_back(BatchResourceBase::BatchTask());
  }
  batch_tasks.push_back(BatchResourceBase::BatchTask());
  EXPECT_EQ(batch_tasks[0].criticality(),
            tsl::criticality::Criticality::kCriticalPlus);
  EXPECT_EQ(batch_tasks[1].criticality(),
            tsl::criticality::Criticality::kCritical);
  EXPECT_EQ(batch_tasks[2].criticality(),
            tsl::criticality::Criticality::kSheddablePlus);
  EXPECT_EQ(batch_tasks[3].criticality(),
            tsl::criticality::Criticality::kSheddable);
  EXPECT_EQ(batch_tasks[4].criticality(),
            tsl::criticality::Criticality::kCritical);
}
#endif

class TestTpuCostMeasurement : public CostMeasurement {
 public:
  using CostMeasurement::CostMeasurement;

  absl::Duration GetTotalCost() override { return absl::Milliseconds(100); }
  absl::string_view GetCostType() const override { return "test_tpu"; }
};
REGISTER_COST_MEASUREMENT("test_tpu", TestTpuCostMeasurement);

class TestGcuCostMeasurement : public CostMeasurement {
 public:
  using CostMeasurement::CostMeasurement;

  absl::Duration GetTotalCost() override { return absl::Milliseconds(200); }
  absl::string_view GetCostType() const override { return "test_gcu"; }
};
REGISTER_COST_MEASUREMENT("test_gcu", TestGcuCostMeasurement);

std::unique_ptr<BatchResourceBase::BatchTask> MakeBatchTask(
    const int64_t task_size, RequestCost* request_cost,
    absl::Time start_time = absl::UnixEpoch()) {
  auto task = std::make_unique<BatchResourceBase::BatchTask>();
  task->inputs.push_back(Tensor(DT_DOUBLE, TensorShape({task_size, 1})));
  task->request_cost = request_cost;
  task->start_time = absl::ToUnixNanos(start_time);
  return task;
}

TEST(SplitBatchCostsAndRecordMetricsTest, SkipOnNoCostMeasurement) {
  BatchResourceBase::BatchT batch;
  RequestCost cost;
  batch.AddTask(MakeBatchTask(/*task_size=*/1, &cost));
  batch.Close();

  std::vector<std::unique_ptr<CostMeasurement>> batch_cost_measurements;
  BatchResourceBase::SplitBatchCostsAndRecordMetrics(
      "model_name", "op_name", batch_cost_measurements, /*processed_size=*/16,
      batch);
  EXPECT_TRUE(batch.task(0).request_cost->GetCosts().empty());
  EXPECT_THAT(batch.task(0).request_cost->GetBatchMetrics(),
              ::testing::ElementsAre(::testing::FieldsAre(
                  /*processed_size=*/16, /*input_size=*/1, /*padding_size=*/15,
                  ::testing::IsEmpty())));
}

TEST(SplitBatchCostsAndRecordMetricsTest, SkipOnZeroCost) {
  BatchResourceBase::BatchT batch;
  RequestCost cost;
  batch.AddTask(MakeBatchTask(/*task_size=*/1, &cost));
  batch.Close();

  CostMeasurement::Context context{/*is_per_query=*/false};
  std::vector<std::unique_ptr<CostMeasurement>> batch_cost_measurements;
  batch_cost_measurements.push_back(
      CostMeasurementRegistry::CreateByNameOrNull("no_op", context));
  BatchResourceBase::SplitBatchCostsAndRecordMetrics(
      "model_name", "op_name", batch_cost_measurements, /*processed_size=*/16,
      batch);
  EXPECT_TRUE(batch.task(0).request_cost->GetCosts().empty());
  EXPECT_THAT(batch.task(0).request_cost->GetBatchMetrics(),
              ::testing::ElementsAre(::testing::FieldsAre(
                  /*processed_size=*/16, /*input_size=*/1, /*padding_size=*/15,
                  ::testing::IsEmpty())));
}

TEST(SplitBatchCostsAndRecordMetricsTest, SkipOnZeroBatchSize) {
  BatchResourceBase::BatchT batch;
  batch.Close();

  CostMeasurement::Context context{/*is_per_query=*/false};
  std::vector<std::unique_ptr<CostMeasurement>> batch_cost_measurements;
  batch_cost_measurements.push_back(
      CostMeasurementRegistry::CreateByNameOrNull("test_tpu", context));
  BatchResourceBase::SplitBatchCostsAndRecordMetrics(
      "model_name", "op_name", batch_cost_measurements, /*processed_size=*/0,
      batch);
}

TEST(SplitBatchCostsAndRecordMetricsTest, SkipOnNoRequestCost) {
  BatchResourceBase::BatchT batch;
  batch.AddTask(MakeBatchTask(/*task_size=*/1, nullptr));
  batch.AddTask(MakeBatchTask(/*task_size=*/9, nullptr));
  batch.Close();

  CostMeasurement::Context context{/*is_per_query=*/false};
  std::vector<std::unique_ptr<CostMeasurement>> batch_cost_measurements;
  batch_cost_measurements.push_back(
      CostMeasurementRegistry::CreateByNameOrNull("test_tpu", context));
  BatchResourceBase::SplitBatchCostsAndRecordMetrics(
      "model_name", "op_name", batch_cost_measurements, /*processed_size=*/16,
      batch);

  EXPECT_EQ(batch.task(0).request_cost, nullptr);
  EXPECT_EQ(batch.task(1).request_cost, nullptr);
}

TEST(SplitBatchCostsAndRecordMetricsTest, SplitSingleCostType) {
  BatchResourceBase::BatchT batch;
  RequestCost cost1, cost2;
  batch.AddTask(MakeBatchTask(/*task_size=*/1, &cost1));
  batch.AddTask(MakeBatchTask(/*task_size=*/9, &cost2));
  batch.Close();

  CostMeasurement::Context context{/*is_per_query=*/false};
  std::vector<std::unique_ptr<CostMeasurement>> batch_cost_measurements;
  batch_cost_measurements.push_back(
      CostMeasurementRegistry::CreateByNameOrNull("test_tpu", context));
  BatchResourceBase::SplitBatchCostsAndRecordMetrics(
      "model_name", "op_name", batch_cost_measurements, /*processed_size=*/20,
      batch);

  EXPECT_THAT(
      batch.task(0).request_cost->GetCosts(),
      UnorderedElementsAre(Pair("test_tpu_with_smear", absl::Milliseconds(10)),
                           Pair("test_tpu_no_smear", absl::Milliseconds(5))));
  EXPECT_THAT(
      batch.task(0).request_cost->GetBatchMetrics(),
      ::testing::ElementsAre(::testing::FieldsAre(
          /*processed_size=*/20, /*input_size=*/1, /*padding_size=*/10,
          UnorderedElementsAre(Pair("test_tpu", absl::Milliseconds(100))))));
  EXPECT_THAT(
      batch.task(1).request_cost->GetCosts(),
      UnorderedElementsAre(Pair("test_tpu_with_smear", absl::Milliseconds(90)),
                           Pair("test_tpu_no_smear", absl::Milliseconds(45))));
  EXPECT_THAT(
      batch.task(1).request_cost->GetBatchMetrics(),
      ::testing::ElementsAre(::testing::FieldsAre(
          /*processed_size=*/20, /*input_size=*/9, /*padding_size=*/10,
          UnorderedElementsAre(Pair("test_tpu", absl::Milliseconds(100))))));
}

TEST(SplitBatchCostsAndRecordMetricsTest, SplitMultiCostTypes) {
  BatchResourceBase::BatchT batch;
  RequestCost cost1, cost2;
  batch.AddTask(MakeBatchTask(/*task_size=*/1, &cost1));
  batch.AddTask(MakeBatchTask(/*task_size=*/9, &cost2));
  batch.Close();

  CostMeasurement::Context context{/*is_per_query=*/false};
  std::vector<std::unique_ptr<CostMeasurement>> batch_cost_measurements;
  batch_cost_measurements.push_back(
      CostMeasurementRegistry::CreateByNameOrNull("test_tpu", context));
  batch_cost_measurements.push_back(
      CostMeasurementRegistry::CreateByNameOrNull("test_gcu", context));
  BatchResourceBase::SplitBatchCostsAndRecordMetrics(
      "model_name", "op_name", batch_cost_measurements, /*processed_size=*/20,
      batch);

  EXPECT_THAT(
      batch.task(0).request_cost->GetCosts(),
      UnorderedElementsAre(Pair("test_tpu_with_smear", absl::Milliseconds(10)),
                           Pair("test_tpu_no_smear", absl::Milliseconds(5)),
                           Pair("test_gcu_with_smear", absl::Milliseconds(20)),
                           Pair("test_gcu_no_smear", absl::Milliseconds(10))));
  EXPECT_THAT(
      batch.task(0).request_cost->GetBatchMetrics(),
      ::testing::ElementsAre(::testing::FieldsAre(
          /*processed_size=*/20, /*input_size=*/1, /*padding_size=*/10,
          UnorderedElementsAre(Pair("test_tpu", absl::Milliseconds(100)),
                               Pair("test_gcu", absl::Milliseconds(200))))));

  EXPECT_THAT(
      batch.task(1).request_cost->GetCosts(),
      UnorderedElementsAre(Pair("test_tpu_with_smear", absl::Milliseconds(90)),
                           Pair("test_tpu_no_smear", absl::Milliseconds(45)),
                           Pair("test_gcu_with_smear", absl::Milliseconds(180)),
                           Pair("test_gcu_no_smear", absl::Milliseconds(90))));
  EXPECT_THAT(
      batch.task(1).request_cost->GetBatchMetrics(),
      ::testing::ElementsAre(::testing::FieldsAre(
          /*processed_size=*/20, /*input_size=*/9, /*padding_size=*/10,
          UnorderedElementsAre(Pair("test_tpu", absl::Milliseconds(100)),
                               Pair("test_gcu", absl::Milliseconds(200))))));
}

TEST(SplitBatchCostsAndRecordMetricsTest, SplitOnlyNonZeroCostTypes) {
  BatchResourceBase::BatchT batch;
  RequestCost cost1, cost2;
  batch.AddTask(MakeBatchTask(/*task_size=*/1, &cost1));
  batch.AddTask(MakeBatchTask(/*task_size=*/9, &cost2));
  batch.Close();

  CostMeasurement::Context context{/*is_per_query=*/false};
  std::vector<std::unique_ptr<CostMeasurement>> batch_cost_measurements;
  batch_cost_measurements.push_back(
      CostMeasurementRegistry::CreateByNameOrNull("no_op", context));
  batch_cost_measurements.push_back(
      CostMeasurementRegistry::CreateByNameOrNull("test_tpu", context));
  BatchResourceBase::SplitBatchCostsAndRecordMetrics(
      "model_name", "op_name", batch_cost_measurements, /*processed_size=*/20,
      batch);

  EXPECT_THAT(
      batch.task(0).request_cost->GetCosts(),
      UnorderedElementsAre(Pair("test_tpu_with_smear", absl::Milliseconds(10)),
                           Pair("test_tpu_no_smear", absl::Milliseconds(5))));
  EXPECT_THAT(
      batch.task(0).request_cost->GetBatchMetrics(),
      ::testing::ElementsAre(::testing::FieldsAre(
          /*processed_size=*/20, /*input_size=*/1, /*padding_size=*/10,
          UnorderedElementsAre(Pair("test_tpu", absl::Milliseconds(100))))));

  EXPECT_THAT(
      batch.task(1).request_cost->GetCosts(),
      UnorderedElementsAre(Pair("test_tpu_with_smear", absl::Milliseconds(90)),
                           Pair("test_tpu_no_smear", absl::Milliseconds(45))));
  EXPECT_THAT(
      batch.task(1).request_cost->GetBatchMetrics(),
      ::testing::ElementsAre(::testing::FieldsAre(
          /*processed_size=*/20, /*input_size=*/9, /*padding_size=*/10,
          UnorderedElementsAre(Pair("test_tpu", absl::Milliseconds(100))))));
}

TEST(SplitBatchCostsAndRecordMetricsTest, UpdatesGlobalBatchStats) {
  // Create batch_cost_measurements with one TPU cost.
  class FakeTpuCostMeasurement : public CostMeasurement {
   public:
    using CostMeasurement::CostMeasurement;
    absl::Duration GetTotalCost() override { return absl::Hours(555); }
    absl::string_view GetCostType() const override { return kTpuCostName; }
  };
  CostMeasurement::Context context{/* is_per_query= */ false};
  std::vector<std::unique_ptr<CostMeasurement>> batch_cost_measurements;
  batch_cost_measurements.push_back(
      std::make_unique<FakeTpuCostMeasurement>(context));

  // Create a non-empty batch.
  BatchResourceBase::BatchT batch;
  batch.AddTask(MakeBatchTask(/* task_size= */ 1, nullptr));
  batch.Close();

  // Pick a model name that no other test would pick. This is so that we are
  // sure that the CPU cost for this model name has either never been reported
  // before or, if this test is executed multiple times, has been reported by
  // this only.
  const char kModelName[] = "test_updates_global_batch_stats";

  BatchResourceBase::SplitBatchCostsAndRecordMetrics(
      /* model_name= */ kModelName, /* op_name= */ "op_name",
      batch_cost_measurements, /* processed_size= */ 17, batch);

  EXPECT_EQ(GlobalBatchStatsRegistry()
                .model(/* model_name= */ kModelName, /* op_name= */ "op_name")
                .batch_size(17)
                .tpu_cost()
                .mean(),
            absl::Hours(555));
}

TEST(SplitBatchCostsAndRecordMetricsTest, GlobalBatchStatsProcessedSize) {
  // Create batch_cost_measurements with one TPU cost.
  class FakeTpuCostMeasurement : public CostMeasurement {
   public:
    using CostMeasurement::CostMeasurement;
    absl::Duration GetTotalCost() override { return absl::Hours(555); }
    absl::string_view GetCostType() const override { return kTpuCostName; }
  };
  CostMeasurement::Context context{/* is_per_query= */ false};
  std::vector<std::unique_ptr<CostMeasurement>> batch_cost_measurements;
  batch_cost_measurements.push_back(
      std::make_unique<FakeTpuCostMeasurement>(context));

  // Create a non-empty batch.
  BatchResourceBase::BatchT batch;
  batch.AddTask(MakeBatchTask(/* task_size= */ 1, nullptr));
  batch.Close();

  // Pick a model name that no other test would pick. This is so that we are
  // sure that the CPU cost for this model name has either never been reported
  // before or, if this test is executed multiple times, has been reported by
  // this only.
  const char kModelName[] = "test_global_batch_stats_processed_size";

  // Get the original cumulative processed size.
  int original_cumulative_processed_size =
      GlobalBatchStatsRegistry()
          .model(/* model_name= */ kModelName, /* op_name= */ "op_name")
          .cumulative_processed_size();

  BatchResourceBase::SplitBatchCostsAndRecordMetrics(
      /* model_name= */ kModelName, /* op_name= */ "op_name",
      batch_cost_measurements, /* processed_size= */ 17, batch);

  // Expect the cumulative processed size to be updated correctly. Note
  // that even though the batch size is 17, there is only one non-padding task,
  // so the cumulative processed size should be
  // original_cumulative_processed_size + 1.
  EXPECT_EQ(GlobalBatchStatsRegistry()
                .model(/* model_name= */ kModelName, /* op_name= */ "op_name")
                .cumulative_processed_size(),
            original_cumulative_processed_size + 1);

  // Add a second processed batch with three non-padding tasks and a different
  // total batch size.
  BatchResourceBase::BatchT batch2;
  batch2.AddTask(MakeBatchTask(/* task_size= */ 1, nullptr));
  batch2.AddTask(MakeBatchTask(/* task_size= */ 1, nullptr));
  batch2.AddTask(MakeBatchTask(/* task_size= */ 1, nullptr));
  batch2.Close();
  BatchResourceBase::SplitBatchCostsAndRecordMetrics(
      /* model_name= */ kModelName, /* op_name= */ "op_name",
      batch_cost_measurements, /* processed_size= */ 8, batch2);

  // Expect the cumulative processed size to be updated correctly.
  EXPECT_EQ(GlobalBatchStatsRegistry()
                .model(/* model_name= */ kModelName, /* op_name= */ "op_name")
                .cumulative_processed_size(),
            original_cumulative_processed_size + 4);
}

TEST(RecordBatchDelayMetricsTest,
     TwoRequestsWithNoQueueingDelayAndSchedulingAtBatchTimeout) {
  const absl::Duration batch_timeout = absl::Seconds(1);
  const absl::Duration task2_delay = batch_timeout / 4;
  const absl::Time task1_start_time = absl::Now();
  const absl::Time task2_start_time = task1_start_time + task2_delay;
  const absl::Time batch_schedule_time = task1_start_time + batch_timeout;

  BatchResourceBase::BatchT batch;
  RequestCost cost1, cost2;
  batch.AddTask(MakeBatchTask(/*task_size=*/1, &cost1, task1_start_time));
  batch.AddTask(MakeBatchTask(/*task_size=*/1, &cost2, task2_start_time));
  batch.Close();

  BatchResourceBase::RecordBatchDelayMetrics(
      batch, "model_name", "op_name", /*processed_size=*/20,
      batch_schedule_time, batch_timeout);

  EXPECT_THAT(
      batch.task(0).request_cost->GetMetrics(),
      UnorderedElementsAre(Pair("batching_delay_msecs",
                                absl::ToDoubleMilliseconds(batch_timeout)),
                           Pair("batch_queueing_delay_msecs", 0)));
  EXPECT_THAT(batch.task(1).request_cost->GetMetrics(),
              UnorderedElementsAre(
                  Pair("batching_delay_msecs",
                       absl::ToDoubleMilliseconds(batch_timeout - task2_delay)),
                  Pair("batch_queueing_delay_msecs", 0)));
}

TEST(RecordBatchDelayMetricsTest,
     TwoRequestsWithNoQueueingDelayAndSchedulingAfterSecondRequest) {
  const absl::Duration batch_timeout = absl::Seconds(1);
  const absl::Duration task2_delay = batch_timeout / 4;
  const absl::Duration scheduling_delay = batch_timeout / 10;
  const absl::Time task1_start_time = absl::Now();
  const absl::Time task2_start_time = task1_start_time + task2_delay;
  const absl::Time batch_schedule_time =
      task1_start_time + task2_delay + scheduling_delay;

  BatchResourceBase::BatchT batch;
  RequestCost cost1, cost2;
  batch.AddTask(MakeBatchTask(/*task_size=*/1, &cost1, task1_start_time));
  batch.AddTask(MakeBatchTask(/*task_size=*/1, &cost2, task2_start_time));
  batch.Close();

  BatchResourceBase::RecordBatchDelayMetrics(
      batch, "model_name", "op_name", /*processed_size=*/20,
      batch_schedule_time, batch_timeout);

  EXPECT_THAT(
      batch.task(0).request_cost->GetMetrics(),
      UnorderedElementsAre(
          Pair("batching_delay_msecs",
               absl::ToDoubleMilliseconds(task2_delay + scheduling_delay)),
          Pair("batch_queueing_delay_msecs", 0)));
  EXPECT_THAT(
      batch.task(1).request_cost->GetMetrics(),
      UnorderedElementsAre(Pair("batching_delay_msecs",
                                absl::ToDoubleMilliseconds(scheduling_delay)),
                           Pair("batch_queueing_delay_msecs", 0)));
}

TEST(RecordBatchDelayMetricsTest, TwoRequestWithQueueingDelay) {
  const absl::Duration batch_timeout = absl::Seconds(1);
  const absl::Duration task2_delay = batch_timeout / 4;
  const absl::Duration queueing_delay = 5 * batch_timeout;
  const absl::Time task1_start_time = absl::Now();
  const absl::Time task2_start_time = task1_start_time + task2_delay;
  const absl::Time batch_schedule_time =
      task1_start_time + batch_timeout + queueing_delay;

  BatchResourceBase::BatchT batch;
  RequestCost cost1, cost2;
  batch.AddTask(MakeBatchTask(/*task_size=*/1, &cost1, task1_start_time));
  batch.AddTask(MakeBatchTask(/*task_size=*/1, &cost2, task2_start_time));
  batch.Close();

  BatchResourceBase::RecordBatchDelayMetrics(
      batch, "model_name", "op_name", /*processed_size=*/20,
      batch_schedule_time, batch_timeout);

  EXPECT_THAT(
      batch.task(0).request_cost->GetMetrics(),
      UnorderedElementsAre(Pair("batching_delay_msecs",
                                absl::ToDoubleMilliseconds(batch_timeout)),
                           Pair("batch_queueing_delay_msecs",
                                absl::ToDoubleMilliseconds(queueing_delay))));
  EXPECT_THAT(batch.task(1).request_cost->GetMetrics(),
              UnorderedElementsAre(
                  Pair("batching_delay_msecs",
                       absl::ToDoubleMilliseconds(batch_timeout - task2_delay)),
                  Pair("batch_queueing_delay_msecs",
                       absl::ToDoubleMilliseconds(queueing_delay))));
}

TEST(RecordBatchDelayMetricsTest,
     TwoRequestsWithQueueingDelayAndSecondArrivingAfterBatchTimeout) {
  const absl::Duration batch_timeout = absl::Seconds(1);
  const absl::Duration task2_delay = 3 * batch_timeout;
  const absl::Duration queueing_delay = 5 * batch_timeout;
  const absl::Time task1_start_time = absl::Now();
  const absl::Time task2_start_time = task1_start_time + task2_delay;
  const absl::Time batch_schedule_time =
      task1_start_time + task2_delay + queueing_delay;

  BatchResourceBase::BatchT batch;
  RequestCost cost1, cost2;
  batch.AddTask(MakeBatchTask(/*task_size=*/1, &cost1, task1_start_time));
  batch.AddTask(MakeBatchTask(/*task_size=*/1, &cost2, task2_start_time));
  batch.Close();

  BatchResourceBase::RecordBatchDelayMetrics(
      batch, "model_name", "op_name", /*processed_size=*/20,
      batch_schedule_time, batch_timeout);

  EXPECT_THAT(batch.task(0).request_cost->GetMetrics(),
              UnorderedElementsAre(
                  Pair("batching_delay_msecs",
                       absl::ToDoubleMilliseconds(batch_timeout)),
                  Pair("batch_queueing_delay_msecs",
                       absl::ToDoubleMilliseconds(task2_delay - batch_timeout +
                                                  queueing_delay))));
  EXPECT_THAT(
      batch.task(1).request_cost->GetMetrics(),
      UnorderedElementsAre(Pair("batching_delay_msecs", 0),
                           Pair("batch_queueing_delay_msecs",
                                absl::ToDoubleMilliseconds(queueing_delay))));
}

class BatchResourceBaseTest : public ::testing::Test {
 protected:
  // Like BatchResourceBase but overrides abstract methods, one of which
  // notifies the exposed process_func_batch_called() notification.
  class MyBatchResource : public BatchResourceBase {
   public:
    using BatchResourceBase::BatchResourceBase;

    std::string DebugString() const override { return ""; }

    void ProcessFuncBatchImpl(
        const BatchResourceBase::BatchTask& /* last_task */,
        absl::Span<const Tensor> /* inputs */,
        std::vector<Tensor>* /* combined_outputs */,
        std::function<void(const absl::Status&)> /* done */) const override {
      process_func_batch_called_.Notify();
    }

    Notification& process_func_batch_called() {
      return process_func_batch_called_;
    }

   private:
    mutable Notification process_func_batch_called_;
  };

  BatchResourceBaseTest() {
    // The whole point of this test fixture is to create a usable batch function
    // context, context_.

    // Create device_.
    device_ = DeviceFactory::NewDevice("CPU", SessionOptions{},
                                       "/job:a/replica:0/task:0");

    // Create batch_kernel_node_def.
    NodeDefBuilder batch_function_builder("my_batch_node", "BatchFunction");
    batch_function_builder.Attr("max_batch_size", 128);
    batch_function_builder.Attr("num_batch_threads", 8);
    batch_function_builder.Attr("allowed_batch_sizes", {2, 4, 8});
    batch_function_builder.Attr("batch_timeout_micros", 100);
    batch_function_builder.Attr("max_enqueued_batches", 100);
    batch_function_builder.Attr("enable_large_batch_splitting", true);
    std::vector<DataType> input_dtypes = {DataType::DT_INT64,
                                          DataType::DT_INT64};
    std::vector<NodeDefBuilder::NodeOut> inputs;
    inputs.push_back(NodeDefBuilder::NodeOut({"n1", 0, DataType::DT_INT64}));
    inputs.push_back(NodeDefBuilder::NodeOut({"n2", 1, DataType::DT_INT64}));
    batch_function_builder.Attr("Tin", input_dtypes);
    batch_function_builder.Input(inputs);
    batch_function_builder.Attr("Tcaptured", {DataType::DT_INT64});
    batch_function_builder.Input(std::vector<NodeDefBuilder::NodeOut>{
        NodeDefBuilder::NodeOut({"n3", 1, DataType::DT_INT64})});
    batch_function_builder.Attr("Tout", {DataType::DT_INT64});
    NameAttrList f;
    f.set_name("func_to_batch");
    batch_function_builder.Attr("f", f);
    NodeDef batch_kernel_node_def;
    TF_CHECK_OK(batch_function_builder.Finalize(&batch_kernel_node_def));

    // Create batch_kernel_.
    absl::Status op_kernel_creation_status;
    batch_kernel_ =
        CreateOpKernel(DEVICE_CPU, device_.get(), device_->GetAllocator({}),
                       batch_kernel_node_def, TF_GRAPH_DEF_VERSION,
                       &op_kernel_creation_status);
    TF_CHECK_OK(op_kernel_creation_status);
    CHECK(batch_kernel_ != nullptr);

    // Create input tensors.
    input_tensor_ = Tensor(DataType::DT_INT64, TensorShape({5, 2, 1}));
    input_tensor_values_ = {
        TensorValue(&input_tensor_),
        TensorValue(&input_tensor_),
        TensorValue(&input_tensor_),
    };

    // Fill-in session_metadata_.
    session_metadata_.set_name("my_model_name");

    // Fill-in params_.
    params_.device = device_.get();
    params_.op_kernel = batch_kernel_.get();
    params_.inputs = input_tensor_values_;
    params_.session_metadata = &session_metadata_;

    // Create context_.
    context_ = std::make_unique<OpKernelContext>(&params_);
  }

  std::unique_ptr<Device> device_;

  std::unique_ptr<OpKernel> batch_kernel_;

  Tensor input_tensor_;
  std::vector<TensorValue> input_tensor_values_;

  SessionMetadata session_metadata_;

  OpKernelContext::Params params_;

  std::unique_ptr<OpKernelContext> context_;
};

TEST_F(BatchResourceBaseTest, PassesCorrectModelBatchStatsToSbs) {
  using BatchTask = BatchResourceBase::BatchTask;
  using SharedBatchScheduler = SharedBatchScheduler<BatchTask>;

  // Like SharedBatchScheduler but exposes the last QueueOptions passed to
  // AddQueue as queue_options().
  class MySharedBatchScheduler : public SharedBatchScheduler {
   public:
    MySharedBatchScheduler() : SharedBatchScheduler::SharedBatchScheduler({}) {}

    absl::Status AddQueue(
        const QueueOptions& options,
        ProcessBatchCallback process_batch_callback,
        std::unique_ptr<BatchScheduler<BatchTask>>* queue) override {
      queue_options_ = options;
      return SharedBatchScheduler::AddQueue(options, process_batch_callback,
                                            queue);
    }

    const QueueOptions& queue_options() const { return queue_options_; }

   private:
    QueueOptions queue_options_;
  };

  auto batcher = std::make_shared<MySharedBatchScheduler>();

  MyBatchResource* my_batch_resource = new MyBatchResource(
      /* has_process_batch_function */ true,
      /* batcher= */ batcher,
      /* batcher_queue_options */ {},
      /* allowed_batch_sizes */ {});

  TF_CHECK_OK(my_batch_resource->RegisterInput(
      /* guid= */
      0,
      /* context= */ context_.get(),
      /* batcher_queue_name= */ "batcher_queue_name",
      /* create_batch_task_fn= */
      []() -> absl::StatusOr<std::unique_ptr<BatchResourceBase::BatchTask>> {
        return std::make_unique<BatchResourceBase::BatchTask>();
      },
      /* done_callback= */ [] {}, /* forced_warmup_batch_size= */ 0));

  EXPECT_EQ(batcher->queue_options().model_batch_stats,
            &GlobalBatchStatsRegistry().model(/* model_name= */ "my_model_name",
                                              /* op_name= */ "my_batch_node"));

  // Wait for the batch timeout to expire and the scheduler to dump the only
  // scheduled task back to the batch resource. If we don't do this, the
  // scheduler will do this itself on destruction, when the resource has already
  // been destroyed.
  my_batch_resource->process_func_batch_called().WaitForNotificationWithTimeout(
      absl::Seconds(1));

  // This is how we have to destroy the BatchResource.
  my_batch_resource->Unref();
}

TEST_F(BatchResourceBaseTest, ConfiguredBatchPaddingPolicyMetric) {
  tensorflow::monitoring::testing::CellReader<std::string> metric(
      "/tensorflow/serving/batching/configured_batch_padding_policy");

  std::shared_ptr<SharedBatchScheduler<BatchResourceBase::BatchTask>> batcher;
  TF_CHECK_OK(
      SharedBatchScheduler<BatchResourceBase::BatchTask>::Create({}, &batcher));

  MyBatchResource* my_batch_resource = new MyBatchResource(
      /* has_process_batch_function */ true,
      /* batcher= */ batcher,
      /* batcher_queue_options */
      MyBatchResource::BatcherT::QueueOptions{
          .batch_padding_policy{kMinimizeTpuCostPerRequestPolicy},
      },
      /* allowed_batch_sizes */ {});

  TF_CHECK_OK(my_batch_resource->RegisterInput(
      /* guid= */
      0, /* context= */ context_.get(),
      /* batcher_queue_name= */ "batcher_queue_name",
      /* create_batch_task_fn= */
      []() -> absl::StatusOr<std::unique_ptr<BatchResourceBase::BatchTask>> {
        return std::make_unique<BatchResourceBase::BatchTask>();
      },
      /* done_callback= */ [] {}, /* forced_warmup_batch_size= */ 0));

  EXPECT_EQ(metric.Read(/* model_name= */ "my_model_name",
                        /* op_name= */ "my_batch_node"),
            kMinimizeTpuCostPerRequestPolicy);

  // Wait for the batch timeout to expire and the scheduler to dump the only
  // scheduled task back to the batch resource. If we don't do this, the
  // scheduler will do this itself on destruction, when the resource has already
  // been destroyed.
  my_batch_resource->process_func_batch_called().WaitForNotificationWithTimeout(
      absl::Seconds(1));

  // This is how we have to destroy the BatchResource.
  my_batch_resource->Unref();
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
