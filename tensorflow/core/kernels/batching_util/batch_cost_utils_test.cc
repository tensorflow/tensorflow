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

#include "tensorflow/core/kernels/batching_util/batch_cost_utils.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "tensorflow/core/common_runtime/cost_constants.h"
#include "tensorflow/core/common_runtime/cost_measurement.h"
#include "tensorflow/core/common_runtime/cost_measurement_registry.h"
#include "tensorflow/core/common_runtime/request_cost.h"
#include "tensorflow/core/kernels/batching_util/batch_scheduler.h"
#include "tensorflow/core/kernels/batching_util/batch_stats.h"

namespace tensorflow {
namespace serving {
namespace {

constexpr char kTestTpuCostName[] = "test_tpu";
constexpr char kTestTpuCostWithSmear[] = "test_tpu_with_smear";
constexpr char kTestTpuCostNoSmear[] = "test_tpu_no_smear";
constexpr char kTestGcuCostName[] = "test_gcu";
constexpr char kTestGcuCostWithSmear[] = "test_gcu_with_smear";
constexpr char kTestGcuCostNoSmear[] = "test_gcu_no_smear";

using ::testing::Pair;
using ::testing::UnorderedElementsAre;

struct SimpleTask : public BatchTask {
  size_t task_size;
  RequestCost* request_cost;

  size_t size() const override { return task_size; }
  ~SimpleTask() override = default;
};

std::unique_ptr<SimpleTask> MakeSimpleTask(const int64_t task_size,
                                           RequestCost* request_cost) {
  auto task = std::make_unique<SimpleTask>();
  task->task_size = task_size;
  task->request_cost = request_cost;
  return task;
}

class TestTpuCostMeasurement : public CostMeasurement {
 public:
  using CostMeasurement::CostMeasurement;

  absl::Duration GetTotalCost() override { return absl::Milliseconds(100); }
  absl::string_view GetCostType() const override { return kTestTpuCostName; }
};
REGISTER_COST_MEASUREMENT(kTestTpuCostName, TestTpuCostMeasurement);

class TestGcuCostMeasurement : public CostMeasurement {
 public:
  using CostMeasurement::CostMeasurement;

  absl::Duration GetTotalCost() override { return absl::Milliseconds(200); }
  absl::string_view GetCostType() const override { return kTestGcuCostName; }
};
REGISTER_COST_MEASUREMENT(kTestGcuCostName, TestGcuCostMeasurement);

TEST(SplitBatchCostsAndRecordMetricsTest, SkipOnNoCostMeasurement) {
  Batch<SimpleTask> batch;
  RequestCost cost;
  batch.AddTask(MakeSimpleTask(/*task_size=*/1, &cost));
  batch.Close();

  std::vector<std::unique_ptr<CostMeasurement>> batch_cost_measurements;
  SplitBatchCostsAndRecordMetrics("model_name", "op_name",
                                  batch_cost_measurements,
                                  /*processed_size=*/16, batch);
  EXPECT_TRUE(batch.task(0).request_cost->GetCosts().empty());
  EXPECT_THAT(batch.task(0).request_cost->GetBatchMetrics(),
              ::testing::ElementsAre(::testing::FieldsAre(
                  /*processed_size=*/16, /*input_size=*/1, /*padding_size=*/15,
                  ::testing::IsEmpty())));
}

TEST(SplitBatchCostsAndRecordMetricsTest, SkipOnZeroCost) {
  Batch<SimpleTask> batch;
  RequestCost cost;
  batch.AddTask(MakeSimpleTask(/*task_size=*/1, &cost));
  batch.Close();

  CostMeasurement::Context context{/*is_per_query=*/false};
  std::vector<std::unique_ptr<CostMeasurement>> batch_cost_measurements;
  batch_cost_measurements.push_back(
      CostMeasurementRegistry::CreateByNameOrNull("no_op", context));
  SplitBatchCostsAndRecordMetrics("model_name", "op_name",
                                  batch_cost_measurements,
                                  /*processed_size=*/16, batch);
  EXPECT_TRUE(batch.task(0).request_cost->GetCosts().empty());
  EXPECT_THAT(batch.task(0).request_cost->GetBatchMetrics(),
              ::testing::ElementsAre(::testing::FieldsAre(
                  /*processed_size=*/16, /*input_size=*/1, /*padding_size=*/15,
                  ::testing::IsEmpty())));
}

TEST(SplitBatchCostsAndRecordMetricsTest, SkipOnZeroBatchSize) {
  Batch<SimpleTask> batch;
  batch.Close();

  CostMeasurement::Context context{/*is_per_query=*/false};
  std::vector<std::unique_ptr<CostMeasurement>> batch_cost_measurements;
  batch_cost_measurements.push_back(
      CostMeasurementRegistry::CreateByNameOrNull(kTestTpuCostName, context));
  SplitBatchCostsAndRecordMetrics("model_name", "op_name",
                                  batch_cost_measurements, /*processed_size=*/0,
                                  batch);
}

TEST(SplitBatchCostsAndRecordMetricsTest, SkipOnNoRequestCost) {
  Batch<SimpleTask> batch;
  batch.AddTask(MakeSimpleTask(/*task_size=*/1, /*request_cost=*/nullptr));
  batch.AddTask(MakeSimpleTask(/*task_size=*/9, /*request_cost=*/nullptr));
  batch.Close();

  CostMeasurement::Context context{/*is_per_query=*/false};
  std::vector<std::unique_ptr<CostMeasurement>> batch_cost_measurements;
  batch_cost_measurements.push_back(
      CostMeasurementRegistry::CreateByNameOrNull(kTestTpuCostName, context));
  SplitBatchCostsAndRecordMetrics("model_name", "op_name",
                                  batch_cost_measurements,
                                  /*processed_size=*/16, batch);

  EXPECT_EQ(batch.task(0).request_cost, nullptr);
  EXPECT_EQ(batch.task(1).request_cost, nullptr);
}

TEST(SplitBatchCostsAndRecordMetricsTest, SplitSingleCostType) {
  Batch<SimpleTask> batch;
  RequestCost cost1, cost2;
  batch.AddTask(MakeSimpleTask(/*task_size=*/1, &cost1));
  batch.AddTask(MakeSimpleTask(/*task_size=*/9, &cost2));
  batch.Close();

  CostMeasurement::Context context{/*is_per_query=*/false};
  std::vector<std::unique_ptr<CostMeasurement>> batch_cost_measurements;
  batch_cost_measurements.push_back(
      CostMeasurementRegistry::CreateByNameOrNull(kTestTpuCostName, context));
  SplitBatchCostsAndRecordMetrics("model_name", "op_name",
                                  batch_cost_measurements,
                                  /*processed_size=*/20, batch);

  EXPECT_THAT(
      batch.task(0).request_cost->GetCosts(),
      UnorderedElementsAre(Pair(kTestTpuCostWithSmear, absl::Milliseconds(10)),
                           Pair(kTestTpuCostNoSmear, absl::Milliseconds(5))));
  EXPECT_THAT(batch.task(0).request_cost->GetBatchMetrics(),
              ::testing::ElementsAre(::testing::FieldsAre(
                  /*processed_size=*/20, /*input_size=*/1, /*padding_size=*/10,
                  UnorderedElementsAre(
                      Pair(kTestTpuCostName, absl::Milliseconds(100))))));
  EXPECT_THAT(
      batch.task(1).request_cost->GetCosts(),
      UnorderedElementsAre(Pair(kTestTpuCostWithSmear, absl::Milliseconds(90)),
                           Pair(kTestTpuCostNoSmear, absl::Milliseconds(45))));
  EXPECT_THAT(batch.task(1).request_cost->GetBatchMetrics(),
              ::testing::ElementsAre(::testing::FieldsAre(
                  /*processed_size=*/20, /*input_size=*/9, /*padding_size=*/10,
                  UnorderedElementsAre(
                      Pair(kTestTpuCostName, absl::Milliseconds(100))))));
}

TEST(SplitBatchCostsAndRecordMetricsTest, SplitMultiCostTypes) {
  Batch<SimpleTask> batch;
  RequestCost cost1, cost2;
  batch.AddTask(MakeSimpleTask(/*task_size=*/1, &cost1));
  batch.AddTask(MakeSimpleTask(/*task_size=*/9, &cost2));
  batch.Close();

  CostMeasurement::Context context{/*is_per_query=*/false};
  std::vector<std::unique_ptr<CostMeasurement>> batch_cost_measurements;
  batch_cost_measurements.push_back(
      CostMeasurementRegistry::CreateByNameOrNull(kTestTpuCostName, context));
  batch_cost_measurements.push_back(
      CostMeasurementRegistry::CreateByNameOrNull(kTestGcuCostName, context));
  SplitBatchCostsAndRecordMetrics("model_name", "op_name",
                                  batch_cost_measurements,
                                  /*processed_size=*/20, batch);

  EXPECT_THAT(
      batch.task(0).request_cost->GetCosts(),
      UnorderedElementsAre(Pair(kTestTpuCostWithSmear, absl::Milliseconds(10)),
                           Pair(kTestTpuCostNoSmear, absl::Milliseconds(5)),
                           Pair(kTestGcuCostWithSmear, absl::Milliseconds(20)),
                           Pair(kTestGcuCostNoSmear, absl::Milliseconds(10))));
  EXPECT_THAT(batch.task(0).request_cost->GetBatchMetrics(),
              ::testing::ElementsAre(::testing::FieldsAre(
                  /*processed_size=*/20, /*input_size=*/1, /*padding_size=*/10,
                  UnorderedElementsAre(
                      Pair(kTestTpuCostName, absl::Milliseconds(100)),
                      Pair(kTestGcuCostName, absl::Milliseconds(200))))));

  EXPECT_THAT(
      batch.task(1).request_cost->GetCosts(),
      UnorderedElementsAre(Pair(kTestTpuCostWithSmear, absl::Milliseconds(90)),
                           Pair(kTestTpuCostNoSmear, absl::Milliseconds(45)),
                           Pair(kTestGcuCostWithSmear, absl::Milliseconds(180)),
                           Pair(kTestGcuCostNoSmear, absl::Milliseconds(90))));
  EXPECT_THAT(batch.task(1).request_cost->GetBatchMetrics(),
              ::testing::ElementsAre(::testing::FieldsAre(
                  /*processed_size=*/20, /*input_size=*/9, /*padding_size=*/10,
                  UnorderedElementsAre(
                      Pair(kTestTpuCostName, absl::Milliseconds(100)),
                      Pair(kTestGcuCostName, absl::Milliseconds(200))))));
}

TEST(SplitBatchCostsAndRecordMetricsTest, SplitOnlyNonZeroCostTypes) {
  Batch<SimpleTask> batch;
  RequestCost cost1, cost2;
  batch.AddTask(MakeSimpleTask(/*task_size=*/1, &cost1));
  batch.AddTask(MakeSimpleTask(/*task_size=*/9, &cost2));
  batch.Close();

  CostMeasurement::Context context{/*is_per_query=*/false};
  std::vector<std::unique_ptr<CostMeasurement>> batch_cost_measurements;
  batch_cost_measurements.push_back(
      CostMeasurementRegistry::CreateByNameOrNull("no_op", context));
  batch_cost_measurements.push_back(
      CostMeasurementRegistry::CreateByNameOrNull(kTestTpuCostName, context));
  SplitBatchCostsAndRecordMetrics("model_name", "op_name",
                                  batch_cost_measurements,
                                  /*processed_size=*/20, batch);

  EXPECT_THAT(
      batch.task(0).request_cost->GetCosts(),
      UnorderedElementsAre(Pair(kTestTpuCostWithSmear, absl::Milliseconds(10)),
                           Pair(kTestTpuCostNoSmear, absl::Milliseconds(5))));
  EXPECT_THAT(batch.task(0).request_cost->GetBatchMetrics(),
              ::testing::ElementsAre(::testing::FieldsAre(
                  /*processed_size=*/20, /*input_size=*/1, /*padding_size=*/10,
                  UnorderedElementsAre(
                      Pair(kTestTpuCostName, absl::Milliseconds(100))))));

  EXPECT_THAT(
      batch.task(1).request_cost->GetCosts(),
      UnorderedElementsAre(Pair(kTestTpuCostWithSmear, absl::Milliseconds(90)),
                           Pair(kTestTpuCostNoSmear, absl::Milliseconds(45))));
  EXPECT_THAT(batch.task(1).request_cost->GetBatchMetrics(),
              ::testing::ElementsAre(::testing::FieldsAre(
                  /*processed_size=*/20, /*input_size=*/9, /*padding_size=*/10,
                  UnorderedElementsAre(
                      Pair(kTestTpuCostName, absl::Milliseconds(100))))));
}

TEST(SplitBatchCostsAndRecordMetricsTest, UpdatesGlobalBatchStats) {
  // Create batch_cost_measurements with one TPU cost.
  class FakeTpuCostMeasurement : public CostMeasurement {
   public:
    using CostMeasurement::CostMeasurement;
    absl::Duration GetTotalCost() override { return absl::Hours(555); }
    absl::string_view GetCostType() const override { return kTpuCostName; }
  };
  CostMeasurement::Context context{/*is_per_query=*/false};
  std::vector<std::unique_ptr<CostMeasurement>> batch_cost_measurements;
  batch_cost_measurements.push_back(
      std::make_unique<FakeTpuCostMeasurement>(context));

  // Create a non-empty batch.
  Batch<SimpleTask> batch;
  batch.AddTask(MakeSimpleTask(/*task_size=*/1, /*request_cost=*/nullptr));
  batch.Close();

  // Pick a model name that no other test would pick. This is so that we are
  // sure that the CPU cost for this model name has either never been reported
  // before or, if this test is executed multiple times, has been reported by
  // this only.
  const char kModelName[] = "test_updates_global_batch_stats";

  SplitBatchCostsAndRecordMetrics(
      /*model_name=*/kModelName, /*op_name=*/"op_name", batch_cost_measurements,
      /*processed_size=*/17, batch);

  EXPECT_EQ(GlobalBatchStatsRegistry()
                .model(kModelName, /*op_name=*/"op_name")
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
  CostMeasurement::Context context{/*is_per_query=*/false};
  std::vector<std::unique_ptr<CostMeasurement>> batch_cost_measurements;
  batch_cost_measurements.push_back(
      std::make_unique<FakeTpuCostMeasurement>(context));

  // Create a non-empty batch.
  Batch<SimpleTask> batch;
  batch.AddTask(MakeSimpleTask(/*task_size=*/1, /*request_cost=*/nullptr));
  batch.Close();

  // Pick a model name that no other test would pick. This is so that we are
  // sure that the CPU cost for this model name has either never been reported
  // before or, if this test is executed multiple times, has been reported by
  // this only.
  const char kModelName[] = "test_global_batch_stats_processed_size";

  // Get the original cumulative processed size.
  int original_cumulative_processed_size =
      GlobalBatchStatsRegistry()
          .model(kModelName, /*op_name=*/"op_name")
          .cumulative_processed_size();

  SplitBatchCostsAndRecordMetrics(
      /*model_name=*/kModelName, /*op_name=*/"op_name", batch_cost_measurements,
      /*processed_size=*/17, batch);

  // Expect the cumulative processed size to be updated correctly. Note
  // that even though the batch size is 17, there is only one non-padding task,
  // so the cumulative processed size should be
  // original_cumulative_processed_size + 1.
  EXPECT_EQ(GlobalBatchStatsRegistry()
                .model(/*model_name=*/kModelName, /*op_name=*/"op_name")
                .cumulative_processed_size(),
            original_cumulative_processed_size + 1);

  // Add a second processed batch with three non-padding tasks and a different
  // total batch size.
  Batch<SimpleTask> batch2;
  batch2.AddTask(MakeSimpleTask(/*task_size=*/1, /*request_cost=*/nullptr));
  batch2.AddTask(MakeSimpleTask(/*task_size=*/1, /*request_cost=*/nullptr));
  batch2.AddTask(MakeSimpleTask(/*task_size=*/1, /*request_cost=*/nullptr));
  batch2.Close();
  SplitBatchCostsAndRecordMetrics(
      /*model_name=*/kModelName, /*op_name=*/"op_name", batch_cost_measurements,
      /*processed_size=*/8, batch2);

  // Expect the cumulative processed size to be updated correctly.
  EXPECT_EQ(GlobalBatchStatsRegistry()
                .model(/*model_name=*/kModelName, /*op_name=*/"op_name")
                .cumulative_processed_size(),
            original_cumulative_processed_size + 4);
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
