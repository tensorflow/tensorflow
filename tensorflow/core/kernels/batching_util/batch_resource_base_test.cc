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

#include "absl/time/time.h"
#include "tensorflow/core/common_runtime/no_op_cost_measurement.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace serving {
namespace {

using ::testing::Pair;
using ::testing::UnorderedElementsAre;

class TestCostMeasurement : public CostMeasurement {
 public:
  absl::Duration GetTotalCost() override { return absl::Milliseconds(100); }
  absl::string_view GetCostType() const override { return "test"; }
};

std::unique_ptr<BatchResourceBase::BatchTask> MakeBatchTask(
    const int64_t task_size, RequestCost* request_cost) {
  auto task = absl::make_unique<BatchResourceBase::BatchTask>();
  task->inputs.push_back(Tensor(DT_DOUBLE, TensorShape({task_size, 1})));
  task->request_cost = request_cost;
  return task;
}

TEST(SplitBatchCostTest, SkipOnNoCostMeasurement) {
  BatchResourceBase::BatchT batch;
  RequestCost cost;
  batch.AddTask(MakeBatchTask(/*task_size=*/1, &cost));
  batch.Close();
  BatchResourceBase::SplitBatchCost(/*batch_cost_measurement*/ nullptr,
                                    /*processed_size=*/16, batch);
  EXPECT_TRUE(batch.task(0).request_cost->GetCosts().empty());
}

TEST(SplitBatchCostTest, SkipOnZeroCost) {
  BatchResourceBase::BatchT batch;
  RequestCost cost;
  batch.AddTask(MakeBatchTask(/*task_size=*/1, &cost));
  batch.Close();

  NoOpCostMeasurement no_op_cost_measurement;
  BatchResourceBase::SplitBatchCost(&no_op_cost_measurement,
                                    /*processed_size=*/16, batch);
  EXPECT_TRUE(batch.task(0).request_cost->GetCosts().empty());
}

TEST(SplitBatchCostTest, SkipOnZeroBatchSize) {
  BatchResourceBase::BatchT batch;
  batch.Close();

  TestCostMeasurement test_cost_measurement;
  BatchResourceBase::SplitBatchCost(&test_cost_measurement,
                                    /*processed_size=*/0, batch);
}

TEST(SplitBatchCostTest, SkipOnNoRequestCost) {
  BatchResourceBase::BatchT batch;
  batch.AddTask(MakeBatchTask(/*task_size=*/1, nullptr));
  batch.AddTask(MakeBatchTask(/*task_size=*/9, nullptr));
  batch.Close();

  TestCostMeasurement test_cost_measurement;
  BatchResourceBase::SplitBatchCost(&test_cost_measurement,
                                    /*processed_size=*/16, batch);

  EXPECT_EQ(batch.task(0).request_cost, nullptr);
  EXPECT_EQ(batch.task(1).request_cost, nullptr);
}

TEST(SplitBatchCostTest, Basic) {
  BatchResourceBase::BatchT batch;
  RequestCost cost1, cost2;
  batch.AddTask(MakeBatchTask(/*task_size=*/1, &cost1));
  batch.AddTask(MakeBatchTask(/*task_size=*/9, &cost2));
  batch.Close();

  TestCostMeasurement test_cost_measurement;
  BatchResourceBase::SplitBatchCost(&test_cost_measurement,
                                    /*processed_size=*/20, batch);

  EXPECT_THAT(
      batch.task(0).request_cost->GetCosts(),
      UnorderedElementsAre(Pair("test_with_smear", absl::Milliseconds(10)),
                           Pair("test_no_smear", absl::Milliseconds(5))));
  EXPECT_THAT(
      batch.task(1).request_cost->GetCosts(),
      UnorderedElementsAre(Pair("test_with_smear", absl::Milliseconds(90)),
                           Pair("test_no_smear", absl::Milliseconds(45))));
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
