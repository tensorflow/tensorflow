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
#include <utility>
#include <vector>

#if defined(PLATFORM_GOOGLE)
#include "base/context.h"
#endif
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/lib/monitoring/cell_reader.h"
#include "xla/tsl/lib/monitoring/test_utils.h"
#include "xla/tsl/platform/criticality.h"
#include "xla/tsl/platform/statusor.h"
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
#include "tensorflow/core/kernels/batching_util/threadsafe_status.h"
#include "tensorflow/core/lib/monitoring/cell_reader.h"
#include "tensorflow/core/platform/context.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"
#include "tsl/platform/refcount.h"
#include "tsl/platform/status.h"

namespace tensorflow {
namespace serving {
namespace {

using ::tensorflow::monitoring::testing::CellReader;
using ::tensorflow::monitoring::testing::Histogram;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

TEST(BatchTaskCriticalityTest, CriticalityDefaultsToCritical) {
  BatchResourceBase::BatchTask batch_task;
  EXPECT_EQ(batch_task.criticality(), tsl::criticality::Criticality::kCritical);
}

struct PriorityTestParams {
  std::string test_name;
  bool enable_large_batch_splitting;
  MixedPriorityBatchingPolicy mixed_priority_batching_policy;
  // The expected number of batches for each allowed batch size.
  absl::flat_hash_map<int, int> expected_batch_size_count;
  // The expected sum of padding sizes for each allowed batch size.
  absl::flat_hash_map<int, int> expected_batch_size_padding_sum;
};

class TestBatchResourceBase : public BatchResourceBase {
 public:
  using BatchResourceBase::BatchResourceBase;

  std::string DebugString() const override { return "TestBatchResourceBase"; }

 protected:
  // Simple function that returns the input tensors as the output tensors.
  void ProcessFuncBatchImpl(
      const BatchResourceBase::BatchTask& last_task,
      absl::Span<const Tensor> inputs, std::vector<Tensor>* combined_outputs,
      std::function<void(const absl::Status&)> done) const override {
    for (const auto& input : inputs) {
      combined_outputs->push_back(input);
    }
    done(absl::OkStatus());
  }
};

class FailBatchResource : public BatchResourceBase {
 public:
  using BatchResourceBase::BatchResourceBase;

  std::string DebugString() const override { return "FailBatchResource"; }

 protected:
  void ProcessFuncBatchImpl(
      const BatchResourceBase::BatchTask& last_task,
      absl::Span<const Tensor> inputs, std::vector<Tensor>* combined_outputs,
      std::function<void(const absl::Status&)> done) const override {
    done(absl::CancelledError("Function was cancelled"));
  }
};

class BatchResourceBaseWithPriorityTest
    : public ::testing::TestWithParam<PriorityTestParams> {
 protected:
  void SetUp() override {
    processed_batch_size_v2_reader_ = std::make_unique<CellReader<int64_t>>(
        "/tensorflow/serving/batching/processed_batch_size_v2");
    padding_size_v2_reader_ = std::make_unique<CellReader<Histogram>>(
        "/tensorflow/serving/batching/padding_size_v2");
    mixed_priority_policy_reader_ = std::make_unique<CellReader<std::string>>(
        "/tensorflow/serving/batching/mixed_priority_batching_policy");
    // Create device_.
    device_ = DeviceFactory::NewDevice("CPU", SessionOptions{},
                                       "/job:a/replica:0/task:0");
    // Create batch_kernel_node_def.
    NodeDefBuilder batch_function_builder("my_batch_node", "BatchFunction");
    batch_function_builder.Attr("max_batch_size", 16);
    batch_function_builder.Attr("num_batch_threads", 6);
    batch_function_builder.Attr("allowed_batch_sizes", {4, 8, 12, 16});
    batch_function_builder.Attr("batch_timeout_micros", 3000000);
    batch_function_builder.Attr("max_enqueued_batches", 6);
    batch_function_builder.Attr("enable_large_batch_splitting", true);
    batch_function_builder.Attr("Tin", {DataType::DT_INT64});
    batch_function_builder.Input(std::vector<NodeDefBuilder::NodeOut>{
        NodeDefBuilder::NodeOut({"n1", 0, DataType::DT_INT64})});
    batch_function_builder.Attr("Tcaptured", std::vector<DataType>{});
    batch_function_builder.Input(std::vector<NodeDefBuilder::NodeOut>{});
    batch_function_builder.Attr("Tout", {DataType::DT_INT64});
    NameAttrList f;
    f.set_name("func_to_batch");
    batch_function_builder.Attr("f", f);
    NodeDef batch_kernel_node_def;
    CHECK_OK(batch_function_builder.Finalize(&batch_kernel_node_def));

    // Create batch_kernel_.
    absl::Status op_kernel_creation_status;
    batch_kernel_ =
        CreateOpKernel(DEVICE_CPU, device_.get(), device_->GetAllocator({}),
                       batch_kernel_node_def, TF_GRAPH_DEF_VERSION,
                       &op_kernel_creation_status);
    CHECK_OK(op_kernel_creation_status);
    CHECK(batch_kernel_ != nullptr);

    // Create input tensors.
    input_tensor_ = Tensor(DataType::DT_INT64, TensorShape({3, 4}));
    input_tensor_.flat<int64_t>().setZero();
    input_tensor_values_ = {
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

  std::unique_ptr<CellReader<int64_t>> processed_batch_size_v2_reader_;
  std::unique_ptr<CellReader<Histogram>> padding_size_v2_reader_;
  std::unique_ptr<CellReader<std::string>> mixed_priority_policy_reader_;
  std::unique_ptr<Device> device_;
  std::unique_ptr<OpKernel> batch_kernel_;
  Tensor input_tensor_;
  std::vector<TensorValue> input_tensor_values_;
  SessionMetadata session_metadata_;
  OpKernelContext::Params params_;
  std::unique_ptr<OpKernelContext> context_;
};

#if defined(PLATFORM_GOOGLE)
TEST(BatchTaskCriticalityTest, CriticalitySuccessfullyPropagated) {
  std::vector<std::unique_ptr<BatchResourceBase::BatchTask>> batch_tasks;
  // Tasks created with the scoped criticalities must have proper criticalities
  // set.
  {
    tsl::criticality::ScopedCriticality scoped_criticality(
        tsl::criticality::Criticality::kCriticalPlus);
    ASSERT_EQ(tsl::criticality::GetCriticality(),
              tsl::criticality::Criticality::kCriticalPlus);
    batch_tasks.push_back(std::make_unique<BatchResourceBase::BatchTask>());
  }
  {
    tsl::criticality::ScopedCriticality scoped_criticality(
        tsl::criticality::Criticality::kCritical);
    ASSERT_EQ(tsl::criticality::GetCriticality(),
              tsl::criticality::Criticality::kCritical);
    batch_tasks.push_back(std::make_unique<BatchResourceBase::BatchTask>());
  }
  {
    tsl::criticality::ScopedCriticality scoped_criticality(
        tsl::criticality::Criticality::kSheddablePlus);
    ASSERT_EQ(tsl::criticality::GetCriticality(),
              tsl::criticality::Criticality::kSheddablePlus);
    batch_tasks.push_back(std::make_unique<BatchResourceBase::BatchTask>());
  }
  {
    tsl::criticality::ScopedCriticality scoped_criticality(
        tsl::criticality::Criticality::kSheddable);
    ASSERT_EQ(tsl::criticality::GetCriticality(),
              tsl::criticality::Criticality::kSheddable);
    batch_tasks.push_back(std::make_unique<BatchResourceBase::BatchTask>());
  }
  batch_tasks.push_back(std::make_unique<BatchResourceBase::BatchTask>());
  EXPECT_EQ(batch_tasks[0]->criticality(),
            tsl::criticality::Criticality::kCriticalPlus);
  EXPECT_EQ(batch_tasks[1]->criticality(),
            tsl::criticality::Criticality::kCritical);
  EXPECT_EQ(batch_tasks[2]->criticality(),
            tsl::criticality::Criticality::kSheddablePlus);
  EXPECT_EQ(batch_tasks[3]->criticality(),
            tsl::criticality::Criticality::kSheddable);
  EXPECT_EQ(batch_tasks[4]->criticality(),
            tsl::criticality::Criticality::kCritical);
}

TEST_P(BatchResourceBaseWithPriorityTest, BatchingWithMixedPriorityPolicy) {
  std::shared_ptr<SharedBatchScheduler<BatchResourceBase::BatchTask>> batcher;
  TF_ASSERT_OK(SharedBatchScheduler<BatchResourceBase::BatchTask>::Create(
      SharedBatchScheduler<BatchResourceBase::BatchTask>::Options(), &batcher));
  std::vector<int32_t> allowed_batch_sizes = {4, 8, 12, 16};
  int max_batch_size = 16;
  int64_t batch_timeout = absl::ToInt64Microseconds(absl::Seconds(3));
  int num_requests = 6;
  // Make the low priority batch timeout longer than the high priority batch
  // so the low priority tasks can be padded to the high priority batch instead
  // of forming a separate batch.
  BatchResourceBase::BatcherT::QueueOptions queue_options =
      TestBatchResourceBase::GetBatcherQueueOptions(
          /*num_batch_threads=*/num_requests, /*max_batch_size=*/max_batch_size,
          /*batch_timeout_micros=*/batch_timeout,
          /*max_enqueued_batches=*/num_requests, allowed_batch_sizes,
          /*enable_large_batch_splitting=*/
          GetParam().enable_large_batch_splitting,
          /*disable_padding=*/false, kPadUpPolicy,
          /*low_priority_max_batch_size=*/max_batch_size,
          /*low_priority_batch_timeout_micros=*/batch_timeout * 3,
          /*low_priority_max_enqueued_batches=*/num_requests,
          /*low_priority_allowed_batch_sizes=*/allowed_batch_sizes,
          /*mixed_priority_batching_policy=*/
          GetParam().mixed_priority_batching_policy,
          /*enable_priority_aware_batch_scheduler=*/false);
  tsl::core::RefCountPtr<BatchResourceBase> batch_resource(
      new TestBatchResourceBase(true, batcher, queue_options,
                                allowed_batch_sizes));

  std::vector<std::unique_ptr<OpKernelContext>> contexts;
  for (int i = 0; i < num_requests; ++i) {
    contexts.push_back(std::make_unique<OpKernelContext>(&params_));
  }

  absl::BlockingCounter blocking_counter(num_requests);
  for (int i = 0; i < num_requests; ++i) {
    auto create_batch_task_fn = [&]() {
      // The first 3 requests are assigned with the default high priority, while
      // the last 3 requests are set to low priority.
      std::unique_ptr<BatchResourceBase::BatchTask> batch_task;
      if (i >= 3) {
        tsl::criticality::ScopedCriticality scoped_criticality(
            tsl::criticality::Criticality::kSheddable);
        batch_task = std::make_unique<BatchResourceBase::BatchTask>();
      } else {
        batch_task = std::make_unique<BatchResourceBase::BatchTask>();
      }
      return batch_task;
    };
    auto done_callback = [&]() { blocking_counter.DecrementCount(); };
    TF_ASSERT_OK(batch_resource->RegisterInput(
        /*guid=*/i, contexts[i].get(),
        /*batcher_queue_name=*/"batcher_queue_name",
        /*create_batch_task_fn=*/create_batch_task_fn,
        /*done_callback=*/done_callback,
        /*forced_warmup_batch_size=*/0));
  }
  blocking_counter.Wait();

  TF_ASSERT_OK_AND_ASSIGN(absl::string_view policy_str,
                          GetMixedPriorityBatchingPolicyString(
                              GetParam().mixed_priority_batching_policy));
  EXPECT_EQ(
      mixed_priority_policy_reader_->Read("my_model_name", "my_batch_node"),
      policy_str);

  for (const auto& [batch_size, expected_count] :
       GetParam().expected_batch_size_count) {
    EXPECT_EQ(processed_batch_size_v2_reader_->Delta(
                  "my_model_name", "my_batch_node", absl::StrCat(batch_size)),
              expected_count);
  }
  for (const auto& [batch_size, expected_padding_sum] :
       GetParam().expected_batch_size_padding_sum) {
    EXPECT_EQ(
        padding_size_v2_reader_
            ->Delta("my_model_name", absl::StrCat(batch_size), "my_batch_node")
            .sum(),
        expected_padding_sum);
  }
}

INSTANTIATE_TEST_SUITE_P(
    BatchResourceBaseWithPriorityTests, BatchResourceBaseWithPriorityTest,
    ::testing::ValuesIn<PriorityTestParams>({
        // allowed_batch_sizes = {4, 8, 12, 16}.
        // 6 requests in total and each request has task size 3.
        // 3 requests with high priority and 3 requests with low priority.
        // With priority_isolation policy, the high priority tasks and low
        // priority tasks are batched separately. There are 2 batches. Each one
        // has 3 tasks and total size is 12. Each batch has 3 paddings.
        {
            "priority_isolation",
            /*enable_large_batch_splitting=*/true,
            MixedPriorityBatchingPolicy::kPriorityIsolation,
            /*expected_batch_size_count=*/
            {{4, 0}, {8, 0}, {12, 2}, {16, 0}},
            /*expected_batch_size_padding_sum=*/
            {{4, 0}, {8, 0}, {12, 6}, {16, 0}},
        },
        // With priority_merge policy, high priority tasks and low priority
        // tasks are batched together. The total size of all tasks is 18 which
        // exceeds the max batch size 16. The last low priority task is split
        // into two tasks of size 1 and size 2. There are 2 batches. First batch
        // has 6 tasks and total size is 16. No padding for the first batch. The
        // second batch has 1 task of size 2 and is padded to size 4.
        {
            "priority_merge_enable_splitting",
            /*enable_large_batch_splitting=*/true,
            MixedPriorityBatchingPolicy::kPriorityMerge,
            /*expected_batch_size_count=*/
            {{4, 1}, {8, 0}, {12, 0}, {16, 1}},
            /*expected_batch_size_padding_sum=*/
            {{4, 2}, {8, 0}, {12, 0}, {16, 0}},
        },
        // With priority_merge policy, high priority tasks and low priority
        // tasks are batched together. Since splitting is disabled, there are 2
        // batches. First batch has 5 tasks, total size is 15 and is padded to
        // size 16. The second batch has 1 low priority task of size 3 and is
        // padded to size 4.
        {
            "priority_merge_disable_splitting",
            /*enable_large_batch_splitting=*/false,
            MixedPriorityBatchingPolicy::kPriorityMerge,
            /*expected_batch_size_count=*/
            {{4, 1}, {8, 0}, {12, 0}, {16, 1}},
            /*expected_batch_size_padding_sum=*/
            {{4, 1}, {8, 0}, {12, 0}, {16, 1}},
        },
        // With padding_with_max_batch_size policy, high priority tasks and low
        // priority tasks are batched to the max batch size and there is no
        // splitting for low priority tasks. 3 high priority tasks and 2 low
        // priority tasks are batched together. The first batch has total size
        // of 15 and is padded to size 16. The second batch has 1 low priority
        // task of size 3 and is padded to size 4.
        {
            "padding_with_max_batch_size",
            /*enable_large_batch_splitting=*/true,
            MixedPriorityBatchingPolicy::kLowPriorityPaddingWithMaxBatchSize,
            /*expected_batch_size_count=*/
            {{4, 1}, {8, 0}, {12, 0}, {16, 1}},
            /*expected_batch_size_padding_sum=*/
            {{4, 1}, {8, 0}, {12, 0}, {16, 1}},
        },
        // With padding_with_next_allowed_batch_size policy, high priority tasks
        // and low priority tasks are batched to the next allowed batch size. 3
        // high priority tasks and 1 low priority tasks are batched together.
        // The first batch has total size of 12. No padding for batch 1. The
        // second batch has 2 low priority tasks (total size 6) and is padded to
        // size 8.
        {
            "low_priority_padding_with_next_allowed_batch_size",
            /*enable_large_batch_splitting=*/true,
            MixedPriorityBatchingPolicy::
                kLowPriorityPaddingWithNextAllowedBatchSize,
            /*expected_batch_size_count=*/
            {{4, 0}, {8, 1}, {12, 1}, {16, 0}},
            /*expected_batch_size_padding_sum=*/
            {{4, 0}, {8, 2}, {12, 0}, {16, 0}},
        },
        // Same as above but disabled large batch splitting.
        {
            "low_priority_padding_with_next_allowed_batch_size_disable_"
            "splitting",
            /*enable_large_batch_splitting=*/false,
            MixedPriorityBatchingPolicy::
                kLowPriorityPaddingWithNextAllowedBatchSize,
            /*expected_batch_size_count=*/
            {{4, 0}, {8, 1}, {12, 1}, {16, 0}},
            /*expected_batch_size_padding_sum=*/
            {{4, 0}, {8, 2}, {12, 0}, {16, 0}},
        },
    }),
    [](const ::testing::TestParamInfo<
        BatchResourceBaseWithPriorityTest::ParamType>& info) {
      return info.param.test_name;
    });
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

    absl::Notification& process_func_batch_called() {
      return process_func_batch_called_;
    }

   private:
    mutable absl::Notification process_func_batch_called_;
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

TEST_F(BatchResourceBaseTest, ProcessFuncBatchPropagatesError) {
  using BatchTask = BatchResourceBase::BatchTask;
  using SharedBatchScheduler = SharedBatchScheduler<BatchTask>;

  std::shared_ptr<SharedBatchScheduler> batcher;
  TF_ASSERT_OK(SharedBatchScheduler::Create({}, &batcher));

  tsl::core::RefCountPtr<FailBatchResource> batch_resource(
      new FailBatchResource(/*has_process_batch_function=*/true, batcher, {},
                            /*allowed_batch_sizes=*/{}));

  absl::Notification done;
  TF_ASSERT_OK(batch_resource->RegisterInput(
      /*guid=*/0, context_.get(), "queue",
      []() -> absl::StatusOr<std::unique_ptr<BatchTask>> {
        return std::make_unique<BatchTask>();
      },
      [&done]() { done.Notify(); },
      /*forced_warmup_batch_size=*/0));

  done.WaitForNotification();
  EXPECT_EQ(context_->status().code(), absl::StatusCode::kCancelled);
  EXPECT_THAT(context_->status().message(),
              ::testing::HasSubstr("Function was cancelled"));
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

struct MaxExecutionBatchSizeTestParams {
  std::string test_name;
  bool enable_large_batch_splitting;
  int max_batch_size;
  std::vector<int> allowed_batch_sizes;
  absl::flat_hash_map<int, int> expected_batch_size_and_count;
  absl::flat_hash_map<int, int> expected_batch_size_and_padding_sum;
};

class BatchResourceBaseMaxExecutionBatchSizeTest
    : public ::testing::TestWithParam<MaxExecutionBatchSizeTestParams> {
 protected:
  void SetUp() override {
    processed_batch_size_v2_reader_ = std::make_unique<CellReader<int64_t>>(
        "/tensorflow/serving/batching/processed_batch_size_v2");
    padding_size_v2_reader_ = std::make_unique<CellReader<Histogram>>(
        "/tensorflow/serving/batching/padding_size_v2");
    // Create device_.
    device_ = DeviceFactory::NewDevice("CPU", SessionOptions{},
                                       "/job:a/replica:0/task:0");
    // Create batch_kernel_node_def.
    NodeDefBuilder batch_function_builder("my_batch_node", "BatchFunction");
    batch_function_builder.Attr("max_batch_size", GetParam().max_batch_size);
    batch_function_builder.Attr("num_batch_threads", 6);
    batch_function_builder.Attr("allowed_batch_sizes",
                                GetParam().allowed_batch_sizes);
    batch_function_builder.Attr("batch_timeout_micros", 3000000);
    batch_function_builder.Attr("max_enqueued_batches", 6);
    batch_function_builder.Attr("enable_large_batch_splitting",
                                GetParam().enable_large_batch_splitting);
    batch_function_builder.Attr("Tin", {DataType::DT_INT64});
    batch_function_builder.Input(std::vector<NodeDefBuilder::NodeOut>{
        NodeDefBuilder::NodeOut({"n1", 0, DataType::DT_INT64})});
    batch_function_builder.Attr("Tcaptured", std::vector<DataType>{});
    batch_function_builder.Input(std::vector<NodeDefBuilder::NodeOut>{});
    batch_function_builder.Attr("Tout", {DataType::DT_INT64});
    NameAttrList f;
    f.set_name("func_to_batch");
    batch_function_builder.Attr("f", f);
    NodeDef batch_kernel_node_def;
    CHECK_OK(batch_function_builder.Finalize(&batch_kernel_node_def));

    // Create batch_kernel_.
    absl::Status op_kernel_creation_status;
    batch_kernel_ =
        CreateOpKernel(DEVICE_CPU, device_.get(), device_->GetAllocator({}),
                       batch_kernel_node_def, TF_GRAPH_DEF_VERSION,
                       &op_kernel_creation_status);
    CHECK_OK(op_kernel_creation_status);
    CHECK(batch_kernel_ != nullptr);

    // Create input tensors.
    input_tensor_ = Tensor(DataType::DT_INT64, TensorShape({1, 4}));
    input_tensor_.flat<int64_t>().setZero();
    input_tensor_values_ = {
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

  std::unique_ptr<CellReader<int64_t>> processed_batch_size_v2_reader_;
  std::unique_ptr<CellReader<Histogram>> padding_size_v2_reader_;
  std::unique_ptr<Device> device_;
  std::unique_ptr<OpKernel> batch_kernel_;
  Tensor input_tensor_;
  std::vector<TensorValue> input_tensor_values_;
  SessionMetadata session_metadata_;
  OpKernelContext::Params params_;
  std::unique_ptr<OpKernelContext> context_;
};

TEST_P(BatchResourceBaseMaxExecutionBatchSizeTest,
       MaxExecutionBatchSizeIsRespected) {
  std::shared_ptr<SharedBatchScheduler<BatchResourceBase::BatchTask>> batcher;
  TF_ASSERT_OK(SharedBatchScheduler<BatchResourceBase::BatchTask>::Create(
      SharedBatchScheduler<BatchResourceBase::BatchTask>::Options(), &batcher));
  int64_t batch_timeout = absl::ToInt64Microseconds(absl::Seconds(3));
  int num_requests = 10;
  BatchResourceBase::BatcherT::QueueOptions queue_options =
      TestBatchResourceBase::GetBatcherQueueOptions(
          /*num_batch_threads=*/num_requests,
          /*max_batch_size=*/GetParam().max_batch_size,
          /*batch_timeout_micros=*/batch_timeout,
          /*max_enqueued_batches=*/num_requests, GetParam().allowed_batch_sizes,
          /*enable_large_batch_splitting=*/
          GetParam().enable_large_batch_splitting,
          /*disable_padding=*/false);
  tsl::core::RefCountPtr<BatchResourceBase> batch_resource(
      new TestBatchResourceBase(true, batcher, queue_options,
                                GetParam().allowed_batch_sizes));

  std::vector<std::unique_ptr<OpKernelContext>> contexts;
  for (int i = 0; i < num_requests; ++i) {
    contexts.push_back(std::make_unique<OpKernelContext>(&params_));
  }

  absl::BlockingCounter blocking_counter(num_requests);
  for (int i = 0; i < num_requests; ++i) {
    auto create_batch_task_fn = [&]() {
      return std::make_unique<BatchResourceBase::BatchTask>();
    };
    auto done_callback = [&]() { blocking_counter.DecrementCount(); };
    TF_ASSERT_OK(batch_resource->RegisterInput(
        /*guid=*/i, contexts[i].get(),
        /*batcher_queue_name=*/"batcher_queue_name",
        /*create_batch_task_fn=*/create_batch_task_fn,
        /*done_callback=*/done_callback,
        /*forced_warmup_batch_size=*/0));
  }
  blocking_counter.Wait();

  for (const auto& [batch_size, expected_count] :
       GetParam().expected_batch_size_and_count) {
    EXPECT_EQ(processed_batch_size_v2_reader_->Delta(
                  "my_model_name", "my_batch_node", absl::StrCat(batch_size)),
              expected_count);
  }
  for (const auto& [batch_size, expected_padding_sum] :
       GetParam().expected_batch_size_and_padding_sum) {
    EXPECT_EQ(
        padding_size_v2_reader_
            ->Delta("my_model_name", absl::StrCat(batch_size), "my_batch_node")
            .sum(),
        expected_padding_sum);
  }
}

INSTANTIATE_TEST_SUITE_P(
    BatchResourceBaseMaxExecutionBatchSizeTests,
    BatchResourceBaseMaxExecutionBatchSizeTest,
    testing::ValuesIn<MaxExecutionBatchSizeTestParams>({
        // There are 10 requests and each request has task size 1. When batch
        // splitting is enabled and allowed_batch_sizes is empty, the
        // max_execution_batch_size is assigned by the max_batch_size 16. Since
        // allowed_batch_sizes is empty, any batch size <= 16 is allowed.
        // Therefore an input batch of size 10 is processed directly with no
        // padding.
        {
            "batch_splitting_enabled_and_allowed_batch_sizes_empty",
            /*enable_large_batch_splitting=*/true,
            /*max_batch_size=*/16,
            /*allowed_batch_sizes=*/{},
            /*expected_batch_size_and_count=*/{{10, 1}},
            /*expected_batch_size_and_padding_sum=*/{{10, 0}},
        },
        // Same requests as above. With batch splitting disabled,
        // max_execution_batch_size is set by input_batch_size_limit, which
        // inherits its value from max_batch_size (16). Since
        // allowed_batch_sizes is empty, any batch size <= 16 is permitted.
        // Therefore, an input batch of size 10 is processed directly with no
        // padding.
        {
            "batch_splitting_disabled_and_allowed_batch_sizes_empty",
            /*enable_large_batch_splitting=*/false,
            /*max_batch_size=*/16,
            /*allowed_batch_sizes=*/{},
            /*expected_batch_size_and_count=*/{{10, 1}},
            /*expected_batch_size_and_padding_sum=*/{{10, 0}},
        },
        // Same requests as above. When batch splitting is enabled and
        // allowed_batch_sizes is not empty, the max_execution_batch_size is
        // assigned to the largest allowed_batch_size 8. There are two batches.
        // The first batch has 8 requests with total size 8, no padding. The
        // second batch has 2 requests with total size 2, padding to size 4
        // with 2 paddings.
        {
            "batch_splitting_enabled_and_allowed_batch_sizes_not_empty",
            /*enable_large_batch_splitting=*/true,
            /*max_batch_size=*/16,
            /*allowed_batch_sizes=*/{4, 8},
            /*expected_batch_size_and_count=*/{{4, 1}, {8, 1}},
            /*expected_batch_size_and_padding_sum=*/{{4, 2}, {8, 0}},
        },
        // Same requests as above. When batch splitting is disabled, the
        // max_execution_batch_size is assigned to the max_batch_size 16. Since
        // allowed_batch_sizes is not empty and the padding policy is pad up,
        // there is one batch of total size 10 which is padded to size 16 with 6
        // paddings.
        {
            "batch_splitting_disabled_and_allowed_batch_sizes_not_empty",
            /*enable_large_batch_splitting=*/false,
            /*max_batch_size=*/16,
            /*allowed_batch_sizes=*/{4, 8, 16},
            /*expected_batch_size_and_count=*/{{4, 0}, {8, 0}, {16, 1}},
            /*expected_batch_size_and_padding_sum=*/{{4, 0}, {8, 0}, {16, 6}},
        },
    }),
    [](const ::testing::TestParamInfo<
        BatchResourceBaseMaxExecutionBatchSizeTest::ParamType>& info) {
      return info.param.test_name;
    });

TEST(BatchTaskTest, FinishTaskPropagatesErrorStatus) {
  auto task = std::make_unique<BatchResourceBase::BatchTask>();
  task->status = std::make_shared<ThreadSafeStatus>();

  bool callback_invoked = false;
  task->set_done_callback([&]() { callback_invoked = true; });

  // Simulate an error
  absl::Status error = absl::InternalError("Something went wrong");
  task->FinishTask(error);

  EXPECT_TRUE(callback_invoked);
  EXPECT_EQ(task->status->status(), error);
}

TEST(BatchTaskTest, FinishTaskIsIdempotent) {
  auto task = std::make_unique<BatchResourceBase::BatchTask>();
  task->status = std::make_shared<ThreadSafeStatus>();

  int call_count = 0;
  task->set_done_callback([&]() { call_count++; });

  // Call it twice
  task->FinishTask(absl::OkStatus());
  task->FinishTask(absl::OkStatus());

  // Should only run once
  EXPECT_EQ(call_count, 1);
}

#if defined(PLATFORM_GOOGLE)
TEST(BatchTaskTest, FinishTaskRestoresContext) {
  base::WithDeadline wd(absl::Now() + absl::Hours(1));

  Context initial_context(ContextKind::kThread);

  auto task = std::make_unique<BatchResourceBase::BatchTask>();
  task->status = std::make_shared<ThreadSafeStatus>();

  // Create a distinct context (default/empty) to propagate.
  Context unique_context(ContextKind::kDefault);

  // Move the unique context into the task.
  task->propagated_context = std::move(unique_context);

  bool callback_invoked = false;
  task->set_done_callback([&]() {
    callback_invoked = true;
    // Verify that the callback runs in the propagated context (kDefault).
    EXPECT_TRUE(Context(ContextKind::kThread) ==
                Context(ContextKind::kDefault));
    // Verify it is NOT running in the initial context.
    EXPECT_FALSE(Context(ContextKind::kThread) == initial_context);
  });

  task->FinishTask(absl::OkStatus());

  EXPECT_TRUE(callback_invoked);
  // Verify that the context was restored to the initial context.
  EXPECT_TRUE(Context(ContextKind::kThread) == initial_context);
}
#endif  // defined(PLATFORM_GOOGLE)

}  // namespace
}  // namespace serving
}  // namespace tensorflow
