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
#include "tensorflow/core/common_runtime/cost_measurement.h"
#include "tensorflow/core/common_runtime/cost_measurement_registry.h"
#include "tensorflow/core/common_runtime/request_cost.h"
#include "tensorflow/core/common_runtime/request_cost_accessor.h"
#include "tensorflow/core/common_runtime/request_cost_accessor_registry.h"
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
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"
#include "tsl/platform/refcount.h"

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

TEST(BatchTaskTest, IsDeadlineExceeded) {
  BatchResourceBase::BatchTask task;
  const absl::Time now = absl::Now();

  task.rpc_deadline = now + absl::Seconds(10);
  EXPECT_FALSE(task.IsDeadlineExceeded(now));

  task.rpc_deadline = now - absl::Seconds(10);
  EXPECT_TRUE(task.IsDeadlineExceeded(now));
}

TEST(BatchTaskTest, IsCancelled) {
  BatchResourceBase::BatchTask task;

  bool cancelled = false;
  task.is_rpc_cancelled = [&cancelled]() { return cancelled; };
  EXPECT_FALSE(task.IsCancelled());

  cancelled = true;
  EXPECT_TRUE(task.IsCancelled());
}

TEST(BatchTaskTest, CreateSplitTaskCopiesFields) {
  BatchResourceBase::BatchTask task;
  task.rpc_deadline = absl::Now() + absl::Seconds(10);
  bool cancelled = false;
  task.is_rpc_cancelled = [&cancelled]() { return cancelled; };

  std::unique_ptr<BatchResourceBase::BatchTask> split_task =
      task.CreateSplitTask(0, []() {});

  EXPECT_EQ(split_task->rpc_deadline, task.rpc_deadline);
  EXPECT_FALSE(split_task->IsCancelled());
  cancelled = true;
  EXPECT_TRUE(split_task->IsCancelled());
}

struct PriorityTestParams {
  std::string test_name;
  bool enable_large_batch_splitting;
  MixedPriorityBatchingPolicy mixed_priority_batching_policy;
  // The expected number of batches for each allowed batch size.
  absl::flat_hash_map<int, int> expected_batch_size_count;
  // The expected sum of padding sizes for each allowed batch size.
  absl::flat_hash_map<int, int> expected_batch_size_padding_sum;
  bool enable_batching_task_lazy_cancellation = false;
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
          /*enable_priority_aware_batch_scheduler=*/false,
          /*enable_priority_aware_batch_scheduler_resplit=*/false,
          /*enable_batching_task_lazy_cancellation=*/
          GetParam().enable_batching_task_lazy_cancellation);
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
  EXPECT_EQ(mixed_priority_policy_reader_->Read("my_model_name",
                                                "my_batch_node", "true"),
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
        // Same as padding_with_max_batch_size but with lazy cancellation
        // enabled. Verifies that the flag is accepted and does not change
        // batching behavior (since priority-aware scheduler is disabled in
        // this test).
        {
            "padding_with_max_batch_size_lazy_cancellation",
            /*enable_large_batch_splitting=*/true,
            MixedPriorityBatchingPolicy::kLowPriorityPaddingWithMaxBatchSize,
            /*expected_batch_size_count=*/
            {{4, 1}, {8, 0}, {12, 0}, {16, 1}},
            /*expected_batch_size_padding_sum=*/
            {{4, 1}, {8, 0}, {12, 0}, {16, 1}},
            /*enable_batching_task_lazy_cancellation=*/true,
        },
    }),
    [](const ::testing::TestParamInfo<
        BatchResourceBaseWithPriorityTest::ParamType>& info) {
      return info.param.test_name;
    });

TEST(GetBatcherQueueOptionsTest,
     LazyCancellationEnabled_PropagatedToSchedulerOptions) {
  std::vector<int32_t> allowed_batch_sizes = {4, 8};
  BatchResourceBase::BatcherT::QueueOptions queue_options =
      TestBatchResourceBase::GetBatcherQueueOptions(
          /*num_batch_threads=*/1, /*max_batch_size=*/8,
          /*batch_timeout_micros=*/1000,
          /*max_enqueued_batches=*/10, allowed_batch_sizes,
          /*enable_large_batch_splitting=*/true,
          /*disable_padding=*/false, kPadUpPolicy,
          /*low_priority_max_batch_size=*/8,
          /*low_priority_batch_timeout_micros=*/1000,
          /*low_priority_max_enqueued_batches=*/10,
          /*low_priority_allowed_batch_sizes=*/allowed_batch_sizes,
          /*mixed_priority_batching_policy=*/
          MixedPriorityBatchingPolicy::kLowPriorityPaddingWithMaxBatchSize,
          /*enable_priority_aware_batch_scheduler=*/true,
          /*enable_priority_aware_batch_scheduler_resplit=*/false,
          /*enable_batching_task_lazy_cancellation=*/true);
  EXPECT_TRUE(queue_options.priority_aware_scheduler_options
                  .enable_lazy_cancellation_filtering);
}
#endif

// Regression test for b/466418871: When SplitInputTask creates subtasks with
// shared TensorMatrix and IncrementalBarrier, and those subtasks are evicted
// from the priority queue (FinishTask called with UnavailableError before
// outputs are populated), the IncrementalBarrier callback must not crash trying
// to Concat uninitialized tensor data.
class SplitInputTaskTest : public ::testing::Test {
 protected:
  void SetUp() override {
    device_ = DeviceFactory::NewDevice("CPU", SessionOptions{},
                                       "/job:a/replica:0/task:0");
    NodeDefBuilder batch_function_builder("my_batch_node", "BatchFunction");
    batch_function_builder.Attr("max_batch_size", 16);
    batch_function_builder.Attr("num_batch_threads", 1);
    batch_function_builder.Attr("allowed_batch_sizes", {4, 8, 16});
    batch_function_builder.Attr("batch_timeout_micros", 1000000);
    batch_function_builder.Attr("max_enqueued_batches", 10);
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

    absl::Status op_kernel_creation_status;
    batch_kernel_ =
        CreateOpKernel(DEVICE_CPU, device_.get(), device_->GetAllocator({}),
                       batch_kernel_node_def, TF_GRAPH_DEF_VERSION,
                       &op_kernel_creation_status);
    CHECK_OK(op_kernel_creation_status);

    input_tensor_ = Tensor(DataType::DT_INT64, TensorShape({6, 4}));
    input_tensor_.flat<int64_t>().setZero();
    input_tensor_values_ = {TensorValue(&input_tensor_)};

    params_.device = device_.get();
    params_.op_kernel = batch_kernel_.get();
    params_.inputs = input_tensor_values_;
    context_ = std::make_unique<OpKernelContext>(&params_);
  }

  std::unique_ptr<Device> device_;
  std::unique_ptr<OpKernel> batch_kernel_;
  Tensor input_tensor_;
  std::vector<TensorValue> input_tensor_values_;
  OpKernelContext::Params params_;
  std::unique_ptr<OpKernelContext> context_;
};

TEST_F(SplitInputTaskTest, EvictedSubtasksDoNotCrash) {
  // Set up a BatchTask with a real OpKernelContext, shared output and status.
  auto task = std::make_unique<BatchResourceBase::BatchTask>();
  task->inputs.push_back(
      Tensor(DataType::DT_INT64, TensorShape({6, 4})));  // size = 6
  task->context = context_.get();
  task->output = std::make_shared<BatchResourceBase::TensorMatrix>();
  task->status = std::make_shared<ThreadSafeStatus>();
  task->guid = 0;
  task->start_time = 0;

  // Capture shared status before SplitInputTask moves the task.
  auto shared_status = task->status;

  absl::Notification done;
  task->set_done_callback([&done]() { done.Notify(); });

  // Split the task: open_batch_remaining_slot=2, max_batch_size=4.
  // Input size 6 > 2, so it splits into [2, 4] = 2 subtasks.
  std::vector<std::unique_ptr<BatchResourceBase::BatchTask>> output_tasks;
  TF_ASSERT_OK(
      BatchResourceBase::SplitInputTask(&task, /*open_batch_remaining_slot=*/2,
                                        /*max_batch_size=*/4, &output_tasks));
  ASSERT_EQ(output_tasks.size(), 2);

  // Simulate eviction: call FinishTask(UnavailableError) on ALL subtasks
  // without populating their outputs. This is exactly what happens when
  // PriorityTaskQueue::AddTask evicts split subtasks.
  for (auto& subtask : output_tasks) {
    subtask->FinishTask(
        absl::UnavailableError("Task evicted due to priority queue full."));
  }

  // The IncrementalBarrier callback should fire after all subtasks finish.
  // Without the fix (checking status before Concat), this would attempt to
  // Concat uninitialized TensorMatrix entries and set_output on the context.
  // With the fix, it skips Concat entirely and no output is set.
  ASSERT_TRUE(done.WaitForNotificationWithTimeout(absl::Seconds(5)));

  // Verify the error was propagated.
  EXPECT_EQ(shared_status->status().code(), absl::StatusCode::kUnavailable);

  // Key behavioral assertion: With the fix, the early exit skips Concat and
  // set_output, so no output is set on the context. Without the fix, Concat
  // runs on empty tensors and set_output IS called.
  EXPECT_EQ(context_->mutable_output(0), nullptr);
}

TEST_F(SplitInputTaskTest, ResplitAlreadySplitTask) {
  auto task = std::make_unique<BatchResourceBase::BatchTask>();
  task->inputs.push_back(Tensor(DataType::DT_INT64, TensorShape({6, 4})));
  task->context = context_.get();
  task->output = std::make_shared<BatchResourceBase::TensorMatrix>();
  task->status = std::make_shared<ThreadSafeStatus>();
  task->guid = 0;
  task->start_time = 0;

  std::shared_ptr<BatchResourceBase::TensorMatrix> parent_output = task->output;
  std::shared_ptr<ThreadSafeStatus> shared_status = task->status;

  absl::Notification done;
  task->set_done_callback([&done]() { done.Notify(); });

  std::vector<std::unique_ptr<BatchResourceBase::BatchTask>> first_split;
  TF_ASSERT_OK(
      BatchResourceBase::SplitInputTask(&task, /*open_batch_remaining_slot=*/2,
                                        /*max_batch_size=*/4, &first_split));
  ASSERT_EQ(first_split.size(), 2);
  EXPECT_EQ(first_split[0]->size(), 2);
  EXPECT_EQ(first_split[1]->size(), 4);
  EXPECT_TRUE(first_split[0]->is_partial);
  EXPECT_TRUE(first_split[1]->is_partial);

  std::vector<std::unique_ptr<BatchResourceBase::BatchTask>> second_split;
  auto subtask1 = std::move(first_split[1]);
  TF_ASSERT_OK(BatchResourceBase::SplitInputTask(
      &subtask1, /*open_batch_remaining_slot=*/1,
      /*max_batch_size=*/4, &second_split));
  ASSERT_EQ(second_split.size(), 2);
  EXPECT_EQ(second_split[0]->size(), 1);
  EXPECT_EQ(second_split[1]->size(), 3);
  EXPECT_TRUE(second_split[0]->is_partial);
  EXPECT_TRUE(second_split[1]->is_partial);

  std::vector<std::unique_ptr<BatchResourceBase::BatchTask>> third_split;
  auto subtask2 = std::move(second_split[1]);
  TF_ASSERT_OK(BatchResourceBase::SplitInputTask(
      &subtask2, /*open_batch_remaining_slot=*/1,
      /*max_batch_size=*/4, &third_split));
  ASSERT_EQ(third_split.size(), 2);
  EXPECT_EQ(third_split[0]->size(), 1);
  EXPECT_EQ(third_split[1]->size(), 2);
  EXPECT_TRUE(third_split[0]->is_partial);
  EXPECT_TRUE(third_split[1]->is_partial);

  EXPECT_EQ(parent_output->size(), 2);

  Tensor out0(DataType::DT_INT64, TensorShape({2, 4}));
  out0.flat<int64_t>().setConstant(10);
  (*first_split[0]->output)[first_split[0]->split_index][0] = out0;

  Tensor out_resplit0(DataType::DT_INT64, TensorShape({1, 4}));
  out_resplit0.flat<int64_t>().setConstant(20);
  (*second_split[0]->output)[second_split[0]->split_index][0] = out_resplit0;

  Tensor out_resplit1(DataType::DT_INT64, TensorShape({1, 4}));
  out_resplit1.flat<int64_t>().setConstant(30);
  (*third_split[0]->output)[third_split[0]->split_index][0] = out_resplit1;

  Tensor out_resplit2(DataType::DT_INT64, TensorShape({2, 4}));
  out_resplit2.flat<int64_t>().setConstant(40);
  (*third_split[1]->output)[third_split[1]->split_index][0] = out_resplit2;

  // Finish the subtasks. Note that the resplitted parent subtasks will be
  // finished when its children are finished.
  first_split[0]->FinishTask(absl::OkStatus());

  second_split[0]->FinishTask(absl::OkStatus());

  for (auto& sub : third_split) {
    sub->FinishTask(absl::OkStatus());
  }

  ASSERT_TRUE(done.WaitForNotificationWithTimeout(absl::Seconds(5)));

  EXPECT_TRUE(shared_status->status().ok());

  const Tensor* final_output = context_->mutable_output(0);
  ASSERT_NE(final_output, nullptr);
  EXPECT_EQ(final_output->shape(), TensorShape({6, 4}));

  auto flat = final_output->flat<int64_t>();
  for (int i = 0; i < 2 * 4; ++i) {
    EXPECT_EQ(flat(i), 10);
  }
  for (int i = 2 * 4; i < 3 * 4; ++i) {
    EXPECT_EQ(flat(i), 20);
  }
  for (int i = 3 * 4; i < 4 * 4; ++i) {
    EXPECT_EQ(flat(i), 30);
  }
  for (int i = 4 * 4; i < 6 * 4; ++i) {
    EXPECT_EQ(flat(i), 40);
  }
}

TEST_F(SplitInputTaskTest, EvictedResplitSubtasksDoNotCrash) {
  auto task = std::make_unique<BatchResourceBase::BatchTask>();
  task->inputs.push_back(
      Tensor(DataType::DT_INT64, TensorShape({6, 4})));  // size = 6
  task->context = context_.get();
  task->output = std::make_shared<BatchResourceBase::TensorMatrix>();
  task->status = std::make_shared<ThreadSafeStatus>();
  task->guid = 0;
  task->start_time = 0;

  auto shared_status = task->status;

  absl::Notification done;
  task->set_done_callback([&done]() { done.Notify(); });

  // First split: [2, 4]
  std::vector<std::unique_ptr<BatchResourceBase::BatchTask>> first_split;
  TF_ASSERT_OK(
      BatchResourceBase::SplitInputTask(&task, /*open_batch_remaining_slot=*/2,
                                        /*max_batch_size=*/4, &first_split));
  ASSERT_EQ(first_split.size(), 2);
  EXPECT_EQ(first_split[0]->size(), 2);
  EXPECT_EQ(first_split[1]->size(), 4);
  EXPECT_TRUE(first_split[0]->is_partial);
  EXPECT_TRUE(first_split[1]->is_partial);

  // Re-split the second subtask (size 4) into [1, 3].
  auto subtask1 = std::move(first_split[1]);
  std::vector<std::unique_ptr<BatchResourceBase::BatchTask>> second_split;
  TF_ASSERT_OK(BatchResourceBase::SplitInputTask(
      &subtask1, /*open_batch_remaining_slot=*/1,
      /*max_batch_size=*/4, &second_split));
  ASSERT_EQ(second_split.size(), 2);
  EXPECT_EQ(second_split[0]->size(), 1);
  EXPECT_EQ(second_split[1]->size(), 3);
  EXPECT_TRUE(second_split[0]->is_partial);
  EXPECT_TRUE(second_split[1]->is_partial);

  // Third split: re-split second_split[1] (size 3) into [1, 2].
  auto subtask2 = std::move(second_split[1]);
  std::vector<std::unique_ptr<BatchResourceBase::BatchTask>> third_split;
  TF_ASSERT_OK(BatchResourceBase::SplitInputTask(
      &subtask2, /*open_batch_remaining_slot=*/1,
      /*max_batch_size=*/4, &third_split));
  ASSERT_EQ(third_split.size(), 2);
  EXPECT_EQ(third_split[0]->size(), 1);
  EXPECT_EQ(third_split[1]->size(), 2);
  EXPECT_TRUE(third_split[0]->is_partial);
  EXPECT_TRUE(third_split[1]->is_partial);

  // first_split[0] and second_split[0] finish OK (outputs unpopulated — they
  // will never be concat'd because the error from the third split causes the
  // barrier callbacks to skip concat).
  first_split[0]->FinishTask(absl::OkStatus());
  second_split[0]->FinishTask(absl::OkStatus());

  // Evict all third-split subtasks without populating outputs.
  for (auto& sub : third_split) {
    sub->FinishTask(
        absl::UnavailableError("Task evicted due to priority queue full."));
  }

  // The chained barrier callbacks should fire without crashing:
  //   third_split barrier  → sees error, skips concat, FinishTask(error)
  //   second_split barrier → sees error, skips concat, FinishTask(error)
  //   first_split barrier  → sees error, skips concat, FinishTask(error)
  //   → done notification
  ASSERT_TRUE(done.WaitForNotificationWithTimeout(absl::Seconds(5)));

  // Verify the error was propagated through all three levels.
  EXPECT_EQ(shared_status->status().code(), absl::StatusCode::kUnavailable);
  EXPECT_EQ(context_->mutable_output(0), nullptr);
}

TEST_F(SplitInputTaskTest, EvictedSiblingTaskAllowsResplit) {
  auto task = std::make_unique<BatchResourceBase::BatchTask>();
  task->inputs.push_back(Tensor(DataType::DT_INT64, TensorShape({12, 4})));
  task->context = context_.get();
  task->output = std::make_shared<BatchResourceBase::TensorMatrix>();
  task->status = std::make_shared<ThreadSafeStatus>();
  task->guid = 0;
  task->start_time = 0;

  auto shared_status = task->status;

  absl::Notification done;
  task->set_done_callback([&done]() { done.Notify(); });

  // Split the task: open_batch_remaining_slot=2, max_batch_size=4.
  // Input size 12 > 2, so it splits into [2, 4, 4, 2] = 4 subtasks.
  std::vector<std::unique_ptr<BatchResourceBase::BatchTask>> first_split;
  TF_ASSERT_OK(
      BatchResourceBase::SplitInputTask(&task, /*open_batch_remaining_slot=*/2,
                                        /*max_batch_size=*/4, &first_split));
  ASSERT_EQ(first_split.size(), 4);
  EXPECT_EQ(first_split[0]->size(), 2);
  EXPECT_EQ(first_split[1]->size(), 4);
  EXPECT_EQ(first_split[2]->size(), 4);
  EXPECT_EQ(first_split[3]->size(), 2);
  EXPECT_TRUE(first_split[0]->is_partial);
  EXPECT_TRUE(first_split[1]->is_partial);
  EXPECT_TRUE(first_split[2]->is_partial);
  EXPECT_TRUE(first_split[3]->is_partial);

  // Finish the first subtask OK.
  first_split[0]->FinishTask(absl::OkStatus());

  // Evict the fourth subtask.
  first_split[3]->FinishTask(
      absl::UnavailableError("Task evicted due to priority queue full."));

  // Verify that the shared status is now an error.
  EXPECT_FALSE(shared_status->status().ok());
  EXPECT_EQ(shared_status->status().code(), absl::StatusCode::kUnavailable);

  // Re-split the second subtask (size 4) into [1, 3].
  auto subtask1 = std::move(first_split[1]);
  std::vector<std::unique_ptr<BatchResourceBase::BatchTask>> second_split;
  TF_ASSERT_OK(BatchResourceBase::SplitInputTask(
      &subtask1, /*open_batch_remaining_slot=*/1,
      /*max_batch_size=*/4, &second_split));
  ASSERT_EQ(second_split.size(), 2);
  EXPECT_EQ(second_split[0]->size(), 1);
  EXPECT_EQ(second_split[1]->size(), 3);
  EXPECT_TRUE(second_split[0]->is_partial);
  EXPECT_TRUE(second_split[1]->is_partial);

  // Verify that the shared status is still an error.
  EXPECT_FALSE(shared_status->status().ok());
  EXPECT_EQ(shared_status->status().code(), absl::StatusCode::kUnavailable);

  first_split[2]->FinishTask(absl::OkStatus());
  // Finish the new subtasks.
  for (auto& sub : second_split) {
    sub->FinishTask(absl::OkStatus());
  }

  // The done callback should fire.
  ASSERT_TRUE(done.WaitForNotificationWithTimeout(absl::Seconds(5)));

  // Verify that the error was preserved.
  EXPECT_EQ(shared_status->status().code(), absl::StatusCode::kUnavailable);
}

#if defined(PLATFORM_GOOGLE)
TEST_F(SplitInputTaskTest, SplitTasksKeepCriticalityOfOriginalRequest) {
  // Create a task under kSheddablePlus criticality.
  std::unique_ptr<BatchResourceBase::BatchTask> task;
  {
    tsl::criticality::ScopedCriticality scoped_criticality(
        tsl::criticality::Criticality::kSheddablePlus);
    task = std::make_unique<BatchResourceBase::BatchTask>();
  }
  ASSERT_EQ(task->criticality(), tsl::criticality::Criticality::kSheddablePlus);

  task->inputs.push_back(Tensor(DataType::DT_INT64, TensorShape({6, 4})));
  task->context = context_.get();
  task->output = std::make_shared<BatchResourceBase::TensorMatrix>();
  task->status = std::make_shared<ThreadSafeStatus>();
  task->guid = 0;
  task->start_time = 0;

  absl::Notification done;
  task->set_done_callback([&done]() { done.Notify(); });

  // Split the task outside the ScopedCriticality scope. The calling thread's
  // criticality is the default (kCritical), but the split tasks should
  // inherit the original task's criticality (kSheddablePlus).
  std::vector<std::unique_ptr<BatchResourceBase::BatchTask>> output_tasks;
  TF_ASSERT_OK(
      BatchResourceBase::SplitInputTask(&task, /*open_batch_remaining_slot=*/2,
                                        /*max_batch_size=*/4, &output_tasks));
  ASSERT_EQ(output_tasks.size(), 2);
  EXPECT_EQ(output_tasks[0]->size(), 2);
  EXPECT_EQ(output_tasks[1]->size(), 4);

  // Verify all subtasks inherited the original criticality.
  for (const auto& subtask : output_tasks) {
    EXPECT_EQ(subtask->criticality(),
              tsl::criticality::Criticality::kSheddablePlus);
  }

  // Re-split the second subtask and verify criticality is preserved through
  // multiple levels of splitting.
  auto subtask1 = std::move(output_tasks[1]);
  std::vector<std::unique_ptr<BatchResourceBase::BatchTask>> second_split;
  TF_ASSERT_OK(BatchResourceBase::SplitInputTask(
      &subtask1, /*open_batch_remaining_slot=*/1,
      /*max_batch_size=*/4, &second_split));
  ASSERT_EQ(second_split.size(), 2);
  EXPECT_EQ(second_split[0]->size(), 1);
  EXPECT_EQ(second_split[1]->size(), 3);

  // Verify criticality is preserved through re-splitting.
  for (const auto& subtask : second_split) {
    EXPECT_EQ(subtask->criticality(),
              tsl::criticality::Criticality::kSheddablePlus);
  }

  // Clean up: finish all subtasks to avoid leaking the IncrementalBarrier.
  output_tasks[0]->FinishTask(absl::OkStatus());
  for (auto& sub : second_split) {
    sub->FinishTask(absl::OkStatus());
  }
  ASSERT_TRUE(done.WaitForNotificationWithTimeout(absl::Seconds(5)));
}
#endif

std::unique_ptr<BatchResourceBase::BatchTask> MakeBatchTask(
    const int64_t task_size, RequestCost* request_cost,
    absl::Time start_time = absl::UnixEpoch()) {
  auto task = std::make_unique<BatchResourceBase::BatchTask>();
  task->inputs.push_back(Tensor(DT_DOUBLE, TensorShape({task_size, 1})));
  task->request_cost = request_cost;
  task->start_time = absl::ToUnixNanos(start_time);
  return task;
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

TEST(RecordBatchDelayMetricsTest, StreamzBatchAndQueueingDelayUsV2) {
  auto batch_delay_v2_reader = std::make_unique<CellReader<Histogram>>(
      "/tensorflow/serving/batching/batch_delay_us_v2");
  auto queueing_delay_v2_reader = std::make_unique<CellReader<Histogram>>(
      "/tensorflow/serving/batching/queueing_delay_us_v2");

  const absl::Duration batch_timeout = absl::Seconds(1);
  const absl::Duration queueing_delay = absl::Seconds(5);
  const absl::Time task_start_time = absl::Now();
  const absl::Time batch_schedule_time =
      task_start_time + batch_timeout + queueing_delay;
  const int64_t processed_size = 20;

  BatchResourceBase::BatchT batch;
  RequestCost cost;
  batch.AddTask(MakeBatchTask(/*task_size=*/1, &cost, task_start_time));
  batch.Close();

  BatchResourceBase::RecordBatchDelayMetrics(
      batch, "model_name", "op_name", processed_size, batch_schedule_time,
      batch_timeout);

  const std::string criticality_str = absl::StrCat(batch.task(0).criticality());
  EXPECT_GT(batch_delay_v2_reader
                ->Delta("model_name", "op_name", std::to_string(processed_size),
                        criticality_str)
                .num(),
            0);
  EXPECT_GT(queueing_delay_v2_reader
                ->Delta("model_name", "op_name", std::to_string(processed_size),
                        criticality_str)
                .num(),
            0);
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

// A CostMeasurement that always reports a fixed TPU cost.
class TestTpuCostMeasurement : public CostMeasurement {
 public:
  using CostMeasurement::CostMeasurement;

  absl::Duration GetTotalCost() override { return absl::Milliseconds(100); }
  absl::string_view GetCostType() const override { return "test_tpu"; }
};
REGISTER_COST_MEASUREMENT("test_tpu", TestTpuCostMeasurement);

// In production, each incoming request runs on its own thread. Using
// thread_local gives each concurrent request its own RequestCost instance
// without locks or map lookups, so costs from different requests never mix.
class TestRequestCostAccessor : public RequestCostAccessor {
 public:
  RequestCost* GetRequestCost() const override {
    thread_local RequestCost request_cost;
    return &request_cost;
  }
};

bool SetBatchResourceCostMeasurementAndAccessorType() {
  setenv("TF_COST_MEASUREMENT_TYPE", "test_tpu", 1 /*overwrite*/);
  setenv("TF_REQUEST_COST_ACCESSOR_TYPE", "test_request_cost_accessor",
         1 /*overwrite*/);
  RequestCostAccessorRegistry::RegisterRequestCostAccessor(
      "test_request_cost_accessor",
      []() { return std::make_unique<TestRequestCostAccessor>(); });
  return true;
}
static bool batch_resource_cost_measurement_and_accessor_registered =
    SetBatchResourceCostMeasurementAndAccessorType();

// Test fixture for cost-related tests. Uses a single input/output to be
// compatible with TestBatchResourceBase (which returns all inputs as outputs).
class BatchResourceCostTest : public ::testing::Test {
 protected:
  BatchResourceCostTest() {
    device_ = DeviceFactory::NewDevice("CPU", SessionOptions{},
                                       "/job:a/replica:0/task:0");

    NodeDefBuilder batch_function_builder("my_batch_node", "BatchFunction");
    batch_function_builder.Attr("max_batch_size", 128);
    batch_function_builder.Attr("num_batch_threads", 8);
    batch_function_builder.Attr("allowed_batch_sizes", {4, 8});
    batch_function_builder.Attr("batch_timeout_micros", 100);
    batch_function_builder.Attr("max_enqueued_batches", 100);
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
    TF_CHECK_OK(batch_function_builder.Finalize(&batch_kernel_node_def));

    absl::Status op_kernel_creation_status;
    batch_kernel_ =
        CreateOpKernel(DEVICE_CPU, device_.get(), device_->GetAllocator({}),
                       batch_kernel_node_def, TF_GRAPH_DEF_VERSION,
                       &op_kernel_creation_status);
    TF_CHECK_OK(op_kernel_creation_status);
    CHECK(batch_kernel_ != nullptr);
  }

  std::unique_ptr<Device> device_;
  std::unique_ptr<OpKernel> batch_kernel_;
};

// Verifies that the RequestCostAccessor-based injection populates costs
// during actual batch processing.
TEST_F(BatchResourceCostTest, ProcessFuncBatchWithRequestCost) {
  using BatchTask = BatchResourceBase::BatchTask;
  using SharedBatchScheduler = SharedBatchScheduler<BatchTask>;

  std::shared_ptr<SharedBatchScheduler> batcher;
  TF_ASSERT_OK(SharedBatchScheduler::Create({}, &batcher));

  Tensor input_tensor(DataType::DT_INT64, TensorShape({5, 2, 1}));
  std::vector<TensorValue> input_tensor_values = {TensorValue(&input_tensor)};
  SessionMetadata session_metadata;
  OpKernelContext::Params params = {
      .op_kernel = batch_kernel_.get(),
      .device = device_.get(),
      .session_metadata = &session_metadata,
      .inputs = input_tensor_values,
  };
  auto context = std::make_unique<OpKernelContext>(&params);

  tsl::core::RefCountPtr<TestBatchResourceBase> batch_resource(
      new TestBatchResourceBase(
          /*has_process_batch_function=*/true, batcher,
          SharedBatchScheduler::QueueOptions{},
          /*allowed_batch_sizes=*/{}));

  absl::Notification task_done;
  absl::flat_hash_map<std::string, absl::Duration> costs;
  // Run the request on a dedicated thread so it gets a fresh thread_local
  // RequestCost. gUnit runs all TEST_F cases on the same main thread, so
  // without this, costs from earlier tests would accumulate in the main
  // thread's thread_local instance and pollute subsequent assertions.
  std::unique_ptr<Thread> request_thread(
      Env::Default()->StartThread(ThreadOptions(), "request_thread", [&]() {
        TF_ASSERT_OK(batch_resource->RegisterInput(
            /*guid=*/0, context.get(), /*batcher_queue_name=*/"queue",
            /*create_batch_task_fn=*/
            []() -> absl::StatusOr<std::unique_ptr<BatchTask>> {
              return std::make_unique<BatchTask>();
            },
            /*done_callback=*/[&]() { task_done.Notify(); },
            /*forced_warmup_batch_size=*/0));

        task_done.WaitForNotification();

        auto accessor = RequestCostAccessorRegistry::CreateByNameOrNull(
            "test_request_cost_accessor");
        ASSERT_NE(accessor, nullptr);
        costs = accessor->GetRequestCost()->GetCosts();
      }));

  request_thread.reset();  // Wait for thread to finish

  // With 1 task of size 5 (from input_tensor_ shape {5,2,1}),
  // batch.size() = 5, processed_size = 5 (no allowed_batch_sizes padding),
  // total cost = 100ms.
  //   with_smear = 100ms / 5 * 5 = 100ms
  //   no_smear   = 100ms / 5 * 5 = 100ms
  EXPECT_THAT(costs, UnorderedElementsAre(
                         Pair("test_tpu_with_smear", absl::Milliseconds(100)),
                         Pair("test_tpu_no_smear", absl::Milliseconds(100))));
}

// Verifies that the RequestCostAccessor-based injection populates costs
// during actual batch processing with padding.
TEST_F(BatchResourceCostTest, ProcessFuncBatchWithPaddingAndRequestCost) {
  using BatchTask = BatchResourceBase::BatchTask;
  using SharedBatchScheduler = SharedBatchScheduler<BatchTask>;

  std::shared_ptr<SharedBatchScheduler> batcher;
  TF_ASSERT_OK(SharedBatchScheduler::Create({}, &batcher));

  Tensor input_tensor(DataType::DT_INT64, TensorShape({5, 2, 1}));
  std::vector<TensorValue> input_tensor_values = {TensorValue(&input_tensor)};
  SessionMetadata session_metadata;
  OpKernelContext::Params params = {
      .op_kernel = batch_kernel_.get(),
      .device = device_.get(),
      .session_metadata = &session_metadata,
      .inputs = input_tensor_values,
  };
  auto context = std::make_unique<OpKernelContext>(&params);

  // Set allowed_batch_sizes to 8, which is larger than the input tensor size 5.
  // This will trigger padding.
  tsl::core::RefCountPtr<TestBatchResourceBase> batch_resource(
      new TestBatchResourceBase(
          /*has_process_batch_function=*/true, batcher,
          SharedBatchScheduler::QueueOptions{},
          /*allowed_batch_sizes=*/{8}));

  absl::Notification task_done;
  absl::flat_hash_map<std::string, absl::Duration> costs;
  // Run the request on a dedicated thread so it gets a fresh thread_local
  // RequestCost. gUnit runs all TEST_F cases on the same main thread, so
  // without this, costs from earlier tests would accumulate in the main
  // thread's thread_local instance and pollute subsequent assertions.
  std::unique_ptr<Thread> request_thread(
      Env::Default()->StartThread(ThreadOptions(), "request_thread", [&]() {
        TF_ASSERT_OK(batch_resource->RegisterInput(
            /*guid=*/0, context.get(), /*batcher_queue_name=*/"queue",
            /*create_batch_task_fn=*/
            []() -> absl::StatusOr<std::unique_ptr<BatchTask>> {
              return std::make_unique<BatchTask>();
            },
            /*done_callback=*/[&]() { task_done.Notify(); },
            /*forced_warmup_batch_size=*/0));

        task_done.WaitForNotification();

        auto accessor = RequestCostAccessorRegistry::CreateByNameOrNull(
            "test_request_cost_accessor");
        ASSERT_NE(accessor, nullptr);
        costs = accessor->GetRequestCost()->GetCosts();
      }));

  request_thread.reset();  // Wait for thread to finish

  // With 1 task of size 5 (from input_tensor_ shape {5,2,1}),
  // batch.size() = 5, processed_size = 8 (due to allowed_batch_sizes={8}),
  // total cost = 100ms.
  //   with_smear = 100ms / 5 * 5 = 100ms
  //   no_smear   = 100ms / 8 * 5 = 62.5ms
  EXPECT_THAT(costs, UnorderedElementsAre(
                         Pair("test_tpu_with_smear", absl::Milliseconds(100)),
                         Pair("test_tpu_no_smear", absl::Milliseconds(62.5))));
}

// Verifies that the RequestCostAccessor-based injection populates costs
// during actual batch processing with splitting and padding.
TEST_F(BatchResourceCostTest, ProcessFuncBatchWithSplitTaskAndRequestCost) {
  using BatchTask = BatchResourceBase::BatchTask;
  using SharedBatchScheduler = SharedBatchScheduler<BatchTask>;

  std::shared_ptr<SharedBatchScheduler> batcher;
  TF_ASSERT_OK(SharedBatchScheduler::Create({}, &batcher));

  SharedBatchScheduler::QueueOptions queue_options;
  queue_options.enable_large_batch_splitting = true;
  // Set max_execution_batch_size to 4, which is smaller than the input tensor
  // size 6. This will trigger splitting.
  queue_options.max_execution_batch_size = 4;
  queue_options.input_batch_size_limit = 10;
  queue_options.split_input_task_func =
      [](std::unique_ptr<BatchTask>* input_task, int open_batch_remaining_slot,
         int max_batch_size,
         std::vector<std::unique_ptr<BatchTask>>* output_tasks)
      -> absl::Status {
    return BatchResourceBase::SplitInputTask(
        input_task, open_batch_remaining_slot, max_batch_size, output_tasks);
  };

  // Use a larger input tensor to trigger splitting (6 > 4).
  Tensor input_tensor(DataType::DT_INT64, TensorShape({6, 2, 1}));
  std::vector<TensorValue> input_tensor_values = {TensorValue(&input_tensor)};
  SessionMetadata session_metadata;
  OpKernelContext::Params params = {
      .op_kernel = batch_kernel_.get(),
      .device = device_.get(),
      .session_metadata = &session_metadata,
      .inputs = input_tensor_values,
  };
  auto context = std::make_unique<OpKernelContext>(&params);

  // Set allowed_batch_sizes to 4 so that the split task is padded to 4.
  tsl::core::RefCountPtr<TestBatchResourceBase> batch_resource(
      new TestBatchResourceBase(
          /*has_process_batch_function=*/true, batcher, queue_options,
          /*allowed_batch_sizes=*/{4}));

  absl::Notification task_done;
  absl::flat_hash_map<std::string, absl::Duration> costs;
  // Run the request on a dedicated thread so it gets a fresh thread_local
  // RequestCost. gUnit runs all TEST_F cases on the same main thread, so
  // without this, costs from earlier tests would accumulate in the main
  // thread's thread_local instance and pollute subsequent assertions.
  std::unique_ptr<Thread> request_thread(
      Env::Default()->StartThread(ThreadOptions(), "request_thread", [&]() {
        TF_ASSERT_OK(batch_resource->RegisterInput(
            /*guid=*/0, context.get(), /*batcher_queue_name=*/"queue",
            /*create_batch_task_fn=*/
            []() -> absl::StatusOr<std::unique_ptr<BatchTask>> {
              return std::make_unique<BatchTask>();
            },
            /*done_callback=*/[&]() { task_done.Notify(); },
            /*forced_warmup_batch_size=*/0));

        task_done.WaitForNotification();

        auto accessor = RequestCostAccessorRegistry::CreateByNameOrNull(
            "test_request_cost_accessor");
        ASSERT_NE(accessor, nullptr);
        costs = accessor->GetRequestCost()->GetCosts();
      }));

  request_thread.reset();  // Wait for thread to finish

  // With 1 task of size 6 (from input_tensor_ shape {6,2,1}),
  // 1 task is split into 2 tasks, size 4 and size 2.
  // The task of size 4 is not padded and processed as is, and the task of size
  // 2 is padded to 4.
  //   with_smear = 100ms / 4 * 4 + 100ms / 4 * 4 = 200ms
  //   no_smear   = 100ms / 4 * 4 + 100ms / 4 * 2 = 150ms
  EXPECT_THAT(costs, UnorderedElementsAre(
                         Pair("test_tpu_with_smear", absl::Milliseconds(200)),
                         Pair("test_tpu_no_smear", absl::Milliseconds(150))));
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
