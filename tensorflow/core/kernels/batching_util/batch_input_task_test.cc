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

#include "tensorflow/core/kernels/batching_util/batch_input_task.h"

#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_factory.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_properties.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/batching_util/batch_resource_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace serving {
namespace internal {

template <typename TaskType>
class BatchInputTaskHandleTestAccess {
 public:
  explicit BatchInputTaskHandleTestAccess(
      BatchInputTaskHandle<TaskType>* handle)
      : handle_(handle) {}

  int split_id() const { return handle_->split_id(); }

 private:
  BatchInputTaskHandle<TaskType>* const handle_;
};

template <typename TaskType>
class BatchInputTaskTestAccess {
 public:
  explicit BatchInputTaskTestAccess(BatchInputTask<TaskType>* task)
      : task_(task) {}

  size_t size() const { return task_->size(); }

  int num_batches() const { return task_->num_batches(); }

  int num_new_batches() const { return task_->num_new_batches(); }

  int tail_batch_task_size() const { return task_->tail_batch_task_size(); }

  int head_batch_task_size() const { return task_->head_batch_task_size(); }

  int GetTaskSize(int split_id) const { return task_->GetTaskSize(split_id); }

 private:
  BatchInputTask<TaskType>* const task_;
};
}  // namespace internal
namespace {

using TensorMatrix = std::vector<std::vector<Tensor>>;

using SplitFunc = std::function<Status(
    std::unique_ptr<BatchResourceBase::BatchTask>* input_task,
    int first_output_task_size, int input_batch_size_limit,
    std::vector<std::unique_ptr<BatchResourceBase::BatchTask>>* output_tasks)>;

// Creates a tensor with the specified dtype, shape, and value.
template <typename T>
static Tensor CreateTensor(const TensorShape& input_shape,
                           gtl::ArraySlice<T> input_data) {
  Tensor tensor(DataTypeToEnum<T>::value, input_shape);
  test::FillValues<T>(&tensor, input_data);
  return tensor;
}

NodeDef CreateBatchKernelNodeDef() {
  NodeDef batch_kernel_node_def;
  NodeDefBuilder batch_function_builder("BatchTPUInput", "BatchFunction");
  batch_function_builder.Attr("max_batch_size", 128);
  batch_function_builder.Attr("num_batch_threads", 8);
  batch_function_builder.Attr("allowed_batch_sizes", {2, 4, 8});
  batch_function_builder.Attr("batch_timeout_micros", 1000);
  batch_function_builder.Attr("max_enqueued_batches", 100);
  batch_function_builder.Attr("enable_large_batch_splitting", true);

  std::vector<DataType> input_dtypes({DataType::DT_INT64, DataType::DT_INT64});
  std::vector<NodeDefBuilder::NodeOut> inputs;
  inputs.resize(2);
  inputs[0] = NodeDefBuilder::NodeOut({"n1", 0, DataType::DT_INT64});
  inputs[1] = NodeDefBuilder::NodeOut({"n2", 1, DataType::DT_INT64});
  batch_function_builder.Attr("Tin", input_dtypes);
  batch_function_builder.Input(inputs);

  batch_function_builder.Attr("Tcaptured",
                              std::vector<DataType>{DataType::DT_INT64});
  batch_function_builder.Input(std::vector<NodeDefBuilder::NodeOut>{
      NodeDefBuilder::NodeOut({"n3", 1, DataType::DT_INT64})});
  batch_function_builder.Attr("Tout",
                              std::vector<DataType>(4, DataType::DT_INT64));

  NameAttrList f;
  f.set_name("func_to_batch");
  batch_function_builder.Attr("f", f);

  TF_CHECK_OK(batch_function_builder.Finalize(&batch_kernel_node_def));

  return batch_kernel_node_def;
}

// Tests that constructor of BatchInputTask would compute following information
// correctly:
// 1) How many slots the batch-input-task will occupy after being enqueued.
// 2) The number of batches after split
//    - The number of slot in a batch is within range [1, batch_size_limit]
// 3) The number of new batches created by enqueueing this input
// 4) The number of slots used by last batch
TEST(BatchInputTask, BatchTaskSplitSize) {
  for (const auto& batch_task_param :
       {std::tuple<int /* input_size */, int /* open_batch_remaining_slot */,
                   int /* batch_size_limit */, int /* expected_num_batches */,
                   int /* expected_num_new_batches */,
                   int /* expected_head_batch_task_size */,
                   int /* expected_tail_batch_task_size */>{5, 1, 1, 5, 4, 1,
                                                            1},
        {10, 3, 4, 3, 2, 3, 3},
        {20, 5, 6, 4, 3, 5, 3},
        {30, 0, 11, 3, 3, 11, 8},
        {5, 6, 8, 1, 0, 5, 5}}) {
    const int input_size = std::get<0>(batch_task_param);
    const int open_batch_remaining_slot = std::get<1>(batch_task_param);
    const int batch_size_limit = std::get<2>(batch_task_param);
    const int expected_num_batches = std::get<3>(batch_task_param);
    const int expected_num_new_batches = std::get<4>(batch_task_param);
    const int expected_head_batch_task_size = std::get<5>(batch_task_param);
    const int expected_tail_batch_task_size = std::get<6>(batch_task_param);

    // The number of remaining slots should be smaller than or equal to
    // batch_size_limit; whearas we allow one input (of `input_size`) to span
    // over multiple batches.
    ASSERT_LE(open_batch_remaining_slot, batch_size_limit);

    auto batch_task = std::make_unique<BatchResourceBase::BatchTask>();
    batch_task->inputs.push_back(CreateTensor<int64>(
        TensorShape({input_size, 1}), std::vector<int64>(input_size, 1)));
    auto batch_input_task =
        std::make_shared<BatchInputTask<BatchResourceBase::BatchTask>>(
            std::move(batch_task), open_batch_remaining_slot, batch_size_limit,
            BatchResourceBase::SplitInputTask);

    internal::BatchInputTaskTestAccess<BatchResourceBase::BatchTask>
        test_access(batch_input_task.get());
    EXPECT_EQ(test_access.size(), input_size);
    EXPECT_EQ(test_access.num_batches(), expected_num_batches);
    EXPECT_EQ(test_access.num_new_batches(), expected_num_new_batches);
    EXPECT_EQ(test_access.head_batch_task_size(),
              expected_head_batch_task_size);
    EXPECT_EQ(test_access.tail_batch_task_size(),
              expected_tail_batch_task_size);
    for (int i = 1; i + 1 < test_access.num_batches(); i++) {
      EXPECT_EQ(test_access.GetTaskSize(i), batch_size_limit);
    }
  }
}

class BatchInputTaskTest : public ::testing::Test {
 protected:
  BatchInputTaskTest() {
    device_ = DeviceFactory::NewDevice("CPU", SessionOptions{},
                                       "/job:a/replica:0/task:0");

    Status op_kernel_creation_status;
    batch_kernel_ = CreateOpKernel(
        DEVICE_CPU, device_.get(), device_->GetAllocator(AllocatorAttributes{}),
        CreateBatchKernelNodeDef(), TF_GRAPH_DEF_VERSION,
        &op_kernel_creation_status);
    TF_CHECK_OK(op_kernel_creation_status);
    EXPECT_NE(batch_kernel_, nullptr);

    op_kernel_context_params_.device = device_.get();
    op_kernel_context_params_.op_kernel = batch_kernel_.get();
    op_kernel_context_ = std::make_unique<OpKernelContext>(
        &op_kernel_context_params_, 4 /* num outputs */);
  }

  OpKernelContext* op_kernel_context() const {
    return op_kernel_context_.get();
  }

 private:
  std::unique_ptr<Device> device_;

  std::unique_ptr<OpKernel> batch_kernel_;

  OpKernelContext::Params op_kernel_context_params_;
  std::unique_ptr<OpKernelContext> op_kernel_context_;
};

TEST_F(BatchInputTaskTest, BatchInputToSplitTasks) {
  auto batch_task = std::make_unique<BatchResourceBase::BatchTask>();

  batch_task->inputs.push_back(CreateTensor<int64>(
      TensorShape({5, 2, 1}), {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));
  batch_task->inputs.push_back(CreateTensor<int64>(
      TensorShape({5, 1, 2}), {11, 12, 13, 14, 15, 16, 17, 18, 19, 20}));

  batch_task->captured_inputs.push_back(
      CreateTensor<int64>(TensorShape{1}, {0}));

  batch_task->context = op_kernel_context();

  bool batch_task_done_callback_executed = false;
  batch_task->output = std::make_shared<TensorMatrix>();
  batch_task->status = std::make_shared<ThreadSafeStatus>();
  batch_task->done_callback = [&batch_task_done_callback_executed]() {
    batch_task_done_callback_executed = true;
  };

  auto batch_input_task =
      std::make_shared<BatchInputTask<BatchResourceBase::BatchTask>>(
          std::move(batch_task), /*open_batch_remaining_slot=*/1,
          /*batch_size_limit=*/3, BatchResourceBase::SplitInputTask);

  internal::BatchInputTaskTestAccess<BatchResourceBase::BatchTask> test_access(
      batch_input_task.get());
  ASSERT_EQ(test_access.size(), 5);

  ASSERT_EQ(test_access.num_batches(), 3);

  std::vector<
      std::unique_ptr<BatchInputTaskHandle<BatchResourceBase::BatchTask>>>
      output_tasks;
  batch_input_task->ToTaskHandles(&output_tasks);

  EXPECT_EQ(output_tasks.size(), test_access.num_batches());

  // Output tasks haven't invoked `done_callback`, so
  // `batch_task->done_callback` hasn't run yet.
  ASSERT_FALSE(batch_task_done_callback_executed);

  const std::vector<int> expected_task_sizes{1, 3, 1};
  // Call `done_callback` for each output task, so `batch_task->done_callback`
  // can be executed and batch_task_done_callback_executed will be updated.
  for (int i = 0; i < output_tasks.size(); i++) {
    // When emitting output tasks, split_id starts from zero and increases by
    // one.
    EXPECT_EQ(
        internal::BatchInputTaskHandleTestAccess<BatchResourceBase::BatchTask>(
            output_tasks[i].get())
            .split_id(),
        i);

    auto batch_task = output_tasks[i]->GetSplitTask();
    ASSERT_NE(batch_task, nullptr);
    EXPECT_EQ(batch_task->size(), expected_task_sizes[i]);
    batch_task->done_callback();

    // `GetSplitTask` returns nullptr from the 2nd call and on.
    EXPECT_EQ(output_tasks[i]->GetSplitTask(), nullptr);
  }

  // Each output task completed, so `batch_task->done_callback` ran.
  ASSERT_TRUE(batch_task_done_callback_executed);
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
