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
#include "tensorflow/core/tfrt/run_handler_thread_pool/run_handler_concurrent_work_queue.h"

#include <memory>

#include "tensorflow/core/tfrt/run_handler_thread_pool/run_handler.h"
#include "tfrt/host_context/async_dispatch.h"  // from @tf_runtime
#include "tfrt/host_context/async_value.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/support/error_util.h"  // from @tf_runtime

namespace tfrt {
namespace tf {

RunHandlerThreadWorkQueue::RunHandlerThreadWorkQueue(const Options& options)
    : options_(options),
      quiescing_state_(std::make_unique<::tfrt::internal::QuiescingState>()),
      non_blocking_work_queue_(quiescing_state_.get(),
                               /*num_threads=*/1),
      blocking_work_queue_(quiescing_state_.get(),
                           /*num_threads=*/1) {
  CHECK(options.num_threads_in_sub_thread_pool.size() ==  // Crash OK.
        options.num_sub_thread_pool);
  CHECK(options.sub_thread_request_percentage.size() ==  // Crash OK.
        options.num_sub_thread_pool);

  RunHandlerPool::Options pool_options;
  pool_options.num_inter_op_threads = options.num_main_threads;
  pool_options.num_intra_op_threads = options.num_complementary_threads;
  pool_options.max_concurrent_handler = options.max_concurrent_handler;
  pool_options.blocking_threads_max_sleep_time_micro_sec =
      options.blocking_threads_max_sleep_time_micro_sec;
  pool_options.non_blocking_threads_sleep_time_micro_sec =
      options.non_blocking_threads_sleep_time_micro_sec;
  pool_options.num_sub_thread_pool = options.num_sub_thread_pool;
  pool_options.num_threads_in_sub_thread_pool =
      options.num_threads_in_sub_thread_pool;
  pool_options.sub_thread_request_percentage =
      options.sub_thread_request_percentage;
  pool_options.enable_wake_up = options.enable_wake_up;
  pool_options.wait_if_no_active_request = options.wait_if_no_active_request;
  pool_options.use_adaptive_waiting_time = options.use_adaptive_waiting_time;
  handler_pool_ = absl::make_unique<RunHandlerPool>(pool_options);
}

tensorflow::StatusOr<std::unique_ptr<tensorflow::tfrt_stub::WorkQueueInterface>>
RunHandlerThreadWorkQueue::InitializeRequest(
    tfrt::RequestContextBuilder* request_context_builder,
    tensorflow::thread::ThreadPoolInterface** intra_op_threadpool) const {
  DCHECK(intra_op_threadpool);
  RunHandlerOptions options;
  options.priority = request_context_builder->request_options().priority;
  std::unique_ptr<RunHandler> handler = handler_pool_->Get(
      request_context_builder->id(), options_.init_timeout_ms, options);
  if (!handler) {
    return tensorflow::errors::Internal(absl::StrCat(
        "Could not obtain RunHandler for request after waiting for ",
        options_.init_timeout_ms, " ms."));
  }

  *intra_op_threadpool = handler->AsIntraThreadPoolInterface();

  return {std::make_unique<RunHandlerWorkQueue>(std::move(handler))};
}

void RunHandlerThreadWorkQueue::AddTask(TaskFunction work) {
  non_blocking_work_queue_.AddTask(std::move(work));
}

Optional<TaskFunction> RunHandlerThreadWorkQueue::AddBlockingTask(
    TaskFunction work, bool allow_queuing) {
  if (allow_queuing) {
    return blocking_work_queue_.EnqueueBlockingTask(std::move(work));
  } else {
    return blocking_work_queue_.RunBlockingTask(std::move(work));
  }
  return llvm::None;
}

void RunHandlerThreadWorkQueue::Quiesce() {
  handler_pool_->Quiesce();
  non_blocking_work_queue_.Quiesce();
  blocking_work_queue_.Quiesce();
}

void RunHandlerThreadWorkQueue::Await(
    ArrayRef<RCReference<AsyncValue>> values) {
  tfrt::Await(values);
}

bool RunHandlerThreadWorkQueue::IsInWorkerThread() const {
  // TODO(b/192247530): Check if we have cases it is not true.
  return true;
}

}  // namespace tf
}  // namespace tfrt
