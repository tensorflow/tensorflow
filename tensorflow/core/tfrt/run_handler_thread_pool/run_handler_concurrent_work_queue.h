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
#ifndef TENSORFLOW_CORE_TFRT_RUN_HANDLER_THREAD_POOL_RUN_HANDLER_CONCURRENT_WORK_QUEUE_H_
#define TENSORFLOW_CORE_TFRT_RUN_HANDLER_THREAD_POOL_RUN_HANDLER_CONCURRENT_WORK_QUEUE_H_

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/tfrt/run_handler_thread_pool/run_handler.h"
#include "tensorflow/core/tfrt/runtime/work_queue_interface.h"
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/support/thread_environment.h"  // from @tf_runtime
#include "third_party/concurrent_work_queue/lib/blocking_work_queue.h"
#include "third_party/concurrent_work_queue/lib/non_blocking_work_queue.h"

namespace tfrt {
namespace tf {

// Concurrent Work Queue based on Run Handler thread Pool. All tasks are queued
// based on requests.
class RunHandlerThreadWorkQueue
    : public tensorflow::tfrt_stub::WorkQueueInterface {
 public:
  struct Options {
    // The number of threads used for the main thread pool.
    int num_main_threads;

    // The number of threads used for complementary thread pool.
    int num_complementary_threads;

    // Timeout for InitRequest().
    // The timeout may trigger as the work queue limits the number of concurrent
    // in-flight requests for better latency.
    int64_t init_timeout_ms;

    // The number of max concurrent handlers.
    int max_concurrent_handler = 128;

    // The number of sub thread pool configed.
    int num_sub_thread_pool = 1;

    // The number of threads in each sub thread pool. The length of the vector
    // should equal to num_sub_thread_pool.
    std::vector<int> num_threads_in_sub_thread_pool = {1};

    // The percentage of requests the first N sub thread pool handles. The
    // length of the vector should equal to num_sub_thread_pool.
    std::vector<double> sub_thread_request_percentage = {1.0};

    // Sleep time for non blocking threads if there is no pending task.
    int non_blocking_threads_sleep_time_micro_sec = 1000;

    // Max sleep time for blocking threads if there is no pending task and no
    // new task wakes up the thread.
    int blocking_threads_max_sleep_time_micro_sec = 1000;

    // If true, use adaptive waiting time.
    bool use_adaptive_waiting_time = true;

    // If true, threads won't wake itself up if there is no active requests.
    bool wait_if_no_active_request = true;

    // If true, threads will be waken up by new tasks.
    bool enable_wake_up = true;
  };

  explicit RunHandlerThreadWorkQueue(const Options& options);
  ~RunHandlerThreadWorkQueue() override = default;

  std::string name() const override {
    return tensorflow::strings::StrCat(
        "RunHandlerThreadWorkQueue C++ work queue (", options_.num_main_threads,
        " main threads, ", options_.num_complementary_threads,
        " complementary threads)");
  }

  absl::StatusOr<std::unique_ptr<tensorflow::tfrt_stub::WorkQueueInterface>>
  InitializeRequest(int64_t request_id) const override;

  int GetParallelismLevel() const override {
    return options_.num_main_threads + options_.num_complementary_threads;
  }

  void AddTask(TaskFunction work) override;

  std::optional<TaskFunction> AddBlockingTask(TaskFunction work,
                                              bool allow_queuing) override;

  void Quiesce() override;

  void Await(ArrayRef<RCReference<AsyncValue>> values) override;

  bool IsInWorkerThread() const override;

 private:
  Options options_;

  // Handler Pool.
  // Each request will require a handler from the pool, and release the handler
  // back to the pool once it is done.
  std::unique_ptr<RunHandlerPool> handler_pool_;

  // An id assigned to each request for tracing purpose.
  static std::atomic_int_fast64_t step_id_counter_;

  // QuiescingState for non_blocking_work_queue_ and blocking_work_queue_.
  std::unique_ptr<::tfrt::internal::QuiescingState> quiescing_state_;

  // Nonblocking queue used for cases without execution context.
  ::tfrt::internal::NonBlockingWorkQueue<ThreadingEnvironment>
      non_blocking_work_queue_;

  // Blocking queue used for cases without execution context.
  ::tfrt::internal::BlockingWorkQueue<ThreadingEnvironment>
      blocking_work_queue_;
};

std::ostream& operator<<(std::ostream& strm,
                         const RunHandlerThreadWorkQueue::Options& options);
}  // namespace tf
}  // namespace tfrt

#endif  // TENSORFLOW_CORE_TFRT_RUN_HANDLER_THREAD_POOL_RUN_HANDLER_CONCURRENT_WORK_QUEUE_H_
