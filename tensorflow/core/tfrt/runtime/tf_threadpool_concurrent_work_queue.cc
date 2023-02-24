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
#include "tensorflow/core/tfrt/runtime/tf_threadpool_concurrent_work_queue.h"

#include <optional>
#include <utility>

#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/core/platform/threadpool_interface.h"
#include "tensorflow/core/tfrt/utils/thread_pool.h"
#include "tfrt/host_context/async_value.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/task_function.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime
#include "tfrt/support/latch.h"  // from @tf_runtime

namespace tensorflow {
namespace tfrt_stub {

using ::tensorflow::thread::ThreadPoolInterface;

StatusOr<std::unique_ptr<WorkQueueInterface>>
TfThreadPoolWorkQueue::InitializeRequest(int64_t request_id) const {
  return {std::make_unique<TfThreadPoolWorkQueue>(
      request_id, intra_op_threadpool_, inter_op_threadpool_)};
}

void TfThreadPoolWorkQueue::AddTask(tfrt::TaskFunction work) {
  auto* copy = new tfrt::TaskFunction(
      tensorflow::tfrt_stub::WrapWork(id(), "inter", std::move(work)));
  inter_op_threadpool_->Schedule([copy] {
    (*copy)();
    delete copy;
  });
}

llvm::Optional<tfrt::TaskFunction> TfThreadPoolWorkQueue::AddBlockingTask(
    tfrt::TaskFunction work, bool allow_queuing) {
  AddTask(std::move(work));
  return std::nullopt;
}

void TfThreadPoolWorkQueue::Quiesce() {
  // TODO(b/186668821): implement this
  CHECK(false);  // Crash OK
}

// From
// third_party/tensorflow/core/tfrt/run_handler_thread_pool/run_handler_concurrent_work_queue.cc
void TfThreadPoolWorkQueue::Await(
    tfrt::ArrayRef<tfrt::RCReference<tfrt::AsyncValue>> values) {
  // We are done when values_remaining drops to zero.
  tfrt::latch values_remaining(values.size());

  // As each value becomes available, we decrement the count.
  for (auto& value : values) {
    value->AndThen([&values_remaining]() { values_remaining.count_down(); });
  }

  // Wait until all values are resolved.
  values_remaining.wait();
}

bool TfThreadPoolWorkQueue::IsInWorkerThread() const {
  // TODO(b/192247530): Check if we have cases it is not true.
  return true;
}

std::unique_ptr<TfThreadPoolWorkQueue> CreateDefaultTfThreadPoolWorkQueue(
    int num_inter_op_threads, int num_intra_op_threads) {
  struct ThreadPools {
    TfThreadPool inter_op_threadpool;
    TfThreadPool intra_op_threadpool;

    ThreadPools(int num_inter_op_threads, int num_intra_op_threads)
        : inter_op_threadpool("default_work_queue_inter", num_inter_op_threads),
          intra_op_threadpool("default_work_queue_intra",
                              num_intra_op_threads) {}
  };

  class Wrapper : public TfThreadPoolWorkQueue {
   public:
    explicit Wrapper(std::unique_ptr<ThreadPools> thread_pools)
        : TfThreadPoolWorkQueue(
              /*intra_op_threadpool=*/&thread_pools->intra_op_threadpool,
              /*inter_op_threadpool=*/&thread_pools->inter_op_threadpool),
          thread_pools_(std::move(thread_pools)) {}

    ~Wrapper() override = default;

   private:
    std::unique_ptr<ThreadPools> thread_pools_;
  };

  return std::make_unique<Wrapper>(std::make_unique<ThreadPools>(
      num_inter_op_threads, num_intra_op_threads));
}

}  // namespace tfrt_stub
}  // namespace tensorflow
