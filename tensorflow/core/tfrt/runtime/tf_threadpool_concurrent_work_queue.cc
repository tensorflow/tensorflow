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

#include <utility>

#include "llvm/ADT/None.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/threadpool_interface.h"
#include "tfrt/host_context/async_value.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/task_function.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime
#include "tfrt/support/latch.h"  // from @tf_runtime

namespace tensorflow {
namespace tfrt_stub {

using ::tensorflow::thread::ThreadPoolInterface;

StatusOr<std::unique_ptr<WorkQueueInterface>>
TfThreadPoolWorkQueue::InitializeRequest(
    ::tfrt::RequestContextBuilder* request_context_builder,
    ThreadPoolInterface** intra_op_threadpool) const {
  DCHECK(intra_op_threadpool);
  *intra_op_threadpool = intra_op_threadpool_;

  return {nullptr};
}

void TfThreadPoolWorkQueue::AddTask(tfrt::TaskFunction work) {
  auto* copy = new tfrt::TaskFunction(std::move(work));
  inter_op_threadpool_->Schedule([copy] {
    (*copy)();
    delete copy;
  });
}

void TfThreadPoolWorkQueue::AddTask(const tfrt::ExecutionContext& exec_ctx,
                                    tfrt::TaskFunction work) {
  int64_t id = 0;
  if (auto* request_context = exec_ctx.request_ctx()) {
    id = request_context->id();
  }
  AddTask(tensorflow::tfrt_stub::WrapWork(id, "inter", std::move(work)));
}

llvm::Optional<tfrt::TaskFunction> TfThreadPoolWorkQueue::AddBlockingTask(
    tfrt::TaskFunction work, bool allow_queuing) {
  AddTask(std::move(work));
  return llvm::None;
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

}  // namespace tfrt_stub
}  // namespace tensorflow
