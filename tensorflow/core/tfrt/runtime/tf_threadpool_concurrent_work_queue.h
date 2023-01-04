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
#ifndef TENSORFLOW_CORE_TFRT_RUNTIME_TF_THREADPOOL_CONCURRENT_WORK_QUEUE_H_
#define TENSORFLOW_CORE_TFRT_RUNTIME_TF_THREADPOOL_CONCURRENT_WORK_QUEUE_H_

#include <string>

#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/threadpool_interface.h"
#include "tensorflow/core/tfrt/runtime/work_queue_interface.h"
#include "tfrt/host_context/async_value.h"  // from @tf_runtime
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/task_function.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime

namespace tensorflow {
namespace tfrt_stub {

// This class defines a work queue based on the WorkQueueInterface that uses the
// Tensorflow threadpools to execute inter-op and intra-op closures.
class TfThreadPoolWorkQueue : public WorkQueueInterface {
 public:
  TfThreadPoolWorkQueue(
      tensorflow::thread::ThreadPoolInterface* intra_op_threadpool,
      tensorflow::thread::ThreadPoolInterface* inter_op_threadpool)
      : TfThreadPoolWorkQueue(/*id=*/0, intra_op_threadpool,
                              inter_op_threadpool) {}

  TfThreadPoolWorkQueue(
      int64_t id, tensorflow::thread::ThreadPoolInterface* intra_op_threadpool,
      tensorflow::thread::ThreadPoolInterface* inter_op_threadpool)
      : WorkQueueInterface(id, intra_op_threadpool),
        intra_op_threadpool_(intra_op_threadpool),
        inter_op_threadpool_(inter_op_threadpool) {}

  StatusOr<std::unique_ptr<WorkQueueInterface>> InitializeRequest(
      int64_t request_id) const override;

  int GetParallelismLevel() const override {
    return tensorflow::port::MaxParallelism();
  }
  std::string name() const override { return "TfThreadPoolWorkQueue"; }

  void AddTask(tfrt::TaskFunction work) override;

  llvm::Optional<tfrt::TaskFunction> AddBlockingTask(
      tfrt::TaskFunction work, bool allow_queuing) override;

  void Quiesce() override;

  void Await(
      tfrt::ArrayRef<::tfrt::RCReference<::tfrt::AsyncValue>> values) override;

  bool IsInWorkerThread() const override;

 private:
  tensorflow::thread::ThreadPoolInterface* intra_op_threadpool_ = nullptr;
  tensorflow::thread::ThreadPoolInterface* inter_op_threadpool_ = nullptr;
};

// Create a default TfThreadPoolWorkQueue that is implemented by
// tensorflow::thread::ThreadPool. `num_inter_op_threads` and
// `num_intra_op_threads` must be larger than zero.
std::unique_ptr<TfThreadPoolWorkQueue> CreateDefaultTfThreadPoolWorkQueue(
    int num_inter_op_threads, int num_intra_op_threads);

}  // namespace tfrt_stub
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_RUNTIME_TF_THREADPOOL_CONCURRENT_WORK_QUEUE_H_
