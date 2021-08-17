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
#ifndef TENSORFLOW_CORE_TFRT_RUNTIME_WORK_QUEUE_INTERFACE_H_
#define TENSORFLOW_CORE_TFRT_RUNTIME_WORK_QUEUE_INTERFACE_H_

#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/threadpool_interface.h"
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime
#include "tfrt/support/error_util.h"  // from @tf_runtime

namespace tensorflow {
namespace tfrt_stub {

// This is an intermediate interface in tensorflow for injecting thread pool
// implementation into TFRT. We can add savedmodel/tensorflow specific
// methods (eg. create an intra op thread pool) without changing TFRT core.
class WorkQueueInterface : public tfrt::ConcurrentWorkQueue {
 public:
  ~WorkQueueInterface() override = 0;

  // TODO(tfrt-devs): Use StatusOr to return error or result once StatusOr is
  // allowed generally in tensorflow.
  virtual tensorflow::Status InitializeRequest(
      tfrt::RequestContextBuilder* request_context_builder,
      thread::ThreadPoolInterface** intra_op_threadpool) const {
    *intra_op_threadpool = nullptr;
    return tensorflow::Status::OK();
  }

 private:
  // TODO(chky): The method below is not very useful right now because we have
  // to initialize tensorflow specific concepts (eg. intra op threadpool) which
  // cannot be known in TFRT core infra including ConcurrentWorkQueue. Consider
  // removing this method from base, and consider whether we can introduce
  // more interfaces in TFRT to support these tensorflow specific concepts.
  llvm::Error InitRequest(tfrt::RequestContextBuilder*) final {
    return llvm::Error::success();
  }
};

inline WorkQueueInterface::~WorkQueueInterface() = default;

// Create a WorkQueueInterface from a ConcurrentWorkQueue. The returned
// WorkQueueInterface simply delegates all its public methods to the specified
// ConcurrentWorkQueue.
std::unique_ptr<WorkQueueInterface> WrapDefaultWorkQueue(
    std::unique_ptr<tfrt::ConcurrentWorkQueue> work_queue);

}  // namespace tfrt_stub
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_RUNTIME_WORK_QUEUE_INTERFACE_H_
