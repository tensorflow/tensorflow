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
#include "tensorflow/core/tfrt/runtime/work_queue_interface.h"

#include <memory>
#include <utility>

#include "tfrt/host_context/execution_context.h"  // from @tf_runtime

namespace tensorflow {
namespace tfrt_stub {
namespace {

class DefaultWorkQueueWrapper : public WorkQueueInterface {
 public:
  explicit DefaultWorkQueueWrapper(
      std::unique_ptr<tfrt::ConcurrentWorkQueue> work_queue)
      : WorkQueueInterface(/*id=*/0),
        work_queue_owner_(std::move(work_queue)),
        work_queue_(work_queue_owner_.get()) {}

  DefaultWorkQueueWrapper(std::unique_ptr<tfrt::ConcurrentWorkQueue> work_queue,
                          thread::ThreadPoolInterface* intra_thread_pool)
      : WorkQueueInterface(/*id=*/0, intra_thread_pool),
        work_queue_owner_(std::move(work_queue)),
        work_queue_(work_queue_owner_.get()) {}

  DefaultWorkQueueWrapper(int64_t request_id,
                          tfrt::ConcurrentWorkQueue* work_queue,
                          thread::ThreadPoolInterface* intra_thread_pool)
      : WorkQueueInterface(request_id, intra_thread_pool),
        work_queue_(work_queue) {}

  ~DefaultWorkQueueWrapper() override = default;

 private:
  std::string name() const override { return work_queue_->name(); }

  void AddTask(tfrt::TaskFunction work) override {
    work_queue_->AddTask(WrapWork(id(), "inter", std::move(work)));
  }

  llvm::Optional<tfrt::TaskFunction> AddBlockingTask(
      tfrt::TaskFunction work, bool allow_queuing) override {
    return work_queue_->AddBlockingTask(
        WrapWork(id(), "blocking", std::move(work)), allow_queuing);
  }

  void Await(
      llvm::ArrayRef<tfrt::RCReference<tfrt::AsyncValue>> values) override {
    work_queue_->Await(values);
  }

  void Quiesce() override { work_queue_->Quiesce(); }

  int GetParallelismLevel() const override {
    return work_queue_->GetParallelismLevel();
  }

  bool IsInWorkerThread() const override {
    return work_queue_->IsInWorkerThread();
  }

  StatusOr<std::unique_ptr<WorkQueueInterface>> InitializeRequest(
      int64_t request_id) const override {
    return {std::make_unique<DefaultWorkQueueWrapper>(request_id, work_queue_,
                                                      GetIntraOpThreadPool())};
  }

 private:
  // Optionally the wrapper can own a work queue. In that case, it is stored in
  // `work_queue_owner_`.
  std::unique_ptr<tfrt::ConcurrentWorkQueue> work_queue_owner_;
  // The non-owning pointer to the underlying work queue. If `work_queue_owner_`
  // is not nullptr, then `work_queue_` is the same as `work_queue_owner_`.
  tfrt::ConcurrentWorkQueue* work_queue_ = nullptr;
};

}  // namespace

std::unique_ptr<WorkQueueInterface> WrapDefaultWorkQueue(
    std::unique_ptr<tfrt::ConcurrentWorkQueue> work_queue) {
  return std::make_unique<DefaultWorkQueueWrapper>(std::move(work_queue));
}

std::unique_ptr<WorkQueueInterface> WrapDefaultWorkQueue(
    std::unique_ptr<tfrt::ConcurrentWorkQueue> work_queue,
    thread::ThreadPoolInterface* intra_thread_pool) {
  return std::make_unique<DefaultWorkQueueWrapper>(std::move(work_queue),
                                                   intra_thread_pool);
}

}  // namespace tfrt_stub
}  // namespace tensorflow
