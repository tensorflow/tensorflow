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

#include <utility>

#include "tfrt/host_context/execution_context.h"  // from @tf_runtime

namespace tensorflow {
namespace tfrt_stub {
namespace {

class DefaultWorkQueueWrapperBase : public WorkQueueInterface {
 public:
  explicit DefaultWorkQueueWrapperBase(int64_t id,
                                       tfrt::ConcurrentWorkQueue* work_queue)
      : id_(id), work_queue_(work_queue) {}

  ~DefaultWorkQueueWrapperBase() override = default;

 private:
  std::string name() const override { return work_queue_->name(); }

  void AddTask(tfrt::TaskFunction work) override {
    work_queue_->AddTask(WrapWork(id_, "inter", std::move(work)));
  }

  llvm::Optional<tfrt::TaskFunction> AddBlockingTask(
      tfrt::TaskFunction work, bool allow_queuing) override {
    return work_queue_->AddBlockingTask(
        WrapWork(id_, "blocking", std::move(work)), allow_queuing);
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

 private:
  int64_t id_ = 0;
  tfrt::ConcurrentWorkQueue* work_queue_ = nullptr;
};

class DefaultWorkQueueWrapper final : public DefaultWorkQueueWrapperBase {
 public:
  explicit DefaultWorkQueueWrapper(
      std::unique_ptr<tfrt::ConcurrentWorkQueue> work_queue)
      : DefaultWorkQueueWrapperBase(/*id=*/0, work_queue.get()),
        work_queue_(std::move(work_queue)) {}

  ~DefaultWorkQueueWrapper() override = default;

  DefaultWorkQueueWrapper(std::unique_ptr<tfrt::ConcurrentWorkQueue> work_queue,
                          thread::ThreadPoolInterface* intra_thread_pool)
      : DefaultWorkQueueWrapperBase(/*id=*/0, work_queue.get()),
        work_queue_(std::move(work_queue)),
        intra_thread_pool_(intra_thread_pool) {}

  StatusOr<std::unique_ptr<WorkQueueInterface>> InitializeRequest(
      tfrt::RequestContextBuilder* request_context_builder,
      thread::ThreadPoolInterface** intra_op_threadpool) const override {
    *intra_op_threadpool = intra_thread_pool_;

    int64_t id = 0;
    if (request_context_builder) {
      id = request_context_builder->id();
    }

    return {
        std::make_unique<DefaultWorkQueueWrapperBase>(id, work_queue_.get())};
  }

 private:
  std::unique_ptr<tfrt::ConcurrentWorkQueue> work_queue_;
  tensorflow::thread::ThreadPoolInterface* intra_thread_pool_ = nullptr;
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
