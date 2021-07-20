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

#include "tensorflow/core/platform/context.h"

namespace tensorflow {
namespace tfrt_stub {
namespace {

class DefaultWorkQueueWrapper final : public WorkQueueInterface {
 public:
  explicit DefaultWorkQueueWrapper(
      std::unique_ptr<tfrt::ConcurrentWorkQueue> work_queue)
      : work_queue_(std::move(work_queue)) {}
  ~DefaultWorkQueueWrapper() override = default;

 private:
  std::string name() const override { return work_queue_->name(); }

  void AddTask(tfrt::TaskFunction work) override {
    tensorflow::Context context(tensorflow::ContextKind::kThread);
    tfrt::TaskFunction wrapped_work(
        [context = std::move(context), work = std::move(work)]() mutable {
          tensorflow::WithContext wc(context);
          work();
        });

    work_queue_->AddTask(std::move(wrapped_work));
  }

  void AddTask(const tfrt::ExecutionContext& exec_ctx,
               tfrt::TaskFunction work) override {
    tensorflow::Context context(tensorflow::ContextKind::kThread);
    tfrt::TaskFunction wrapped_work(
        [context = std::move(context), work = std::move(work)]() mutable {
          tensorflow::WithContext wc(context);
          work();
        });

    work_queue_->AddTask(exec_ctx, std::move(wrapped_work));
  }

  llvm::Optional<tfrt::TaskFunction> AddBlockingTask(
      tfrt::TaskFunction work, bool allow_queuing) override {
    tensorflow::Context context(tensorflow::ContextKind::kThread);
    tfrt::TaskFunction wrapped_work(
        [context = std::move(context), work = std::move(work)]() mutable {
          tensorflow::WithContext wc(context);
          work();
        });

    return work_queue_->AddBlockingTask(std::move(wrapped_work), allow_queuing);
  }

  llvm::Optional<tfrt::TaskFunction> AddBlockingTask(
      const tfrt::ExecutionContext& exec_ctx, tfrt::TaskFunction work,
      bool allow_queuing) override {
    tensorflow::Context context(tensorflow::ContextKind::kThread);
    tfrt::TaskFunction wrapped_work(
        [context = std::move(context), work = std::move(work)]() mutable {
          tensorflow::WithContext wc(context);
          work();
        });

    return work_queue_->AddBlockingTask(exec_ctx, std::move(wrapped_work),
                                        allow_queuing);
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
  std::unique_ptr<tfrt::ConcurrentWorkQueue> work_queue_;
};

}  // namespace

std::unique_ptr<WorkQueueInterface> WrapDefaultWorkQueue(
    std::unique_ptr<tfrt::ConcurrentWorkQueue> work_queue) {
  return std::make_unique<DefaultWorkQueueWrapper>(std::move(work_queue));
}

}  // namespace tfrt_stub
}  // namespace tensorflow
