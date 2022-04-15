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
#ifndef TENSORFLOW_CORE_TFRT_EAGER_TFRT_CONTEXT_H_
#define TENSORFLOW_CORE_TFRT_EAGER_TFRT_CONTEXT_H_

#include <functional>
#include <utility>

#include "tensorflow/c/eager/immediate_execution_context.h"
#include "tensorflow/core/platform/threadpool_interface.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/tfrt/runtime/tf_threadpool_concurrent_work_queue.h"
#include "tfrt/host_context/resource_context.h"  // from @tf_runtime

namespace tensorflow {
class EagerContext;
class DynamicDeviceMgr;
}
namespace tfrt {
class HostContext;
class CoreRuntime;
class OpHandler;

namespace tf {

// Wraps an `Eigen::ThreadPoolInterface` as a
// `tensorflow::thread::ThreadPoolInterface`.
//
// Copied from internal directory: http://shortn/_jsmzLpQu7q
class ThreadPoolInterfaceWrapper
    : public tensorflow::thread::ThreadPoolInterface {
 public:
  explicit ThreadPoolInterfaceWrapper(Eigen::ThreadPoolInterface* thread_pool)
      : thread_pool_{thread_pool} {
    DCHECK(thread_pool);
  }

  void Schedule(std::function<void()> fn) override {
    return thread_pool().Schedule(std::move(fn));
  }

  void ScheduleWithHint(std::function<void()> fn, int start, int end) override {
    return thread_pool().ScheduleWithHint(std::move(fn), start, end);
  }

  void Cancel() override { thread_pool().Cancel(); }

  int NumThreads() const override { return thread_pool().NumThreads(); }

  int CurrentThreadId() const override {
    return thread_pool().CurrentThreadId();
  }

 private:
  Eigen::ThreadPoolInterface& thread_pool() const {
    DCHECK(thread_pool_);
    return *thread_pool_;
  }

  // Not owning pointer to the thread pool.
  Eigen::ThreadPoolInterface* thread_pool_ = nullptr;
};

// This class defines a list of objects needed to support execution with TFRT.
class TfrtContext {
 public:
  TfrtContext(
      const tensorflow::SessionOptions& opts,
      tensorflow::ContextDevicePlacementPolicy default_device_placement_policy,
      bool is_async);
  ~TfrtContext();

  HostContext* GetHostContext() { return host_context_; }
  CoreRuntime* GetCoreRuntime() { return corert_.get(); }
  tensorflow::EagerContext* GetEagerContext() { return eager_context_; }
  const tensorflow::EagerContext* GetEagerContext() const {
    return eager_context_;
  }
  OpHandler* GetFallbackOpHandler() { return fallback_op_handler_; }

  ResourceContext* GetResourceContext() { return &resource_context_; }

  tensorflow::tfrt_stub::TfThreadPoolWorkQueue* GetTfThreadPoolWorkQueue() {
    return tf_thread_pool_work_queue_.get();
  }

  const tensorflow::DeviceNameUtils::ParsedName& HostCPUParsedName() const;

  bool IsAsync() const;

 private:
  std::unique_ptr<CoreRuntime> corert_;
  ::tfrt::HostContext* host_context_;
  OpHandler* fallback_op_handler_;
  ResourceContext resource_context_;
  tensorflow::EagerContext* eager_context_;
  std::unique_ptr<ThreadPoolInterfaceWrapper> eager_ctx_thread_pool_;

  // Manage the local thread pool's lifetime because the wrapper does not own
  // the thread pool.
  std::unique_ptr<tensorflow::thread::ThreadPool> local_thread_pool_;
  std::unique_ptr<ThreadPoolInterfaceWrapper> local_thread_pool_wrapper_;
  std::unique_ptr<tensorflow::tfrt_stub::TfThreadPoolWorkQueue>
      tf_thread_pool_work_queue_;
};

}  // namespace tf
}  // namespace tfrt

#endif  // TENSORFLOW_CORE_TFRT_EAGER_TFRT_CONTEXT_H_
