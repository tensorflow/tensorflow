/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_RUNTIME_ONEDNN_ONEDNN_THREADPOOL_H_
#define XLA_BACKENDS_CPU_RUNTIME_ONEDNN_ONEDNN_THREADPOOL_H_

#include <cstddef>
#include <cstdint>
#include <functional>

#include "dnnl_threadpool.hpp"
#include "oneapi/dnnl/dnnl_threadpool_iface.hpp"
#include "xla/backends/cpu/runtime/work_queue.h"

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla::cpu {

static tsl::AsyncValueRef<tsl::Chain> OkDoneEventSingleton() {
  static tsl::AsyncValueOwningRef<tsl::Chain>* singleton = [] {
    auto* storage = new tsl::internal::AsyncValueStorage<tsl::Chain>();
    return new tsl::AsyncValueOwningRef<tsl::Chain>(
        tsl::MakeAvailableAsyncValueRef<tsl::Chain>(*storage));
  }();
  return singleton->AsRef();
}

class OneDnnThreadPool final
    : public dnnl::threadpool_interop::threadpool_iface {
 public:
  explicit OneDnnThreadPool(Eigen::ThreadPoolInterface* thread_pool,
                            bool is_async = false)
      : thread_pool_(thread_pool), is_async_(is_async) {
    if (is_async_) {
      done_event_ = OkDoneEventSingleton();
      dnnl_threadpool_interop_set_max_concurrency(thread_pool_->NumThreads());
    }
  }

  int get_num_threads() const final { return thread_pool_->NumThreads(); }

  bool get_in_parallel() const final {
    if (is_async_) {
      // TODO(intel-tf): this is a temporary fix without which oneDNN runs
      // single-threaded.
      return false;
    }
    return thread_pool_->CurrentThreadId() >= 0;
  }

  uint64_t get_flags() const final { return is_async_ ? ASYNCHRONOUS : 0; }

#ifdef ENABLE_ONEDNN_ASYNC
  // The wait() method only exists with oneDNN's experimental support for
  // asynchronous execution determined by the ENABLE_ONEDNN_ASYNC.
  void wait() override {
    if (is_async_) {
      // While performing asynchronous execution, wait() method is needed to
      // notify the user that the output is ready. oneDNN will not call wait()
      // inside the library to avoid deadlock.
      tsl::BlockUntilReady(done_event_);
    }
  }
#endif  // ENABLE_ONEDNN_ASYNC

  void parallel_for(int n, const std::function<void(int, int)>& fn) final {
    if (is_async_) {
      // If we are using oneDNN with async support, we need to schedule the
      // parallel loop using the done_event_. This allows us to return
      // immediately and not block the caller thread.
      auto parallelize = [this, n, fn](tsl::Chain) {
        return Worker::Parallelize(
            thread_pool_, thread_pool_->NumThreads(), n,
            [fn, n](size_t i) { fn(static_cast<int>(i), n); });
      };

      done_event_ = done_event_.FlatMap(parallelize);
      return;
    }

    // If we are not using oneDNN with async support, it is perfectly safe to
    // block here as Worker implements work stealing that guarantees forward
    // progress and deadlock freedom, even if we are running in the same thread
    // pool as the Eigen thread_pool.
    tsl::BlockUntilReady(Worker::Parallelize(thread_pool_,
                                             thread_pool_->NumThreads(), n,
                                             [fn, n](size_t i) { fn(i, n); }));
  }

  const void set_thread_pool(Eigen::ThreadPoolInterface* thread_pool) {
    thread_pool_ = thread_pool;
  }

  tsl::AsyncValueRef<tsl::Chain> done_event() const { return done_event_; }

 private:
  Eigen::ThreadPoolInterface* thread_pool_;

  // Indicates if we are using oneDNN with async support. TODO(intel-tf): Remove
  // this flag when oneDNN supports asynchronous execution by default.
  bool is_async_ = false;

  // Async value that signals completion of the last scheduled parallel loop.
  // This is used only when is_async_ is true.
  tsl::AsyncValueRef<tsl::Chain> done_event_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_ONEDNN_ONEDNN_THREADPOOL_H_
