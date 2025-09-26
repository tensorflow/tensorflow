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

#include "oneapi/dnnl/dnnl_threadpool_iface.hpp"
#include "xla/backends/cpu/runtime/work_queue.h"

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla::cpu {

class OneDnnThreadPool final
    : public dnnl::threadpool_interop::threadpool_iface {
 public:
  explicit OneDnnThreadPool(Eigen::ThreadPoolInterface* thread_pool)
      : thread_pool_(thread_pool) {}

  int get_num_threads() const final { return thread_pool_->NumThreads(); }

  bool get_in_parallel() const final {
    return thread_pool_->CurrentThreadId() >= 0;
  }

  uint64_t get_flags() const final { return 0; }

#ifdef ENABLE_ONEDNN_ASYNC
  // This is a placeholder implementation for the wait method, as we
  // need to satisfy the interface requirements of the
  // dnnl::threadpool_interop::threadpool_iface with the experimental
  // asynchronous runtime support in oneDNN.
  // TODO(intel-tf): Implement proper wait logic when thunk runtime
  // with oneDNN is enabled.
  void wait() final {}
#endif  // ENABLE_ONEDNN_ASYNC

  void parallel_for(int n, const std::function<void(int, int)>& fn) final {
    // It is perfectly safe to block here as Worker implements work stealing
    // that guarantees forward progress and deadlock freedom, even if we are
    // running in the same thread pool as the Eigen thread_pool.
    tsl::BlockUntilReady(Worker::Parallelize(thread_pool_,
                                             thread_pool_->NumThreads(), n,
                                             [fn, n](size_t i) { fn(i, n); }));
  }

  const void set_thread_pool(Eigen::ThreadPoolInterface* thread_pool) {
    thread_pool_ = thread_pool;
  }

 private:
  Eigen::ThreadPoolInterface* thread_pool_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_ONEDNN_ONEDNN_THREADPOOL_H_
