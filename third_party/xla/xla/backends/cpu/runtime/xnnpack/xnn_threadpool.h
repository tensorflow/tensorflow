/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_RUNTIME_XNNPACK_XNN_THREADPOOL_H_
#define XLA_BACKENDS_CPU_RUNTIME_XNNPACK_XNN_THREADPOOL_H_

#include <atomic>
#include <cstddef>
#include <cstdint>

#include "pthreadpool.h"
#include "third_party/slinky/base/function_ref.h"
#include "third_party/slinky/base/ref_count.h"
#include "third_party/slinky/base/thread_pool.h"
#include "third_party/slinky/base/thread_pool_impl.h"
#include "xla/backends/cpu/runtime/parallel_loop_runner.h"

namespace Eigen {

class ThreadPoolInterface;

}  // namespace Eigen

struct xnn_runtime;

namespace xla::cpu {

// Creates a `pthreadpool` that uses the given `runner` to execute work.
pthreadpool_t CreateCustomPthreadpool(xla::cpu::ParallelLoopRunner* runner);

// Destroys the given `pthreadpool`.
//
// IMPORTANT: Thread pool must be created with `CreateCustomPthreadpool`.
void DestroyCustomPthreadpool(pthreadpool_t threadpool);

// Returns the parallel loop runner associated with the given `pthreadpool`.
//
// IMPORTANT: Thread pool must be created with `CreateCustomPthreadpool`.
xla::cpu::ParallelLoopRunner* GetParallelLoopRunner(pthreadpool_t threadpool);

//===----------------------------------------------------------------------===//
// Slinky Thread pool API.
//===----------------------------------------------------------------------===//

class SlinkyEigenThreadPool : public slinky::thread_pool {
 public:
  explicit SlinkyEigenThreadPool(Eigen::ThreadPoolInterface* eigen_thread_pool);
  explicit SlinkyEigenThreadPool(
      const Eigen::ThreadPoolDevice* eigen_thread_pool);

  int thread_count() const override;
  slinky::ref_count<task> enqueue(size_t n, task_body t,
                                  int32_t max_workers) override;
  void wait_for(task* t) override;
  void wait_for(predicate_ref condition) override;
  void atomic_call(slinky::function_ref<void()> t) override;

 private:
  slinky::thread_pool_impl thread_pool_;
  Eigen::ThreadPoolInterface* eigen_thread_pool_;
  std::atomic<int> worker_count_{0};
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_XNNPACK_XNN_THREADPOOL_H_
