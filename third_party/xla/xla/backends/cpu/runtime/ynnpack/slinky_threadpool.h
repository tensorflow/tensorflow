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

#ifndef XLA_BACKENDS_CPU_RUNTIME_YNNPACK_SLINKY_THREADPOOL_H_
#define XLA_BACKENDS_CPU_RUNTIME_YNNPACK_SLINKY_THREADPOOL_H_

#include <cstddef>
#include <cstdint>

#include "slinky/base/function_ref.h"
#include "slinky/base/ref_count.h"
#include "slinky/base/thread_pool.h"

namespace Eigen {
struct ThreadPoolDevice;
class ThreadPoolInterface;
}  // namespace Eigen

namespace xla::cpu {

// This is an implementation of slinky::thread_pool, using absl::Mutex for
// synchronization, and dispatches work to Eigen::ThreadPoolInterface.
class SlinkyThreadPool final : public slinky::thread_pool {
 public:
  explicit SlinkyThreadPool(Eigen::ThreadPoolDevice* device);
  explicit SlinkyThreadPool(Eigen::ThreadPoolInterface* threadpool);
  ~SlinkyThreadPool() final;

  SlinkyThreadPool(SlinkyThreadPool&&);
  SlinkyThreadPool& operator=(SlinkyThreadPool&&);

  slinky::ref_count<task> enqueue(size_t n, task_body t,
                                  int32_t max_workers) final;

  void wait_for(task* t) final;
  void wait_for(predicate_ref condition) final;

  void atomic_call(slinky::function_ref<void()> t) final;

  int thread_count() const final;

 private:
  class Impl;
  slinky::ref_count<Impl> impl_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_YNNPACK_SLINKY_THREADPOOL_H_
