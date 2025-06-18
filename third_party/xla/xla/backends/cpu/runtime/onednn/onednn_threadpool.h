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
#include "xla/backends/cpu/runtime/parallel_loop_runner.h"

namespace xla::cpu {

class OneDnnThreadPool final
    : public dnnl::threadpool_interop::threadpool_iface {
 public:
  explicit OneDnnThreadPool(ParallelLoopRunner* runner) : runner_(runner) {}

  int get_num_threads() const final;
  bool get_in_parallel() const final;
  uint64_t get_flags() const final;

  void parallel_for(int n, const std::function<void(int, int)>& fn) final;

 private:
  ParallelLoopRunner* runner_;
};

inline int OneDnnThreadPool::get_num_threads() const {
  return runner_->num_threads();
}

inline bool OneDnnThreadPool::get_in_parallel() const {
  return runner_->is_in_runner();
}

inline uint64_t OneDnnThreadPool::get_flags() const { return 0; }

inline void OneDnnThreadPool::parallel_for(
    int n, const std::function<void(int, int)>& fn) {
  runner_->Parallelize(
      ParallelLoopRunner::RangeDim{static_cast<size_t>(n)},
      [fn, n](ParallelLoopRunner::RangeIndex i) { fn(i.offset, n); });
}

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_ONEDNN_ONEDNN_THREADPOOL_H_
