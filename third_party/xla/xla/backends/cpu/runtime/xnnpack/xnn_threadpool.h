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

#include "pthreadpool.h"
#include "xla/backends/cpu/runtime/parallel_loop_runner.h"

namespace Eigen {

class ThreadPoolInterface;

}  // namespace Eigen

struct xnn_scheduler;

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

// A wrapper to redirect xnn_scheduler operations to Eigen::ThreadPoolInterface.
using XnnSchedulerPtr =
    std::unique_ptr<xnn_scheduler, void (*)(xnn_scheduler*)>;
XnnSchedulerPtr CreateXnnEigenScheduler(
    Eigen::ThreadPoolInterface* eigen_thread_pool);
XnnSchedulerPtr CreateXnnEigenScheduler(
    const Eigen::ThreadPoolDevice* eigen_thread_pool);

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_XNNPACK_XNN_THREADPOOL_H_
