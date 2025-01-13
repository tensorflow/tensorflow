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
#include "xla/backends/cpu/runtime/xnnpack/parallel_loop_runner.h"

namespace xla::cpu {

// Returns true if the custom pthreadpool is enabled.
bool IsCustomPthreadpoolEnabled();

// Returns the default per-process pthreadpool. If custom `pthreadpool` is
// enabled, it will return nullptr.
pthreadpool_t DefaultPthreadpool();

// Creates a `pthreadpool` that uses the given `runner` to execute work. If
// custom `pthreadpool` is disabled, it will kill the process.
pthreadpool_t CreateCustomPthreadpool(xla::cpu::ParallelLoopRunner* runner);

// Returns the parallel loop runner associated with the given `pthreadpool`. If
// the `pthreadpool` is not associated with a parallel loop runner, returns
// nullptr.
xla::cpu::ParallelLoopRunner* GetParallelLoopRunner(pthreadpool_t threadpool);

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_XNNPACK_XNN_THREADPOOL_H_
