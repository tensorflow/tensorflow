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

#ifndef XLA_PJRT_THREAD_POOL_ASYNC_WORK_RUNNER_H_
#define XLA_PJRT_THREAD_POOL_ASYNC_WORK_RUNNER_H_

#include <memory>
#include <string>

#include "xla/pjrt/async_work_runner.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla {

std::unique_ptr<AsyncWorkRunner> MakeThreadPoolAsyncWorkRunner(
    tsl::thread::ThreadPool* pool);

std::unique_ptr<AsyncWorkRunner> MakeUnboundedAsyncWorkRunner(
    const std::string& name, const tsl::ThreadOptions& thread_options);

}  // namespace xla

#endif  // XLA_PJRT_THREAD_POOL_ASYNC_WORK_RUNNER_H_
