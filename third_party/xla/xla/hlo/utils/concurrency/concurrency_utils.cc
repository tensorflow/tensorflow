/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/hlo/utils/concurrency/concurrency_utils.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>

#include "absl/base/no_destructor.h"
#include "absl/strings/numbers.h"
#include "xla/tsl/concurrency/executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"
#include "tsl/platform/cpu_info.h"

namespace xla::concurrency {

static int32_t DefaultThreadPoolSize() {
  // Google's CI system exposes an environment variable NPROC that describes
  // a CPU reservation for tests.
  const char* nproc_str = std::getenv("NPROC");
  if (int32_t nproc = 0; nproc_str && absl::SimpleAtoi(nproc_str, &nproc)) {
    return std::max(0, nproc);
  }
  return tsl::port::MaxParallelism();
}

tsl::Executor& DefaultExecutor() {
  static absl::NoDestructor<tsl::thread::ThreadPool> pool(
      tsl::Env::Default(), "xla-concurrency", DefaultThreadPoolSize());
  return *pool->AsExecutor();
}

}  // namespace xla::concurrency
