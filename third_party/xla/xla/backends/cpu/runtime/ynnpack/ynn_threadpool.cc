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

#include "xla/backends/cpu/runtime/ynnpack/ynn_threadpool.h"

#include <cstdint>

#include "ynnpack/include/ynnpack.h"
#include "absl/base/optimization.h"
#include "absl/status/statusor.h"
#include "xla/backends/cpu/runtime/ynnpack/ynn_interop.h"

#define EIGEN_USE_THREADS
#include "Eigen/ThreadPool"
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla::cpu {

static int32_t NumThreads(void* pool) {
  if (ABSL_PREDICT_FALSE(pool == nullptr)) {
    return 0;
  }
  return reinterpret_cast<Eigen::ThreadPoolInterface*>(pool)->NumThreads();
}

static void Schedule(void* pool, void* context, void (*task)(void* context)) {
  if (ABSL_PREDICT_FALSE(pool == nullptr)) {
    (*task)(context);
  }
  reinterpret_cast<Eigen::ThreadPoolInterface*>(pool)->Schedule(
      [task, context]() { (*task)(context); });
}

// An adaptor from Eigen::ThreadPoolInterface to xnn_threadpool_t.
static constexpr ynn_scheduler kYnnScheduler = {&NumThreads, &Schedule};

absl::StatusOr<YnnThreadpool> CreateYnnThreadpool(
    Eigen::ThreadPoolInterface* threadpool) {
  return CreateYnnThreadpool([&](ynn_threadpool_t* ynn_threadpool) {
    return ynn_create_threadpool(&kYnnScheduler, threadpool, /*flags=*/1,
                                 ynn_threadpool);
  });
}

absl::StatusOr<YnnThreadpool> CreateYnnThreadpool(
    const Eigen::ThreadPoolDevice* device) {
  return CreateYnnThreadpool(device->getPool());
}

}  // namespace xla::cpu
