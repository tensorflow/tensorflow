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

#include <cassert>

#include "ynnpack/include/ynnpack.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/backends/cpu/runtime/ynnpack/slinky_threadpool.h"
#include "xla/backends/cpu/runtime/ynnpack/ynn_interop.h"

#define EIGEN_USE_THREADS
#include "Eigen/ThreadPool"
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla::cpu {

absl::StatusOr<YnnThreadpool> CreateYnnThreadpool(
    Eigen::ThreadPoolInterface* threadpool) {
  return CreateYnnThreadpool([&](ynn_threadpool_t* ynn_threadpool) {
    *ynn_threadpool =
        reinterpret_cast<ynn_threadpool_t>(new SlinkyThreadPool(threadpool));
    return ynn_status_success;
  });
}

absl::StatusOr<YnnThreadpool> CreateYnnThreadpool(
    const Eigen::ThreadPoolDevice* device) {
  return CreateYnnThreadpool(device->getPool());
}

}  // namespace xla::cpu
