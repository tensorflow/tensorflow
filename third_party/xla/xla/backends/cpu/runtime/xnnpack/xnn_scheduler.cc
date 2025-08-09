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

#include "xla/backends/cpu/runtime/xnnpack/xnn_scheduler.h"

#include <cstdint>

#include "experimental.h"  // xnnpack

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla::cpu {

static int32_t NumThreads(xnn_scheduler* self) {
  return reinterpret_cast<XnnScheduler*>(self)->thread_pool()->NumThreads();
}

static void Schedule(xnn_scheduler* self, void* context,
                     void (*task)(void* context)) {
  reinterpret_cast<XnnScheduler*>(self)->thread_pool()->Schedule(
      [task, context]() { (*task)(context); });
}

XnnScheduler::XnnScheduler(const Eigen::ThreadPoolDevice* device)
    : XnnScheduler(device->getPool()) {}

XnnScheduler::XnnScheduler(Eigen::ThreadPoolInterface* thread_pool)
    : thread_pool_(thread_pool) {
  xnn_scheduler::num_threads = &NumThreads;
  xnn_scheduler::schedule = &Schedule;
}

}  // namespace xla::cpu
