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

#ifndef XLA_BACKENDS_CPU_RUNTIME_XNNPACK_XNN_SCHEDULER_H_
#define XLA_BACKENDS_CPU_RUNTIME_XNNPACK_XNN_SCHEDULER_H_

#include "experimental.h"  // xnnpack

namespace Eigen {
struct ThreadPoolDevice;
class ThreadPoolInterface;
}  // namespace Eigen

namespace xla::cpu {

// An adaptor from Eigen thread pool to XNNPACK scheduler.
class XnnScheduler : public xnn_scheduler {
 public:
  explicit XnnScheduler(const Eigen::ThreadPoolDevice* device);
  explicit XnnScheduler(Eigen::ThreadPoolInterface* thread_pool);

  void set_thread_pool(Eigen::ThreadPoolInterface* thread_pool) {
    thread_pool_ = thread_pool;
  }

  Eigen::ThreadPoolInterface* thread_pool() const { return thread_pool_; }

 private:
  Eigen::ThreadPoolInterface* thread_pool_ = nullptr;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_XNNPACK_XNN_SCHEDULER_H_
