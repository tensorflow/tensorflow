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

#include "xla/backends/cpu/runtime/xnnpack/xnn_threadpool.h"

#include "experimental.h"  // xnnpack

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla::cpu {

class XnnEigenScheduler : public xnn_scheduler {
 public:
  explicit XnnEigenScheduler(Eigen::ThreadPoolInterface* eigen_thread_pool);

 private:
  static int NumThreads(xnn_scheduler* self);
  static void Schedule(xnn_scheduler* self, void* context,
                       void (*task)(void* context));

  Eigen::ThreadPoolInterface* eigen_thread_pool_ = nullptr;
};

XnnEigenScheduler::XnnEigenScheduler(
    Eigen::ThreadPoolInterface* eigen_thread_pool) {
  eigen_thread_pool_ = eigen_thread_pool;
  num_threads = &NumThreads;
  schedule = &Schedule;
}

int XnnEigenScheduler::NumThreads(xnn_scheduler* self) {
  return reinterpret_cast<XnnEigenScheduler*>(self)
      ->eigen_thread_pool_->NumThreads();
}

void XnnEigenScheduler::Schedule(xnn_scheduler* self, void* context,
                                 void (*task)(void* context)) {
  reinterpret_cast<XnnEigenScheduler*>(self)->eigen_thread_pool_->Schedule(
      [task, context]() { (*task)(context); });
}

namespace {

void DestroyXnnEigenScheduler(xnn_scheduler* scheduler) {
  delete reinterpret_cast<XnnEigenScheduler*>(scheduler);
}

}  // namespace

XnnScheduler CreateXnnEigenScheduler(Eigen::ThreadPoolInterface* threads) {
  return XnnScheduler(new XnnEigenScheduler(threads), DestroyXnnEigenScheduler);
}

XnnScheduler CreateXnnEigenScheduler(const Eigen::ThreadPoolDevice* device) {
  return CreateXnnEigenScheduler(device->getPool());
}

}  // namespace xla::cpu
