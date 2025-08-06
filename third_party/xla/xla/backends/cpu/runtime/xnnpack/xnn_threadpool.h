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

#include <memory>

struct xnn_scheduler;

namespace Eigen {
struct ThreadPoolDevice;
class ThreadPoolInterface;
}  // namespace Eigen

namespace xla::cpu {

// A wrapper to redirect xnn_scheduler operations to Eigen::ThreadPoolInterface.
using XnnScheduler = std::unique_ptr<xnn_scheduler, void (*)(xnn_scheduler*)>;

// Creates an XnnScheduler that uses the given Eigen thread pool to launch tasks
// submitted by the XNNPACK.
XnnScheduler CreateXnnEigenScheduler(Eigen::ThreadPoolInterface* threads);
XnnScheduler CreateXnnEigenScheduler(const Eigen::ThreadPoolDevice* device);

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_XNNPACK_XNN_THREADPOOL_H_
