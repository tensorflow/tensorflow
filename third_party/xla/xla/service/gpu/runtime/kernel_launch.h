/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_GPU_RUNTIME_KERNEL_LAUNCH_H_
#define XLA_SERVICE_GPU_RUNTIME_KERNEL_LAUNCH_H_

#include <memory>
#include <string_view>
#include <tuple>

#include "absl/container/node_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "xla/runtime/custom_call_registry.h"
#include "xla/runtime/state.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {

// Registers XLA Gpu runtime kernel launch custom calls.
void RegisterKernelLaunchCustomCalls(
    runtime::DirectCustomCallRegistry& registry);

// Kernels loaded by Gpu executable for a single stream executor.
class StreamExecutorKernels
    : public runtime::StateVector<std::unique_ptr<se::Kernel>> {};

// Xla runtime Gpu executable owns the pre-compiled device module (PTX and
// Cubin for Nvidia Gpus) for all device kernels, and the cache keeps a mapping
// from stream executor to pre-loaded kernels
class GpuExecutableKernels {
 public:
  StreamExecutorKernels* operator()(se::StreamExecutor* executor);

 private:
  mutable absl::Mutex mutex_;
  absl::node_hash_map<se::StreamExecutor*, StreamExecutorKernels> kernels_
      ABSL_GUARDED_BY(mutex_);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_RUNTIME_KERNEL_LAUNCH_H_
