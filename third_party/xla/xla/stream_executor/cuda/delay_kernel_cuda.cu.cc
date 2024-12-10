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

#include <cstddef>

#include "xla/stream_executor/cuda/delay_kernel.h"
#include "xla/stream_executor/gpu/gpu_semaphore.h"
#include "xla/stream_executor/typed_kernel_factory.h"

namespace stream_executor::gpu {
namespace {
// Wait for the value pointed to by `semaphore` to have value `target`, timing
// out after approximately `APPROX_TIMEOUT_SECONDS` seconds if that value is
// not reached. This can happen if, for example, blocking launches are enabled
// via CUDA_LAUNCH_BLOCKING=1. It can also happen if launching a kernel after
// this delay kernel causes synchronisation, e.g. because of lazy loading.
__global__ void DelayKernel(volatile GpuSemaphoreState* semaphore,
                            GpuSemaphoreState target) {
  constexpr int64_t WAIT_CYCLES{1024};
  constexpr int64_t TIMEOUT_CYCLES{200000000};  // 100ms at 2GHz
  const int64_t tstart{clock64()};
  bool target_not_reached;
  while ((target_not_reached = (*semaphore != target)) &&
         (clock64() - tstart) < TIMEOUT_CYCLES) {
    int64_t elapsed{};
    const int64_t t0{clock64()};
    do {
      elapsed = clock64() - t0;
    } while (elapsed < WAIT_CYCLES);
  }
  if (target_not_reached) {
    // We are exiting due to the timeout. Signal this back to the host so that
    // we can emit a warning, as it probably indicates suboptimal usage.
    *semaphore = GpuSemaphoreState::kTimedOut;
  }
}
}  // namespace

absl::StatusOr<GpuSemaphore> LaunchDelayKernel(Stream* stream) {
  StreamExecutor* executor = stream->parent();

  // Allocate a semaphore value that will be used to signal to the delay
  // kernel that it may exit.
  TF_ASSIGN_OR_RETURN(auto semaphore, GpuSemaphore::Create(executor));
  *semaphore = GpuSemaphoreState::kHold;
  // In principle the kernel could be loaded lazily and shared across
  // multiple GpuTimer objects.
  TF_ASSIGN_OR_RETURN(
      auto kernel,
      (TypedKernelFactory<DeviceMemory<GpuSemaphoreState>,
                          GpuSemaphoreState>::Create(executor, "DelayKernel",
                                                     reinterpret_cast<void*>(
                                                         DelayKernel))));
  // Launch a delay kernel into this stream, which will spin until
  // GetElapsedDuration() is called, the timer is destroyed, or the timeout
  // in the kernel is reached.
  TF_RETURN_IF_ERROR(stream->ThenLaunch(ThreadDim(1, 1, 1), BlockDim(1, 1, 1),
                                        kernel, semaphore.device(),
                                        GpuSemaphoreState::kRelease));

  return semaphore;
}

namespace delay_kernel {
void* kernel() { return reinterpret_cast<void*>(DelayKernel); }
}  // namespace delay_kernel

}  // namespace stream_executor::gpu
