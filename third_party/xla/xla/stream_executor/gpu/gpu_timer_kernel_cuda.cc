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

#include "xla/stream_executor/gpu/gpu_driver.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/gpu/gpu_semaphore.h"
#include "xla/stream_executor/gpu/gpu_timer_kernel.h"
#include "xla/stream_executor/gpu/gpu_timer_kernel_cuda.h"
#include "xla/stream_executor/typed_kernel_factory.h"

namespace stream_executor::gpu {
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
                                                     delay_kernel::kernel())));
  // Launch a delay kernel into this stream, which will spin until
  // GetElapsedDuration() is called, the timer is destroyed, or the timeout
  // in the kernel is reached.
  TF_RETURN_IF_ERROR(stream->ThenLaunch(ThreadDim(1, 1, 1), BlockDim(1, 1, 1),
                                        kernel, semaphore.device(),
                                        GpuSemaphoreState::kRelease));

  return semaphore;
}

absl::StatusOr<bool> DelayKernelIsSupported(GpuStream* stream) {
  // Check the assumption that this device supports unified addressing,
  // otherwise skip the delay kernel
  TF_ASSIGN_OR_RETURN(int status, GpuDriver::GetDeviceAttribute(
                                      CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING,
                                      stream->parent()->device()));
  if (!status) {
    LOG(WARNING) << "Skipping the delay kernel because the device does not "
                    "support unified addressing";
  }

  return static_cast<bool>(status);
}
}  // namespace stream_executor::gpu
