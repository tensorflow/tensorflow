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

#ifndef XLA_STREAM_EXECUTOR_CUDA_COMMAND_BUFFER_KERNELS_H_
#define XLA_STREAM_EXECUTOR_CUDA_COMMAND_BUFFER_KERNELS_H_

#include "absl/status/statusor.h"
#include "xla/stream_executor/kernel_spec.h"

namespace stream_executor::cuda {

// These are various kernels that update Gpu conditionals based on the device
// memory values, and allow implementing on-device control flow via conditional
// command buffers.
absl::StatusOr<KernelLoaderSpec> GetSetCaseConditionKernelLoaderSpec();
absl::StatusOr<KernelLoaderSpec> GetSetWhileConditionKernelLoaderSpec();
absl::StatusOr<KernelLoaderSpec> GetNoOpKernelLoaderSpec();

}  // namespace stream_executor::cuda

#endif  // XLA_STREAM_EXECUTOR_CUDA_COMMAND_BUFFER_KERNELS_H_
