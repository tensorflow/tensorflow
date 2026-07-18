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

#ifndef XLA_STREAM_EXECUTOR_GPU_KERNEL_SERIALIZATION_CHECK_H_
#define XLA_STREAM_EXECUTOR_GPU_KERNEL_SERIALIZATION_CHECK_H_

#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/platform.h"
namespace stream_executor::gpu {

// Verifies that a the given KernelLoaderSpec can be serialized and deserialized
// correctly for the given platform id.
// The check is best-effort and won't actually try to run or load the kernel.
// It's just verifying that the necessary information is present.
void VerifyKernelIsSerializable(const KernelLoaderSpec& kernel_spec,
                                Platform::Id platform_id);

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_KERNEL_SERIALIZATION_CHECK_H_
