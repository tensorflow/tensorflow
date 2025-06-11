/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_GPU_GPU_TEST_KERNELS_H_
#define XLA_STREAM_EXECUTOR_GPU_GPU_TEST_KERNELS_H_

#include "absl/status/statusor.h"
#include "xla/stream_executor/gpu/gpu_test_kernel_traits.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::gpu {

// This is a collection of gpu kernels for writing simple StreamExecutor tests.
//
// Some of the kernels available as pre-compiled PTX blobs (can be loaded with
// CUDA driver API), and
// some of the kernels are written directly in CUDA C++ and can be loaded from a
// symbol pointer (to test StreamExecutor CUDA runtime integration).

absl::StatusOr<internal::AddI32Kernel::KernelType> LoadAddI32TestKernel(
    StreamExecutor* executor);

absl::StatusOr<internal::MulI32Kernel::KernelType> LoadMulI32TestKernel(
    StreamExecutor* executor);

absl::StatusOr<internal::IncAndCmpKernel::KernelType> LoadCmpAndIncTestKernel(
    StreamExecutor* executor);

absl::StatusOr<internal::AddI32Ptrs3Kernel::KernelType>
LoadAddI32Ptrs3TestKernel(StreamExecutor* executor);

absl::StatusOr<internal::CopyKernel::KernelType> LoadCopyTestKernel(
    StreamExecutor* executor);

absl::StatusOr<MultiKernelLoaderSpec> GetAddI32TestKernelSpec(
    Platform::Id platform_id);

// Returns a PTX kernel loader spec for the `AddI32` PTX kernel above.
MultiKernelLoaderSpec GetAddI32PtxKernelSpec();

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_GPU_TEST_KERNELS_H_
