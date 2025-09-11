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

#ifndef XLA_STREAM_EXECUTOR_CUDA_CHECKSUM_KERNEL_CUDA_H_
#define XLA_STREAM_EXECUTOR_CUDA_CHECKSUM_KERNEL_CUDA_H_

#include <cstdint>

#include "absl/status/status.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream.h"

// Launches a kernel to compute the HalfSipHash-1-3 checksum of the given input
// buffer.
//
// `key` is an arbitrary value used to initialize the hash function.
absl::Status LaunchHalfSipHash13Kernel(
    stream_executor::Stream* stream,
    stream_executor::DeviceMemory<uint8_t>* input, uint64_t key,
    stream_executor::DeviceMemory<uint64_t>* output);

#endif  // XLA_STREAM_EXECUTOR_CUDA_CHECKSUM_KERNEL_CUDA_H_
