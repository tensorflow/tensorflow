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

#ifndef XLA_STREAM_EXECUTOR_CUDA_NVSHMEM_H_
#define XLA_STREAM_EXECUTOR_CUDA_NVSHMEM_H_

#include <cstddef>
#include <memory>

#include "absl/status/status.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/runtime/process_id.h"

namespace stream_executor::gpu::nvshmem {

// Set environment information for NVSHMEM library.
void SetEnvInfo(xla::ProcessId process_id, size_t num_processes,
                size_t device_count_per_process,
                std::weak_ptr<xla::KeyValueStoreInterface> kv_store);

// Returns true if NVSHMEM library is initialized.
bool IsInitialized();

// Initializes NVSHMEM library once per process.
absl::Status InitializeOnce();

// Finalizes NVSHMEM library
void Finalize();

}  // namespace stream_executor::gpu::nvshmem

#endif  // XLA_STREAM_EXECUTOR_CUDA_NVSHMEM_H_
