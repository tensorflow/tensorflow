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

// CUDA/ROCm runtime library wrapper functionality.

#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCM_RUNTIME_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCM_RUNTIME_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "xla/stream_executor/rocm/rocm_driver_wrapper.h"

namespace stream_executor::gpu {

// Rocm runtime returns types defined in the stream_executor::gpu namespace, and
// they usually correspond to the driver types, as driver API is the primary
// integration API of Gpus into StreamExecutor.
class RocmRuntime {
 public:
  // Get pointer to device entry function that matches entry function `symbol`.
  //
  // WARNING: This will load all fatbins statically registered with the
  // underlying runtime into runtime modules for the current context. If no
  // context is current, the runtime will use the primary context for the
  // current device (and create it if it doesn't exist yet).
  static absl::StatusOr<hipFunction_t> GetFuncBySymbol(void* symbol);

  // Returns the Gpu Runtime version.
  static absl::StatusOr<int32_t> GetRuntimeVersion();
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_RUNTIME_H_
