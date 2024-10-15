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

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUDA_RUNTIME_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUDA_RUNTIME_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "third_party/gpus/cuda/include/cuda.h"

namespace stream_executor::gpu {

//===----------------------------------------------------------------------===//
//                         !!!!!  W A R N I N G !!!!!
//===----------------------------------------------------------------------===//
//
// We should not be using GPU runtime API inside StreamExecutor or XLA as
// it comes with a lot of extra complexity of context management. We use only
// one GPU (CUDA) runtime API that is not available in the driver API to
// simplify in-process GPU (CUDA) kernels integration with StreamExecutor.
//
// Do not add new APIs to GpuRuntime if they have alternatives in the driver
// API.
//
//===----------------------------------------------------------------------===//

// Cuda runtime returns types defined in the stream_executor::gpu namespace, and
// they usually correspond to the driver types, as driver API is the primary
// integration API of Gpus into StreamExecutor.
class CudaRuntime {
 public:
  // Get pointer to device entry function that matches entry function `symbol`.
  //
  // WARNING: This will load all fatbins statically registered with the
  // underlying runtime into runtime modules for the current context. If no
  // context is current, the runtime will use the primary context for the
  // current device (and create it if it doesn't exist yet).
  //
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DRIVER.html#group__CUDART__DRIVER_1gaba6f8d01e745f0c8d8776ceb18be617
  static absl::StatusOr<CUfunction> GetFuncBySymbol(void* symbol);

  // Returns the Gpu Runtime version.
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART____VERSION.html#group__CUDART____VERSION_1g0e3952c7802fd730432180f1f4a6cdc6
  static absl::StatusOr<int32_t> GetRuntimeVersion();
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_RUNTIME_H_
