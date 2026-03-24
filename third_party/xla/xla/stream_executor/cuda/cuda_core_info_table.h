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

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUDA_CORE_INFO_TABLE_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUDA_CORE_INFO_TABLE_H_

#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"

namespace stream_executor {
namespace gpu {

// Fills the scalar and matrix unit fields in `desc` with CUDA Core and Tensor
// Core descriptions if available for the given compute capability.
void FillExecutionUnitDesc(CudaComputeCapability cc, float base_clock_rate_ghz,
                           DeviceDescription& desc);

// Gets the number of FPUs (CUDA Cores) per SM. Assumes FP32 cores.
int GetFpusPerCore(CudaComputeCapability cc);

}  // namespace gpu
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_CORE_INFO_TABLE_H_
