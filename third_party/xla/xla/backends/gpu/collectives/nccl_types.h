/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_NCCL_TYPES_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_NCCL_TYPES_H_

#include <cstddef>

#include "absl/status/statusor.h"
#include "third_party/nccl/nccl.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

//===----------------------------------------------------------------------===//
// Conversions between XLA and NCCL data types
//===----------------------------------------------------------------------===//

size_t ToNcclCount(PrimitiveType dtype, size_t count);

absl::StatusOr<ncclDataType_t> ToNcclDataType(
    PrimitiveType dtype, bool is_reduction_op,
    stream_executor::CudaComputeCapability cc);

ncclRedOp_t ToNcclReduction(ReductionKind kind);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_NCCL_TYPES_H_
