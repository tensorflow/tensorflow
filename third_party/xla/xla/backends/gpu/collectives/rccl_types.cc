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

#include "xla/backends/gpu/collectives/rccl_types.h"

#include <cstddef>

#include "absl/status/statusor.h"
#include "rocm/include/rccl/rccl.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

size_t ToNcclCount(PrimitiveType dtype, size_t count) {
  return primitive_util::IsComplexType(dtype) ? count * 2 : count;
}

absl::StatusOr<ncclDataType_t> ToNcclDataType(
    PrimitiveType dtype, bool is_reduction_op,
    stream_executor::RocmComputeCapability rocm_cc) {
  switch (dtype) {
    case F8E5M2:
      return rocm_cc.has_ocp_fp8_support() ? ncclFloat8e5m2 : ncclInt8;
    case F8E4M3FN:
      return rocm_cc.has_ocp_fp8_support() ? ncclFloat8e4m3 : ncclInt8;
    case S8:
    case F8E5M2FNUZ:
    case F8E4M3FNUZ:
    case F8E8M0FNU:
      return ncclInt8;
    case PRED:
    case U8:
      return ncclUint8;
    case S32:
      return ncclInt32;
    case U32:
      return ncclUint32;
    case S64:
      return ncclInt64;
    case U64:
      return ncclUint64;
    case F16:
      return ncclFloat16;
    case F32:
    case C64:
      return ncclFloat32;
    case F64:
    case C128:
      return ncclFloat64;
    case S16:
    case U16:
      // For reductions we expect 16 bit integer types to be promoted to 32-bit.
      if (is_reduction_op) {
        return InvalidArgument(
            "Unsupported data type for reduction operation: %s",
            primitive_util::LowercasePrimitiveTypeName(dtype));
      }
      // For collectives that just move data around, we can use ncclFloat16 for
      // 16-bit integer data types.
      return ncclFloat16;
    case BF16:
      return ncclBfloat16;
    default:
      return InvalidArgument("Unsupported data type: %s",
                             primitive_util::LowercasePrimitiveTypeName(dtype));
  }
}

ncclRedOp_t ToNcclReduction(ReductionKind kind) {
  switch (kind) {
    case ReductionKind::SUM:
      return ncclSum;
    case ReductionKind::PRODUCT:
      return ncclProd;
    case ReductionKind::MIN:
      return ncclMin;
    case ReductionKind::MAX:
      return ncclMax;
  }
}

}  // namespace xla::gpu
