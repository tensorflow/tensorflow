/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/nccl_utils.h"

#include <cstdlib>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xla/primitive_util.h"
#include "xla/service/global_device_id.h"
#include "xla/service/gpu/thunk.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

#if XLA_ENABLE_XCCL
#include "third_party/nccl/nccl.h"
#endif  // XLA_ENABLE_XCCL

namespace xla::gpu {

#if XLA_ENABLE_XCCL
static absl::StatusOr<ncclDataType_t> ToNcclDataType(PrimitiveType element_type,
                                                     Thunk::Kind reduction_op) {
  switch (element_type) {
    case S8:
    case F8E5M2:
    case F8E4M3FN:
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
      // For all-reduce and reduce-scatter, we expect 16 bit integer types to be
      // promoted to 32-bit.
      if (reduction_op == Thunk::kNcclAllReduce ||
          reduction_op == Thunk::kNcclAllReduceStart ||
          reduction_op == Thunk::kNcclReduceScatter) {
        return tsl::errors::InvalidArgument(absl::StrFormat(
            "Unsupported data type: %s", PrimitiveType_Name(element_type)));
      }
      // For collectives that just move data around, we can use ncclFloat16 for
      // 16-bit integer data types.
      return ncclFloat16;
#if defined(__CUDA_BF16_TYPES_EXIST__) || TENSORFLOW_USE_ROCM
    case BF16:
      return ncclBfloat16;
#endif
    default:
      return tsl::errors::InvalidArgument(absl::StrFormat(
          "Unsupported data type: %s", PrimitiveType_Name(element_type)));
  }
}

absl::StatusOr<std::pair<ncclDataType_t, int>> ToNcclDataTypeAndCountMultiplier(
    PrimitiveType element_type, Thunk::Kind reduction_op) {
  TF_ASSIGN_OR_RETURN(ncclDataType_t dtype,
                      ToNcclDataType(element_type, reduction_op));
  bool is_complex = primitive_util::IsComplexType(element_type);
  return std::make_pair(dtype, is_complex ? 2 : 1);
}
#endif  // XLA_ENABLE_XCCL

size_t GetNumLocalParticipants(
    const std::vector<GlobalDeviceId>& participants,
    const std::vector<GlobalDeviceId>* local_devices) {
  if (local_devices == nullptr) return participants.size();

  return absl::c_count_if(participants, [&](const GlobalDeviceId& device_id) {
    return absl::c_linear_search(*local_devices, device_id);
  });
}

}  // namespace xla::gpu
