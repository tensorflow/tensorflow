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

#ifndef XLA_SERVICE_GPU_NCCL_UTILS_H_
#define XLA_SERVICE_GPU_NCCL_UTILS_H_

#include <cstddef>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "xla/service/global_device_id.h"
#include "xla/service/gpu/nccl_clique.h"  // IWYU pragma: export
#include "xla/service/gpu/thunk.h"
#include "xla/xla_data.pb.h"

#if XLA_ENABLE_XCCL
#include "third_party/nccl/nccl.h"
#endif  // XLA_ENABLE_XCCL

namespace xla {
namespace gpu {

#if XLA_ENABLE_XCCL
absl::StatusOr<std::pair<ncclDataType_t, int>> ToNcclDataTypeAndCountMultiplier(
    PrimitiveType element_type, Thunk::Kind reduction_op);
#endif  // XLA_ENABLE_XCCL

size_t GetNumLocalParticipants(
    const std::vector<GlobalDeviceId>& participants,
    const std::vector<GlobalDeviceId>* local_devices);  // may be null

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_NCCL_UTILS_H_
