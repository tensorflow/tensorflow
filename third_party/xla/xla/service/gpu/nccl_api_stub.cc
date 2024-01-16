/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/service/gpu/nccl_api.h"
#include "xla/service/gpu/nccl_clique_key.h"

namespace xla::gpu {

absl::StatusOr<NcclCliqueId> NcclApi::GetUniqueId() {
  return absl::UnimplementedError("XLA compiled without NCCL support");
}

absl::StatusOr<NcclCommHandle> NcclApi::CommInitRank(int32_t,
                                                     const NcclCliqueId&,
                                                     int32_t) {
  return absl::UnimplementedError("XLA compiled without NCCL support");
}

absl::Status NcclApi::CommAbort(NcclCommHandle comm) {
  return absl::UnimplementedError("XLA compiled without NCCL support");
}

absl::Status NcclApi::CommGetAsyncError(NcclCommHandle comm) {
  return absl::UnimplementedError("XLA compiled without NCCL support");
}

}  // namespace xla::gpu
