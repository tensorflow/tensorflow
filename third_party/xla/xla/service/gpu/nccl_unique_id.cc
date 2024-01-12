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

#include "xla/service/gpu/nccl_unique_id.h"

#include <cstdlib>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/service/gpu/nccl_errors.h"
#include "xla/service/gpu/nccl_types.h"
#include "xla/status_macros.h"

#ifdef XLA_ENABLE_XCCL
#include "third_party/nccl/nccl.h"
#endif  // XLA_ENABLE_XCCL

namespace xla::gpu {

bool IsGlobalNcclConfig() {
  static const char* const nccl_comm_id = std::getenv("NCCL_COMM_ID");
  return nccl_comm_id != nullptr;
}

//===----------------------------------------------------------------------===//
// NcclUniqueId
//===----------------------------------------------------------------------===//

// Creates a new NCCL unique id for local communication.
static absl::StatusOr<std::string> LocalNcclUniqueId(const NcclCliqueKey&) {
#ifdef XLA_ENABLE_XCCL
  NcclUniqueId id;
  XLA_NCCL_RETURN_IF_ERROR(ncclGetUniqueId(&id));
  return std::string(id.internal, NCCL_UNIQUE_ID_BYTES);
#endif
  return absl::InternalError("XLA compiled without NCCL support.");
}

absl::StatusOr<const NcclUniqueIdCallback*> GetNcclUniqueIdCallback(
    const NcclUniqueIdCallback* unique_id_callback, bool is_local) {
  if (unique_id_callback != nullptr) return unique_id_callback;

  TF_RET_CHECK(is_local || IsGlobalNcclConfig())
      << "If non-local devices are taking part of a collective API on "
         "GPU, the nccl_unique_id_callback must be provided by the client.";

  static auto* local_callback = new NcclUniqueIdCallback(LocalNcclUniqueId);
  return local_callback;
}

}  // namespace xla::gpu
