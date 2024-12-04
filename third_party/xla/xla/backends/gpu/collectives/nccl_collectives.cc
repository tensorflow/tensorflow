/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/gpu/collectives/nccl_collectives.h"

#include <cstdlib>
#include <string_view>

#include "absl/status/statusor.h"
#include "xla/backends/gpu/collectives/nccl_errors.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/clique_key.h"
#include "xla/status_macros.h"
#include "tsl/platform/logging.h"

#if TENSORFLOW_USE_ROCM
#include "rocm/rocm_config.h"
#if (TF_ROCM_VERSION >= 50200)
#include "rocm/include/rccl/rccl.h"
#else
#include "rocm/include/rccl.h"
#endif  // TF_ROCM_VERSION >= 50200
#else
#include "third_party/nccl/nccl.h"
#endif  // TENSORFLOW_USE_ROCM

namespace xla::gpu {

absl::StatusOr<CliqueId> NcclCollectives::CreateUniqueCliqueId() const {
  VLOG(3) << "Create NCCL unique clique id";
  ncclUniqueId id;
  XLA_NCCL_RETURN_IF_ERROR(ncclGetUniqueId(&id));
  return CliqueId(std::string_view(id.internal, NCCL_UNIQUE_ID_BYTES));
}

bool NcclCollectives::IsGlobalConfig() const {
  static const char* const nccl_comm_id = std::getenv("NCCL_COMM_ID");
  return nccl_comm_id != nullptr;
}

absl::StatusOr<const NcclCollectives::CliqueIdCallback*>
NcclCollectives::GetCliqueIdCallback(const CliqueIdCallback* clique_id_callback,
                                     bool is_local) {
  if (clique_id_callback != nullptr) return clique_id_callback;

  TF_RET_CHECK(is_local || IsGlobalConfig())
      << "If non-local devices are taking part of a collective API on "
         "GPU, the clique_id_callback must be provided by the client.";

  static auto* local_callback = new CliqueIdCallback(
      [this](const CliqueKey&) { return CreateUniqueCliqueId(); });
  return local_callback;
}

}  // namespace xla::gpu
