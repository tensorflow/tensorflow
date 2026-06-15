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

#include "xla/backends/gpu/collectives/rccl_group.h"

#include <cstdint>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/tsl/platform/status_macros.h"
#include "rocm/rocm_config.h"
#include "xla/backends/gpu/collectives/rccl_errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/util.h"

#if (TF_ROCM_VERSION >= 50200)
#include "rocm/include/rccl/rccl.h"
#else
#include "rocm/include/rccl.h"
#endif  // TF_ROCM_VERSION >= 50200

namespace xla::gpu {
namespace {

static thread_local int32_t rccl_group_nesting = 0;

absl::Status RcclGroupStart() {
  VLOG(5) << "Start RCCL group";
  XLA_RCCL_RETURN_IF_ERROR(ncclGroupStart());
  rccl_group_nesting++;
  return absl::OkStatus();
}

absl::StatusOr<bool> RcclGroupEnd() {
  VLOG(5) << "End RCCL group";
  if (rccl_group_nesting <= 0) {
    return Internal("RCCL group end called without a matching group start");
  }

  XLA_RCCL_RETURN_IF_ERROR(ncclGroupEnd());
  rccl_group_nesting--;
  if (rccl_group_nesting > 0) {
    // Though NCCL allows groups to be nested, no operations are actually
    // performed until the outermost group ends. The inner calls to
    // ncclGroupStart and ncclGroupEnd are effectively noops.
    //
    // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/groups.html
    return false;
  }

  return true;
}

}  // namespace

bool IsInsideRcclGroupLaunch() { return rccl_group_nesting > 0; }

absl::StatusOr<bool> RcclGroupLaunch(absl::FunctionRef<absl::Status()> group) {
  RETURN_IF_ERROR(RcclGroupStart());
  absl::Status group_status = group();
  absl::StatusOr<bool> launched = RcclGroupEnd();
  if (!group_status.ok()) {
    return group_status;
  }
  return launched;
}

}  // namespace xla::gpu
