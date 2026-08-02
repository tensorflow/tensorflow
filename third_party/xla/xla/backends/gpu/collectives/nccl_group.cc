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

#include "xla/backends/gpu/collectives/nccl_group.h"

#include <cstdint>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/tsl/platform/status_macros.h"
#include "third_party/nccl/nccl.h"
#include "xla/backends/gpu/collectives/nccl_errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/util.h"

namespace xla::gpu {
namespace {

static thread_local int32_t nccl_group_nesting = 0;

absl::Status NcclGroupStart() {
  VLOG(5) << "Start NCCL group";
  XLA_NCCL_RETURN_IF_ERROR(ncclGroupStart());
  nccl_group_nesting++;
  return absl::OkStatus();
}

absl::StatusOr<bool> NcclGroupEnd() {
  VLOG(5) << "End NCCL group";
  if (nccl_group_nesting <= 0) {
    return Internal("NCCL group end called without a matching group start");
  }

  XLA_NCCL_RETURN_IF_ERROR(ncclGroupEnd());
  nccl_group_nesting--;
  if (nccl_group_nesting > 0) {
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

bool IsInsideNcclGroupLaunch() { return nccl_group_nesting > 0; }

absl::StatusOr<bool> NcclGroupLaunch(absl::FunctionRef<absl::Status()> group) {
  RETURN_IF_ERROR(NcclGroupStart());
  absl::Status group_status = group();
  absl::StatusOr<bool> launched = NcclGroupEnd();
  if (!group_status.ok()) {
    return group_status;
  }
  return launched;
}

}  // namespace xla::gpu
