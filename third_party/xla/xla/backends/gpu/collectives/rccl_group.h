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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_RCCL_GROUP_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_RCCL_GROUP_H_

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace xla::gpu {

// Executes `group` between ncclGroupStart and ncclGroupEnd. Returns true when
// this call closes the outermost RCCL group and launches actual operations.
// Returns false when this call only closes a nested group.
absl::StatusOr<bool> RcclGroupLaunch(absl::FunctionRef<absl::Status()> group);

// Returns true when the current thread is inside an RCCL group launch.
bool IsInsideRcclGroupLaunch();

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_RCCL_GROUP_H_
