/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/stream_executor/sycl/sycl_solver_context.h"

#include <memory>

#include "absl/status/status.h"
#include "xla/stream_executor/gpu_solver_context.h"
#include "xla/stream_executor/platform/platform_object_registry.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/sycl/sycl_platform_id.h"
#include "xla/tsl/platform/logging.h"

namespace stream_executor {

// A stub for SyclSolverContext to enable sycl build. Proper functionality
// will be implemented soon.
absl::StatusOr<std::unique_ptr<GpuSolverContext>> SyclSolverContext::Create() {
  return absl::UnimplementedError("Unimplemented");
}

absl::Status SyclSolverContext::SetStream(Stream* stream) {
  return absl::UnimplementedError("Unimplemented");
}

SyclSolverContext::~SyclSolverContext() {
  LOG(ERROR) << "GpuSolverDestroy: "
             << absl::UnimplementedError("Unimplemented");
}

STREAM_EXECUTOR_REGISTER_OBJECT_STATICALLY(SyclSolverContextFactory,
                                           GpuSolverContextFactory,
                                           sycl::kSyclPlatformId,
                                           SyclSolverContext::Create);

}  // namespace stream_executor
