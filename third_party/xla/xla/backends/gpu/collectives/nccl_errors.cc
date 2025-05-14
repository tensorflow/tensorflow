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

#include "xla/backends/gpu/collectives/nccl_errors.h"

#include <atomic>

#include "absl/log/log.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/util.h"

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

absl::Status PollUntilDone(ncclComm_t comm, const std::atomic_bool& aborted) {
  auto poll = [](ncclComm_t comm,
                 const std::atomic_bool& aborted) -> absl::Status {
    ncclResult_t state = ncclInProgress;
    while (state == ncclInProgress && !aborted.load()) {
      XLA_NCCL_RETURN_IF_ERROR(ncclCommGetAsyncError(comm, &state));
    }
    if (aborted.load()) {
      return Cancelled("NcclCommunicator aborted");
    }
    return XLA_NCCL_STATUS(state);
  };

  if (!VLOG_IS_ON(1)) {
    return poll(comm, aborted);
  }

  absl::Time start = absl::Now();
  absl::Status s = poll(comm, aborted);
  absl::Time stop = absl::Now();
  VLOG(1) << "Polled NCCL communicator " << comm << " for " << (stop - start)
          << ": " << s;
  return s;
}

}  // namespace xla::gpu
