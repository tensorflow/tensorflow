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

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "third_party/nccl/nccl.h"
#include "xla/backends/gpu/collectives/cancellation_token.h"
#include "xla/util.h"

namespace xla::gpu {

absl::Status PollUntilDone(ncclComm_t comm, const CancellationToken& cancel) {
  auto poll = [](ncclComm_t comm,
                 const CancellationToken& cancel) -> absl::Status {
    ncclResult_t state = ncclInProgress;
    while (state == ncclInProgress && !cancel.IsCancelled()) {
      XLA_NCCL_RETURN_IF_ERROR(ncclCommGetAsyncError(comm, &state));
    }
    if (cancel.IsCancelled()) {
      return Cancelled("NcclCommunicator cancelled");
    }
    return XLA_NCCL_STATUS(state);
  };

  if (!VLOG_IS_ON(1)) {
    return poll(comm, cancel);
  }

  absl::Time start = absl::Now();
  absl::Status s = poll(comm, cancel);
  absl::Time stop = absl::Now();
  VLOG(1) << "Polled NCCL communicator " << comm << " for " << (stop - start)
          << ": " << s;
  return s;
}

}  // namespace xla::gpu
