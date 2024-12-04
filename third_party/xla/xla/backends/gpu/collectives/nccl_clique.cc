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

#include "xla/backends/gpu/collectives/nccl_clique.h"

#include <cstdint>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/service/gpu/runtime/nccl_api.h"
#include "tsl/platform/logging.h"

namespace xla::gpu {

std::string NcclCliqueImpl::DebugString() const {
  std::string out = absl::StrFormat(
      "clique_key: %s; fingerprint(id): %d; size: %d; communicators: ",
      clique_key().ToString(),
      clique_id().has_value() ? clique_id()->fingerprint() : 0,
      num_communicators());
  int32_t cnt = 0;
  ForEachComm([&](RankId rank, Communicator* comm) {
    if (cnt++) absl::StrAppend(&out, ", ");
    absl::StrAppendFormat(&out, "[rank=%d, comm=%p]", rank.value(), comm);
  });
  return out;
}

absl::Status NcclCliqueImpl::HealthCheck() const {
  absl::Status health_check = absl::OkStatus();
  ForEachComm([&health_check](RankId rank, Communicator* comm) {
    // TODO(b/380457503): Move error checking API to communicator base class.
    if (auto s = NcclApi::Default()->CommGetAsyncError(comm); !s.ok()) {
      LOG(ERROR) << "NCCL async error (rank " << rank << "): " << s;
      if (health_check.ok()) health_check = std::move(s);  // return first error
    }
  });
  return health_check;
}

}  // namespace xla::gpu
