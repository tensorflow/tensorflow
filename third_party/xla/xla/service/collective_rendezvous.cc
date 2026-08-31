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

#include "xla/service/collective_rendezvous.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xla/executable_run_options.h"
#include "xla/runtime/device_id.h"

namespace xla {

RendezvousKey::RendezvousKey(const RunId& run_id,
                             std::vector<GlobalDeviceId> global_devices,
                             int num_local_participants,
                             CollectiveOpKind collective_op_kind, int64_t op_id)
    : run_id(run_id),
      global_devices(std::move(global_devices)),
      num_local_participants(num_local_participants),
      collective_op_kind(collective_op_kind),
      op_id(op_id) {}

absl::string_view RendezvousKey::CollectiveOpKindString() const {
  switch (collective_op_kind) {
    case kCrossModule:
      return "cross_module";
    case kCrossReplica:
      return "cross_replica";
  }
}

std::string RendezvousKey::ToString() const {
  return absl::StrFormat(
      "RendezvousKey{run_id=%s, global_devices=[%s], "
      "num_local_participants=%d, collective_op_kind=%s, op_id=%d}",
      run_id.ToString(), absl::StrJoin(global_devices, ", "),
      num_local_participants, CollectiveOpKindString(), op_id);
}

}  // namespace xla
