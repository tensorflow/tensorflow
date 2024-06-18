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

#ifndef XLA_SERVICE_CPU_RUNTIME_COLLECTIVE_THUNK_H_
#define XLA_SERVICE_CPU_RUNTIME_COLLECTIVE_THUNK_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/status/statusor.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/cpu/runtime/thunk.h"
#include "xla/service/global_device_id.h"

namespace xla::cpu {

class CollectiveThunk : public Thunk {
  using Thunk::Thunk;

 public:
  // Parameters of the collective operation behind the collective thunk. We rely
  // on them to construct the rendezvous key and to find a thunk "location" in
  // the collective operation "clique" (group of communicating devices).
  struct OpParams {
    int64_t op_id;
    bool has_channel_id;
    std::optional<bool> use_global_device_ids;
    std::vector<ReplicaGroup> group;
  };

  CollectiveThunk(Kind kind, Thunk::Info info, OpParams op_params);

  const OpParams& op_params() const { return op_params_; }

 protected:
  absl::StatusOr<RendezvousKey> GetRendezvousKey(
      const Thunk::CollectiveExecuteParams& params);

  absl::StatusOr<int32_t> RankInGlobalDevices(const RendezvousKey& key,
                                              GlobalDeviceId device);

 private:
  OpParams op_params_;
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_RUNTIME_COLLECTIVE_THUNK_H_
