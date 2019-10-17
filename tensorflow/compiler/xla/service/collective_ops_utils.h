/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_COLLECTIVE_OPS_UTILS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_COLLECTIVE_OPS_UTILS_H_

#include <vector>

#include "tensorflow/compiler/xla/executable_run_options.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

// Figures out which devices (named by their replica-ids) are participating in
// the all-reduce subgroup that contains device_ordinal.
StatusOr<std::vector<int64>> GetParticipatingReplicas(
    int64 device_ordinal, const HloInstruction* instr,
    int64 total_replica_count, const DeviceAssignment& device_assn);

// Key that identifies a particular Rendezvous object in our global hashtable.
// This determines which calls to ExecuteOnStream communicate with each other.
// The rules are as follows.
//
// * Only ops with the same RunId can communicate with each other. (This is the
//   whole purpose of RunId).
//
// * Only ops with the same set of participating replicas can communicate with
//   each other.  This is how we separate out different replica groups (e.g. a
//   single AllReduce HLO might do two reductions, between say GPUs {0,2} and
//   {1,3}).
//
// * Only ops with the same opcode can communicate with each other.  At the
//   moment we only support kAllReduce, so we don't check for this explicitly.
//
// * For cross-module all-reduces (i.e. instr->channel_id().has_value()),
//   only ops with the same value for channel_id() can communicate with each
//   other.
//
// * For cross-replica (i.e. same-module) all-reduces (i.e.
//   !channel_id().has_value()), only ops from the same module (as
//   identified by its unique_id()) can communicate with each other.
//
struct RendezvousKey {
  enum CollectiveOpKind {
    kCrossModule,
    kCrossReplica,
  };

  explicit RendezvousKey(const RunId& run_id,
                         std::vector<int64> participating_replicas,
                         const HloInstruction* instr)
      : run_id(run_id), participating_replicas(participating_replicas) {
    std::tie(collective_op_kind, op_id) =
        instr->channel_id().has_value()
            ? std::make_pair(kCrossModule, instr->channel_id().value())
            : std::make_pair(
                  kCrossReplica,
                  static_cast<int64>(instr->GetModule()->unique_id()));
  }

  int num_participants() const { return participating_replicas.size(); }

  template <typename H>
  friend H AbslHashValue(H h, const RendezvousKey& k) {
    return H::combine(std::move(h), k.run_id, k.participating_replicas,
                      static_cast<int>(k.collective_op_kind), k.op_id);
  }
  friend bool operator==(const RendezvousKey& a, const RendezvousKey& b) {
    return a.run_id == b.run_id &&
           a.participating_replicas == b.participating_replicas &&
           a.collective_op_kind == b.collective_op_kind &&  //
           a.op_id == b.op_id;
  }
  friend bool operator!=(const RendezvousKey& a, const RendezvousKey& b) {
    return !(a == b);
  }

  string ToString() const {
    return absl::StrFormat(
        "RendezvousKey{run_id=%s, participating_replicas=[%s], "
        "collective_op_kind=%d, op_id=%d}",
        run_id.ToString(), absl::StrJoin(participating_replicas, ","),
        static_cast<int>(collective_op_kind), op_id);
  }

  RunId run_id;
  std::vector<int64> participating_replicas;
  CollectiveOpKind collective_op_kind;
  int64 op_id;
};

// Encapsulates parameters to Rendezvous::SubmitParticipant.
struct ParticipantData {
  explicit ParticipantData(RendezvousKey rendezvous_key)
      : rendezvous_key(rendezvous_key) {}

  int64 element_count;
  int64 device_ordinal;
  RendezvousKey rendezvous_key;

  // TODO(b/125951860): We should vet that we're buffer allocating such that
  // source_buffer == destination_buffer if that avoids a NCCL copy (will depend
  // on how well the NCCL in-place implementation performs vs the out-of-place
  // implementation).
  se::DeviceMemoryBase source_data;
  se::DeviceMemoryBase destination_data;
  se::Stream* stream;

  int num_participants() const { return rendezvous_key.num_participants(); }

  string ToString() const {
    return absl::StrFormat(
        "ParticipantData{element_count=%d, rendezvous_key=%s, "
        "device_ordinal=%d, stream=%p}",
        element_count, rendezvous_key.ToString(), device_ordinal, stream);
  }
};

}  // end namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_COLLECTIVE_OPS_UTILS_H_
