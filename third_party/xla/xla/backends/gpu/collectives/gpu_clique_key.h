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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_GPU_CLIQUE_KEY_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_GPU_CLIQUE_KEY_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/hash/hash.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/core/collectives/clique_key.h"
#include "xla/runtime/device_id.h"
#include "xla/tsl/lib/gtl/int_type.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

// CommunicationId is an opaque strongly-typed integer wrapper that represents
// different kinds of communications for the same set of global devices.
//
// Underlying collective communication library typically doesn't allow to run
// multiple concurrent operations using the same set of communicators, however
// some operations use disjoint set of hardware resources and can safely run in
// parallel, i.e. all-reduce is likely to require computation resources to do
// the actual reduction computation, and sending/receiving data can be done
// using copy engines, in such case it is possible to request two cliques for
// the same set of devices, but with a different communication id.
//
// Communication id is an opaque integer type, and how to assign different types
// of communication to ids is a decision made by individual thunks. Today XLA
// assigns most of the collective operations to id `0` and peer-to-peer
// communication (essentially operations decomposed to send and recv) to id `1`.
//
// IMPORTANT: CommunicationId is not the same as CommunicationStreamId!
// Assigning communication streams based on communication id is one of the valid
// strategies, however runtime might make more or less actual streams, as long
// as runtime guarantees that all collective operations launched for a given
// clique have a well defined total execution order, enforced with events.
TSL_LIB_GTL_DEFINE_INT_TYPE(CommunicationId, uint64_t);

// StrJoin for device groups that shortens long list of devices for readability.
std::string HumanReadableDeviceGroups(
    absl::Span<const std::vector<GlobalDeviceId>> device_groups,
    absl::string_view separator = ",", size_t first = 2, size_t last = 1);

// Clique key for identifying a particular collectives clique on a GPU backend.
class GpuCliqueKey : public CliqueKey {
 public:
  GpuCliqueKey(std::vector<GlobalDeviceId> devices,
               int64_t num_local_participants,
               CommunicationId communication_id = CommunicationId(0),
               std::vector<IncarnationId> incarnations = {});

  GpuCliqueKey(const GpuCliqueKey&) = default;
  GpuCliqueKey& operator=(const GpuCliqueKey&) = default;

  GpuCliqueKey(GpuCliqueKey&&) = default;
  GpuCliqueKey& operator=(GpuCliqueKey&&) = default;

  // Returns true if this clique is a subset of `other`: both cliques have the
  // same `stream_id` and all clique devices are part of `other` clique.
  bool IsSubsetOf(const CliqueKey& other) const final;

  // Returns root devices that are responsible for bootstrapping the GPU clique
  // during initialization. Root devices are distributed evenly across all ranks
  // in the clique. XLA processes owning the root devices are responsible for
  // generating clique id and exchanging it with other ranks that share the same
  // root device (this is done via shared KV store).
  std::vector<GlobalDeviceId> GetRootDevices(int64_t nroots) const;

  // The number of participant devices that are local to the current process (in
  // multi-host environments this is likely to be all devices on the same host).
  // This number should never be different in two cliques over the same sets of
  // devices.
  int64_t num_local_participants() const { return num_local_participants_; }

  // Returns the communication id assigned to the clique.
  CommunicationId communication_id() const { return communication_id_; }

  // Returns true if this clique is local to the current process (in multi-host
  // environments this is likely to be all devices on the same host).
  bool is_local() const { return num_local_participants_ == devices().size(); }

  // Returns the incarnation ids of the participating processes.
  absl::Span<const IncarnationId> incarnations() const { return incarnations_; }

  std::string ToString() const final;

  // GPU clique keys have a total order on which we rely on for acquiring
  // cliques in the same order across all participating devices.
  friend bool operator==(const GpuCliqueKey& a, const GpuCliqueKey& b);
  friend bool operator!=(const GpuCliqueKey& a, const GpuCliqueKey& b);
  friend bool operator<(const GpuCliqueKey& a, const GpuCliqueKey& b);
  friend bool operator>(const GpuCliqueKey& a, const GpuCliqueKey& b);

 private:
  void HashValue(absl::HashState state) const final;

  int64_t num_local_participants_;
  CommunicationId communication_id_;

  std::vector<IncarnationId> incarnations_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_GPU_CLIQUE_KEY_H_
