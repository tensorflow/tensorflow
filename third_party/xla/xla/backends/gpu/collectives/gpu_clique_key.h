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

#include <cstdint>
#include <string>
#include <vector>

#include "absl/hash/hash.h"
#include "xla/core/collectives/clique_key.h"
#include "xla/service/global_device_id.h"
#include "xla/tsl/lib/gtl/int_type.h"

namespace xla::gpu {

// In XLA:GPU we use different streams for different kinds of collective
// operations, and include the async stream kind into the GPU clique key.
//
// We carefully isolate different kinds of collectives using separate
// communicators and guarantee that all collective operations have a total order
// that will not create a deadlock.
enum class AsyncStreamKind : int64_t {
  kCollective = 0,  // Stream for asynchronous collective ops.
  kP2P0 = 1,        // One Stream for P2P Send and Recv ops.
  kP2P1 = 2,        // Another Stream for P2P Send and Recv ops.
  kMemCpyP2P = 3,   // Stream for MemCpyP2P
};

inline constexpr int64_t kAsyncStreamTotal =
    static_cast<int64_t>(AsyncStreamKind::kMemCpyP2P) + 1;

// Strongly-typed wrapper to represent collective stream ID.
TSL_LIB_GTL_DEFINE_INT_TYPE(CollectiveStreamId, uint64_t);

// Assigns a unique ID to a stream for asynchronous or synchronous execution.
// These IDs can be used, for example, to look up the NCCL communicator.
CollectiveStreamId GetCollectiveStreamId(
    bool is_async, AsyncStreamKind stream_kind = AsyncStreamKind::kCollective);

// Clique key for identifying a particular collectives clique on a GPU backend.
class GpuCliqueKey : public CliqueKey {
 public:
  explicit GpuCliqueKey(
      std::vector<GlobalDeviceId> devices, int64_t num_local_participants,
      CollectiveStreamId stream_id = CollectiveStreamId(0),
      AsyncStreamKind stream_kind = AsyncStreamKind::kCollective,
      std::vector<std::vector<GlobalDeviceId>> participant_groups = {},
      GlobalDeviceId root_device = GlobalDeviceId(-1));

  GpuCliqueKey(const GpuCliqueKey&) = default;
  GpuCliqueKey& operator=(const GpuCliqueKey&) = default;

  GpuCliqueKey(GpuCliqueKey&&) = default;
  GpuCliqueKey& operator=(GpuCliqueKey&&) = default;

  CollectiveStreamId stream_id() const;

  // Device generating the unique id for this key
  GlobalDeviceId root_device() const;

  // Returns true if this clique is a subset of `other`: both cliques have the
  // same `stream_id` and all clique devices are part of `other` clique.
  bool IsSubsetOf(const CliqueKey& other) const final;

  // For multi-root initialization, generate `nroots` copies (subkeys) of the
  // key each with a different root device. Root devices are distributed evenly
  // across the ranks. The subkeys are used to exchange the CliqueIds during
  // clique initialization.
  std::vector<GpuCliqueKey> GetSubKeys(int64_t nroots) const;

  // Returns the stream kind for this clique key, stream kind will be used to
  // specify what configuration to pass for each type of operation.
  AsyncStreamKind stream_kind() const { return stream_kind_; }

  // The number of participant devices that are local to the current process (in
  // multi-host environments this likely to be all devices on the same host).
  // This number should never be different in two cliques over the same sets of
  // devices.
  int64_t num_local_participants() const { return num_local_participants_; }

  // Returns true if this clique is local to the current process (in multi-host
  // environments this likely to be all devices on the same host).
  bool is_local() const { return num_local_participants_ == devices().size(); }

  std::string ToString() const final;

  // GPU clique keys have a total order on which we rely on for acquiring
  // cliques in the same order across all participating devices.
  friend bool operator==(const GpuCliqueKey& a, const GpuCliqueKey& b);
  friend bool operator<(const GpuCliqueKey& a, const GpuCliqueKey& b);
  friend bool operator>(const GpuCliqueKey& a, const GpuCliqueKey& b);

 private:
  void HashValue(absl::HashState state) const final;

  // See comment on `num_local_participants()`.
  int64_t num_local_participants_;

  CollectiveStreamId stream_id_;
  AsyncStreamKind stream_kind_;

  // The full list of groups across all devices which this clique is a part of.
  //
  // When GPU communicator splitting is enabled, this is used to distinguish
  // which cliques can be reused from the cache or must be split in order to
  // prevent a deadlock situation.
  //
  // For example, imagine we have a communicator with devices = [0,1] and
  // groups = [0, 1] Later on, we may want to create communicators [0, 1] and
  // [2, 3] by splitting [0, 1, 2, 3] If ranks 0 and 1 reuse the existing
  // [0, 1] clique but ranks 2 and 3 initiate a split, there will be a deadlock
  // since ranks 2, 3 and will be waiting forever for 0, 1 to join the split.
  //
  // Having the participating groups as part of the cache key will prevent such
  // situations
  std::vector<std::vector<GlobalDeviceId>> participant_groups_;

  GlobalDeviceId root_device_;
};

bool operator==(const GpuCliqueKey& a, const GpuCliqueKey& b);
bool operator<(const GpuCliqueKey& a, const GpuCliqueKey& b);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_GPU_CLIQUE_KEY_H_
