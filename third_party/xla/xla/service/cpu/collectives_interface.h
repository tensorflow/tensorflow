/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_CPU_COLLECTIVES_INTERFACE_H_
#define XLA_SERVICE_CPU_COLLECTIVES_INTERFACE_H_

#include <cstddef>
#include <memory>
#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/global_device_id.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

class CollectivesCommunicator {
 public:
  virtual ~CollectivesCommunicator() = default;

  // Performs an all-reduce.
  virtual absl::Status AllReduce(const RendezvousKey& key,
                                 ReductionKind reduction_kind,
                                 PrimitiveType element_type,
                                 size_t num_elements, const void* input_buffer,
                                 void* output_buffer,
                                 absl::Duration timeout) = 0;

  // Performs a collective permute.
  // Arguments:
  //  source_rank: the rank from which this rank should receive its data.
  //    Optional; if absent, then the output is filled with zeros.
  //  target_rank: the ranks to which this rank should send its data.
  virtual absl::Status CollectivePermute(const RendezvousKey& key,
                                         size_t num_bytes,
                                         std::optional<int> source_rank,
                                         absl::Span<int const> target_ranks,
                                         const void* input_buffer,
                                         void* output_buffer,
                                         absl::Duration timeout) = 0;

  // Performs an all-to-all.
  // The all-to-all chunks are passed separately and do not have to be
  // contiguous in memory.
  virtual absl::Status AllToAll(const RendezvousKey& key, size_t chunk_bytes,
                                absl::Span<const void* const> input_buffers,
                                absl::Span<void* const> output_buffers,
                                absl::Duration timeout) = 0;

  // Performs an all-gather.
  virtual absl::Status AllGather(const RendezvousKey& key, size_t chunk_bytes,
                                 const void* input_buffer, void* output_buffer,
                                 absl::Duration timeout) = 0;

  // Performs a reduce-scatter
  virtual absl::Status ReduceScatter(
      const RendezvousKey& key, ReductionKind reduction_kind,
      PrimitiveType element_type, size_t chunk_elems, const void* input_buffer,
      void* output_buffer, absl::Duration timeout) = 0;
};

class CollectivesInterface {
 public:
  virtual ~CollectivesInterface() = default;

  // Builds a context for a collective group.
  // Args:
  //  devices: the devices participating in this collective.
  //  rank: the rank of this process.
  virtual absl::StatusOr<std::shared_ptr<CollectivesCommunicator>>
  GetCommunicator(absl::Span<GlobalDeviceId const> devices, int rank) = 0;
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_COLLECTIVES_INTERFACE_H_
