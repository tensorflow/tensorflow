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

#ifndef XLA_SERVICE_CPU_IN_PROCESS_COLLECTIVES_H_
#define XLA_SERVICE_CPU_IN_PROCESS_COLLECTIVES_H_

#include <cstddef>
#include <memory>
#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/cpu/collectives_interface.h"
#include "xla/service/global_device_id.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu::runtime {

struct InProcessCollectivesState;

class InProcessCollectivesCommunicator : public CollectivesCommunicator {
 public:
  InProcessCollectivesCommunicator(InProcessCollectivesState* state, int rank,
                                   int size);
  ~InProcessCollectivesCommunicator() override;

  absl::Status AllReduce(const RendezvousKey& key, ReductionKind reduction_kind,
                         PrimitiveType element_type, size_t num_elements,
                         const void* input_buffer, void* output_buffer,
                         absl::Duration timeout) override;

  absl::Status CollectivePermute(const RendezvousKey& key, size_t num_bytes,
                                 std::optional<int> source_rank,
                                 absl::Span<int const> target_ranks,
                                 const void* input_buffer, void* output_buffer,
                                 absl::Duration timeout) override;

  absl::Status AllToAll(const RendezvousKey& key, size_t chunk_bytes,
                        absl::Span<const void* const> input_buffers,
                        absl::Span<void* const> output_buffers,
                        absl::Duration timeout) override;

  absl::Status AllGather(const RendezvousKey& key, size_t chunk_bytes,
                         const void* input_buffer, void* output_buffer,
                         absl::Duration timeout) override;

  absl::Status ReduceScatter(const RendezvousKey& key,
                             ReductionKind reduction_kind,
                             PrimitiveType element_type, size_t chunk_elems,
                             const void* input_buffer, void* output_buffer,
                             absl::Duration timeout) override;

 private:
  InProcessCollectivesState* state_;
  int rank_;
};

class InProcessCollectives : public CollectivesInterface {
 public:
  InProcessCollectives();
  ~InProcessCollectives() override;

  // Thread-safe.
  absl::StatusOr<std::shared_ptr<CollectivesCommunicator>> GetCommunicator(
      absl::Span<GlobalDeviceId const> devices, int rank) override;

 private:
  std::unique_ptr<InProcessCollectivesState> state_;
};

}  // namespace xla::cpu::runtime

#endif  // XLA_SERVICE_CPU_IN_PROCESS_COLLECTIVES_H_
