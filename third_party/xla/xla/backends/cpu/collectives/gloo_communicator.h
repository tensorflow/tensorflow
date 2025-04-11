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

#ifndef XLA_BACKENDS_CPU_COLLECTIVES_GLOO_COMMUNICATOR_H_
#define XLA_BACKENDS_CPU_COLLECTIVES_GLOO_COMMUNICATOR_H_

#include <cstddef>
#include <memory>
#include <optional>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "gloo/context.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

// XLA communicator implemented using Gloo communication library.
class GlooCommunicator : public Communicator {
 public:
  GlooCommunicator(std::shared_ptr<gloo::Context> context, size_t rank,
                   size_t num_ranks);
  ~GlooCommunicator() override;

  tsl::AsyncValueRef<Event> AllReduce(se::DeviceMemoryBase send_buffer,
                                      se::DeviceMemoryBase recv_buffer,
                                      PrimitiveType dtype, size_t count,
                                      ReductionKind reduction_kind,
                                      const Executor& executor) override;

  tsl::AsyncValueRef<Event> CollectivePermute(
      se::DeviceMemoryBase send_buffer, se::DeviceMemoryBase recv_buffer,
      PrimitiveType dtype, size_t count, std::optional<RankId> source_rank,
      absl::Span<const RankId> target_ranks, const Executor& executor) override;

  tsl::AsyncValueRef<Event> AllToAll(
      absl::Span<const se::DeviceMemoryBase> send_buffers,
      absl::Span<const se::DeviceMemoryBase> recv_buffers, PrimitiveType dtype,
      size_t count, const Executor& executor) override;

  tsl::AsyncValueRef<Event> AllGather(se::DeviceMemoryBase send_buffer,
                                      se::DeviceMemoryBase recv_buffer,
                                      PrimitiveType dtype, size_t count,
                                      const Executor& executor) override;

  tsl::AsyncValueRef<Event> ReduceScatter(se::DeviceMemoryBase send_buffer,
                                          se::DeviceMemoryBase recv_buffer,
                                          PrimitiveType dtype, size_t count,
                                          ReductionKind reduction_kind,
                                          const Executor& executor) override;

  tsl::AsyncValueRef<Event> Broadcast(se::DeviceMemoryBase,
                                      se::DeviceMemoryBase, PrimitiveType,
                                      size_t, RankId,
                                      const Executor&) override {
    return Unimplemented("Broadcast is not implemented");
  }

  tsl::AsyncValueRef<Event> Send(se::DeviceMemoryBase, PrimitiveType, size_t,
                                 RankId, const Executor&) override {
    return Unimplemented("Send is not implemented");
  }

  tsl::AsyncValueRef<Event> Recv(se::DeviceMemoryBase, PrimitiveType, size_t,
                                 RankId, const Executor&) override {
    return Unimplemented("Recv is not implemented");
  }

  absl::StatusOr<size_t> NumRanks() const override { return num_ranks_; }

  std::string ToString() const override {
    return absl::StrCat("GlooCommunicator [rank: ", rank_,
                        " num_ranks: ", num_ranks_, "]");
  }

 private:
  std::shared_ptr<gloo::Context> context_;
  size_t rank_;
  size_t num_ranks_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_COLLECTIVES_GLOO_COMMUNICATOR_H_
