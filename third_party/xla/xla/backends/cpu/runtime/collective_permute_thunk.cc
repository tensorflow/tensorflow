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

#include "xla/backends/cpu/runtime/collective_permute_thunk.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/collective_thunk.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/computation_placer.h"
#include "xla/service/cpu/collectives_interface.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla::cpu {

absl::StatusOr<std::unique_ptr<CollectivePermuteThunk>>
CollectivePermuteThunk::Create(
    Info info, OpParams op_params, OpBuffers op_buffers,
    OpResources op_resources,
    absl::Span<const SourceTargetPair> source_target_pairs) {
  return absl::WrapUnique(new CollectivePermuteThunk(
      std::move(info), std::move(op_params), std::move(op_buffers),
      std::move(op_resources), source_target_pairs));
}

CollectivePermuteThunk::CollectivePermuteThunk(
    Info info, OpParams op_params, OpBuffers op_buffers,
    OpResources op_resources,
    absl::Span<const SourceTargetPair> source_target_pairs)
    : CollectiveThunk(Kind::kCollectivePermute, std::move(info),
                      std::move(op_params), std::move(op_buffers),
                      std::move(op_resources)),
      source_target_pairs_(source_target_pairs.begin(),
                           source_target_pairs.end()) {}

tsl::AsyncValueRef<CollectivePermuteThunk::ExecuteEvent>
CollectivePermuteThunk::Execute(const ExecuteParams& params) {
  tsl::profiler::TraceMe trace([&] { return TraceMeEncode(); });

  TF_ASSIGN_OR_RETURN(OpDeviceMemory data, GetOpDeviceMemory(params));

  Thunk::CollectiveExecuteParams* collective_params = params.collective_params;
  TF_RET_CHECK(collective_params) << "Collectives parameters are not set";

  TF_ASSIGN_OR_RETURN(DeviceAssignment::LogicalID logical_id,
                      collective_params->device_assignment->LogicalIdForDevice(
                          collective_params->global_device_id));

  int32_t logical_device_id = op_params().has_channel_id
                                  ? logical_id.computation_id
                                  : logical_id.replica_id;

  // Find replicas that we will communicate with.
  std::optional<int32_t> source_replica_id;
  std::vector<int32_t> copy_to;

  for (auto& [from, to] : source_target_pairs_) {
    if (from == logical_device_id) {
      copy_to.push_back(to);
    }
    if (to == logical_device_id) {
      TF_RET_CHECK(!source_replica_id.has_value())
          << "Duplicate source replica: " << from << ". "
          << "Previous source replica: " << *source_replica_id;
      source_replica_id = from;
    }
  }

  VLOG(3) << absl::StreamFormat(
      "CollectivePermute: #source_buffers=%d, #destination_buffers=%d, "
      "source_target_pairs=[%s], logical_device_id=%d (%s), "
      "source_replica_id=%d, copy_to=[%s]",
      data.source.size(), data.destination.size(),
      absl::StrJoin(source_target_pairs_, ", ", absl::PairFormatter("->")),
      logical_device_id,
      op_params().has_channel_id ? "computation id" : "replica id",
      source_replica_id.value_or(-1), absl::StrJoin(copy_to, ","));

  for (int i = 0; i < data.source.size(); ++i) {
    VLOG(3) << absl::StreamFormat(
        "  src: %s in slice %s (%p)", source_shape(i).ToString(true),
        source_buffer(i).ToString(), data.source[i].opaque());
  }

  for (int i = 0; i < data.destination.size(); ++i) {
    VLOG(3) << absl::StreamFormat(
        "  dst: %s in slice %s (%p)", destination_shape(i).ToString(true),
        destination_buffer(i).ToString(), data.destination[i].opaque());
  }

  return ExecuteWithCommunicator(
      params.collective_params,
      [&](const RendezvousKey& key, CollectivesCommunicator& comm) {
        for (int32_t i = 0; i < data.source.size(); ++i) {
          const Shape& shape = source_shape(i);
          TF_RETURN_IF_ERROR(comm.CollectivePermute(
              key, ShapeUtil::ByteSizeOf(shape), source_replica_id, copy_to,
              data.source[i].opaque(), data.destination[i].opaque(),
              DefaultCollectiveTimeout()));
        }
        return absl::OkStatus();
      });
}

}  // namespace xla::cpu
