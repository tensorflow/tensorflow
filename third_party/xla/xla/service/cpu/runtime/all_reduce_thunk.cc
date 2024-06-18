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

#include "xla/service/cpu/runtime/all_reduce_thunk.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/primitive_util.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/cpu/runtime/collective_thunk.h"
#include "xla/service/cpu/runtime/thunk.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/util.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla::cpu {
namespace {

static bool IsDataTypeSupportedByCollectiveReduce(PrimitiveType datatype) {
  switch (datatype) {
    case PRED:
    case S8:
    case U8:
    case S16:
    case U16:
    case S32:
    case U32:
    case S64:
    case U64:
    case F16:
    case F32:
    case F64:
    case C64:
    case C128:
      return true;
    default:
      return false;
  }
}

}  // namespace

absl::StatusOr<std::unique_ptr<AllReduceThunk>> AllReduceThunk::Create(
    Info info, ReductionKind reduction_kind, OpParams op_params,
    absl::Span<const BufferAllocation::Slice> source_buffers,
    absl::Span<const Shape> source_shapes,
    absl::Span<const BufferAllocation::Slice> destination_buffers,
    absl::Span<const Shape> destination_shapes, bool single_replica) {
  auto datatype = source_shapes[0].element_type();

  // Check that the data types are supported.
  if (!IsDataTypeSupportedByCollectiveReduce(datatype)) {
    return Unimplemented("AllReduce for datatype '%s' is not supported",
                         primitive_util::LowercasePrimitiveTypeName(datatype));
  }

  return absl::WrapUnique(new AllReduceThunk(
      std::move(info), reduction_kind, op_params, source_buffers, source_shapes,
      destination_buffers, destination_shapes, single_replica));
}

AllReduceThunk::AllReduceThunk(
    Info info, ReductionKind reduction_kind, OpParams op_params,
    absl::Span<const BufferAllocation::Slice> source_buffers,
    absl::Span<const Shape> source_shapes,
    absl::Span<const BufferAllocation::Slice> destination_buffers,
    absl::Span<const Shape> destination_shapes, bool single_replica)
    : CollectiveThunk(Kind::kAllReduce, info, op_params),
      reduction_kind_(reduction_kind),
      source_buffers_(source_buffers.begin(), source_buffers.end()),
      source_shapes_(source_shapes.begin(), source_shapes.end()),
      destination_buffers_(destination_buffers.begin(),
                           destination_buffers.end()),
      destination_shapes_(destination_shapes.begin(), destination_shapes.end()),
      single_replica_(single_replica) {}

tsl::AsyncValueRef<AllReduceThunk::ExecuteEvent> AllReduceThunk::Execute(
    const ExecuteParams& params) {
  tsl::profiler::TraceMe trace([&] { return TraceMeEncode(); });

  size_t num_srcs = source_buffers_.size();
  size_t num_dsts = destination_buffers_.size();
  DCHECK_EQ(num_srcs, num_dsts) << "Number of src and dst buffers must match";

  VLOG(3) << absl::StreamFormat(
      "AllReduce: #source_buffers=%d, #destination_buffers=%d, "
      "reduction_kind=%s, single_replica=%v",
      num_srcs, num_dsts, ReductionKindToString(reduction_kind_),
      single_replica_);

  absl::InlinedVector<se::DeviceMemoryBase, 4> source_data(num_srcs);
  for (int i = 0; i < num_srcs; ++i) {
    TF_ASSIGN_OR_RETURN(
        source_data[i],
        params.buffer_allocations->GetDeviceAddress(source_buffers_[i]));
    VLOG(3) << absl::StreamFormat(
        "  src: %s in slice %s (%p)", source_shapes_[i].ToString(true),
        source_buffers_[i].ToString(), source_data[i].opaque());
  }

  absl::InlinedVector<se::DeviceMemoryBase, 4> destination_data(num_dsts);
  for (int i = 0; i < num_dsts; ++i) {
    TF_ASSIGN_OR_RETURN(
        destination_data[i],
        params.buffer_allocations->GetDeviceAddress(destination_buffers_[i]));
    VLOG(3) << absl::StreamFormat(
        "  dst: %s in slice %s (%p)", destination_shapes_[i].ToString(true),
        destination_buffers_[i].ToString(), destination_data[i].opaque());
  }

  // Handle single-replica case by copying the source to the destination.
  if (single_replica_) {
    DCHECK_EQ(source_data.size(), destination_data.size());
    for (int i = 0; i < num_srcs; ++i) {
      std::memcpy(destination_data[i].opaque(), source_data[i].opaque(),
                  destination_data[i].size());
    }
    return OkExecuteEvent();
  }

  // For multi-replica case, we need collective parameters to be able to
  // perform the all-reduce operation collectively with other replicas.
  CollectiveExecuteParams* collective_params = params.collective_params;
  if (collective_params == nullptr) {
    return Internal(
        "Collective parameters are not set for all-reduce operation");
  }

  TF_ASSIGN_OR_RETURN(RendezvousKey key, GetRendezvousKey(*collective_params));
  TF_ASSIGN_OR_RETURN(
      int32_t rank,
      RankInGlobalDevices(key, collective_params->global_device_id));

  VLOG(3) << absl::StreamFormat("  rank=%d, key=%s", rank, key.ToString());

  return absl::UnimplementedError("AllReduceThunk::Execute not implemented");
}

Thunk::BufferUses AllReduceThunk::buffer_uses() const {
  BufferUses uses;
  uses.reserve(source_buffers_.size() + destination_buffers_.size());
  for (auto& source_buffer : source_buffers_) {
    uses.push_back(BufferUse::Read(source_buffer));
  }
  for (auto& destination_buffer : destination_buffers_) {
    uses.push_back(BufferUse::Write(destination_buffer));
  }
  return uses;
}

}  // namespace xla::cpu
