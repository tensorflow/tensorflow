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

#include "xla/backends/cpu/runtime/all_reduce_thunk.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>

#include "absl/base/optimization.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/backends/cpu/collectives/cpu_collectives.h"
#include "xla/backends/cpu/runtime/collective_thunk.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/core/collectives/communicator.h"
#include "xla/primitive_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla::cpu {

absl::StatusOr<std::unique_ptr<AllReduceThunk>> AllReduceThunk::Create(
    Info info, ReductionKind reduction_kind, OpParams op_params,
    OpBuffers op_buffers, OpResources op_resources, bool single_replica) {
  auto datatype = op_buffers.source_shapes[0].element_type();
  if (!IsDataTypeSupportedByCollectiveReduce(datatype)) {
    return Unimplemented("AllReduce for datatype '%s' is not supported",
                         primitive_util::LowercasePrimitiveTypeName(datatype));
  }

  return absl::WrapUnique(new AllReduceThunk(
      std::move(info), reduction_kind, std::move(op_params),
      std::move(op_buffers), std::move(op_resources), single_replica));
}

AllReduceThunk::AllReduceThunk(Info info, ReductionKind reduction_kind,
                               OpParams op_params, OpBuffers op_buffers,
                               OpResources op_resources, bool single_replica)
    : CollectiveThunk(CollectiveKind::kAllReduce, std::move(info),
                      std::move(op_params), std::move(op_buffers),
                      std::move(op_resources)),
      reduction_kind_(reduction_kind),
      single_replica_(single_replica) {}

tsl::AsyncValueRef<AllReduceThunk::ExecuteEvent> AllReduceThunk::Execute(
    const ExecuteParams& params) {
  TF_ASSIGN_OR_RETURN(OpDeviceMemory data, GetOpDeviceMemory(params));

  VLOG(3) << absl::StreamFormat(
      "AllReduce: #source_buffers=%d, #destination_buffers=%d, "
      "reduction_kind=%s, single_replica=%v",
      data.source.size(), data.destination.size(),
      ReductionKindToString(reduction_kind_), single_replica_);

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

  // Handle single-replica case by copying the source to the destination.
  if (single_replica_) {
    DCHECK_EQ(data.source.size(), data.destination.size());
    for (int i = 0; i < data.source.size(); ++i) {
      std::memcpy(data.destination[i].opaque(), data.source[i].opaque(),
                  data.destination[i].size());
    }
    return OkExecuteEvent();
  }

  return ExecuteWithCommunicator(
      params.collective_params,
      [&, data = std::move(data)](const RendezvousKey& key, Communicator& comm)
          -> tsl::AsyncValueRef<Communicator::Event> {
        tsl::CountDownAsyncValueRef<Communicator::Event> state(
            data.source.size());

        CpuCollectives::Executor executor(key, DefaultCollectiveTimeout());
        for (int32_t i = 0; i < data.source.size(); ++i) {
          const Shape& shape = destination_shape(i);

          auto communicator_event = comm.AllReduce(
              data.source[i], data.destination[i], shape.element_type(),
              ShapeUtil::ElementsIn(shape), reduction_kind_, executor);

          communicator_event.AndThen([state, communicator_event]() mutable {
            if (ABSL_PREDICT_FALSE(communicator_event.IsError())) {
              state.CountDown(communicator_event.GetError());
            } else {
              state.CountDown();
            }
          });
        }
        return state.AsRef();
      });
}

}  // namespace xla::cpu
