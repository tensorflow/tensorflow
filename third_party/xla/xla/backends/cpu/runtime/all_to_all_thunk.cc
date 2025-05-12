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

#include "xla/backends/cpu/runtime/all_to_all_thunk.h"

#include <memory>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/backends/cpu/collectives/cpu_collectives.h"
#include "xla/backends/cpu/runtime/collective_thunk.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/core/collectives/communicator.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::cpu {

absl::StatusOr<std::unique_ptr<AllToAllThunk>> AllToAllThunk::Create(
    Info info, OpParams op_params, OpBuffers op_buffers,
    OpResources op_resources) {
  return absl::WrapUnique(
      new AllToAllThunk(std::move(info), std::move(op_params),
                        std::move(op_buffers), std::move(op_resources)));
}

AllToAllThunk::AllToAllThunk(Info info, OpParams op_params,
                             OpBuffers op_buffers, OpResources op_resources)
    : CollectiveThunk(CollectiveKind::kAllToAll, std::move(info),
                      std::move(op_params), std::move(op_buffers),
                      std::move(op_resources)) {}

tsl::AsyncValueRef<AllToAllThunk::ExecuteEvent> AllToAllThunk::Execute(
    const ExecuteParams& params) {
  TF_ASSIGN_OR_RETURN(OpDeviceMemory data, GetOpDeviceMemory(params));

  VLOG(3) << absl::StreamFormat(
      "AllToAll: #source_buffers=%d, #destination_buffers=%d",
      data.source.size(), data.destination.size());

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
      [&](const RendezvousKey& key, Communicator& comm) {
        CpuCollectives::Executor executor(key, DefaultCollectiveTimeout());
        const Shape& shape = destination_shape(0);

        return comm.AllToAll(std::move(data.source),
                             std::move(data.destination), shape.element_type(),
                             ShapeUtil::ElementsIn(shape), executor);
      });
}

}  // namespace xla::cpu
