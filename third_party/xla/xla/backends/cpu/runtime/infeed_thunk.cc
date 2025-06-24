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

#include "xla/backends/cpu/runtime/infeed_thunk.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/runtime/buffer_use.h"
#include "xla/runtime/resource_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/xfeed_manager.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla::cpu {

absl::StatusOr<std::unique_ptr<InfeedThunk>> InfeedThunk::Create(
    Info info, absl::Span<const InfeedBuffer> infeed_buffers,
    InfeedResources infeed_resources) {
  return absl::WrapUnique(
      new InfeedThunk(info, infeed_buffers, std::move(infeed_resources)));
}

InfeedThunk::InfeedThunk(Info info,
                         absl::Span<const InfeedBuffer> infeed_buffers,
                         InfeedResources infeed_resources)
    : Thunk(Kind::kInfeed, info),
      infeed_buffers_(infeed_buffers.begin(), infeed_buffers.end()),
      infeed_resources_(std::move(infeed_resources)) {}

tsl::AsyncValueRef<Thunk::ExecuteEvent> InfeedThunk::Execute(
    const ExecuteParams& params) {
  VLOG(3) << absl::StreamFormat("Infeed %d buffers", infeed_buffers_.size());

  runtime::XfeedManager* xfeed = params.xfeed;
  if (xfeed == nullptr) {
    return InvalidArgument("Xfeed must be not null to execute infeed thunk");
  }

  int64_t infeed_num = 0;

  for (InfeedBuffer& infeed_buffer : infeed_buffers_) {
    TF_ASSIGN_OR_RETURN(
        se::DeviceMemoryBase infeed_data,
        params.buffer_allocations->GetDeviceAddress(infeed_buffer.slice));

    VLOG(3) << absl::StreamFormat("  infeed #%d: %s into slice %s (%p)",
                                  infeed_num++, infeed_buffer.shape.ToString(),
                                  infeed_buffer.slice.ToString(),
                                  infeed_data.opaque());

    // Acquire infeed buffer from the runtime.
    runtime::XfeedBuffer* buffer = xfeed->infeed()->BlockingDequeueBuffer();
    if (buffer->length() != infeed_buffer.slice.size()) {
      return Internal(
          "XLA runtime-managed infeed buffer size %d did not match the "
          "infeed operation result buffer size %d",
          buffer->length(), infeed_buffer.slice.size());
    }

    // TODO(ezhulenev): Benchmark multi threaded memcpy and consider add a
    // parallel memcpy that can be reused in Copy thunk.

    // Copy data from infeed buffer to the destination.
    std::memcpy(infeed_data.opaque(), buffer->data(), buffer->length());

    // Release infeed buffer back to the runtime.
    xfeed->infeed()->ReleaseCurrentBuffer(buffer->length(), buffer->data(),
                                          infeed_buffer.shape);
  }

  return OkExecuteEvent();
}

InfeedThunk::BufferUses InfeedThunk::buffer_uses() const {
  BufferUses buffer_uses;
  for (const InfeedBuffer& infeed_buffer : infeed_buffers_) {
    buffer_uses.emplace_back(infeed_buffer.slice, BufferUse::kWrite);
  }
  return buffer_uses;
}

InfeedThunk::ResourceUses InfeedThunk::resource_uses() const {
  return {ResourceUse::Read(infeed_resources_.consume_token),
          ResourceUse::Write(infeed_resources_.produce_token)};
}

}  // namespace xla::cpu
