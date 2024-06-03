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

#include "xla/service/cpu/runtime/infeed_thunk.h"

#include <cstring>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/runtime/thunk.h"
#include "xla/service/cpu/xfeed_manager.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/util.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla::cpu {

InfeedThunk::InfeedThunk(Info info,
                         absl::Span<const InfeedBuffer> infeed_buffers)
    : Thunk(Kind::kInfeed, info),
      infeed_buffers_(infeed_buffers.begin(), infeed_buffers.end()) {}

absl::Status InfeedThunk::Execute(const ExecuteParams& params) {
  tsl::profiler::TraceMe trace([&] { return TraceMeEncode(); });

  VLOG(3) << absl::StreamFormat("Infeed %d buffers", infeed_buffers_.size());

  runtime::XfeedManager* xfeed = params.xfeed;
  if (xfeed == nullptr) {
    return InvalidArgument("Xfeed must be not null to execute infeed thunk");
  }

  for (InfeedBuffer& infeed_buffer : infeed_buffers_) {
    TF_ASSIGN_OR_RETURN(
        se::DeviceMemoryBase infeed_data,
        params.buffer_allocations->GetDeviceAddress(infeed_buffer.slice));

    VLOG(3) << absl::StreamFormat(
        "  infeed #%d: %s into slice %s (%p)", infeed_data.size() - 1,
        infeed_buffer.shape.ToString(), infeed_buffer.slice.ToString(),
        infeed_data.opaque());

    // Acquire infeed buffer from the runtime.
    runtime::XfeedBuffer* buffer = xfeed->infeed()->BlockingDequeueBuffer();
    if (buffer->length() != infeed_buffer.slice.size()) {
      return Internal(
          "XLA program infeed request buffer size %d did not match the "
          "runtime's infed buffer length %d",
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

  return absl::OkStatus();
}

}  // namespace xla::cpu
