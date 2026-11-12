/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/pjrt/raw_pjrt_client.h"

#include <cstddef>
#include <optional>
#include <string>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "tsl/profiler/lib/scoped_memory_debug_annotation.h"

namespace xla {

absl::StatusOr<absl::InlinedVector<PjRtRawBufferRef, 4>>
PjRtRawClient::AllocateRawBuffersForExecute(
    PJRT_Memory* memory_space, absl::Span<const size_t> on_device_bytes_counts,
    bool retry_on_oom) {
  const tsl::profiler::MemoryDebugAnnotation outer_anno =
      tsl::profiler::ScopedMemoryDebugAnnotation::CurrentAnnotation();
  absl::InlinedVector<PjRtRawBufferRef, 4> buffers;
  buffers.reserve(on_device_bytes_counts.size());
  for (size_t i = 0; i < on_device_bytes_counts.size(); ++i) {
    std::optional<tsl::profiler::ScopedMemoryDebugAnnotation> anno;
    if (outer_anno.pending_shape_func != nullptr) {
      anno.emplace("dummy", outer_anno.pending_region_type,
                   outer_anno.pending_data_type, [&outer_anno, i](size_t) {
                     return outer_anno.pending_shape_func(i);
                   });
    }
    ABSL_ASSIGN_OR_RETURN(
        PjRtRawBufferRef buffer,
        AllocateRawBufferForExecute(memory_space, on_device_bytes_counts[i],
                                    retry_on_oom));
    buffers.push_back(std::move(buffer));
  }
  return buffers;
}

void PjRtRawClient::ScheduleRemoteSend(
    LocalDeviceId local_device_id, int memory_kind_id,
    PjRtRawBufferRef raw_buffer, PjRtDeviceEventRefVector definition_events,
    PjRtDeviceEventPromiseRef usage_event_promise,
    Future<std::string> serialized_descriptor,
    PjRtRawBufferInterface::RemoteSendCallback on_done) {
  auto error = absl::UnimplementedError(
      absl::StrCat("ScheduleRemoteSend is not implemented for device ",
                   local_device_id.value(), " kind ", memory_kind_id));
  on_done(error, /*sends_were_enqueued=*/false);
  usage_event_promise.SetError(error);
}

}  // namespace xla
