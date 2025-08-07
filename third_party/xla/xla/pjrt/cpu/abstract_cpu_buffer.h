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

#ifndef XLA_PJRT_CPU_ABSTRACT_CPU_BUFFER_H_
#define XLA_PJRT_CPU_ABSTRACT_CPU_BUFFER_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/literal.h"
#include "xla/pjrt/abstract_tracked_device_buffer.h"
#include "xla/pjrt/async_work_runner.h"
#include "xla/pjrt/cpu/cpu_event.h"
#include "xla/pjrt/cpu/tracked_cpu_device_buffer.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/transpose.h"
#include "xla/shape.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

// A RAII helper class used to set an AsyncValueRef<CpuEvent> to a ready state
// upon destruction. In many cases in PjRt implementation, there will be
// multiple return statements in the function, all of which require setting some
// AsyncValueRef<CpuEvent> to be ready. This class could make such code more
// robust by using setting the AsyncValue in the destructor.
class MarkEventReadyOnExit {
 public:
  explicit MarkEventReadyOnExit(tsl::AsyncValueRef<CpuEvent> event)
      : event_(std::move(event)) {}

  MarkEventReadyOnExit(const MarkEventReadyOnExit&) = delete;
  MarkEventReadyOnExit& operator=(const MarkEventReadyOnExit&) = delete;
  MarkEventReadyOnExit(MarkEventReadyOnExit&&) noexcept = default;
  MarkEventReadyOnExit& operator=(MarkEventReadyOnExit&&) noexcept = default;

  ~MarkEventReadyOnExit() {
    if (event_) event_.SetStateConcrete();
  }

  tsl::AsyncValueRef<CpuEvent> Release() && { return std::move(event_); }

 private:
  tsl::AsyncValueRef<CpuEvent> event_;
};

class AbstractCpuBuffer {
 public:
  // Allocates a new `TrackedCpuDeviceBuffer` with the given shape and
  // definition events.
  static absl::StatusOr<std::unique_ptr<TrackedCpuDeviceBuffer>>
  AllocateTrackedDeviceBuffer(
      const Shape& on_device_shape,
      absl::InlinedVector<tsl::AsyncValueRef<CpuEvent>, 4> definition_events);

  // Allocates new cpu events to `avs` and `definition_events`. If `shape` is a
  // tuple, multiple events will be allocated. Otherwise, `avs` and
  // `definition_events` will only contain one event.
  static void AllocateAvsAndEvents(
      const Shape& shape,
      absl::InlinedVector<tsl::RCReference<tsl::AsyncValue>, 4>* avs,
      absl::InlinedVector<tsl::AsyncValueRef<CpuEvent>, 4>* definition_events);

  // A helper function to determine if a BufferFromHostBuffer call is elligable
  // for zero copy construction.
  static bool BufferFromHostBufferSupportsZeroCopy(
      const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides,
      const Shape& shape);
};

// Helper for copying into potentially sub-byte packed literals.
void PackOrCopy(PrimitiveType element_type, const LiteralSlice& literal,
                void* data, int64_t size);

}  // namespace xla

#endif  // XLA_PJRT_CPU_ABSTRACT_CPU_BUFFER_H_
