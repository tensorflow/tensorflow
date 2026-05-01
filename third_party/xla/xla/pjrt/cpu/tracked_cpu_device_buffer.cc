/* Copyright 2021 The OpenXLA Authors.

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

#include "xla/pjrt/cpu/tracked_cpu_device_buffer.h"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/base/no_destructor.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/alignment.h"
#include "xla/future.h"
#include "xla/pjrt/abstract_tracked_device_buffer.h"
#include "xla/pjrt/common_pjrt_client.h"
#include "xla/pjrt/cpu/cpu_event.h"
#include "xla/pjrt/cpu/raw_buffer.h"
#include "xla/pjrt/device_event.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/mem.h"

namespace xla {
namespace {

//===----------------------------------------------------------------------===//
// Default CpuDeviceMemory::RawMemory allocator.
//===----------------------------------------------------------------------===//

class AlignedMemory final : public CpuDeviceMemory::RawMemory {
 public:
  AlignedMemory(void* base, size_t size_bytes)
      : base_(base), size_bytes_(size_bytes) {}

  ~AlignedMemory() final {
    tsl::port::AlignedSizedFree(base_, size_bytes_,
                                static_cast<std::align_val_t>(cpu::MinAlign()));
  }

  void* base() const final { return base_; }
  size_t size_bytes() const final { return size_bytes_; }

 private:
  void* base_;
  size_t size_bytes_;
};

class AlignedAllocator final : public CpuDeviceMemory::Allocator {
 public:
  absl::StatusOr<std::unique_ptr<CpuDeviceMemory::RawMemory>> Allocate(
      size_t size_bytes, size_t alignment) const final {
    if (void* base = tsl::port::AlignedMalloc(
            size_bytes, static_cast<std::align_val_t>(alignment))) {
      return std::make_unique<AlignedMemory>(base, size_bytes);
    }
    return ResourceExhausted("Out of memory allocating %d bytes.", size_bytes);
  }
};

}  // namespace

CpuDeviceMemory::Allocator& CpuDeviceMemory::DefaultAllocator() {
  static absl::NoDestructor<AlignedAllocator> allocator;
  return *allocator;
}

std::unique_ptr<CpuDeviceMemory::Allocator>
CpuDeviceMemory::MakeDefaultAllocator() {
  return std::make_unique<AlignedAllocator>();
}

//===----------------------------------------------------------------------===//
// CpuDeviceMemory implementations.
//===----------------------------------------------------------------------===//

class CpuDeviceMemoryOwned final : public CpuDeviceMemory {
 public:
  explicit CpuDeviceMemoryOwned(std::unique_ptr<RawMemory> mem)
      : mem_(std::move(mem)) {}

  void* untyped_data() const final { return mem_->base(); }
  size_t size_bytes() const final { return mem_->size_bytes(); }

 private:
  std::unique_ptr<RawMemory> mem_;
};

class CpuDeviceMemoryForeign final : public CpuDeviceMemory {
 public:
  CpuDeviceMemoryForeign(void* base, size_t size,
                         absl::AnyInvocable<void() &&> on_delete_callback)
      : base_(base),
        size_bytes_(size),
        on_delete_callback_(std::move(on_delete_callback)) {}

  ~CpuDeviceMemoryForeign() final {
    if (on_delete_callback_) {
      std::move(on_delete_callback_)();
    }
  }

  void* untyped_data() const final { return base_; }
  size_t size_bytes() const final { return size_bytes_; }

 private:
  void* base_;
  size_t size_bytes_;
  absl::AnyInvocable<void() &&> on_delete_callback_;
};

class CpuDeviceMemoryConstant final : public CpuDeviceMemory {
 public:
  CpuDeviceMemoryConstant(void* base, size_t size)
      : base_(base), size_bytes_(size) {}

  void* untyped_data() const final { return base_; }
  size_t size_bytes() const final { return size_bytes_; }

 private:
  void* base_;
  size_t size_bytes_;
};

tsl::AsyncValueRef<CpuDeviceMemory> CpuDeviceMemory::CreateDelayedMemory() {
  return tsl::MakeUnconstructedAsyncValueRef<CpuDeviceMemoryOwned>();
}

tsl::AsyncValueRef<CpuDeviceMemory> CpuDeviceMemory::CreateForeignMemory(
    void* base, size_t size, absl::AnyInvocable<void() &&> on_delete_callback) {
  return tsl::MakeAvailableAsyncValueRef<CpuDeviceMemoryForeign>(
      base, size, std::move(on_delete_callback));
}

tsl::AsyncValueRef<CpuDeviceMemory> CpuDeviceMemory::CreateConstantMemory(
    void* base, size_t size) {
  return tsl::MakeAvailableAsyncValueRef<CpuDeviceMemoryConstant>(base, size);
}

// Allocates owning memory wrapped in an available `AsyncValueRef`.
absl::StatusOr<tsl::AsyncValueRef<CpuDeviceMemory>> CpuDeviceMemory::Allocate(
    size_t size_bytes, const Allocator& allocator) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<RawMemory> mem,
                      allocator.Allocate(size_bytes, cpu::MinAlign()));
  return tsl::MakeAvailableAsyncValueRef<CpuDeviceMemoryOwned>(std::move(mem));
}

absl::Status CpuDeviceMemory::AllocateInto(
    size_t size_bytes, tsl::AsyncValuePtr<CpuDeviceMemory> delayed_memory,
    const Allocator& allocator) {
  auto owned_memory = delayed_memory.DynCast<CpuDeviceMemoryOwned>();
  if (!owned_memory) {
    return Internal("Delayed memory is not a CpuDeviceMemoryOwned");
  }

  TF_ASSIGN_OR_RETURN(std::unique_ptr<RawMemory> mem,
                      allocator.Allocate(size_bytes, cpu::MinAlign()));
  owned_memory.emplace(std::move(mem));
  return absl::OkStatus();
}

//===----------------------------------------------------------------------===//
// TrackedCpuDeviceBuffer.
//===----------------------------------------------------------------------===//

TrackedCpuDeviceBuffer::TrackedCpuDeviceBuffer(
    PjRtRawBufferRef raw_buffer, tsl::AsyncValueRef<CpuEvent> definition_event,
    std::unique_ptr<PjRtDeviceEventSet> usage_events)
    : AbstractTrackedDeviceBuffer(
          std::move(raw_buffer),
          [definition_event]() {
            absl::InlinedVector<PjRtDeviceEventRef, 2> events;
            if (definition_event) {
              events.push_back(PjRtDeviceEventRef(std::move(definition_event)));
            }
            return events;
          }(),
          usage_events ? std::move(usage_events)
                       : std::make_unique<CpuUsageEventSet>()) {
  DCHECK(!definition_events().empty());
}

TrackedCpuDeviceBuffer::TrackedCpuDeviceBuffer(
    tsl::RCReference<CommonPjRtRawBuffer> raw_buffer,
    absl::InlinedVector<PjRtDeviceEventRef, 2> definition_events,
    std::unique_ptr<PjRtDeviceEventSet> usage_events)
    : AbstractTrackedDeviceBuffer(
          std::move(raw_buffer), std::move(definition_events),
          usage_events ? std::move(usage_events)
                       : std::make_unique<CpuUsageEventSet>()) {}

TrackedCpuDeviceBuffer::~TrackedCpuDeviceBuffer() = default;

void CpuUsageEventSet::AddEvent(PjRtDeviceEventRef event) {
  if (event) {
    usage_events_.push_back(event.down_cast<CpuEvent>());
  }
}

void CpuUsageEventSet::AddEvent(tsl::AsyncValueRef<CpuEvent> event) {
  // Periodically remove available usage events to prevent memory blowup.
  if (usage_events_.size() + 1 > usage_events_.capacity()) {
    int i = 0;
    while (i < usage_events_.size()) {
      auto& ev = usage_events_.at(i);
      if (ev.IsAvailable()) {
        using std::swap;
        swap(ev, usage_events_.back());
        usage_events_.pop_back();
        continue;
      }
      ++i;
    }
  }
  usage_events_.push_back(std::move(event));
}

void CpuUsageEventSet::AppendTo(
    std::vector<tsl::RCReference<tsl::AsyncValue>>& events) {
  for (const auto& ev : usage_events_) {
    events.push_back(ev.CopyRef());
  }
}

void CpuUsageEventSet::AppendTo(PjRtDeviceEventSet& events) {
  for (const auto& ev : usage_events_) {
    events.AddEvent(PjRtDeviceEventRef(ev.CopyRef()));
  }
}

std::unique_ptr<PjRtDeviceEventSet> CpuUsageEventSet::Clone() const {
  return std::make_unique<CpuUsageEventSet>(*this);
}

void TrackedCpuDeviceBuffer::Delete(PjRtMemorySpace* memory_space) {
  std::unique_ptr<TrackedCpuDeviceBuffer> device_buffer(this);
  std::vector<tsl::RCReference<tsl::AsyncValue>> avs =
      device_buffer->GetAsyncValueDefinitionAndUsageEvents();
  auto usage_event = device_buffer->LockUseAndTransferUsageEvents();
  std::vector<tsl::AsyncValue*> event_avs;
  event_avs.reserve(avs.size());
  for (auto& av : avs) {
    event_avs.push_back(av.get());
  }

  RunWhenReady(event_avs, [device_buffer = std::move(device_buffer)]() mutable {
    device_buffer.reset();
  });
}

}  // namespace xla
