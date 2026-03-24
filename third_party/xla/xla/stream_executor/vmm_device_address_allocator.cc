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

#include "xla/stream_executor/vmm_device_address_allocator.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_reservation.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor {

static absl::Status DeviceNotFoundError(int device_ordinal) {
  return absl::NotFoundError(
      absl::StrFormat("No device with ordinal %d registered in "
                      "DeviceAddressVmmAllocator",
                      device_ordinal));
}

// Returns the completed timeline value from pinned host memory using an
// acquire load, so all GPU writes prior to this value are visible.
// Uses __atomic_load_n rather than std::atomic<> because the pointer is
// volatile (GPU-written pinned memory) and reinterpret_cast to
// std::atomic<uint64_t>* would discard the volatile qualifier.
static uint64_t LoadTimeline(const volatile uint64_t* pinned_timeline) {
  return __atomic_load_n(pinned_timeline, __ATOMIC_ACQUIRE);
}

DeviceAddressVmmAllocator::DeviceAddressVmmAllocator(const Platform* platform)
    : DeviceAddressAllocator(platform) {}

absl::Status DeviceAddressVmmAllocator::PopulateDevices(
    DeviceAddressVmmAllocator* allocator,
    absl::Span<const DeviceConfig> devices) {
  absl::flat_hash_set<int> seen_ordinals;
  for (const DeviceConfig& cfg : devices) {
    DCHECK_NE(cfg.executor, nullptr);
    DCHECK_NE(cfg.stream, nullptr);
    int ordinal = cfg.executor->device_ordinal();
    DCHECK(seen_ordinals.insert(ordinal).second)
        << "Duplicate device ordinal: " << ordinal;
  }

  for (const DeviceConfig& cfg : devices) {
    int ordinal = cfg.executor->device_ordinal();

    auto state = std::make_unique<PerDeviceState>();
    state->executor = cfg.executor;
    state->stream = cfg.stream;
    state->pa_budget = cfg.pa_budget;

    TF_RETURN_IF_ERROR(allocator->InitializeDeviceState(*state));

    VLOG(3) << "DeviceAddressVmmAllocator: registering device " << ordinal
            << " with pa_budget " << cfg.pa_budget;
    allocator->per_device_.emplace(ordinal, std::move(state));
  }

  return absl::OkStatus();
}

DeviceAddressVmmAllocator::~DeviceAddressVmmAllocator() {
  for (auto& [ordinal, state] : per_device_) {
    // Briefly acquire the lock to read the last pending seqno.
    uint64_t last_seqno = 0;
    {
      absl::MutexLock lock(&state->mu);
      if (!state->pending_deallocations.empty()) {
        last_seqno = state->pending_deallocations.back().seqno;
      }
    }

    // Spin-wait for any pending GPU work to complete before freeing physical
    // memory. pinned_timeline is not ABSL_GUARDED_BY and last_seqno is a local.
    if (state->pinned_timeline != nullptr && last_seqno > 0) {
      while (LoadTimeline(state->pinned_timeline) < last_seqno) {
        absl::SleepFor(absl::Microseconds(50));
      }
    }

    {
      absl::MutexLock lock(&state->mu);
      for (auto& pending : state->pending_deallocations) {
        DoDeallocate(*state, pending.mem);
      }
      state->pending_deallocations.clear();
    }

    // Free platform-specific per-device resources (e.g. pinned timeline).
    if (state->destroy_fn) {
      state->destroy_fn();
    }
  }
}

DeviceAddressVmmAllocator::PerDeviceState*
DeviceAddressVmmAllocator::GetPerDeviceState(int device_ordinal) const {
  auto it = per_device_.find(device_ordinal);
  if (it == per_device_.end()) {
    return nullptr;
  }
  return it->second.get();
}

void DeviceAddressVmmAllocator::ProcessCompletedPendingDeallocations(
    PerDeviceState& state) {
  // Single atomic read covers all entries whose seqno is <= completed.
  uint64_t completed = LoadTimeline(state.pinned_timeline);
  while (!state.pending_deallocations.empty()) {
    if (state.pending_deallocations.front().seqno > completed) {
      break;
    }
    DoDeallocate(state, state.pending_deallocations.front().mem);
    state.pending_deallocations.pop_front();
  }
}

void DeviceAddressVmmAllocator::WaitPendingDeallocationsToComplete(
    PerDeviceState& state, uint64_t size) {
  if (state.pending_deallocations.empty()) {
    return;
  }

  uint64_t accumulated_size = 0;
  size_t count_to_wait = 0;
  uint64_t rounded_size = RoundUpToGranularity(state, size);
  uint64_t target_seqno = 0;

  // Target 1.1x the requested size to provide some headroom.
  uint64_t target_size = rounded_size + rounded_size / 10;

  for (const auto& pending : state.pending_deallocations) {
    accumulated_size += RoundUpToGranularity(state, pending.mem.size());
    target_seqno = pending.seqno;
    ++count_to_wait;
    if (accumulated_size >= target_size) {
      break;
    }
  }

  // Move selected entries out of the deque while holding the lock, so no
  // other thread can observe or free them.
  std::vector<PendingDeallocation> selected;
  selected.reserve(count_to_wait);
  for (size_t i = 0; i < count_to_wait; ++i) {
    selected.push_back(std::move(state.pending_deallocations.front()));
    state.pending_deallocations.pop_front();
  }

  // Release the lock before spin-waiting to avoid stalling other threads for
  // potentially milliseconds while the GPU drains its work queue.
  state.mu.Unlock();

  // Poll until the GPU writes a timeline value >= target_seqno.
  // Since timeline values are written in stream order, this guarantees all
  // earlier pending deallocations have also completed.
  // Sleep 50us per iteration to release the CPU core while waiting rather
  // than hot-spinning.
  while (LoadTimeline(state.pinned_timeline) < target_seqno) {
    absl::SleepFor(absl::Microseconds(50));
  }

  // Reacquire the lock before modifying the maps.
  state.mu.Lock();

  for (auto& item : selected) {
    DoDeallocate(state, item.mem);
  }
}

void DeviceAddressVmmAllocator::DoDeallocate(PerDeviceState& state,
                                             DeviceAddressBase mem) {
  VLOG(3) << absl::StreamFormat(
      "Actually freeing virtual address %p (size=%uB) on device ordinal %d",
      mem.opaque(), mem.size(), state.executor->device_ordinal());

  // Erase the ScopedMapping first: its destructor unmaps the physical memory
  // from the virtual address range.
  state.scoped_mappings.erase(mem.opaque());
  // Erase the reservation next: its destructor frees the virtual address range.
  state.reservations.erase(mem.opaque());
  // Erase the raw allocation last: its destructor releases the physical memory.
  state.raw_allocations.erase(mem.opaque());

  uint64_t rounded_size = RoundUpToGranularity(state, mem.size());
  DCHECK_GE(state.pa_allocated, rounded_size);
  state.pa_allocated -= rounded_size;
}

absl::StatusOr<DeviceAddressBase> DeviceAddressVmmAllocator::AllocateWithBudget(
    PerDeviceState& state, uint64_t size) {
  uint64_t rounded_size = RoundUpToGranularity(state, size);
  if (state.pa_allocated + rounded_size > state.pa_budget) {
    return absl::ResourceExhaustedError(absl::StrFormat(
        "Not enough PA budget for allocation: pa_allocated=%uB, "
        "rounded_size=%uB, pa_budget=%uB",
        state.pa_allocated, rounded_size, state.pa_budget));
  }

  // Create physical memory allocation (e.g. cuMemCreate).
  TF_ASSIGN_OR_RETURN(auto raw_alloc, CreateAllocation(state.executor, size));
  const uint64_t padded_size = raw_alloc->address().size();

  // Reserve virtual address range (e.g. cuMemAddressReserve).
  TF_ASSIGN_OR_RETURN(auto reservation,
                      CreateReservation(state.executor, size));

  // Map physical memory into the virtual address range and enable access.
  TF_ASSIGN_OR_RETURN(
      auto scoped_mapping,
      reservation->MapTo(/*reservation_offset=*/0, /*allocation_offset=*/0,
                         padded_size, *raw_alloc));

  void* va_ptr = reservation->address().opaque();

  // Store tracking entries. Destruction order matters: scoped_mappings must
  // be erased before reservations, which must be erased before raw_allocations.
  state.raw_allocations.emplace(va_ptr, std::move(raw_alloc));
  state.reservations.emplace(va_ptr, std::move(reservation));
  state.scoped_mappings.emplace(va_ptr, std::move(scoped_mapping));

  state.pa_allocated += rounded_size;
  // Return the original requested size, not the padded size.
  return DeviceAddressBase(va_ptr, size);
}

// Allocation flow with retry:
//
// Allocate(device_ordinal, size)
//           │
//           ▼
// ┌─────────────────────────────────┐
// │ Reuse pending deallocation      │──found──► return
// │ with matching size?             │
// └─────────────────────────────────┘
//           │ not found
//           ▼
// ┌─────────────────────────────────┐
// │ Allocate new physical +         │──OK──► return
// │ virtual memory                  │
// └─────────────────────────────────┘
//           │ failed
//           ▼
// ┌─────────────────────────────────┐
// │ Free any GPU-completed          │
// │ pending deallocations           │
// │ (non-blocking)                  │
// └─────────────────────────────────┘
//           │
//           ▼
// ┌─────────────────────────────────┐
// │ Allocate new physical +         │──OK──► return
// │ virtual memory                  │
// └─────────────────────────────────┘
//           │ failed
//           ▼
// ┌─────────────────────────────────┐
// │ Block until GPU frees           │
// │ enough pending memory           │
// └─────────────────────────────────┘
//           │
//           ▼
// ┌─────────────────────────────────┐
// │ Allocate new physical +         │──OK──► return
// │ virtual memory                  │
// └─────────────────────────────────┘
//           │ failed
//           ▼
//    ResourceExhaustedError
absl::StatusOr<ScopedDeviceAddress<uint8_t>>
DeviceAddressVmmAllocator::Allocate(int device_ordinal, uint64_t size,
                                    bool /*retry_on_failure*/,
                                    int64_t /*memory_space*/) {
  if (size == 0) {
    return ScopedDeviceAddress<uint8_t>(DeviceAddressBase(), device_ordinal,
                                        this);
  }

  PerDeviceState* state = GetPerDeviceState(device_ordinal);
  if (state == nullptr) {
    return DeviceNotFoundError(device_ordinal);
  }

  absl::MutexLock lock(&state->mu);

  // Try to reuse a completed pending deallocation with matching size.
  std::optional<DeviceAddressBase> reused =
      TryReusePendingDeallocation(*state, size);
  if (reused.has_value()) {
    return ScopedDeviceAddress<uint8_t>(*reused, device_ordinal, this);
  }

  absl::StatusOr<DeviceAddressBase> result = AllocateWithBudget(*state, size);

  // If allocation failed (e.g., out of memory), try processing pending
  // deallocations to free memory, then retry.
  if (!result.ok()) {
    ProcessCompletedPendingDeallocations(*state);
    result = AllocateWithBudget(*state, size);
  }

  if (!result.ok()) {
    WaitPendingDeallocationsToComplete(*state, size);
    result = AllocateWithBudget(*state, size);
  }

  if (!result.ok()) {
    return result.status();
  }

  VLOG(3) << absl::StreamFormat(
      "Allocated virtual address %p (%uB) on device ordinal %d",
      result->opaque(), size, device_ordinal);

  return ScopedDeviceAddress<uint8_t>(*result, device_ordinal, this);
}

absl::Status DeviceAddressVmmAllocator::Deallocate(int device_ordinal,
                                                   DeviceAddressBase mem) {
  if (mem.is_null()) {
    return absl::OkStatus();
  }

  PerDeviceState* state = GetPerDeviceState(device_ordinal);
  if (state == nullptr) {
    return DeviceNotFoundError(device_ordinal);
  }

  absl::MutexLock lock(&state->mu);

  VLOG(3) << absl::StreamFormat(
      "Queueing deferred deallocation for virtual address %p (size=%uB) "
      "on device ordinal %d",
      mem.opaque(), mem.size(), device_ordinal);

  // Assign the next sequence number and enqueue a GPU write to the pinned
  // timeline when the stream reaches this point. The CPU polls the timeline
  // value to know when it is safe to free the memory.
  uint64_t seqno = state->next_seqno++;
  TF_RETURN_IF_ERROR(EnqueueDeferredDeallocation(*state, seqno));

  state->pending_deallocations.push_back({mem, seqno});

  return absl::OkStatus();
}

absl::StatusOr<Stream*> DeviceAddressVmmAllocator::GetStream(
    int device_ordinal) {
  PerDeviceState* state = GetPerDeviceState(device_ordinal);
  if (state == nullptr) {
    return DeviceNotFoundError(device_ordinal);
  }
  return state->stream;
}

absl::StatusOr<StreamExecutor*> DeviceAddressVmmAllocator::GetStreamExecutor(
    int device_ordinal) const {
  PerDeviceState* state = GetPerDeviceState(device_ordinal);
  if (state == nullptr) {
    return DeviceNotFoundError(device_ordinal);
  }
  return state->executor;
}

MemoryAllocation* DeviceAddressVmmAllocator::GetRawAllocation(
    int device_ordinal, DeviceAddressBase addr) const {
  PerDeviceState* state = GetPerDeviceState(device_ordinal);
  if (state == nullptr) {
    return nullptr;
  }
  absl::MutexLock lock(&state->mu);
  auto it = state->raw_allocations.find(addr.opaque());
  if (it == state->raw_allocations.end()) {
    return nullptr;
  }
  return it->second.get();
}

MemoryReservation* DeviceAddressVmmAllocator::GetReservation(
    int device_ordinal, DeviceAddressBase addr) const {
  PerDeviceState* state = GetPerDeviceState(device_ordinal);
  if (state == nullptr) {
    return nullptr;
  }
  absl::MutexLock lock(&state->mu);
  auto it = state->reservations.find(addr.opaque());
  if (it == state->reservations.end()) {
    return nullptr;
  }
  return it->second.get();
}

std::optional<DeviceAddressBase>
DeviceAddressVmmAllocator::TryReusePendingDeallocation(PerDeviceState& state,
                                                       uint64_t size) {
  uint64_t rounded_size = RoundUpToGranularity(state, size);
  for (auto it = state.pending_deallocations.begin();
       it != state.pending_deallocations.end(); ++it) {
    if (RoundUpToGranularity(state, it->mem.size()) != rounded_size) {
      continue;
    }

    DeviceAddressBase reused_mem(it->mem.opaque(), size);
    VLOG(3) << absl::StreamFormat(
        "Reusing pending deallocation: address=%p original_size=%uB "
        "new_size=%uB rounded_size=%uB device=%d",
        reused_mem.opaque(), it->mem.size(), size, rounded_size,
        state.executor->device_ordinal());
    state.pending_deallocations.erase(it);

    return reused_mem;
  }

  return std::nullopt;
}

uint64_t DeviceAddressVmmAllocator::RoundUpToGranularity(
    const PerDeviceState& state, uint64_t size) const {
  if (state.allocation_granularity == 0) {
    return size;
  }
  return ((size + state.allocation_granularity - 1) /
          state.allocation_granularity) *
         state.allocation_granularity;
}

}  // namespace stream_executor
