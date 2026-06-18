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

#include "xla/stream_executor/device_address_vmm_allocator.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
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
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/service/computation_placer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_reservation.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor {

namespace {
thread_local const xla::DeviceAssignment* current_device_assignment = nullptr;
}  // namespace

DeviceAddressVmmAllocator::DeviceAssignmentScope::DeviceAssignmentScope(
    const xla::DeviceAssignment* device_assignment)
    : previous_(current_device_assignment) {
  current_device_assignment = device_assignment;
}

DeviceAddressVmmAllocator::DeviceAssignmentScope::~DeviceAssignmentScope() {
  current_device_assignment = previous_;
}

bool DeviceAddressVmmAllocator::CurrentMultiDevice() {
  const xla::DeviceAssignment* device_assignment = current_device_assignment;
  return device_assignment != nullptr &&
         device_assignment->replica_count() *
                 device_assignment->computation_count() >
             1;
}

static absl::Status DeviceNotFoundError(int device_ordinal) {
  return absl::NotFoundError(
      absl::StrFormat("No device with ordinal %d registered in "
                      "DeviceAddressVmmAllocator",
                      device_ordinal));
}

// Interval between CPU polls of the GPU-written deallocation timeline while
// waiting for deferred frees to become safe. The 50us value is a conservative
// initial tradeoff: long enough to avoid busy-spinning a CPU core and short
// enough to keep forced allocator synchronization responsive; it has not been
// benchmark-tuned, so workload-specific tests could refine it if this wait
// shows up in profiles.
static constexpr absl::Duration kGpuTimelinePollInterval =
    absl::Microseconds(50);

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

    RETURN_IF_ERROR(allocator->InitializeDeviceState(*state));

    VLOG(3) << "DeviceAddressVmmAllocator: registering device " << ordinal
            << " with pa_budget " << cfg.pa_budget;
    allocator->per_device_.emplace(ordinal, std::move(state));
  }

  return absl::OkStatus();
}

DeviceAddressVmmAllocator::~DeviceAddressVmmAllocator() {
  absl::Status status = SynchronizeAllPendingOperations();
  CHECK(status.ok()) << status;

  for (auto& [ordinal, state] : per_device_) {
    // Free platform-specific per-device resources (e.g. pinned timeline).
    if (state->destroy_fn) {
      state->destroy_fn();
    }
  }
}

absl::Status DeviceAddressVmmAllocator::SynchronizeAllPendingOperations() {
  for (auto& [ordinal, state] : per_device_) {
    RETURN_IF_ERROR(SynchronizePendingOperations(ordinal));
  }
  return absl::OkStatus();
}

DeviceAddressVmmAllocator::PerDeviceState*
DeviceAddressVmmAllocator::GetPerDeviceState(int device_ordinal) const {
  auto it = per_device_.find(device_ordinal);
  if (it == per_device_.end()) {
    return nullptr;
  }
  return it->second.get();
}

absl::StatusOr<DeviceAddressBase>
DeviceAddressVmmAllocator::ValidateReservationRange(
    MemoryReservation* reservation, uint64_t reservation_offset,
    uint64_t size) const {
  if (reservation == nullptr) {
    return absl::InvalidArgumentError("reservation must not be null");
  }

  DeviceAddressBase address = reservation->address();
  if (reservation_offset > address.size() ||
      size > address.size() - reservation_offset) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "reservation range is out of bounds: offset=%uB, size=%uB, "
        "reservation_size=%uB",
        reservation_offset, size, address.size()));
  }

  return address.GetByteSlice(reservation_offset, size);
}

void DeviceAddressVmmAllocator::ProcessCompletedPendingDeallocations(
    PerDeviceState& state) {
  // Single atomic read covers all entries whose seqno is <= completed.
  uint64_t completed = LoadTimeline(state.pinned_timeline);
  while (!state.pending_deallocations.empty()) {
    if (state.pending_deallocations.front().seqno > completed) {
      break;
    }
    if (state.pending_deallocations.front().kind ==
        PendingDeallocationKind::kAllocate) {
      DoDeallocate(state, state.pending_deallocations.front().mem);
    } else {
      DoUnMap(state, state.pending_deallocations.front().mem);
    }
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
    if (pending.kind == PendingDeallocationKind::kAllocate) {
      accumulated_size += RoundUpToGranularity(state, pending.mem.size());
    }
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
  state.mu.unlock();

  // Poll until the GPU writes a timeline value >= target_seqno.
  // Since timeline values are written in stream order, this guarantees all
  // earlier pending deallocations have also completed.
  while (LoadTimeline(state.pinned_timeline) < target_seqno) {
    absl::SleepFor(kGpuTimelinePollInterval);
  }

  // Reacquire the lock before modifying the maps.
  state.mu.lock();

  for (auto& item : selected) {
    if (item.kind == PendingDeallocationKind::kAllocate) {
      DoDeallocate(state, item.mem);
    } else {
      DoUnMap(state, item.mem);
    }
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
  state.multi_device_allocations.erase(mem.opaque());

  uint64_t rounded_size = RoundUpToGranularity(state, mem.size());
  DCHECK_GE(state.pa_allocated, rounded_size);
  state.pa_allocated -= rounded_size;
}

void DeviceAddressVmmAllocator::DoUnMap(PerDeviceState& state,
                                        DeviceAddressBase mem) {
  VLOG(3) << absl::StreamFormat(
      "Actually unmapping reservation address %p (size=%uB) on device ordinal "
      "%d",
      mem.opaque(), mem.size(), state.executor->device_ordinal());
  state.stale_reservation_mappings.erase(mem.opaque());
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
  ASSIGN_OR_RETURN(auto raw_alloc, CreateAllocation(state.executor, size));
  const uint64_t padded_size = raw_alloc->address().size();

  // Reserve virtual address range (e.g. cuMemAddressReserve).
  ASSIGN_OR_RETURN(auto reservation, CreateReservation(state.executor, size));

  // Map physical memory into the virtual address range and enable access.
  ASSIGN_OR_RETURN(
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

absl::StatusOr<ScopedDeviceAddress<uint8_t>>
DeviceAddressVmmAllocator::Allocate(
    int device_ordinal, uint64_t allocation_size, bool /*retry_on_failure*/,
    int64_t /*memory_space*/, MemoryReservation* reservation,
    uint64_t reservation_offset, uint64_t mapping_size,
    bool return_reservation_address) {
  if (allocation_size != mapping_size) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "VMM mapped allocation size (%u) must equal mapping size (%u)",
        allocation_size, mapping_size));
  }
  if (allocation_size == 0) {
    return ScopedDeviceAddress<uint8_t>(DeviceAddressBase(), device_ordinal,
                                        this);
  }
  ASSIGN_OR_RETURN(
      DeviceAddressBase reservation_address,
      ValidateReservationRange(reservation, reservation_offset, mapping_size));

  PerDeviceState* state = GetPerDeviceState(device_ordinal);
  if (state == nullptr) {
    return DeviceNotFoundError(device_ordinal);
  }

  const bool multi_device = CurrentMultiDevice();

  absl::MutexLock lock(state->mu);
  if (state->active_reservation_mappings.contains(
          reservation_address.opaque()) ||
      state->stale_reservation_mappings.contains(
          reservation_address.opaque()) ||
      state->raw_allocations.contains(reservation_address.opaque())) {
    return absl::FailedPreconditionError(
        "Reservation address is already tracked by this allocator");
  }

  uint64_t rounded_size = RoundUpToGranularity(*state, allocation_size);
  if (state->pa_allocated + rounded_size > state->pa_budget) {
    return absl::ResourceExhaustedError(absl::StrFormat(
        "Not enough PA budget for allocation: pa_allocated=%uB, "
        "rounded_size=%uB, pa_budget=%uB",
        state->pa_allocated, rounded_size, state->pa_budget));
  }

  ASSIGN_OR_RETURN(auto raw_alloc,
                   CreateAllocation(state->executor, allocation_size));
  const uint64_t padded_size = raw_alloc->address().size();
  if (mapping_size > padded_size) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Mapping size %u exceeds raw allocation size %u",
                        mapping_size, padded_size));
  }

  ASSIGN_OR_RETURN(
      MemoryReservation::ScopedMapping reservation_mapping,
      reservation->MapTo(reservation_offset, /*allocation_offset=*/0,
                         mapping_size, *raw_alloc));

  std::unique_ptr<MemoryReservation> allocator_reservation;
  MemoryReservation::ScopedMapping allocator_mapping;
  DeviceAddressBase allocator_address = reservation_address;

  if (!return_reservation_address) {
    ASSIGN_OR_RETURN(allocator_reservation,
                     CreateReservation(state->executor, allocation_size));
    ASSIGN_OR_RETURN(allocator_mapping,
                     allocator_reservation->MapTo(
                         /*reservation_offset=*/0, /*allocation_offset=*/0,
                         padded_size, *raw_alloc));
    allocator_address = DeviceAddressBase(
        allocator_reservation->address().opaque(), allocation_size);
  }

  void* allocator_ptr = allocator_address.opaque();
  state->raw_allocations.emplace(allocator_ptr, std::move(raw_alloc));
  state->scoped_mappings.emplace(allocator_ptr, std::move(allocator_mapping));
  if (allocator_reservation != nullptr) {
    state->reservations.emplace(allocator_ptr,
                                std::move(allocator_reservation));
    state->active_reservation_mappings.emplace(
        reservation_address.opaque(),
        ReservationMapping{allocator_address, reservation_address, reservation,
                           reservation_offset, mapping_size,
                           std::move(reservation_mapping)});
  } else {
    state->scoped_mappings[allocator_ptr] = std::move(reservation_mapping);
  }

  state->pa_allocated += rounded_size;
  if (multi_device) {
    state->multi_device_allocations.insert({allocator_ptr, true});
  }
  return ScopedDeviceAddress<uint8_t>(allocator_address, device_ordinal, this);
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

  const bool multi_device = CurrentMultiDevice();

  absl::MutexLock lock(state->mu);

  // Try to reuse a completed pending deallocation with matching size.
  std::optional<DeviceAddressBase> reused =
      TryReusePendingDeallocation(*state, size, multi_device);
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

  if (multi_device)
    state->multi_device_allocations.insert({result->opaque(), true});

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

  absl::MutexLock lock(state->mu);

  if (!state->raw_allocations.contains(mem.opaque())) {
    if (state->active_reservation_mappings.contains(mem.opaque()) ||
        state->stale_reservation_mappings.contains(mem.opaque())) {
      return absl::InvalidArgumentError(
          "DeviceAddressVmmAllocator::Deallocate does not accept reservation "
          "alias addresses; use UnMap instead");
    }
    return absl::InvalidArgumentError(absl::StrFormat(
        "DeviceAddressVmmAllocator::Deallocate received an unknown address %p",
        mem.opaque()));
  }

  for (const auto& [_, mapping] : state->active_reservation_mappings) {
    if (mapping.allocator_address.IsSameAs(mem)) {
      return absl::FailedPreconditionError(
          "DeviceAddressVmmAllocator::Deallocate requires active reservation "
          "aliases to be released with UnMap first");
    }
  }

  VLOG(3) << absl::StreamFormat(
      "Queueing deferred deallocation for virtual address %p (size=%uB) "
      "on device ordinal %d",
      mem.opaque(), mem.size(), device_ordinal);

  bool multi_device = state->multi_device_allocations.erase(mem.opaque()) > 0;

  // Assign the next sequence number and enqueue a GPU write to the pinned
  // timeline when the stream reaches this point. The CPU polls the timeline
  // value to know when it is safe to free the memory.
  uint64_t seqno = state->next_seqno++;
  RETURN_IF_ERROR(EnqueueDeferredDeallocation(*state, seqno));

  state->pending_deallocations.push_back(
      {PendingDeallocationKind::kAllocate, mem, seqno, multi_device});

  return absl::OkStatus();
}

absl::Status DeviceAddressVmmAllocator::Map(int device_ordinal,
                                            DeviceAddressBase addr,
                                            MemoryReservation* reservation,
                                            uint64_t reservation_offset,
                                            uint64_t size) {
  if (size == 0) {
    return absl::OkStatus();
  }
  if (addr.is_null()) {
    return absl::InvalidArgumentError(
        "DeviceAddressVmmAllocator::Map requires a non-null source address");
  }
  ASSIGN_OR_RETURN(
      DeviceAddressBase reservation_address,
      ValidateReservationRange(reservation, reservation_offset, size));

  PerDeviceState* state = GetPerDeviceState(device_ordinal);
  if (state == nullptr) {
    return DeviceNotFoundError(device_ordinal);
  }

  absl::MutexLock lock(state->mu);
  auto raw_it = state->raw_allocations.find(addr.opaque());
  if (raw_it == state->raw_allocations.end()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "DeviceAddressVmmAllocator::Map received an unknown allocator address "
        "%p",
        addr.opaque()));
  }
  if (size > raw_it->second->address().size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "DeviceAddressVmmAllocator::Map size %u exceeds raw allocation size "
        "%u",
        size, raw_it->second->address().size()));
  }
  if (state->active_reservation_mappings.contains(
          reservation_address.opaque()) ||
      state->stale_reservation_mappings.contains(
          reservation_address.opaque())) {
    return absl::FailedPreconditionError(
        "Reservation address is already tracked by this allocator");
  }
  for (const auto& [_, mapping] : state->active_reservation_mappings) {
    if (mapping.allocator_address.IsSameAs(addr)) {
      return absl::FailedPreconditionError(
          "Allocator address already has an active reservation alias");
    }
  }

  ASSIGN_OR_RETURN(
      MemoryReservation::ScopedMapping scoped_mapping,
      reservation->MapTo(reservation_offset, /*allocation_offset=*/0, size,
                         *raw_it->second));
  state->active_reservation_mappings.emplace(
      reservation_address.opaque(),
      ReservationMapping{addr, reservation_address, reservation,
                         reservation_offset, size, std::move(scoped_mapping)});
  return absl::OkStatus();
}

absl::Status DeviceAddressVmmAllocator::UnMap(int device_ordinal,
                                              MemoryReservation* reservation,
                                              uint64_t reservation_offset,
                                              uint64_t size) {
  if (size == 0) {
    return absl::OkStatus();
  }
  ASSIGN_OR_RETURN(
      DeviceAddressBase reservation_address,
      ValidateReservationRange(reservation, reservation_offset, size));

  PerDeviceState* state = GetPerDeviceState(device_ordinal);
  if (state == nullptr) {
    return DeviceNotFoundError(device_ordinal);
  }

  absl::MutexLock lock(state->mu);
  auto it =
      state->active_reservation_mappings.find(reservation_address.opaque());
  if (it == state->active_reservation_mappings.end()) {
    return absl::InvalidArgumentError(
        "DeviceAddressVmmAllocator::UnMap received an untracked reservation "
        "address");
  }
  if (it->second.reservation != reservation ||
      it->second.reservation_offset != reservation_offset ||
      it->second.size != size) {
    return absl::InvalidArgumentError(
        "DeviceAddressVmmAllocator::UnMap requires the same full reservation "
        "range passed to Map");
  }

  uint64_t seqno = state->next_seqno++;
  RETURN_IF_ERROR(EnqueueDeferredDeallocation(*state, seqno));

  ReservationMapping mapping = std::move(it->second);
  state->active_reservation_mappings.erase(it);
  state->stale_reservation_mappings.emplace(reservation_address.opaque(),
                                            std::move(mapping));
  state->pending_deallocations.push_back(
      {PendingDeallocationKind::kMap, reservation_address, seqno});
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

absl::Status DeviceAddressVmmAllocator::SynchronizePendingOperations(
    int device_ordinal) {
  PerDeviceState* state = GetPerDeviceState(device_ordinal);
  if (state == nullptr) {
    return DeviceNotFoundError(device_ordinal);
  }

  uint64_t target_seqno;
  {
    absl::MutexLock lock(state->mu);
    if (state->pending_deallocations.empty()) {
      return absl::OkStatus();
    }
    target_seqno = state->pending_deallocations.back().seqno;
  }

  while (LoadTimeline(state->pinned_timeline) < target_seqno) {
    absl::SleepFor(kGpuTimelinePollInterval);
  }

  {
    absl::MutexLock lock(state->mu);
    while (!state->pending_deallocations.empty() &&
           state->pending_deallocations.front().seqno <= target_seqno) {
      if (state->pending_deallocations.front().kind ==
          PendingDeallocationKind::kAllocate) {
        DoDeallocate(*state, state->pending_deallocations.front().mem);
      } else {
        DoUnMap(*state, state->pending_deallocations.front().mem);
      }
      state->pending_deallocations.pop_front();
    }
  }

  return absl::OkStatus();
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
  absl::MutexLock lock(state->mu);
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
  absl::MutexLock lock(state->mu);
  auto it = state->reservations.find(addr.opaque());
  if (it == state->reservations.end()) {
    return nullptr;
  }
  return it->second.get();
}

uint64_t DeviceAddressVmmAllocator::GetAllocationGranularity(
    StreamExecutor* executor) const {
  PerDeviceState* state = GetPerDeviceState(executor->device_ordinal());
  if (state == nullptr) {
    return 0;
  }
  return state->allocation_granularity;
}

std::optional<DeviceAddressBase>
DeviceAddressVmmAllocator::TryReusePendingDeallocation(PerDeviceState& state,
                                                       uint64_t size,
                                                       bool multi_device) {
  uint64_t rounded_size = RoundUpToGranularity(state, size);
  for (auto it = state.pending_deallocations.begin();
       it != state.pending_deallocations.end(); ++it) {
    if (it->kind != PendingDeallocationKind::kAllocate) {
      continue;
    }
    if (it->multi_device != multi_device) {
      continue;
    }
    if (!state.reservations.contains(it->mem.opaque())) {
      continue;
    }
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
    if (multi_device)
      state.multi_device_allocations.insert({reused_mem.opaque(), true});

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
