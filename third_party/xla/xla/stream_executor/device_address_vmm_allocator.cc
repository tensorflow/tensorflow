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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
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
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

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

DeviceAddressVmmAllocator::AllocationRecord::AllocationRecord(
    Kind kind, DeviceAddressBase allocator_address,
    std::shared_ptr<MemoryAllocation> raw_allocation,
    std::unique_ptr<MemoryReservation> allocator_address_reservation,
    MemoryReservation::ScopedMapping allocator_address_mapping,
    bool multi_device)
    : kind_(kind),
      allocator_address_(allocator_address),
      raw_allocation_(std::move(raw_allocation)),
      multi_device_(multi_device),
      allocator_address_reservation_(std::move(allocator_address_reservation)),
      allocator_address_mapping_(std::move(allocator_address_mapping)) {
  CHECK(raw_allocation_ != nullptr);
  CHECK(!allocator_address_.is_null());
  switch (kind_) {
    case Kind::kAllocate:
    case Kind::kAllocateAndMapReturnNewAddr:
      CHECK(allocator_address_reservation_ != nullptr);
      break;
    case Kind::kAllocateAndMapReturnMapAddr:
      CHECK(allocator_address_reservation_ == nullptr);
      break;
  }
  CHECK(allocator_address_mapping_.has_value());
}

DeviceAddressVmmAllocator::PendingDeallocationKind
DeviceAddressVmmAllocator::AllocationRecord::pending_deallocation_kind() const {
  switch (kind_) {
    case Kind::kAllocate:
      return PendingDeallocationKind::kAllocate;
    case Kind::kAllocateAndMapReturnMapAddr:
      return PendingDeallocationKind::kAllocateAndMapReturnMapAddr;
    case Kind::kAllocateAndMapReturnNewAddr:
      return PendingDeallocationKind::kAllocateAndMapReturnNewAddr;
  }
  LOG(FATAL) << "Unknown AllocationRecord kind";
  return PendingDeallocationKind::kAllocate;
}

DeviceAddressBase
DeviceAddressVmmAllocator::AllocationRecord::reservation_address() const {
  CHECK(reservation_address_.has_value());
  return *reservation_address_;
}

bool DeviceAddressVmmAllocator::AllocationRecord::reservation_mapping_matches(
    DeviceAddressBase address) const {
  return reservation_address_mapping_.has_value() &&
         reservation_address_mapping_->mapped_address().IsSameAs(address);
}

void DeviceAddressVmmAllocator::AllocationRecord::MarkAllocatorStale(
    uint64_t seqno) {
  CHECK(allocator_active());
  CHECK(!allocator_stale());
  CHECK(has_allocator_address_mapping());
  CHECK_NE(seqno, 0);
  allocator_state_ = AllocatorState::kStale;
  allocator_stale_seqno_ = seqno;
}

void DeviceAddressVmmAllocator::AllocationRecord::ReactivateAllocator(
    uint64_t new_size) {
  CHECK(!allocator_active());
  CHECK(allocator_stale());
  CHECK(has_allocator_address_mapping());
  allocator_address_ = DeviceAddressBase(allocator_address_.opaque(), new_size);
  allocator_state_ = AllocatorState::kActive;
  allocator_stale_seqno_ = 0;
}

void DeviceAddressVmmAllocator::AllocationRecord::CompleteStaleAllocator() {
  CHECK(!allocator_active());
  CHECK(allocator_stale());
  CHECK(!reservation_active());
  allocator_address_mapping_.reset();
  allocator_address_reservation_.reset();
  raw_allocation_.reset();
}

void DeviceAddressVmmAllocator::AllocationRecord::AddActiveReservationAlias(
    DeviceAddressBase reservation_address,
    MemoryReservation::ScopedMapping reservation_address_mapping) {
  CHECK(!has_reservation_alias());
  CHECK(!reservation_address.is_null());
  reservation_address_ = reservation_address;
  reservation_address_mapping_.emplace(std::move(reservation_address_mapping));
  reservation_state_ = ReservationState::kActive;
}

void DeviceAddressVmmAllocator::AllocationRecord::MarkReservationStale(
    uint64_t seqno) {
  CHECK(reservation_active());
  CHECK(!reservation_stale());
  CHECK(has_reservation_address());
  CHECK(reservation_address_mapping_.has_value());
  CHECK_NE(seqno, 0);
  reservation_state_ = ReservationState::kStale;
  reservation_stale_seqno_ = seqno;
}

void DeviceAddressVmmAllocator::AllocationRecord::ReactivateReservation() {
  CHECK(!reservation_active());
  CHECK(reservation_stale());
  CHECK(has_reservation_address());
  CHECK(reservation_address_mapping_.has_value());
  reservation_state_ = ReservationState::kActive;
  reservation_stale_seqno_ = 0;
}

void DeviceAddressVmmAllocator::AllocationRecord::CompleteStaleReservation() {
  if (!reservation_stale()) {
    return;
  }
  CHECK(!reservation_active());
  CHECK(has_reservation_address());
  reservation_address_mapping_.reset();
  reservation_address_.reset();
  reservation_state_ = ReservationState::kNone;
  reservation_stale_seqno_ = 0;
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

  for (auto& device : per_device_) {
    auto& state = device.second;
    // Free platform-specific per-device resources (e.g. pinned timeline).
    if (state->destroy_fn) {
      state->destroy_fn();
    }
  }
}

absl::Status DeviceAddressVmmAllocator::SynchronizeAllPendingOperations() {
  for (auto& device : per_device_) {
    RETURN_IF_ERROR(SynchronizePendingOperations(device.first));
  }
  return absl::OkStatus();
}

// Common helpers and accessors.

absl::StatusOr<DeviceAddressVmmAllocator::PerDeviceState*>
DeviceAddressVmmAllocator::GetPerDeviceState(int device_ordinal) const {
  auto it = per_device_.find(device_ordinal);
  if (it == per_device_.end()) {
    return absl::NotFoundError(
        absl::StrFormat("No device with ordinal %d registered in "
                        "DeviceAddressVmmAllocator",
                        device_ordinal));
  }
  return it->second.get();
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

absl::StatusOr<Stream*> DeviceAddressVmmAllocator::GetStream(
    int device_ordinal) {
  ASSIGN_OR_RETURN(auto state, GetPerDeviceState(device_ordinal));
  return state->stream;
}

absl::Status DeviceAddressVmmAllocator::SynchronizePendingOperations(
    int device_ordinal) {
  ASSIGN_OR_RETURN(auto state, GetPerDeviceState(device_ordinal));
  absl::MutexLock lock(state->mu);
  if (state->pending_deallocations.empty()) {
    return absl::OkStatus();
  }
  return WaitAndDrainPendingDeallocationsUntilSeqno(
      *state, state->pending_deallocations.back().seqno);
}

absl::StatusOr<StreamExecutor*> DeviceAddressVmmAllocator::GetStreamExecutor(
    int device_ordinal) const {
  ASSIGN_OR_RETURN(auto state, GetPerDeviceState(device_ordinal));
  return state->executor;
}

MemoryAllocation* DeviceAddressVmmAllocator::GetRawAllocation(
    int device_ordinal, DeviceAddressBase addr) const {
  absl::StatusOr<PerDeviceState*> state_or = GetPerDeviceState(device_ordinal);
  if (!state_or.ok()) {
    return nullptr;
  }
  PerDeviceState* state = *state_or;
  absl::MutexLock lock(state->mu);

  // Allocator addresses are keyed directly by their VA. Stale records remain in
  // this map until deferred teardown completes, so require both active state
  // and an exact address-range match before exposing the backing allocation.
  auto allocation_it = state->records_by_allocator_address.find(addr.opaque());
  if (allocation_it != state->records_by_allocator_address.end() &&
      allocation_it->second->allocator_active() &&
      allocation_it->second->allocator_matches(addr)) {
    return allocation_it->second->raw_allocation();
  }

  // Reservation aliases created by Map() or by Allocate(...,
  // return_reservation_address=false) are tracked in a separate active-only
  // index. Stale or already-unmapped aliases intentionally return nullptr.
  auto reservation_it = state->active_reservation_records.find(addr.opaque());
  if (reservation_it != state->active_reservation_records.end()) {
    return reservation_it->second->raw_allocation();
  }
  return nullptr;
}

MemoryReservation* DeviceAddressVmmAllocator::GetReservation(
    int device_ordinal, DeviceAddressBase addr) const {
  absl::StatusOr<PerDeviceState*> state_or = GetPerDeviceState(device_ordinal);
  if (!state_or.ok()) {
    return nullptr;
  }
  PerDeviceState* state = *state_or;
  absl::MutexLock lock(state->mu);

  auto allocation_it = state->records_by_allocator_address.find(addr.opaque());
  if (allocation_it != state->records_by_allocator_address.end() &&
      allocation_it->second->allocator_active() &&
      allocation_it->second->allocator_matches(addr)) {
    return allocation_it->second->allocator_address_reservation();
  }

  return nullptr;
}

uint64_t DeviceAddressVmmAllocator::GetAllocationGranularity(
    StreamExecutor* executor) const {
  absl::StatusOr<PerDeviceState*> state_or =
      GetPerDeviceState(executor->device_ordinal());
  if (!state_or.ok()) {
    return 0;
  }
  PerDeviceState* state = *state_or;
  return state->allocation_granularity;
}

// Allocate helpers.

void* DeviceAddressVmmAllocator::TrackAllocatorAddressMappedAllocation(
    PerDeviceState& state, AllocationRecord::Kind kind,
    DeviceAddressBase allocator_address,
    std::shared_ptr<MemoryAllocation> raw_allocation,
    std::unique_ptr<MemoryReservation> reservation,
    MemoryReservation::ScopedMapping mapping, uint64_t allocated_size,
    bool multi_device) {
  void* va_ptr = allocator_address.opaque();
  auto record = std::make_unique<AllocationRecord>(
      kind, allocator_address, std::move(raw_allocation),
      std::move(reservation), std::move(mapping), multi_device);
  auto insert_result =
      state.records_by_allocator_address.emplace(va_ptr, std::move(record));
  CHECK(insert_result.second);
  state.pa_allocated += allocated_size;
  return va_ptr;
}

// Shared pending-reclaim retry flow:
//
// TryWithPendingReclaim(reclaim_size, try_reuse, try_fresh)
//           │
//           ▼
// ┌─────────────────────────────────┐
// │ try_reuse()                     │──found──► return reused address
// └─────────────────────────────────┘
//           │ not found
//           ▼
// ┌─────────────────────────────────┐
// │ try_fresh()                     │──OK──► return fresh address
// └─────────────────────────────────┘
//           │ ResourceExhausted
//           ▼
// ┌─────────────────────────────────┐
// │ Process completed pending       │
// │ operations                      │
// └─────────────────────────────────┘
//           │
//           ▼
// ┌─────────────────────────────────┐
// │ try_fresh()                     │──OK──► return fresh address
// └─────────────────────────────────┘
//           │ ResourceExhausted
//           ▼
// ┌─────────────────────────────────┐
// │ Wait for pending operations     │
// │ to reclaim enough memory        │
// └─────────────────────────────────┘
//           │
//           ▼
// ┌─────────────────────────────────┐
// │ try_fresh()                     │──OK──► return fresh address
// └─────────────────────────────────┘
//           │ failed
//           ▼
//       return error
template <typename TryReuseFn, typename TryFreshFn>
absl::StatusOr<DeviceAddressBase>
DeviceAddressVmmAllocator::TryWithPendingReclaim(PerDeviceState& state,
                                                 uint64_t reclaim_size,
                                                 TryReuseFn try_reuse,
                                                 TryFreshFn try_fresh) {
  // First try to reactivate a compatible pending deallocation without waiting.
  // Reuse is stream-order safe and avoids both a fresh VMM allocation and any
  // host-side wait for the GPU timeline.
  ASSIGN_OR_RETURN(std::optional<DeviceAddressBase> reused, try_reuse());
  if (reused.has_value()) {
    return *reused;
  }

  // If no pending entry matches, try the normal fresh allocation path. Most
  // calls should finish here; the reclaim paths below are only for PA budget
  // pressure or allocator-level allocation failures.
  absl::StatusOr<DeviceAddressBase> result = try_fresh();

  if (absl::IsResourceExhausted(result.status())) {
    // A ResourceExhausted error may be stale: some pending deallocations can
    // already be past their stream timeline point. Complete ready allocator
    // deallocations first, without blocking for later pending work and without
    // destroying unrelated stale reservation mappings that may be reused.
    CompleteReadyAllocatorDeallocationsForReclaim(
        state, LoadTimeline(state.pinned_timeline));
    result = try_fresh();
  }

  if (absl::IsResourceExhausted(result.status())) {
    // If completed pending work was not enough, wait until enough queued frees
    // should be reclaimable for this request, then retry once more. This is the
    // only path that may block while the GPU drains earlier stream work.
    // Select enough pending allocator-address deallocations to cover this
    // request, then wait for the selected tail seqno to become safe. Unrelated
    // kMap entries do not own physical memory, so leave them stale and
    // reusable.
    if (!state.pending_deallocations.empty()) {
      uint64_t accumulated_size = 0;
      uint64_t rounded_size = RoundUpToGranularity(state, reclaim_size);
      uint64_t target_seqno = 0;
      std::vector<PendingDeallocationKey> selected;

      // Target 1.1x the requested size to provide some headroom.
      uint64_t target_size = rounded_size + rounded_size / 10;

      for (const PendingDeallocation& pending : state.pending_deallocations) {
        if (pending.kind == PendingDeallocationKind::kMap) {
          continue;
        }
        auto record_it =
            state.records_by_allocator_address.find(pending.addr.opaque());
        CHECK(record_it != state.records_by_allocator_address.end());
        CHECK(record_it->second->allocator_stale());
        CHECK(record_it->second->allocator_matches(pending.addr));
        CHECK(record_it->second->raw_allocation() != nullptr);
        accumulated_size += RoundUpToGranularity(
            state, record_it->second->raw_allocation()->address().size());
        target_seqno = std::max(target_seqno, pending.seqno);
        selected.push_back(
            PendingDeallocationKey{pending.kind, pending.seqno, pending.addr});
        if (accumulated_size >= target_size) {
          break;
        }
      }

      if (!selected.empty()) {
        RETURN_IF_ERROR(WaitUntilSeqno(state, target_seqno));
        for (const PendingDeallocationKey& key : selected) {
          CompletePendingDeallocationByKey(state, key);
        }
      }
    }
    result = try_fresh();
  }

  return result;
}

// Allocate() reuses pending kAllocate entries, otherwise tries a fresh
// allocator-address mapping.
absl::StatusOr<ScopedDeviceAddress<uint8_t>>
DeviceAddressVmmAllocator::Allocate(int device_ordinal, uint64_t size,
                                    bool /*retry_on_failure*/,
                                    int64_t /*memory_space*/) {
  if (size == 0) {
    return ScopedDeviceAddress<uint8_t>(DeviceAddressBase(), device_ordinal,
                                        this);
  }

  ASSIGN_OR_RETURN(auto state, GetPerDeviceState(device_ordinal));
  const bool multi_device = CurrentMultiDevice();

  absl::MutexLock lock(state->mu);
  auto try_reuse = [&]() ABSL_NO_THREAD_SAFETY_ANALYSIS
      -> absl::StatusOr<std::optional<DeviceAddressBase>> {
    uint64_t rounded_size = RoundUpToGranularity(*state, size);
    for (auto it = state->pending_deallocations.begin();
         it != state->pending_deallocations.end(); ++it) {
      if (it->kind != PendingDeallocationKind::kAllocate) {
        continue;
      }
      auto record_it =
          state->records_by_allocator_address.find(it->addr.opaque());
      CHECK(record_it != state->records_by_allocator_address.end());
      AllocationRecord& record = *record_it->second;
      CHECK(record.allocator_stale());
      CHECK(record.allocator_matches(it->addr));
      if (record.multi_device() != multi_device) {
        continue;
      }
      if (RoundUpToGranularity(*state, record.allocator_address().size()) !=
          rounded_size) {
        continue;
      }

      DeviceAddressBase reused_mem(record.allocator_key(), size);
      MoveAllocatorRecordToActive(*state, record, size);
      ErasePendingDeallocationAt(*state, it);

      return std::optional<DeviceAddressBase>(reused_mem);
    }

    return std::optional<DeviceAddressBase>();
  };
  auto try_fresh =
      [&]()
          ABSL_NO_THREAD_SAFETY_ANALYSIS -> absl::StatusOr<DeviceAddressBase> {
    uint64_t rounded_size = RoundUpToGranularity(*state, size);
    if (state->pa_allocated + rounded_size > state->pa_budget) {
      return absl::StatusOr<DeviceAddressBase>(
          absl::ResourceExhaustedError(absl::StrFormat(
              "Not enough PA budget for allocation: pa_allocated=%uB, "
              "rounded_size=%uB, pa_budget=%uB",
              state->pa_allocated, rounded_size, state->pa_budget)));
    }

    ASSIGN_OR_RETURN(auto raw_alloc, CreateAllocation(state->executor, size));
    const uint64_t padded_size = raw_alloc->address().size();

    ASSIGN_OR_RETURN(auto reservation,
                     CreateReservation(state->executor, size));

    ASSIGN_OR_RETURN(
        auto scoped_mapping,
        reservation->MapTo(/*reservation_offset=*/0, /*allocation_offset=*/0,
                           padded_size, *raw_alloc));

    auto shared_raw = std::shared_ptr<MemoryAllocation>(std::move(raw_alloc));
    DeviceAddressBase allocator_address(reservation->address().opaque(), size);
    void* va_ptr = TrackAllocatorAddressMappedAllocation(
        *state, AllocationRecord::Kind::kAllocate, allocator_address,
        std::move(shared_raw), std::move(reservation),
        std::move(scoped_mapping), rounded_size, multi_device);
    // Return the original requested size, not the padded size.
    return absl::StatusOr<DeviceAddressBase>(DeviceAddressBase(va_ptr, size));
  };

  absl::StatusOr<DeviceAddressBase> result =
      TryWithPendingReclaim(*state, size, try_reuse, try_fresh);

  if (!result.ok()) {
    return result.status();
  }

  VLOG(3) << absl::StreamFormat(
      "Allocated virtual address %p (%uB) on device ordinal %d",
      result->opaque(), size, device_ordinal);

  return ScopedDeviceAddress<uint8_t>(*result, device_ordinal, this);
}

// Mapped Allocate() creates fresh physical memory and maps it into the caller
// reservation. It keeps the same externally visible ownership model as the
// previous map-based bookkeeping, but records the lifetime in AllocationRecord.
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

  ASSIGN_OR_RETURN(auto state, GetPerDeviceState(device_ordinal));
  const bool multi_device = CurrentMultiDevice();

  ASSIGN_OR_RETURN(
      DeviceAddressBase reservation_address,
      ValidateReservationRange(reservation, reservation_offset, mapping_size));

  absl::MutexLock lock(state->mu);
  if (state->active_reservation_records.contains(
          reservation_address.opaque()) ||
      state->stale_reservation_records.contains(reservation_address.opaque()) ||
      state->records_by_allocator_address.contains(
          reservation_address.opaque())) {
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
  auto shared_raw = std::shared_ptr<MemoryAllocation>(std::move(raw_alloc));

  if (return_reservation_address) {
    TrackAllocatorAddressMappedAllocation(
        *state, AllocationRecord::Kind::kAllocateAndMapReturnMapAddr,
        reservation_address, std::move(shared_raw), nullptr,
        std::move(reservation_mapping), rounded_size, multi_device);
    return ScopedDeviceAddress<uint8_t>(reservation_address, device_ordinal,
                                        this);
  }

  ASSIGN_OR_RETURN(auto allocator_address_reservation,
                   CreateReservation(state->executor, allocation_size));
  ASSIGN_OR_RETURN(auto allocator_address_mapping,
                   allocator_address_reservation->MapTo(
                       /*reservation_offset=*/0, /*allocation_offset=*/0,
                       padded_size, *shared_raw));
  void* allocator_va = allocator_address_reservation->address().opaque();
  DeviceAddressBase allocator_address(allocator_va, allocation_size);
  auto record = std::make_unique<AllocationRecord>(
      AllocationRecord::Kind::kAllocateAndMapReturnNewAddr, allocator_address,
      std::move(shared_raw), std::move(allocator_address_reservation),
      std::move(allocator_address_mapping), multi_device);
  record->AddActiveReservationAlias(reservation_address,
                                    std::move(reservation_mapping));
  AllocationRecord* record_ptr = record.get();
  auto record_insert = state->records_by_allocator_address.emplace(
      allocator_va, std::move(record));
  CHECK(record_insert.second);
  auto reservation_insert = state->active_reservation_records.emplace(
      reservation_address.opaque(), record_ptr);
  CHECK(reservation_insert.second);
  state->pa_allocated += rounded_size;

  return ScopedDeviceAddress<uint8_t>(allocator_address, device_ordinal, this);
}

absl::Status DeviceAddressVmmAllocator::Deallocate(int device_ordinal,
                                                   DeviceAddressBase mem) {
  if (mem.is_null()) {
    return absl::OkStatus();
  }

  ASSIGN_OR_RETURN(auto state, GetPerDeviceState(device_ordinal));

  absl::MutexLock lock(state->mu);

  auto record_it = state->records_by_allocator_address.find(mem.opaque());
  if (record_it == state->records_by_allocator_address.end() ||
      !record_it->second->allocator_active() ||
      !record_it->second->allocator_matches(mem)) {
    return absl::NotFoundError(absl::StrFormat(
        "virtual address %p is not an active allocator address returned by "
        "Allocate()",
        mem.opaque()));
  }
  AllocationRecord& record = *record_it->second;
  CHECK(!state->active_reservation_records.contains(mem.opaque()));
  if (record.reservation_active()) {
    CHECK(record.has_reservation_address());
    return absl::FailedPreconditionError(absl::StrFormat(
        "Deallocate() requires the active reservation alias at virtual address "
        "%p (%uB) to be released with UnMap() first",
        record.reservation_address().opaque(),
        record.reservation_address().size()));
  }

  VLOG(3) << absl::StreamFormat(
      "Queueing deferred deallocation for virtual address %p (size=%uB) "
      "on device ordinal %d",
      mem.opaque(), mem.size(), state->executor->device_ordinal());

  const uint64_t reclaimable_bytes =
      RoundUpToGranularity(*state, record.raw_allocation()->address().size());

  // Assign the next sequence number and enqueue a GPU write to the pinned
  // timeline when the stream reaches this point. The CPU polls the timeline
  // value to know when it is safe to free the memory.
  uint64_t seqno = state->next_seqno++;
  RETURN_IF_ERROR(EnqueueDeferredDeallocation(*state, seqno));
  // Move the returned allocator address out of active ownership and keep its
  // mapping alive as stale state until the stream reaches `seqno`.
  CHECK(record.allocator_active());
  CHECK(!record.allocator_stale());
  CHECK(record.has_allocator_address_mapping());
  void* allocator_va = record.allocator_key();
  auto allocator_record_it =
      state->records_by_allocator_address.find(allocator_va);
  CHECK(allocator_record_it != state->records_by_allocator_address.end());
  CHECK_EQ(allocator_record_it->second.get(), &record);
  record.MarkAllocatorStale(seqno);
  state->pending_deallocations.push_back(
      PendingDeallocation{record.pending_deallocation_kind(), seqno,
                          record.allocator_address(), reclaimable_bytes});
  return absl::OkStatus();
}

// Map helpers.

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

absl::Status DeviceAddressVmmAllocator::Map(int device_ordinal,
                                            DeviceAddressBase addr,
                                            MemoryReservation* reservation,
                                            uint64_t reservation_offset,
                                            uint64_t size) {
  ASSIGN_OR_RETURN(auto state, GetPerDeviceState(device_ordinal));
  if (size == 0) {
    return absl::OkStatus();
  }
  if (addr.is_null()) {
    return absl::InvalidArgumentError("addr must not be null");
  }

  // Map() does not allocate a VA range. It maps the physical allocation backing
  // `addr` into the caller-owned reservation slice, so validate the slice
  // before taking the allocator lock.
  ASSIGN_OR_RETURN(
      DeviceAddressBase reservation_address,
      ValidateReservationRange(reservation, reservation_offset, size));

  absl::MutexLock lock(state->mu);
  auto resolve_source_record =
      [&]()
          ABSL_NO_THREAD_SAFETY_ANALYSIS -> absl::StatusOr<AllocationRecord*> {
    auto allocation_it =
        state->records_by_allocator_address.find(addr.opaque());
    if (allocation_it == state->records_by_allocator_address.end() ||
        !allocation_it->second->allocator_active() ||
        !allocation_it->second->allocator_matches(addr)) {
      return absl::NotFoundError(absl::StrFormat(
          "addr %p is not an active allocator address, when trying to "
          "do map of VA reservation to existing physical allocation, we "
          "requires the buffer being mapped to is being allocated through "
          "DeviceAddressVmmAllocator, check the allocator type for the "
          "buffer.",
          addr.opaque()));
    }
    return allocation_it->second.get();
  };

  // Resolve the source address to the raw physical allocation that is currently
  // mapped there. Any active allocator address returned by this allocator is
  // accepted as a Map() source.
  ASSIGN_OR_RETURN(AllocationRecord * source_record, resolve_source_record());
  MemoryAllocation* raw_allocation = source_record->raw_allocation();
  if (size > raw_allocation->address().size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "mapping size must not exceed physical allocation size: "
        "mapping_size=%uB, allocation_size=%uB",
        size, raw_allocation->address().size()));
  }
  if (state->active_reservation_records.contains(
          reservation_address.opaque()) ||
      state->stale_reservation_records.contains(reservation_address.opaque())) {
    return absl::FailedPreconditionError(
        "Reservation address is already tracked by this allocator");
  }
  if (source_record->reservation_active()) {
    return absl::FailedPreconditionError(
        "Allocator address already has an active reservation alias");
  }
  if (source_record->reservation_stale()) {
    return absl::FailedPreconditionError(
        "Allocator address already has a pending reservation alias");
  }

  // Install the reservation address mapping to the raw physical allocation. The
  // allocation_offset is zero because Map() aliases the beginning of
  // the source allocation; callers pass the target VA location through
  // `reservation_offset`.
  ASSIGN_OR_RETURN(auto mapping, reservation->MapTo(reservation_offset,
                                                    /*allocation_offset=*/0,
                                                    size, *raw_allocation));
  DeviceAddressBase mapped = mapping.mapped_address();
  // The reservation slice was computed before locking. Verify the platform
  // returned the exact reservation address before recording allocator
  // bookkeeping.
  if (!mapped.IsSameAs(reservation_address)) {
    return absl::InternalError(absl::StrFormat(
        "Map() mapped unexpected virtual address: expected=%p, actual=%p",
        reservation_address.opaque(), mapped.opaque()));
  }
  // Track this as a Map()-owned reservation alias. This only updates the
  // reservation-address index; no new physical allocation is created, so
  // pa_allocated does not change.
  CHECK(!source_record->reservation_active());
  CHECK(!source_record->reservation_stale());
  source_record->AddActiveReservationAlias(mapped, std::move(mapping));
  auto mapping_insert_result =
      state->active_reservation_records.emplace(mapped.opaque(), source_record);
  CHECK(mapping_insert_result.second);
  return absl::OkStatus();
}

// UnMap/deferred teardown helpers.

void DeviceAddressVmmAllocator::ErasePendingDeallocationAt(
    PerDeviceState& state, std::deque<PendingDeallocation>::iterator it) {
  CHECK(it != state.pending_deallocations.end());
  state.pending_deallocations.erase(it);
}

void DeviceAddressVmmAllocator::MoveAllocatorRecordToActive(
    PerDeviceState& state, AllocationRecord& record, uint64_t new_size) {
  CHECK(!record.allocator_active());
  CHECK(record.allocator_stale());
  CHECK(record.has_allocator_address_mapping());
  void* allocator_va = record.allocator_key();
  auto record_it = state.records_by_allocator_address.find(allocator_va);
  CHECK(record_it != state.records_by_allocator_address.end());
  CHECK_EQ(record_it->second.get(), &record);
  record.ReactivateAllocator(new_size);
}

void DeviceAddressVmmAllocator::MoveReservationRecordToStale(
    PerDeviceState& state, AllocationRecord& record, uint64_t seqno) {
  CHECK(record.reservation_active());
  CHECK(!record.reservation_stale());
  CHECK(record.has_reservation_address());
  void* reservation_va = record.reservation_key();
  CHECK_EQ(state.active_reservation_records.erase(reservation_va), 1);
  auto insert_result =
      state.stale_reservation_records.emplace(reservation_va, &record);
  CHECK(insert_result.second);
  record.MarkReservationStale(seqno);
}

void DeviceAddressVmmAllocator::CompleteStaleReservationMapping(
    PerDeviceState& state, AllocationRecord& record) {
  if (!record.reservation_stale()) {
    return;
  }
  CHECK(!record.reservation_active());
  CHECK(record.has_reservation_address());
  void* reservation_va = record.reservation_key();
  auto stale_it = state.stale_reservation_records.find(reservation_va);
  if (stale_it != state.stale_reservation_records.end()) {
    CHECK_EQ(stale_it->second, &record);
    state.stale_reservation_records.erase(stale_it);
  }
  record.CompleteStaleReservation();
}

absl::Status DeviceAddressVmmAllocator::WaitUntilSeqno(PerDeviceState& state,
                                                       uint64_t target_seqno) {
  // Release the lock before spin-waiting to avoid stalling other threads for
  // potentially milliseconds while the GPU drains its work queue.
  state.mu.unlock();

  // Poll until the GPU writes a timeline value >= target_seqno.
  // Since timeline values are written in stream order, this guarantees all
  // selected pending operations have completed.
  while (LoadTimeline(state.pinned_timeline) < target_seqno) {
    absl::SleepFor(kGpuTimelinePollInterval);
  }

  state.mu.lock();
  return absl::OkStatus();
}

absl::Status
DeviceAddressVmmAllocator::WaitAndDrainPendingDeallocationsUntilSeqno(
    PerDeviceState& state, uint64_t target_seqno) {
  RETURN_IF_ERROR(WaitUntilSeqno(state, target_seqno));
  while (!state.pending_deallocations.empty() &&
         state.pending_deallocations.front().seqno <= target_seqno) {
    PendingDeallocation pending = state.pending_deallocations.front();
    state.pending_deallocations.pop_front();
    CompletePendingDeallocation(state, pending);
  }
  return absl::OkStatus();
}

void DeviceAddressVmmAllocator::CompleteReadyAllocatorDeallocationsForReclaim(
    PerDeviceState& state, uint64_t completed_seqno) {
  std::vector<PendingDeallocationKey> selected;
  for (const PendingDeallocation& pending : state.pending_deallocations) {
    if (pending.seqno > completed_seqno ||
        pending.kind == PendingDeallocationKind::kMap) {
      continue;
    }
    selected.push_back(
        PendingDeallocationKey{pending.kind, pending.seqno, pending.addr});
  }
  for (const PendingDeallocationKey& key : selected) {
    CompletePendingDeallocationByKey(state, key);
  }
}

bool DeviceAddressVmmAllocator::CompletePendingDeallocationByKey(
    PerDeviceState& state, const PendingDeallocationKey& key) {
  for (auto it = state.pending_deallocations.begin();
       it != state.pending_deallocations.end(); ++it) {
    if (it->kind == key.kind && it->seqno == key.seqno &&
        it->addr.IsSameAs(key.addr)) {
      PendingDeallocation pending = *it;
      state.pending_deallocations.erase(it);
      CompletePendingDeallocation(state, pending);
      return true;
    }
  }
  return false;
}

void DeviceAddressVmmAllocator::CompletePendingDeallocation(
    PerDeviceState& state, const PendingDeallocation& pending) {
  if (pending.kind == PendingDeallocationKind::kMap) {
    auto record_it =
        state.stale_reservation_records.find(pending.addr.opaque());
    CHECK(record_it != state.stale_reservation_records.end());
    CHECK_EQ(record_it->second->reservation_stale_seqno(), pending.seqno);
    CompleteStaleReservationMapping(state, *record_it->second);
    return;
  }

  auto record_it =
      state.records_by_allocator_address.find(pending.addr.opaque());
  CHECK(record_it != state.records_by_allocator_address.end());
  CHECK(record_it->second->allocator_stale());
  CHECK(record_it->second->allocator_matches(pending.addr));
  CHECK_EQ(record_it->second->pending_deallocation_kind(), pending.kind);
  CHECK_EQ(record_it->second->allocator_stale_seqno(), pending.seqno);
  // Complete allocator-address teardown. If this allocation still has an
  // explicitly unmapped stale reservation alias, drop that mapping first, then
  // release allocator VA state and physical allocation accounting.
  AllocationRecord& record = *record_it->second;
  CHECK(!record.allocator_active());
  CHECK(!record.reservation_active());
  if (record.reservation_stale()) {
    CHECK(record.has_reservation_address());
    PendingDeallocationKey reservation_key{PendingDeallocationKind::kMap,
                                           record.reservation_stale_seqno(),
                                           record.reservation_address()};
    for (auto it = state.pending_deallocations.begin();
         it != state.pending_deallocations.end(); ++it) {
      if (it->kind == reservation_key.kind &&
          it->seqno == reservation_key.seqno &&
          it->addr.IsSameAs(reservation_key.addr)) {
        state.pending_deallocations.erase(it);
        break;
      }
    }
    CompleteStaleReservationMapping(state, record);
  }
  void* allocator_va = record.allocator_key();
  auto owning_record_it = state.records_by_allocator_address.find(allocator_va);
  CHECK(owning_record_it != state.records_by_allocator_address.end());
  CHECK_EQ(owning_record_it->second.get(), &record);

  if (record.raw_allocation() != nullptr) {
    uint64_t released_size =
        RoundUpToGranularity(state, record.raw_allocation()->address().size());
    DCHECK_GE(state.pa_allocated, released_size);
    state.pa_allocated -= released_size;
  }
  record.CompleteStaleAllocator();
  CHECK_EQ(state.records_by_allocator_address.erase(allocator_va), 1);
}

absl::Status DeviceAddressVmmAllocator::UnMap(int device_ordinal,
                                              MemoryReservation* reservation,
                                              uint64_t reservation_offset,
                                              uint64_t size) {
  ASSIGN_OR_RETURN(auto state, GetPerDeviceState(device_ordinal));
  if (size == 0) {
    return absl::OkStatus();
  }

  // Map() and Allocate(..., return_reservation_address=false) record
  // reservation mappings by the mapped reservation VA. Reconstruct the same
  // reservation slice here so callers do not need to hold a ScopedMapping.
  ASSIGN_OR_RETURN(
      DeviceAddressBase reservation_address,
      ValidateReservationRange(reservation, reservation_offset, size));

  absl::MutexLock lock(state->mu);
  // UnMap() only accepts the exact active reservation range previously created
  // by Map() or Allocate(..., return_reservation_address=false). Allocator
  // addresses and subranges are not valid UnMap() inputs.
  auto active_it =
      state->active_reservation_records.find(reservation_address.opaque());
  if (active_it == state->active_reservation_records.end()) {
    auto stale_it =
        state->stale_reservation_records.find(reservation_address.opaque());
    if (stale_it != state->stale_reservation_records.end()) {
      CHECK(stale_it->second->has_reservation_address());
    }
    if (stale_it != state->stale_reservation_records.end() &&
        stale_it->second->reservation_matches(reservation_address)) {
      return absl::FailedPreconditionError(absl::StrFormat(
          "reservation range at virtual address %p (%uB) is already pending "
          "UnMap()",
          reservation_address.opaque(), reservation_address.size()));
    }
    return absl::NotFoundError(absl::StrFormat(
        "UnMap() requires an exact active reservation range created by Map() "
        "or Allocate(..., return_reservation_address=false): virtual address "
        "%p (%uB)",
        reservation_address.opaque(), reservation_address.size()));
  }
  AllocationRecord* record = active_it->second;
  CHECK(record->reservation_active());
  CHECK(!record->reservation_stale());
  CHECK(record->has_reservation_address());
  if (!record->reservation_matches(reservation_address)) {
    return absl::InvalidArgumentError(
        "DeviceAddressVmmAllocator::UnMap requires the same full reservation "
        "range passed to Map");
  }
  CHECK(record->reservation_mapping_matches(reservation_address));

  uint64_t seqno = state->next_seqno++;
  RETURN_IF_ERROR(EnqueueDeferredDeallocation(*state, seqno));
  MoveReservationRecordToStale(*state, *record, seqno);
  state->pending_deallocations.push_back(
      PendingDeallocation{PendingDeallocationKind::kMap, seqno,
                          reservation_address, /*reclaimable_bytes=*/0});
  return absl::OkStatus();
}

}  // namespace stream_executor
