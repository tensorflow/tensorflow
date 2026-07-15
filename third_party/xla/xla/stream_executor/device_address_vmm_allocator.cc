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
#include <cstdint>
#include <deque>
#include <limits>
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
         (device_assignment->replica_count() > 1 ||
          device_assignment->computation_count() > 1);
}

DeviceAddressVmmAllocator::AllocationRecord::AllocationRecord(
    Kind kind, DeviceAddressBase allocator_address,
    std::unique_ptr<MemoryAllocation> raw_allocation,
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
  CHECK_GT(raw_allocation_->address().size(), 0);
  switch (kind_) {
    case Kind::kAllocate:
    case Kind::kAllocateAndMapReturnNewAddr:
      CHECK(allocator_address_reservation_ != nullptr);
      break;
    case Kind::kAllocateAndMapReturnMapAddr:
      CHECK(allocator_address_reservation_ == nullptr);
      break;
  }
  CHECK(!allocator_address_mapping_.is_null());
}

DeviceAddressBase
DeviceAddressVmmAllocator::AllocationRecord::reservation_address() const {
  CHECK(has_reservation_alias());
  return reservation_alias_->mapping.mapped_address();
}

uint64_t DeviceAddressVmmAllocator::AllocationRecord::reservation_stale_seqno()
    const {
  CHECK(has_reservation_alias());
  return reservation_alias_->stale_seqno;
}

void DeviceAddressVmmAllocator::AllocationRecord::MarkAllocatorStale(
    uint64_t seqno) {
  CHECK(allocator_active());
  CHECK(!allocator_address_mapping_.is_null());
  CHECK_NE(seqno, 0);
  allocator_stale_seqno_ = seqno;
}

void DeviceAddressVmmAllocator::AllocationRecord::ReactivateAllocator(
    uint64_t new_size) {
  CHECK(allocator_stale());
  CHECK(!allocator_address_mapping_.is_null());
  allocator_address_ = DeviceAddressBase(allocator_address_.opaque(), new_size);
  allocator_stale_seqno_ = 0;
}

void DeviceAddressVmmAllocator::AllocationRecord::AddActiveReservationAlias(
    MemoryReservation::ScopedMapping reservation_address_mapping) {
  CHECK(!has_reservation_alias());
  CHECK(!reservation_address_mapping.is_null());
  reservation_alias_.emplace(
      ReservationAlias{std::move(reservation_address_mapping)});
}

void DeviceAddressVmmAllocator::AllocationRecord::MarkReservationStale(
    uint64_t seqno) {
  CHECK(reservation_active());
  CHECK_NE(seqno, 0);
  reservation_alias_->stale_seqno = seqno;
}

void DeviceAddressVmmAllocator::AllocationRecord::ReactivateReservation() {
  CHECK(reservation_stale());
  reservation_alias_->stale_seqno = 0;
}

void DeviceAddressVmmAllocator::AllocationRecord::CompleteStaleReservation() {
  CHECK(reservation_stale());
  reservation_alias_.reset();
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

static uintptr_t AddressStart(DeviceAddressBase address) {
  return reinterpret_cast<uintptr_t>(address.opaque());
}

static uintptr_t AddressEnd(DeviceAddressBase address) {
  uintptr_t start = AddressStart(address);
  if (std::numeric_limits<uintptr_t>::max() - start < address.size()) {
    return std::numeric_limits<uintptr_t>::max();
  }
  return start + address.size();
}

static bool AddressRangesOverlap(DeviceAddressBase lhs, DeviceAddressBase rhs) {
  if (lhs.is_null() || rhs.is_null() || lhs.size() == 0 || rhs.size() == 0) {
    return false;
  }
  return AddressStart(lhs) < AddressEnd(rhs) &&
         AddressStart(rhs) < AddressEnd(lhs);
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
    DCHECK_EQ(cfg.stream->parent(), cfg.executor);
    int ordinal = cfg.executor->device_ordinal();
    DCHECK(seen_ordinals.insert(ordinal).second);
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
    auto insert_result =
        allocator->per_device_.emplace(ordinal, std::move(state));
    CHECK(insert_result.second);
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
  uint64_t target_seqno = state->pending_deallocations.back().seqno;
  WaitUntilSeqno(*state, target_seqno);
  while (!state->pending_deallocations.empty() &&
         state->pending_deallocations.front().seqno <= target_seqno) {
    PendingDeallocation pending = state->pending_deallocations.front();
    state->pending_deallocations.pop_front();
    CompletePendingDeallocation(*state, pending);
  }
  return absl::OkStatus();
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
  // this map until deferred teardown completes, so expose their backing
  // allocation for diagnostics/reuse checks until the pending operation drains.
  auto allocation_it = state->records_by_allocator_address.find(addr.opaque());
  if (allocation_it != state->records_by_allocator_address.end() &&
      allocation_it->second->allocator_matches(addr)) {
    return allocation_it->second->raw_allocation();
  }

  // Reservation aliases created by Map() or by Allocate(...,
  // return_reservation_address=false) share one index. Only active aliases are
  // exposed; stale or already-unmapped aliases intentionally return nullptr.
  auto reservation_it = state->reservation_records.find(addr.opaque());
  if (reservation_it != state->reservation_records.end() &&
      reservation_it->second->reservation_active() &&
      reservation_it->second->reservation_matches(addr)) {
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

absl::StatusOr<std::unique_ptr<MemoryAllocation>>
DeviceAddressVmmAllocator::AllocatePhysicalWithinBudget(
    PerDeviceState& state, uint64_t size, uint64_t& physical_size) {
  // Returns ResourceExhausted if charging `bytes` more physical bytes would
  // exceed the configured PA budget. Subtraction avoids unsigned overflow.
  auto check_pa_budget = [&state](uint64_t bytes) -> absl::Status {
    state.mu.AssertHeld();
    if (state.pa_allocated <= state.pa_budget &&
        bytes <= state.pa_budget - state.pa_allocated) {
      return absl::OkStatus();
    }
    return absl::ResourceExhaustedError(absl::StrFormat(
        "Not enough PA budget: pa_allocated=%uB, allocation_size=%uB, "
        "pa_budget=%uB",
        state.pa_allocated, bytes, state.pa_budget));
  };

  // Fail fast on an obvious budget miss before asking the driver to reserve
  // physical memory. rounded_size only estimates what the driver will return;
  // the authoritative check below uses the real committed size.
  RETURN_IF_ERROR(check_pa_budget(RoundUpToGranularity(state, size)));
  ASSIGN_OR_RETURN(auto raw_alloc, CreateAllocation(state.executor, size));
  physical_size = raw_alloc->address().size();
  // CreateAllocation rounds the request up to the allocation granularity, so
  // the physical allocation is never smaller than requested.
  DCHECK_GE(physical_size, size);
  // Re-check against the actual size that will be charged to pa_allocated.
  RETURN_IF_ERROR(check_pa_budget(physical_size));
  return raw_alloc;
}

DeviceAddressVmmAllocator::AllocationRecord&
DeviceAddressVmmAllocator::TrackAllocatorAddressMappedAllocation(
    PerDeviceState& state, AllocationRecord::Kind kind,
    DeviceAddressBase allocator_address,
    std::unique_ptr<MemoryAllocation> raw_allocation,
    std::unique_ptr<MemoryReservation> reservation,
    MemoryReservation::ScopedMapping mapping, bool multi_device) {
  void* va_ptr = allocator_address.opaque();
  CHECK(raw_allocation != nullptr);
  uint64_t physical_size = raw_allocation->address().size();
  auto record = std::make_unique<AllocationRecord>(
      kind, allocator_address, std::move(raw_allocation),
      std::move(reservation), std::move(mapping), multi_device);
  AllocationRecord* record_ptr = record.get();
  auto insert_result =
      state.records_by_allocator_address.emplace(va_ptr, std::move(record));
  CHECK(insert_result.second);
  state.pa_allocated += physical_size;
  return *record_ptr;
}

std::optional<DeviceAddressBase>
DeviceAddressVmmAllocator::TryReuseMappedAllocation(
    PerDeviceState& state, const MappedAllocateRequest& request) {
  const bool separate_address =
      request.mode == MappedAddressMode::kSeparateAllocatorAddress;
  AllocationRecord* record;
  if (separate_address) {
    auto record_it =
        state.reservation_records.find(request.reservation_address.opaque());
    if (record_it == state.reservation_records.end()) {
      return std::nullopt;
    }
    record = record_it->second;
  } else {
    auto record_it = state.records_by_allocator_address.find(
        request.reservation_address.opaque());
    if (record_it == state.records_by_allocator_address.end()) {
      return std::nullopt;
    }
    record = record_it->second.get();
  }

  AllocationRecord::Kind expected_kind =
      separate_address ? AllocationRecord::Kind::kAllocateAndMapReturnNewAddr
                       : AllocationRecord::Kind::kAllocateAndMapReturnMapAddr;
  bool address_matches =
      separate_address
          ? record->reservation_matches(request.reservation_address)
          : record->allocator_matches(request.reservation_address);
  if (record->kind() != expected_kind || !record->allocator_stale() ||
      record->multi_device() != request.multi_device || !address_matches ||
      (separate_address && !record->reservation_stale())) {
    return std::nullopt;
  }

  auto pending_it = state.pending_deallocations.end();
  for (auto it = state.pending_deallocations.begin();
       it != state.pending_deallocations.end(); ++it) {
    if (it->kind == PendingDeallocationKind::kAllocation &&
        it->addr.IsSameAs(record->allocator_address())) {
      pending_it = it;
      break;
    }
  }
  CHECK(pending_it != state.pending_deallocations.end());

  DeviceAddressBase reused_mem(record->allocator_key(), request.size);
  MoveAllocatorRecordToActive(state, *record, request.size);
  if (separate_address) {
    record->ReactivateReservation();
  }
  ErasePendingDeallocationAt(state, pending_it);
  if (separate_address) {
    ErasePendingDeallocation(state, PendingDeallocationKind::kMap,
                             request.reservation_address);
  }
  return reused_mem;
}

absl::Status
DeviceAddressVmmAllocator::EnsureReservationAvailableForFreshMapping(
    PerDeviceState& state, const MappedAllocateRequest& request) {
  // If the requested reservation VA is only present in the deferred queue,
  // wait for that queued unmap/deallocation to complete before installing a
  // fresh mapping. Active mappings are still rejected below.
  while (true) {
    // Partial overlaps are never reusable: the allocator tracks whole mapped
    // ranges, so a caller must request the exact same reservation slice before
    // stale state can be waited on or reactivated.
    RETURN_IF_ERROR(
        CheckNoPartialReservationOverlap(state, request.reservation_address));
    // An exact stale overlap means the previous mapping for this reservation
    // VA is still protected by stream order. Complete only that conflicting
    // stale record, then rescan because another thread may have changed the
    // allocator state while the lock was released.
    uint64_t pending_completion_seqno = 0;
    {
      auto stale_overlap = FindOverlappingRecord(
          state, request.reservation_address, AddressRole::kBoth,
          RecordState::kStale, OverlapKind::kExact);
      if (stale_overlap.has_value()) {
        CHECK(!stale_overlap->is_active);
        AllocationRecord& record = *stale_overlap->record;
        if (stale_overlap->is_allocator) {
          pending_completion_seqno = record.allocator_stale_seqno();
        } else {
          CHECK(record.reservation_stale());
          pending_completion_seqno = record.reservation_stale_seqno();
        }
      }
    }
    if (pending_completion_seqno == 0) {
      break;
    }
    WaitUntilSeqno(state, pending_completion_seqno);
    CompletePendingDeallocationBySeqno(state, pending_completion_seqno);
  }

  // At this point stale exact overlaps have been drained. Any remaining
  // overlap is active ownership of the requested reservation range and must be
  // reported as a duplicate mapping attempt instead of remapped underneath
  // existing users.
  if (FindOverlappingRecord(state, request.reservation_address,
                            AddressRole::kBoth, RecordState::kActive,
                            OverlapKind::kExact)
          .has_value()) {
    return absl::AlreadyExistsError(absl::StrFormat(
        "reservation range is already tracked at virtual address %p",
        request.reservation_address.opaque()));
  }
  return absl::OkStatus();
}

absl::StatusOr<DeviceAddressBase>
DeviceAddressVmmAllocator::CreateMappedAllocation(
    PerDeviceState& state, const MappedAllocateRequest& request) {
  uint64_t physical_size = 0;
  ASSIGN_OR_RETURN(auto raw_alloc, AllocatePhysicalWithinBudget(
                                       state, request.size, physical_size));

  const bool separate_address =
      request.mode == MappedAddressMode::kSeparateAllocatorAddress;
  AllocationRecord::Kind kind =
      AllocationRecord::Kind::kAllocateAndMapReturnMapAddr;
  DeviceAddressBase allocator_address = request.reservation_address;
  std::unique_ptr<MemoryReservation> allocator_address_reservation;
  MemoryReservation::ScopedMapping allocator_address_mapping;

  if (separate_address) {
    kind = AllocationRecord::Kind::kAllocateAndMapReturnNewAddr;
    ASSIGN_OR_RETURN(allocator_address_reservation,
                     CreateReservation(state.executor, request.size));
    allocator_address = DeviceAddressBase(
        allocator_address_reservation->address().opaque(), request.size);
    ASSIGN_OR_RETURN(allocator_address_mapping,
                     allocator_address_reservation->MapTo(
                         /*reservation_offset=*/0, /*allocation_offset=*/0,
                         physical_size, *raw_alloc));
  }

  ASSIGN_OR_RETURN(auto reservation_address_mapping,
                   request.reservation->MapTo(request.reservation_offset,
                                              /*allocation_offset=*/0,
                                              request.size, *raw_alloc));

  if (!separate_address) {
    allocator_address_mapping = std::move(reservation_address_mapping);
    reservation_address_mapping = MemoryReservation::ScopedMapping();
  }
  AllocationRecord& record = TrackAllocatorAddressMappedAllocation(
      state, kind, allocator_address, std::move(raw_alloc),
      std::move(allocator_address_reservation),
      std::move(allocator_address_mapping), request.multi_device);
  if (separate_address) {
    record.AddActiveReservationAlias(std::move(reservation_address_mapping));
    auto reservation_insert = state.reservation_records.emplace(
        request.reservation_address.opaque(), &record);
    CHECK(reservation_insert.second);
  }

  return allocator_address;
}

// Reuses compatible pending state before trying a fresh allocation. On a
// ResourceExhausted failure, first reclaim completed work, then wait for enough
// queued allocator deallocations and try once more.
template <typename TryReuseFn, typename TryFreshFn>
absl::StatusOr<DeviceAddressBase>
DeviceAddressVmmAllocator::TryWithPendingReclaim(PerDeviceState& state,
                                                 uint64_t reclaim_size,
                                                 TryReuseFn try_reuse,
                                                 TryFreshFn try_fresh) {
  // First try to reactivate a compatible pending deallocation without waiting.
  // Reuse is stream-order safe and avoids both a fresh VMM allocation and any
  // host-side wait for the GPU timeline.
  std::optional<DeviceAddressBase> reused = try_reuse();
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
    uint64_t accumulated_size = 0;
    uint64_t rounded_size = RoundUpToGranularity(state, reclaim_size);
    uint64_t target_seqno = 0;
    std::vector<uint64_t> selected;

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
      uint64_t reclaimable_bytes =
          record_it->second->raw_allocation()->address().size();
      CHECK_GT(reclaimable_bytes, 0);
      accumulated_size += reclaimable_bytes;
      target_seqno = std::max(target_seqno, pending.seqno);
      selected.push_back(pending.seqno);
      if (accumulated_size >= target_size) {
        break;
      }
    }

    if (!selected.empty()) {
      WaitUntilSeqno(state, target_seqno);
      for (uint64_t seqno : selected) {
        CompletePendingDeallocationBySeqno(state, seqno);
      }
    }
    result = try_fresh();
  }

  return result;
}

// Allocate() reuses compatible pending regular-allocation records, otherwise
// tries a fresh allocator-address mapping.
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
  // Clang cannot propagate TryWithPendingReclaim's state.mu lock requirement
  // into its callbacks. AssertHeld makes that invariant explicit within each
  // independently analyzed lambda.
  auto try_reuse = [&]() -> std::optional<DeviceAddressBase> {
    state->mu.AssertHeld();
    uint64_t rounded_size = RoundUpToGranularity(*state, size);
    for (auto it = state->pending_deallocations.begin();
         it != state->pending_deallocations.end(); ++it) {
      if (it->kind != PendingDeallocationKind::kAllocation) {
        continue;
      }
      auto record_it =
          state->records_by_allocator_address.find(it->addr.opaque());
      CHECK(record_it != state->records_by_allocator_address.end());
      AllocationRecord& record = *record_it->second;
      CHECK(record.allocator_stale());
      CHECK(record.allocator_matches(it->addr));
      if (record.kind() != AllocationRecord::Kind::kAllocate) {
        continue;
      }
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

      return reused_mem;
    }

    return std::nullopt;
  };
  auto try_fresh = [&]() -> absl::StatusOr<DeviceAddressBase> {
    state->mu.AssertHeld();
    uint64_t physical_size = 0;
    ASSIGN_OR_RETURN(auto raw_alloc,
                     AllocatePhysicalWithinBudget(*state, size, physical_size));

    ASSIGN_OR_RETURN(auto reservation,
                     CreateReservation(state->executor, size));

    ASSIGN_OR_RETURN(
        auto scoped_mapping,
        reservation->MapTo(/*reservation_offset=*/0, /*allocation_offset=*/0,
                           physical_size, *raw_alloc));

    DeviceAddressBase allocator_address(reservation->address().opaque(), size);
    TrackAllocatorAddressMappedAllocation(
        *state, AllocationRecord::Kind::kAllocate, allocator_address,
        std::move(raw_alloc), std::move(reservation), std::move(scoped_mapping),
        multi_device);
    // Return the original requested size, not the padded size.
    return absl::StatusOr<DeviceAddressBase>(allocator_address);
  };

  ASSIGN_OR_RETURN(DeviceAddressBase result,
                   TryWithPendingReclaim(*state, size, try_reuse, try_fresh));

  VLOG(3) << absl::StreamFormat(
      "Allocated virtual address %p (%uB) on device ordinal %d",
      result.opaque(), size, device_ordinal);

  return ScopedDeviceAddress<uint8_t>(result, device_ordinal, this);
}

// Mapped Allocate() reuses matching pending mapped deallocations, otherwise
// tries fresh physical allocation and maps it into the caller reservation.
absl::StatusOr<ScopedDeviceAddress<uint8_t>>
DeviceAddressVmmAllocator::Allocate(
    int device_ordinal, uint64_t allocation_size, bool /*retry_on_failure*/,
    int64_t /*memory_space*/, MemoryReservation* reservation,
    uint64_t reservation_offset, uint64_t mapping_size,
    bool return_reservation_address) {
  if (allocation_size != mapping_size) {
    return absl::InvalidArgumentError(
        "allocation_size must equal mapping_size for mapped Allocate");
  }
  // Keep zero-sized mapped allocation consistent with regular Allocate(): no
  // physical allocation or mapping is created.
  if (allocation_size == 0) {
    return ScopedDeviceAddress<uint8_t>(DeviceAddressBase(), device_ordinal,
                                        this);
  }

  ASSIGN_OR_RETURN(auto state, GetPerDeviceState(device_ordinal));
  const bool multi_device = CurrentMultiDevice();

  // Validate the caller-owned reservation slice before taking the allocator
  // lock. `reservation_address` is the VA that must either be reactivated from
  // a pending deallocation or freshly mapped below.
  ASSIGN_OR_RETURN(
      DeviceAddressBase reservation_address,
      ValidateReservationRange(reservation, reservation_offset, mapping_size));

  const MappedAddressMode mode =
      return_reservation_address ? MappedAddressMode::kReservationAddress
                                 : MappedAddressMode::kSeparateAllocatorAddress;
  const MappedAllocateRequest request{reservation,     reservation_address,
                                      allocation_size, reservation_offset,
                                      multi_device,    mode};

  absl::MutexLock lock(state->mu);
  // Clang cannot propagate TryWithPendingReclaim's state.mu lock requirement
  // into its callbacks. AssertHeld makes that invariant explicit within each
  // independently analyzed lambda; stateful work stays in annotated helpers.
  auto try_reuse = [&]() -> std::optional<DeviceAddressBase> {
    state->mu.AssertHeld();
    return TryReuseMappedAllocation(*state, request);
  };
  auto try_fresh = [&]() -> absl::StatusOr<DeviceAddressBase> {
    state->mu.AssertHeld();
    RETURN_IF_ERROR(EnsureReservationAvailableForFreshMapping(*state, request));
    return CreateMappedAllocation(*state, request);
  };

  // The shared retry helper handles PA-budget pressure: try reuse, try fresh,
  // complete already-finished pending work on ResourceExhausted, and finally
  // wait for enough pending deallocations only if necessary.
  ASSIGN_OR_RETURN(
      DeviceAddressBase result,
      TryWithPendingReclaim(*state, allocation_size, try_reuse, try_fresh));

  // For return_reservation_address=true this is `reservation_address`; for
  // return_reservation_address=false it is the allocator-owned address paired
  // with the reservation mapping.
  return ScopedDeviceAddress<uint8_t>(result, device_ordinal, this);
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
  auto reservation_it = state->reservation_records.find(mem.opaque());
  CHECK(reservation_it == state->reservation_records.end() ||
        !reservation_it->second->reservation_active());
  if (record.reservation_active()) {
    CHECK(record.has_reservation_alias());
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

  // Assign the next sequence number and enqueue a GPU write to the pinned
  // timeline when the stream reaches this point. The CPU polls the timeline
  // value to know when it is safe to free the memory.
  uint64_t seqno = state->next_seqno++;
  RETURN_IF_ERROR(EnqueueDeferredDeallocation(*state, seqno));
  // Move the returned allocator address out of active ownership and keep its
  // mapping alive as stale state until the stream reaches `seqno`.
  record.MarkAllocatorStale(seqno);
  state->pending_deallocations.push_back(PendingDeallocation{
      PendingDeallocationKind::kAllocation, seqno, record.allocator_address()});
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

std::optional<DeviceAddressVmmAllocator::OverlappingRecord>
DeviceAddressVmmAllocator::FindOverlappingRecord(
    PerDeviceState& state, DeviceAddressBase address, AddressRole role,
    RecordState record_state, OverlapKind overlap_kind) const {
  const bool include_allocator = role != AddressRole::kReservation;
  const bool include_reservation = role != AddressRole::kAllocator;
  const bool include_active = record_state != RecordState::kStale;
  const bool include_stale = record_state != RecordState::kActive;

  auto matches = [&](DeviceAddressBase tracked_address) {
    switch (overlap_kind) {
      case OverlapKind::kExact:
        return tracked_address.IsSameAs(address);
      case OverlapKind::kPartial:
        // Partial overlap means the ranges intersect but are not the same full
        // ownership range.
        return AddressRangesOverlap(tracked_address, address) &&
               !tracked_address.IsSameAs(address);
    }
  };

  auto check_record = [&](AllocationRecord* record,
                          DeviceAddressBase tracked_address, bool is_allocator,
                          bool is_active) -> std::optional<OverlappingRecord> {
    if (matches(tracked_address)) {
      return OverlappingRecord{record, tracked_address, is_allocator,
                               is_active};
    }
    return std::nullopt;
  };

  if (include_allocator) {
    for (const auto& [_, record_owner] : state.records_by_allocator_address) {
      AllocationRecord* record = record_owner.get();
      bool include_record = (include_active && record->allocator_active()) ||
                            (include_stale && record->allocator_stale());
      if (!include_record) {
        continue;
      }
      if (auto overlap =
              check_record(record, record->allocator_address(),
                           /*is_allocator=*/true,
                           /*is_active=*/record->allocator_active())) {
        return overlap;
      }
    }
  }
  if (include_reservation) {
    for (const auto& [_, record] : state.reservation_records) {
      CHECK(record->has_reservation_alias());
      bool include_record = (include_active && record->reservation_active()) ||
                            (include_stale && record->reservation_stale());
      if (!include_record) {
        continue;
      }
      if (auto overlap = check_record(
              record, record->reservation_address(), /*is_allocator=*/false,
              /*is_active=*/record->reservation_active())) {
        return overlap;
      }
    }
  }

  return std::nullopt;
}

absl::StatusOr<DeviceAddressVmmAllocator::AllocationRecord*>
DeviceAddressVmmAllocator::ResolveMapSourceRecord(
    PerDeviceState& state, DeviceAddressBase source_address,
    uint64_t size) const {
  auto allocation_it =
      state.records_by_allocator_address.find(source_address.opaque());
  if (allocation_it == state.records_by_allocator_address.end() ||
      !allocation_it->second->allocator_active() ||
      !allocation_it->second->allocator_matches(source_address)) {
    return absl::NotFoundError(absl::StrFormat(
        "addr %p is not an active allocator address; Map() sources must be "
        "allocated by this DeviceAddressVmmAllocator",
        source_address.opaque()));
  }
  AllocationRecord* source_record = allocation_it->second.get();
  MemoryAllocation* raw_allocation = source_record->raw_allocation();
  if (size > raw_allocation->address().size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "mapping size must not exceed physical allocation size: "
        "mapping_size=%uB, allocation_size=%uB",
        size, raw_allocation->address().size()));
  }
  return source_record;
}

absl::Status DeviceAddressVmmAllocator::CheckNoPartialReservationOverlap(
    PerDeviceState& state, DeviceAddressBase reservation_address) const {
  if (auto overlap =
          FindOverlappingRecord(state, reservation_address, AddressRole::kBoth,
                                RecordState::kBoth, OverlapKind::kPartial)) {
    return absl::FailedPreconditionError(absl::StrFormat(
        "reservation range at %p (%uB) partially overlaps %s %s range at %p "
        "(%uB); reservation mappings must be managed with the same full "
        "range",
        reservation_address.opaque(), reservation_address.size(),
        overlap->is_active ? "active" : "stale",
        overlap->is_allocator ? "allocator" : "reservation",
        overlap->tracked_address.opaque(), overlap->tracked_address.size()));
  }
  return absl::OkStatus();
}

absl::Status DeviceAddressVmmAllocator::ResolveAndMapAlias(
    PerDeviceState& state, const MapRequest& request) {
  // One source can have one stale alias and the destination can have one stale
  // occupant. Without concurrent changes, at most two waits are needed before
  // a third attempt can install or reuse the requested mapping.
  constexpr int kMaxStaleMappingWaits = 2;
  for (int wait_count = 0;; ++wait_count) {
    uint64_t pending_completion_seqno = 0;
    {
      // Keep record pointers inside this scope so none survives a wait that
      // releases state.mu.
      AllocationRecord* source_record;
      ASSIGN_OR_RETURN(
          source_record,
          ResolveMapSourceRecord(state, request.source_address, request.size));
      RETURN_IF_ERROR(
          CheckNoPartialReservationOverlap(state, request.reservation_address));

      if (source_record->reservation_active()) {
        return absl::AlreadyExistsError(absl::StrFormat(
            "allocator address %p already has an active reservation mapping "
            "at %p",
            request.source_address.opaque(),
            source_record->reservation_address().opaque()));
      }

      // Reject an active destination before waiting for a stale source alias. A
      // failed Map() must not drain unrelated pending state.
      if (FindOverlappingRecord(state, request.reservation_address,
                                AddressRole::kBoth, RecordState::kActive,
                                OverlapKind::kExact)
              .has_value()) {
        return absl::AlreadyExistsError(absl::StrFormat(
            "reservation range is already tracked at virtual address %p",
            request.reservation_address.opaque()));
      }

      if (source_record->reservation_stale()) {
        CHECK(source_record->has_reservation_alias());
        if (!source_record->reservation_matches(request.reservation_address)) {
          pending_completion_seqno = source_record->reservation_stale_seqno();
        }
      }

      if (pending_completion_seqno == 0) {
        auto stale_reservation_overlap = FindOverlappingRecord(
            state, request.reservation_address, AddressRole::kReservation,
            RecordState::kStale, OverlapKind::kExact);
        if (stale_reservation_overlap.has_value()) {
          AllocationRecord& stale_record = *stale_reservation_overlap->record;

          // A deferred UnMap() leaves the old mapping valid. Reuse it when it
          // aliases the requested physical allocation; otherwise wait before
          // overwriting it.
          CHECK(stale_record.has_reservation_alias());
          CHECK(stale_record.reservation_matches(request.reservation_address));
          if (stale_record.raw_allocation() ==
              source_record->raw_allocation()) {
            stale_record.ReactivateReservation();
            ErasePendingDeallocation(state, PendingDeallocationKind::kMap,
                                     request.reservation_address);
            return absl::OkStatus();
          }
          pending_completion_seqno = stale_record.reservation_stale_seqno();
        }
      }

      if (pending_completion_seqno == 0) {
        auto stale_allocator_overlap = FindOverlappingRecord(
            state, request.reservation_address, AddressRole::kAllocator,
            RecordState::kStale, OverlapKind::kExact);
        if (stale_allocator_overlap.has_value()) {
          AllocationRecord& stale_record = *stale_allocator_overlap->record;
          pending_completion_seqno = stale_record.allocator_stale_seqno();
        }
      }

      if (pending_completion_seqno == 0) {
        // Map() aliases the beginning of the source allocation into the
        // caller's VA slice. No physical allocation or PA accounting is added.
        ASSIGN_OR_RETURN(
            auto mapping,
            request.reservation->MapTo(request.reservation_offset,
                                       /*allocation_offset=*/0, request.size,
                                       *source_record->raw_allocation()));
        DeviceAddressBase mapped = mapping.mapped_address();
        DCHECK(mapped.IsSameAs(request.reservation_address))
            << "Map() mapped unexpected virtual address: expected="
            << request.reservation_address.opaque()
            << ", actual=" << mapped.opaque();

        source_record->AddActiveReservationAlias(std::move(mapping));
        auto mapping_insert_result =
            state.reservation_records.emplace(mapped.opaque(), source_record);
        CHECK(mapping_insert_result.second);
        return absl::OkStatus();
      }
      if (wait_count == kMaxStaleMappingWaits) {
        return absl::AbortedError(
            "Map() allocator state changed repeatedly while waiting for stale "
            "mappings; retry Map()");
      }
    }

    CHECK_NE(pending_completion_seqno, 0);
    WaitUntilSeqno(state, pending_completion_seqno);
    CompletePendingDeallocationBySeqno(state, pending_completion_seqno);
  }
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

  // Map() does not allocate a VA range. Validate the caller-owned slice before
  // taking the allocator lock.
  ASSIGN_OR_RETURN(
      DeviceAddressBase reservation_address,
      ValidateReservationRange(reservation, reservation_offset, size));
  MapRequest request{addr, reservation, reservation_offset, size,
                     reservation_address};

  absl::MutexLock lock(state->mu);
  return ResolveAndMapAlias(*state, request);
}

// UnMap/deferred teardown helpers.

void DeviceAddressVmmAllocator::ErasePendingDeallocationAt(
    PerDeviceState& state, std::deque<PendingDeallocation>::iterator it) {
  CHECK(it != state.pending_deallocations.end());
  state.pending_deallocations.erase(it);
}

void DeviceAddressVmmAllocator::ErasePendingDeallocation(
    PerDeviceState& state, PendingDeallocationKind kind,
    DeviceAddressBase addr) {
  for (auto it = state.pending_deallocations.begin();
       it != state.pending_deallocations.end(); ++it) {
    if (it->kind == kind && it->addr.IsSameAs(addr)) {
      ErasePendingDeallocationAt(state, it);
      return;
    }
  }
}

void DeviceAddressVmmAllocator::MoveAllocatorRecordToActive(
    PerDeviceState& state, AllocationRecord& record, uint64_t new_size) {
  void* allocator_va = record.allocator_key();
  auto record_it = state.records_by_allocator_address.find(allocator_va);
  CHECK(record_it != state.records_by_allocator_address.end());
  CHECK_EQ(record_it->second.get(), &record);
  record.ReactivateAllocator(new_size);
}

void DeviceAddressVmmAllocator::WaitUntilSeqno(PerDeviceState& state,
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
}

void DeviceAddressVmmAllocator::CompleteReadyAllocatorDeallocationsForReclaim(
    PerDeviceState& state, uint64_t completed_seqno) {
  std::vector<uint64_t> selected;
  for (const PendingDeallocation& pending : state.pending_deallocations) {
    if (pending.seqno > completed_seqno ||
        pending.kind == PendingDeallocationKind::kMap) {
      continue;
    }
    selected.push_back(pending.seqno);
  }
  for (uint64_t seqno : selected) {
    CompletePendingDeallocationBySeqno(state, seqno);
  }
}

void DeviceAddressVmmAllocator::CompletePendingDeallocationBySeqno(
    PerDeviceState& state, uint64_t seqno) {
  for (auto it = state.pending_deallocations.begin();
       it != state.pending_deallocations.end(); ++it) {
    if (it->seqno == seqno) {
      PendingDeallocation pending = *it;
      state.pending_deallocations.erase(it);
      CompletePendingDeallocation(state, pending);
      return;
    }
  }
}

void DeviceAddressVmmAllocator::CompletePendingDeallocation(
    PerDeviceState& state, const PendingDeallocation& pending) {
  if (pending.kind == PendingDeallocationKind::kMap) {
    auto record_it = state.reservation_records.find(pending.addr.opaque());
    CHECK(record_it != state.reservation_records.end());
    AllocationRecord& record = *record_it->second;
    CHECK(record.reservation_stale());
    CHECK_EQ(record.reservation_stale_seqno(), pending.seqno);
    CHECK(record.has_reservation_alias());
    CHECK_EQ(record.reservation_key(), pending.addr.opaque());
    state.reservation_records.erase(record_it);
    record.CompleteStaleReservation();
    return;
  }

  auto record_it =
      state.records_by_allocator_address.find(pending.addr.opaque());
  CHECK(record_it != state.records_by_allocator_address.end());
  CHECK_EQ(pending.kind, PendingDeallocationKind::kAllocation);
  CHECK(record_it->second->allocator_stale());
  CHECK(record_it->second->allocator_matches(pending.addr));
  CHECK_EQ(record_it->second->allocator_stale_seqno(), pending.seqno);
  // Complete allocator-address teardown. If this allocation still has an
  // explicitly unmapped stale reservation alias, drop that mapping first, then
  // release allocator VA state and physical allocation accounting.
  AllocationRecord& record = *record_it->second;
  CHECK(!record.reservation_active());
  if (record.reservation_stale()) {
    CHECK(record.has_reservation_alias());
    CompletePendingDeallocationBySeqno(state, record.reservation_stale_seqno());
    CHECK(!record.has_reservation_alias());
  }
  uint64_t physical_size = record.raw_allocation()->address().size();
  CHECK_GT(physical_size, 0);
  CHECK_GE(state.pa_allocated, physical_size);
  state.pa_allocated -= physical_size;
  state.records_by_allocator_address.erase(record_it);
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
  auto reservation_it =
      state->reservation_records.find(reservation_address.opaque());
  if (reservation_it == state->reservation_records.end()) {
    return absl::NotFoundError(absl::StrFormat(
        "UnMap() requires an exact active reservation range created by Map() "
        "or Allocate(..., return_reservation_address=false): virtual address "
        "%p (%uB)",
        reservation_address.opaque(), reservation_address.size()));
  }
  AllocationRecord* record = reservation_it->second;
  CHECK(record->has_reservation_alias());
  if (record->reservation_stale()) {
    if (record->reservation_matches(reservation_address)) {
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
  CHECK(record->reservation_active());
  if (!record->reservation_matches(reservation_address)) {
    return absl::InvalidArgumentError(
        "DeviceAddressVmmAllocator::UnMap requires the same full reservation "
        "range passed to Map");
  }

  uint64_t seqno = state->next_seqno++;
  RETURN_IF_ERROR(EnqueueDeferredDeallocation(*state, seqno));
  record->MarkReservationStale(seqno);
  state->pending_deallocations.push_back(PendingDeallocation{
      PendingDeallocationKind::kMap, seqno, reservation_address});
  return absl::OkStatus();
}

}  // namespace stream_executor
