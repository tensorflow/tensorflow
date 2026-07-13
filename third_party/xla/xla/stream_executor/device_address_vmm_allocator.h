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

#ifndef XLA_STREAM_EXECUTOR_DEVICE_ADDRESS_VMM_ALLOCATOR_H_
#define XLA_STREAM_EXECUTOR_DEVICE_ADDRESS_VMM_ALLOCATOR_H_

#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_reservation.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
class DeviceAssignment;
}  // namespace xla

namespace stream_executor {

// Abstract base class for a DeviceAddressAllocator backed by virtual memory
// management (VMM). VMM lets the allocator manage device memory in three
// separate steps:
//
//  1. Allocate raw physical memory. This is the real device memory capacity.
//  2. Reserve a virtual address (VA) range. This creates addresses but does not
//     make them usable yet.
//  3. Map a VA range to raw physical memory. Device kernels access memory
//     through the mapped VA.
//
// A concrete subclass provides the platform-specific operations for those
// steps, plus a stream-ordered timeline used to know when old mappings and
// allocations are safe to release.
//
// Caller-visible address roles:
//
//  * Allocator address: any address returned by Allocate(). It owns the raw
//    physical allocation, can be used as the source address for Map(), and must
//    eventually be released with Deallocate().
//  * Reservation address: a caller-owned MemoryReservation slice
//    [reservation_base + offset, reservation_base + offset + size) that is
//    mapped as a non-owning alias of an allocator address. It must be released
//    with UnMap(), not Deallocate().
//
// clang-format off
// NOLINTBEGIN(whitespace/line_length)
// Allowed address behavior:
//
// +--------------------------------------------------------+---------------------+------------+-----+-------+
// | Address                                                | Role                | Deallocate | Map | UnMap |
// +--------------------------------------------------------+---------------------+------------+-----+-------+
// | Allocate() return                                      | allocator address   | yes        | yes | no    |
// | Allocate(..., return_reservation_address=true) return  | allocator address   | yes        | yes | no    |
// | Allocate(..., return_reservation_address=false) return | allocator address   | yes        | yes | no    |
// | reservation slice from Allocate(..., false)            | reservation address | no         | no  | yes   |
// | reservation slice from Map()                           | reservation address | no         | no  | yes   |
// +--------------------------------------------------------+---------------------+------------+-----+-------+
// NOLINTEND(whitespace/line_length)
// clang-format on
//
// The table uses "yes" for API calls that accept the address in that row. For
// example, Map() takes an allocator address as its source, while UnMap() takes
// a reservation address to tear down. Map() still requires the allocator
// address to have no active reservation-address alias; for example, an
// Allocate(..., return_reservation_address=false) result can be remapped only
// after its initial reservation-address alias is released with UnMap().
//
// The main API flows are:
//
//  1. Allocate(size) creates an allocator-owned VA reservation, allocates raw
//     physical memory, maps that memory into the owned reservation, and returns
//     the allocator address.
//  2. Allocate(..., return_reservation_address=true) allocates raw physical
//     memory and maps it directly into the caller reservation. The returned VA
//     comes from the caller reservation, but it is still the allocator address
//     for this allocation.
//  3. Allocate(..., return_reservation_address=false) returns a separate
//     allocator-owned address and also maps the same raw physical allocation
//     into the caller reservation as a reservation address.
//  4. Map(addr, reservation, ...) maps the raw physical allocation currently
//     backing allocator address `addr` into one caller reservation slice.
//     UnMap(reservation, ...) removes that reservation-address alias.
//
// Deallocate() accepts only allocator addresses and requires any active
// reservation-address alias to be released with UnMap() first. UnMap() accepts
// only reservation addresses created by Map() or by
// Allocate(..., return_reservation_address=false). Passing an allocator address
// to UnMap(), or a reservation address to Deallocate(), is an error.
// Each allocator address may have at most one active reservation-address alias.
// A reservation mapping is owned as the same full range that created it:
// partial UnMap(), Map(), or Allocate() operations that overlap an active or
// stale reservation mapping are rejected.
//
// Deallocate() and UnMap() are stream-ordered deferred operations. The
// allocator assigns the affected address record a per-device sequence number,
// moves it from active tracking to stale tracking, and appends a pending entry
// with the operation kind, sequence number, and address. The stale
// AllocationRecord keeps the raw allocation, any allocator-owned reservation,
// and ScopedMapping objects alive until the stream reaches that sequence
// number, so kernels already submitted to the stream can keep using the old VA.
// When the sequence completes, dropping the ScopedMapping objects performs the
// real unmap, then the allocator releases any owned reservation and raw
// physical memory.
//
// Stale records are also the fast reuse path. Allocate() first looks for a
// compatible stale allocator address before creating new VMM state. Map() does
// the same for a stale reservation mapping: if the requested reservation
// address is still mapped to the same raw physical allocation, the allocator
// reactivates the old mapping instead of unmapping and remapping. If a
// requested reservation address is still stale for a different raw allocation,
// Map() waits for that deferred unmap to complete before installing the new
// mapping. If the source allocation has a stale alias in a different
// reservation, Map() likewise waits for that alias before mapping the source
// into the requested reservation.
//
// Each registered device has independent state protected by its own mutex, so
// operations on different devices can proceed in parallel. The per-device map
// is populated at construction time and is not modified afterward. Concrete
// subclasses implement the platform-specific virtual methods
// (InitializeDeviceState, CreateAllocation, CreateReservation,
// EnqueueDeferredDeallocation) and expose platform-specific Create() factories.
// Subclasses must also set PerDeviceState::destroy_fn in InitializeDeviceState
// to release platform-specific resources such as pinned timeline memory.
class DeviceAddressVmmAllocator : public DeviceAddressAllocator {
 public:
  // Per-device configuration supplied at construction.
  struct DeviceConfig {
    // StreamExecutor for this device. Must outlive the allocator.
    StreamExecutor* executor;
    // Stream used for deferred deallocation. Must outlive the allocator.
    Stream* stream;
    // Maximum bytes of physical memory that may be allocated simultaneously on
    // this device. Defaults to unlimited.
    uint64_t pa_budget = UINT64_MAX;
  };

  // Creates a platform-appropriate VMM allocator for the given devices,
  // dispatching to the CUDA or ROCm implementation based on the build platform.
  // The pa_budget for each device is computed from `memory_fraction` (or
  // overridden by `gpu_system_memory_size` when set). Returns an error on
  // platforms without a VMM implementation.
  //
  // Defined in device_address_vmm_allocator_factory.cc so this base library
  // does not depend on the platform-specific subclasses.
  static absl::StatusOr<std::unique_ptr<DeviceAddressVmmAllocator>> Create(
      const Platform* platform, double memory_fraction,
      std::optional<int64_t> gpu_system_memory_size,
      absl::Span<const std::pair<StreamExecutor*, Stream*>> devices);

  ~DeviceAddressVmmAllocator() override;

  absl::StatusOr<ScopedDeviceAddress<uint8_t>> Allocate(
      int device_ordinal, uint64_t size, bool retry_on_failure,
      int64_t memory_space) override;

  // Allocates raw physical memory and maps it into a caller-owned
  // MemoryReservation range.
  // `allocation_size` and `mapping_size` must be equal.
  //
  // There are two modes:
  //
  //  * `return_reservation_address=true`: the mapped reservation slice is
  //    returned and is treated as the allocator address. The caller releases it
  //    with Deallocate(), may use it as a Map() source, and must not pass it to
  //    UnMap().
  //  * `return_reservation_address=false`: the allocator creates and returns a
  //    separate allocator-owned address. The same raw physical allocation is
  //    also mapped into the caller reservation as a reservation address. The
  //    returned allocator address is released with Deallocate(); the
  //    reservation-address alias may be released earlier with UnMap().
  //
  // The caller owns `reservation` and must keep it alive while any mapping into
  // it is active or waiting for deferred unmap completion. Deallocate() never
  // destroys or takes ownership of `reservation`.
  absl::StatusOr<ScopedDeviceAddress<uint8_t>> Allocate(
      int device_ordinal, uint64_t allocation_size, bool retry_on_failure,
      int64_t memory_space, MemoryReservation* reservation,
      uint64_t reservation_offset, uint64_t mapping_size,
      bool return_reservation_address);

  // Pull in two-arg overload that sets retry_on_failure to true.
  using DeviceAddressAllocator::Allocate;

  // RAII: while in scope, Allocate() treats allocations as multi-device iff
  // the assignment has replica * computation > 1.
  class DeviceAssignmentScope {
   public:
    explicit DeviceAssignmentScope(
        const xla::DeviceAssignment* device_assignment);
    ~DeviceAssignmentScope();
    DeviceAssignmentScope(const DeviceAssignmentScope&) = delete;
    DeviceAssignmentScope& operator=(const DeviceAssignmentScope&) = delete;

   private:
    const xla::DeviceAssignment* previous_;
  };

  // Deallocates an allocator address asynchronously. `mem` must be an address
  // returned by Allocate(), including reservation-derived addresses returned by
  // Allocate(..., return_reservation_address=true). Reservation addresses
  // created by Map() or by Allocate(..., return_reservation_address=false) must
  // not be passed to Deallocate(). If `mem` has an active reservation-address
  // alias, the caller must release that alias with UnMap() before calling
  // Deallocate(). The caller can call this function while device kernels are
  // still consuming the data; the actual release is deferred until earlier work
  // on the device stream completes.
  absl::Status Deallocate(int device_ordinal, DeviceAddressBase mem) override;

  // Adds a reservation-address alias for an existing allocator address by
  // mapping the physical allocation currently backing `addr` into
  // `reservation` at `reservation_offset`.
  //
  // `addr` must be an active allocator address returned by this allocator,
  // including reservation-derived addresses returned by
  // Allocate(..., return_reservation_address=true). Non-owning reservation
  // addresses created by Map() or by
  // Allocate(..., return_reservation_address=false), and addresses from other
  // allocators, are not supported. The physical allocation backing `addr` must
  // be at least `size` bytes. Each allocator address may have at most one
  // active reservation-address alias at a time. The caller owns `reservation`
  // and must keep it alive until UnMap() is called and the allocator stream
  // reaches that deferred unmap point.
  absl::Status Map(int device_ordinal, DeviceAddressBase addr,
                   MemoryReservation* reservation, uint64_t reservation_offset,
                   uint64_t size);

  // Defers unmapping the reservation address created by Map() or by
  // Allocate(..., return_reservation_address=false) for the given reservation
  // range until all previously enqueued work on the allocator stream has
  // completed.
  // The caller must pass the same full reservation range that created the
  // mapping; partial ranges that overlap a tracked mapping are rejected.
  // The reservation-derived allocator address returned by
  // Allocate(..., return_reservation_address=true) is not a reservation
  // address for this API and must be released with Deallocate() instead.
  //
  // On success this method moves the active mapping to the deferred unmap
  // queue. On error, active bookkeeping is unchanged. Empty mappings, such as
  // zero-size Map(), are treated as no-ops.
  absl::Status UnMap(int device_ordinal, MemoryReservation* reservation,
                     uint64_t reservation_offset, uint64_t size);

  // Returns true: this allocator supports asynchronous deallocation.
  bool AllowsAsynchronousDeallocation() const override { return true; }

  // Returns the stream for the given device ordinal.
  absl::StatusOr<Stream*> GetStream(int device_ordinal) override;

  // Waits for all pending stream-ordered deallocations and unmaps on the given
  // device to complete, then drops the corresponding deferred bookkeeping.
  absl::Status SynchronizePendingOperations(int device_ordinal);

  // Returns the StreamExecutor for the given device ordinal.
  absl::StatusOr<StreamExecutor*> GetStreamExecutor(int device_ordinal) const;

  // Returns the MemoryAllocation (physical memory) backing the given virtual
  // address on the specified device, or nullptr if the address was not
  // allocated by this allocator. The returned pointer is valid until the
  // allocation is deallocated.
  MemoryAllocation* GetRawAllocation(int device_ordinal,
                                     DeviceAddressBase addr) const;

  // Returns the MemoryReservation (virtual address range) for the given
  // virtual address on the specified device, or nullptr if the address was not
  // allocated by this allocator. The returned pointer is valid until the
  // allocation is deallocated.
  MemoryReservation* GetReservation(int device_ordinal,
                                    DeviceAddressBase addr) const;

  // Returns the VMM allocation granularity for the device associated with
  // `executor`, or 0 if the device is not registered or granularity is unknown.
  uint64_t GetAllocationGranularity(StreamExecutor* executor) const;

  // Creates a virtual address reservation of the given size.
  virtual absl::StatusOr<std::unique_ptr<MemoryReservation>> CreateReservation(
      StreamExecutor* executor, uint64_t size) = 0;

 protected:
  enum class PendingDeallocationKind {
    // Deferred Deallocate() of an Allocate() result backed by an
    // allocator-owned reservation.
    kAllocate,
    // Deferred Deallocate() of an
    // Allocate(..., return_reservation_address=true) result. The allocator
    // address is a caller-owned reservation range.
    kAllocateAndMapReturnMapAddr,
    // Deferred Deallocate() of an
    // Allocate(..., return_reservation_address=false) result. The record has
    // an allocator-owned returned address and may also have a non-owning caller
    // reservation mapping to unmap.
    kAllocateAndMapReturnNewAddr,
    // Deferred completion of a Map()-owned reservation address. The reservation
    // address is a non-owning alias of an existing raw allocation.
    kMap,
  };

  // Lifetime record for one raw physical allocation.
  //
  // The record is owned by records_by_allocator_address while either the
  // allocator address or a reservation-address alias is active, stale, or
  // pending completion. Active indexes are callable by public APIs. Stale
  // indexes are no longer callable by users, but still keep mappings alive
  // until the stream-ordered deferred operation completes or a later Allocate()
  // or Map() reuses them.
  class AllocationRecord {
   public:
    enum class Kind {
      kAllocate,
      kAllocateAndMapReturnMapAddr,
      kAllocateAndMapReturnNewAddr,
    };

    enum class AllocatorState {
      kActive,
      kStale,
    };

    enum class ReservationState {
      kNone,
      kActive,
      kStale,
    };

    AllocationRecord(
        Kind kind, DeviceAddressBase allocator_address,
        std::unique_ptr<MemoryAllocation> raw_allocation,
        std::unique_ptr<MemoryReservation> allocator_address_reservation,
        MemoryReservation::ScopedMapping allocator_address_mapping,
        bool multi_device);
    AllocationRecord(const AllocationRecord&) = delete;
    AllocationRecord& operator=(const AllocationRecord&) = delete;
    AllocationRecord(AllocationRecord&&) = default;
    AllocationRecord& operator=(AllocationRecord&&) = default;

    Kind kind() const { return kind_; }
    PendingDeallocationKind pending_deallocation_kind() const;
    DeviceAddressBase allocator_address() const { return allocator_address_; }
    void* allocator_key() const { return allocator_address_.opaque(); }
    bool allocator_active() const {
      return allocator_state_ == AllocatorState::kActive;
    }
    bool allocator_stale() const {
      return allocator_state_ == AllocatorState::kStale;
    }
    bool allocator_matches(DeviceAddressBase address) const {
      return allocator_address_.IsSameAs(address);
    }
    uint64_t allocator_stale_seqno() const { return allocator_stale_seqno_; }
    bool multi_device() const { return multi_device_; }
    MemoryAllocation* raw_allocation() const { return raw_allocation_.get(); }
    MemoryReservation* allocator_address_reservation() const {
      return allocator_address_reservation_.get();
    }
    bool has_allocator_address_mapping() const {
      return allocator_address_mapping_.has_value();
    }

    bool has_reservation_alias() const {
      return reservation_state_ != ReservationState::kNone;
    }
    bool reservation_active() const {
      return reservation_state_ == ReservationState::kActive;
    }
    bool reservation_stale() const {
      return reservation_state_ == ReservationState::kStale;
    }
    bool has_reservation_address() const {
      return reservation_address_.has_value();
    }
    DeviceAddressBase reservation_address() const;
    void* reservation_key() const { return reservation_address().opaque(); }
    bool reservation_matches(DeviceAddressBase address) const {
      return has_reservation_address() &&
             reservation_address_->IsSameAs(address);
    }
    uint64_t reservation_stale_seqno() const {
      return reservation_stale_seqno_;
    }
    bool reservation_mapping_matches(DeviceAddressBase address) const;

    void MarkAllocatorStale(uint64_t seqno);
    void ReactivateAllocator(uint64_t new_size);
    void CompleteStaleAllocator();

    void AddActiveReservationAlias(
        DeviceAddressBase reservation_address,
        MemoryReservation::ScopedMapping reservation_address_mapping);
    void MarkReservationStale(uint64_t seqno);
    void ReactivateReservation();
    void CompleteStaleReservation();

   private:
    Kind kind_;
    DeviceAddressBase allocator_address_;
    std::unique_ptr<MemoryAllocation> raw_allocation_;
    bool multi_device_;

    // Present for Allocate() and
    // Allocate(..., return_reservation_address=false).
    std::unique_ptr<MemoryReservation> allocator_address_reservation_;
    // Present while the allocator address is active or stale.
    std::optional<MemoryReservation::ScopedMapping> allocator_address_mapping_;

    // Present while a reservation alias is active or stale.
    std::optional<DeviceAddressBase> reservation_address_;
    std::optional<MemoryReservation::ScopedMapping>
        reservation_address_mapping_;

    AllocatorState allocator_state_ = AllocatorState::kActive;
    ReservationState reservation_state_ = ReservationState::kNone;
    uint64_t allocator_stale_seqno_ = 0;
    uint64_t reservation_stale_seqno_ = 0;
  };

  // Queue entry for a stream-ordered deferred operation. The heavy resources
  // live in AllocationRecord; this entry only says which stale address becomes
  // safe to complete when the GPU timeline reaches `seqno`.
  struct PendingDeallocation {
    PendingDeallocationKind kind = PendingDeallocationKind::kAllocate;
    // GPU stream sequence number recorded at deallocation time. When the
    // pinned_timeline value reaches this seqno, the memory is safe to free.
    uint64_t seqno = 0;
    // Allocator address for allocation deallocations; reservation address for
    // kMap.
    DeviceAddressBase addr;
    // Physical-allocation bytes charged to the PA budget that become
    // reclaimable when this pending operation completes. kMap entries do not
    // own physical memory and therefore use zero.
    uint64_t reclaimable_bytes = 0;
  };

  // Stable identity for a pending operation. Iterators into
  // pending_deallocations must not be kept across waits because
  // WaitUntilSeqno() releases state.mu.
  struct PendingDeallocationKey {
    PendingDeallocationKind kind = PendingDeallocationKind::kAllocate;
    uint64_t seqno = 0;
    DeviceAddressBase addr;
  };

  struct PerDeviceState {
    StreamExecutor* executor;
    Stream* stream;
    uint64_t pa_budget;
    // VMM allocation granularity for this device. Set once in
    // InitializeDeviceState(); 0 means the query failed.
    uint64_t allocation_granularity = 0;

    // Host-visible timeline counter. The GPU writes an increasing sequence
    // number to this location as each deallocation point is reached in the
    // stream. The CPU reads it atomically to determine which pending
    // deallocations are safe to execute.
    // Allocated at construction in InitializeDeviceState(); freed via
    // destroy_fn in ~DeviceAddressVmmAllocator().
    // Never modified after construction other than by the GPU.
    volatile uint64_t* pinned_timeline = nullptr;
    // Device-mapped pointer to pinned_timeline (as uint64_t to avoid
    // platform-specific types in this header). Passed to the platform-specific
    // stream write operation.
    uint64_t timeline_dev_ptr = 0;

    // Called at the end of ~DeviceAddressVmmAllocator() to release
    // platform-specific resources (e.g. pinned timeline memory). Set once by
    // InitializeDeviceState(); must not reference the subclass instance.
    std::function<void()> destroy_fn;

    mutable absl::Mutex mu;
    uint64_t pa_allocated ABSL_GUARDED_BY(mu) = 0;
    // Monotonically increasing counter for timeline sequence numbers.
    uint64_t next_seqno ABSL_GUARDED_BY(mu) = 1;
    std::deque<PendingDeallocation> pending_deallocations ABSL_GUARDED_BY(mu);
    // Owns AllocationRecord objects. Key is the allocator address pointer
    // (`AllocationRecord::allocator_address().opaque()`), including the
    // reservation-derived allocator address returned by
    // Allocate(..., return_reservation_address=true). Allocator-address
    // active/stale state is stored in
    // AllocationRecord::allocator_active()/allocator_stale().
    absl::flat_hash_map<void*, std::unique_ptr<AllocationRecord>>
        records_by_allocator_address ABSL_GUARDED_BY(mu);

    // Active/stale reservation-address indexes. Keys are reservation alias
    // pointers (`AllocationRecord::reservation_address().opaque()`) created by
    // Map() or by Allocate(..., return_reservation_address=false).
    absl::flat_hash_map<void*, AllocationRecord*> active_reservation_records
        ABSL_GUARDED_BY(mu);
    absl::flat_hash_map<void*, AllocationRecord*> stale_reservation_records
        ABSL_GUARDED_BY(mu);
  };

  explicit DeviceAddressVmmAllocator(const Platform* platform);

  // Validates no duplicate ordinals in `devices`, then iterates over each
  // device config, constructs a PerDeviceState (setting executor, stream,
  // pa_budget), calls InitializeDeviceState() for platform-specific
  // initialization, and registers the state in allocator->per_device_.
  //
  // Called by platform-specific Create() factories.
  static absl::Status PopulateDevices(DeviceAddressVmmAllocator* allocator,
                                      absl::Span<const DeviceConfig> devices);

  // Drains all pending operations for all devices.
  absl::Status SynchronizeAllPendingOperations();

  // Validates device capabilities and initializes timeline fields
  // (pinned_timeline, timeline_dev_ptr, allocation_granularity) in state.
  // state.executor, state.stream, and state.pa_budget are already set.
  virtual absl::Status InitializeDeviceState(PerDeviceState& state) = 0;

  // Creates a physical memory allocation of the given size.
  virtual absl::StatusOr<std::unique_ptr<MemoryAllocation>> CreateAllocation(
      StreamExecutor* executor, uint64_t size) = 0;

  // Enqueues a GPU timeline write at the given seqno on the device's stream.
  virtual absl::Status EnqueueDeferredDeallocation(PerDeviceState& state,
                                                   uint64_t seqno) = 0;

 private:
  // Common helpers.

  // Returns pointer into per_device_ map, or NotFound if device_ordinal is not
  // registered. No lock needed: per_device_ is read-only after construction.
  absl::StatusOr<PerDeviceState*> GetPerDeviceState(int device_ordinal) const;

  // Round up size to the device's allocation granularity.
  uint64_t RoundUpToGranularity(const PerDeviceState& state,
                                uint64_t size) const;

  // True iff the calling thread is inside a multi-device DeviceAssignmentScope.
  static bool CurrentMultiDevice();

  // Allocate helpers.

  // Checks the PA budget for `size`, creates a physical allocation, and
  // re-checks the budget against the actual (granularity-padded) size the
  // driver returned, which is the size charged to pa_allocated. Returns the
  // allocation and reports that size via `physical_size`.
  absl::StatusOr<std::unique_ptr<MemoryAllocation>>
  AllocatePhysicalWithinBudget(PerDeviceState& state, uint64_t size,
                               uint64_t& physical_size)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  // Records a raw allocation mapped at an owning allocator address. Takes
  // ownership of `reservation` when the allocator address was allocator-owned;
  // reservation-backed returned addresses pass nullptr here. Charges the raw
  // allocation's committed size to the PA budget and returns the allocator VA
  // pointer.
  void* TrackAllocatorAddressMappedAllocation(
      PerDeviceState& state, AllocationRecord::Kind kind,
      DeviceAddressBase allocator_address,
      std::unique_ptr<MemoryAllocation> raw_allocation,
      std::unique_ptr<MemoryReservation> reservation,
      MemoryReservation::ScopedMapping mapping, bool multi_device)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  struct MappedAllocateRequest {
    MemoryReservation* reservation;
    DeviceAddressBase reservation_address;
    uint64_t allocation_size;
    uint64_t reservation_offset;
    uint64_t mapping_size;
    bool multi_device;
  };

  // Reactivates a stale mapped allocation whose returned allocator address is
  // the requested caller-owned reservation address.
  absl::StatusOr<std::optional<DeviceAddressBase>>
  TryReuseMappedAllocationAtReservationAddress(
      PerDeviceState& state, const MappedAllocateRequest& request)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  // Reactivates a stale mapped allocation with both a separate allocator-owned
  // returned address and an alias at the requested reservation address.
  absl::StatusOr<std::optional<DeviceAddressBase>>
  TryReuseMappedAllocationWithSeparateAddress(
      PerDeviceState& state, const MappedAllocateRequest& request)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  // Waits for exact stale overlaps with the requested reservation range and
  // rejects partial or active overlaps before a fresh mapping is installed.
  absl::Status EnsureReservationAvailableForFreshMapping(
      PerDeviceState& state, const MappedAllocateRequest& request)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  // Creates a mapped allocation that uses the caller-owned reservation address
  // as its returned allocator address.
  absl::StatusOr<DeviceAddressBase> CreateMappedAllocationAtReservationAddress(
      PerDeviceState& state, const MappedAllocateRequest& request)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  // Creates a mapped allocation with a separate allocator-owned returned
  // address and a non-owning alias in the caller-owned reservation.
  absl::StatusOr<DeviceAddressBase> CreateMappedAllocationWithSeparateAddress(
      PerDeviceState& state, const MappedAllocateRequest& request)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  // Shared allocation retry policy. First calls `try_reuse` to reactivate
  // compatible pending state without blocking, then calls `try_fresh`. On
  // ResourceExhausted, it completes ready pending entries and, if needed, waits
  // for enough pending frees to reclaim approximately `reclaim_size` bytes.
  template <typename TryReuseFn, typename TryFreshFn>
  absl::StatusOr<DeviceAddressBase> TryWithPendingReclaim(PerDeviceState& state,
                                                          uint64_t reclaim_size,
                                                          TryReuseFn try_reuse,
                                                          TryFreshFn try_fresh)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  // Map helpers.

  // Validates a caller-owned reservation slice and returns the corresponding
  // DeviceAddressBase. Rejects null reservations and out-of-bounds
  // offset/size pairs before any allocator bookkeeping is mutated.
  absl::StatusOr<DeviceAddressBase> ValidateReservationRange(
      MemoryReservation* reservation, uint64_t reservation_offset,
      uint64_t size) const;

  struct OverlappingRecord {
    AllocationRecord* record = nullptr;
    DeviceAddressBase tracked_address;
    bool is_allocator = false;
    bool is_active = false;
  };

  enum class AddressRole { kAllocator, kReservation, kBoth };
  enum class RecordState { kActive, kStale, kBoth };
  enum class OverlapKind { kExact, kPartial };

  // Immutable inputs shared by the Map() preparation and installation
  // helpers.
  struct MapRequest {
    DeviceAddressBase source_address;
    MemoryReservation* reservation = nullptr;
    uint64_t reservation_offset = 0;
    uint64_t size = 0;
    DeviceAddressBase reservation_address;
  };

  // Result of inspecting the requested Map() target while holding state.mu.
  // A stale record is consumed immediately for kReactivateStale; a pending key
  // is copied out before kWait releases state.mu.
  struct MapTargetEvaluation {
    enum class Action { kInstallFresh, kReactivateStale, kWait };

    Action action;
    AllocationRecord* stale_record = nullptr;
    std::optional<PendingDeallocationKey> pending_completion_key;
  };

  // Result of preparing a Map() target. kInstallFresh carries the current
  // source record; kReusedStaleMapping means preparation already reactivated
  // the requested mapping.
  struct PreparedMapTarget {
    enum class Action { kInstallFresh, kReusedStaleMapping };

    Action action;
    AllocationRecord* source_record = nullptr;
  };

  // Finds a tracked allocator or reservation range that overlaps `address`.
  std::optional<OverlappingRecord> FindOverlappingRecord(
      PerDeviceState& state, DeviceAddressBase address, AddressRole role,
      RecordState record_state, OverlapKind overlap_kind) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  // Resolves an active allocator address to its allocation record.
  absl::StatusOr<AllocationRecord*> ResolveMapSourceRecord(
      PerDeviceState& state, DeviceAddressBase source_address) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  // Validates that `size` fits in the source record's physical allocation.
  absl::Status ValidateMapSourceSize(PerDeviceState& state,
                                     const AllocationRecord& source_record,
                                     uint64_t size) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  // Resolves and validates the Map() source before target preparation starts.
  absl::StatusOr<AllocationRecord*> ResolveAndValidateMapSource(
      PerDeviceState& state, DeviceAddressBase source_address,
      uint64_t size) const ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  // Rejects non-identical overlaps with any active or stale tracked range.
  absl::Status CheckNoPartialReservationOverlap(
      PerDeviceState& state, DeviceAddressBase reservation_address) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  // Determines whether Map() can install a fresh mapping, reactivate a stale
  // mapping, or must wait for a conflicting pending operation.
  absl::StatusOr<MapTargetEvaluation> EvaluateMapTarget(
      PerDeviceState& state, const MapRequest& request,
      AllocationRecord& source_record) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  // Resolves up to two stale conflicts for a Map() request. Every wait is
  // followed by a fresh source lookup and validation because waiting
  // temporarily releases state.mu.
  absl::StatusOr<PreparedMapTarget> PrepareMapTarget(PerDeviceState& state,
                                                     const MapRequest& request)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  // Installs and records a fresh reservation-address alias.
  absl::Status InstallMapAlias(PerDeviceState& state, const MapRequest& request,
                               AllocationRecord& source_record)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  // UnMap/deferred teardown helpers.

  // Removes a pending entry when a stale record is reused.
  void ErasePendingDeallocationAt(PerDeviceState& state,
                                  std::deque<PendingDeallocation>::iterator it)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  // Removes the matching pending entry when a stale record is reused.
  void ErasePendingDeallocation(PerDeviceState& state,
                                PendingDeallocationKind kind,
                                DeviceAddressBase addr)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  void MoveAllocatorRecordToActive(PerDeviceState& state,
                                   AllocationRecord& record, uint64_t new_size)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  void MoveReservationRecordToStale(PerDeviceState& state,
                                    AllocationRecord& record, uint64_t seqno)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  void MoveReservationRecordToActive(PerDeviceState& state,
                                     AllocationRecord& record)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  void CompleteStaleReservationMapping(PerDeviceState& state,
                                       AllocationRecord& record)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  // Waits for the device timeline to reach `target_seqno`. Temporarily releases
  // and reacquires state.mu around the blocking wait. This does not complete
  // pending entries by itself.
  absl::Status WaitUntilSeqno(PerDeviceState& state, uint64_t target_seqno)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  // Waits for pending operations through `target_seqno`, then completes all
  // still-pending operations up to that sequence. Used only when preserving
  // stale mappings for future reuse is no longer useful.
  absl::Status WaitAndDrainPendingDeallocationsUntilSeqno(PerDeviceState& state,
                                                          uint64_t target_seqno)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  // Completes ready allocator-address deallocations for PA reclaim while
  // leaving unrelated kMap entries stale and reusable.
  void CompleteReadyAllocatorDeallocationsForReclaim(PerDeviceState& state,
                                                     uint64_t completed_seqno)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  // Completes a pending operation whose stream sequence has passed by dropping
  // its ScopedMappings, allocator-owned reservation, and raw allocation
  // reference. This is where VA unmap, reservation release, and PA budget
  // accounting happen.
  void CompletePendingDeallocation(PerDeviceState& state,
                                   const PendingDeallocation& pending)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  // Finds, erases, and completes the selected pending entry if it is still
  // present. Returns false if another thread already reused or completed it
  // while state.mu was released.
  bool CompletePendingDeallocationByKey(PerDeviceState& state,
                                        const PendingDeallocationKey& key)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  // Waits for and completes the selected allocator-address deallocation, if it
  // is still pending after the wait.
  absl::Status WaitAndCompleteStaleAllocatorDeallocation(
      PerDeviceState& state, const PendingDeallocationKey& key)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  // Waits for and completes a stale reservation-address mapping queued by
  // UnMap().
  absl::Status WaitAndCompleteStaleReservationMapping(
      PerDeviceState& state, const PendingDeallocationKey& key)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  // Completes only the stale allocator or reservation mapping that conflicts
  // with the current request, leaving unrelated stale mappings reusable.
  absl::Status WaitAndCompleteStaleOverlap(PerDeviceState& state,
                                           const OverlappingRecord& overlap)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mu);

  // Device ordinal -> per-device allocator state. Populated at construction by
  // PopulateDevices() and never modified afterward, so map lookup is safe
  // without an allocator-wide lock. Each PerDeviceState owns its own mutex for
  // mutable allocation and pending-deallocation state.
  absl::flat_hash_map<int, std::unique_ptr<PerDeviceState>> per_device_;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_DEVICE_ADDRESS_VMM_ALLOCATOR_H_
