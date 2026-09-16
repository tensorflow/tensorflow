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

#ifndef XLA_STREAM_EXECUTOR_MEMORY_RESERVATION_H_
#define XLA_STREAM_EXECUTOR_MEMORY_RESERVATION_H_

#include <cstddef>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/memory_allocation.h"

namespace stream_executor {

// A MemoryReservation represents a reserved virtual address range on a
// StreamExecutor device. The range is not backed by physical memory until
// physical allocations are mapped into it via MapTo.
//
// MemoryReservation is the base class for platform-specific implementations
// (e.g. CUDA virtual memory management via cuMemAddressReserve).
class MemoryReservation {
 public:
  MemoryReservation() = default;
  virtual ~MemoryReservation() = default;

  MemoryReservation(MemoryReservation&&) = delete;
  MemoryReservation& operator=(MemoryReservation&&) = delete;

  // Returns the base address and size of the reserved virtual address range.
  // The returned address is a valid device virtual address, but may not be
  // accessible until physical memory is mapped via MapTo.
  virtual DeviceAddressBase address() const = 0;

  // Describes a mapping from a memory reservation range
  // [reservation_offset, reservation_offset + size) to a physical allocation
  // range [allocation_offset, allocation_offset + size).
  struct MappingDescriptor {
    size_t reservation_offset;
    size_t allocation_offset;
    size_t size;
    MemoryAllocation* allocation;
  };

  // Describes a desired post-Remap mapping for a slice of the reservation.
  struct RemappingDescriptor {
    size_t reservation_offset;
    size_t allocation_offset;
    size_t size;
    MemoryAllocation* allocation;  // current physical handle, never null
    // Whether this slice needs UnMap/Map/SetAccess calls. When false, the
    // existing mapping is preserved.
    bool remap_required;
  };

  // An RAII wrapper that gives access to a contiguous slice of a memory
  // reservation backed by one or more physical memory allocations.
  // Unmaps the mapped range from the reservation on destruction.
  class ScopedMapping {
   public:
    ScopedMapping() = default;
    ~ScopedMapping();

    ScopedMapping(ScopedMapping&&) noexcept;
    ScopedMapping& operator=(ScopedMapping&&) noexcept;

    // Returns the device address range that is mapped to the physical memory
    // allocation(s) and can be accessed from the device.
    DeviceAddressBase mapped_address() const;

    // Re-maps this mapping's reservation range as described by `mappings`.
    // The descriptors must cover the same full range. For each mapping where
    // remap_required is false, the current mapping is preserved. For the rest,
    // the slice is unmapped and mapped to `allocation`; SetAccess is invoked
    // once per maximal run of consecutive remapped mappings. On failure after
    // remapping starts, Remap unmaps all slices described by `mappings`,
    // leaving the reservation with no active mapping for the range.
    absl::StatusOr<ScopedMapping> Remap(
        absl::Span<const RemappingDescriptor> mappings) &&;

    // Returns true if this object does not own a mapping.
    bool is_null() const { return reservation_ == nullptr; }

   private:
    // Unmaps the given range on the reservation and logs on failure.
    static void UnmapAndLogIfError(MemoryReservation* reservation,
                                   size_t reservation_offset, size_t size);

    // Remap helpers. ----------------------------------------------------------

    // Validates that `mappings` are contiguous, have non-null allocations, and
    // exactly cover [existing_reservation_offset,
    // existing_reservation_offset + existing_size). Returns the covered size.
    static absl::StatusOr<size_t> ValidateRemapDescriptors(
        absl::Span<const RemappingDescriptor> mappings,
        size_t existing_reservation_offset, size_t existing_size);

    // Unmaps the slices whose backing handle changed, coalescing contiguous
    // remapped runs into a single UnMap. Clears the corresponding entries in
    // `slice_mapped` and returns the number of changed slices.
    static absl::StatusOr<int> UnmapChangedRuns(
        MemoryReservation* reservation,
        absl::Span<const RemappingDescriptor> mappings,
        std::vector<bool>& slice_mapped);

    // Maps each changed slice to its new physical handle and sets the
    // corresponding entries in `slice_mapped`.
    static absl::Status MapChangedSlices(
        MemoryReservation* reservation,
        absl::Span<const RemappingDescriptor> mappings,
        std::vector<bool>& slice_mapped);

    friend class MemoryReservation;
    ScopedMapping(MemoryReservation* reservation, size_t reservation_offset,
                  size_t size);

    MemoryReservation* reservation_ = nullptr;
    size_t reservation_offset_ = 0;
    size_t size_ = 0;
  };

  // Maps a single physical allocation to the reservation range
  // [reservation_offset, reservation_offset + size), backed by the allocation
  // range [allocation_offset, allocation_offset + size). Enables device access
  // to the mapped range before returning.
  absl::StatusOr<ScopedMapping> MapTo(size_t reservation_offset,
                                      size_t allocation_offset, size_t size,
                                      MemoryAllocation& allocation);

  // Maps multiple physical allocations to a contiguous memory reservation
  // range. The descriptors must form a contiguous range in the reservation.
  // Enables device access to the full contiguous range before returning.
  // For non-contiguous mappings, use separate MapTo calls instead.
  absl::StatusOr<ScopedMapping> MapTo(
      absl::Span<const MappingDescriptor> mappings);

 private:
  virtual absl::Status Map(size_t reservation_offset, size_t allocation_offset,
                           size_t size, MemoryAllocation& allocation) = 0;

  // Enable read/write access to the reservation range specified by the offset
  // and size.
  virtual absl::Status SetAccess(uint64_t reservation_offset, size_t size) = 0;

  virtual absl::Status UnMap(size_t reservation_offset, size_t size) = 0;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_MEMORY_RESERVATION_H_
