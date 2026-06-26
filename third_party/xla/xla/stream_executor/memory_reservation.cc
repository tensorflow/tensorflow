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

#include "xla/stream_executor/memory_reservation.h"

#include <cstddef>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "absl/status/status_macros.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/tsl/platform/errors.h"

namespace stream_executor {

// ScopedMapping

void MemoryReservation::ScopedMapping::UnmapAndLogIfError(
    MemoryReservation* reservation, size_t reservation_offset, size_t size) {
  absl::Status status = reservation->UnMap(reservation_offset, size);
  if (!status.ok()) {
    LOG(ERROR) << "ScopedMapping: failed to unmap reservation range: "
               << status.message();
  }
}

MemoryReservation::ScopedMapping::ScopedMapping(MemoryReservation* reservation,
                                                size_t reservation_offset,
                                                size_t size)
    : reservation_(reservation),
      reservation_offset_(reservation_offset),
      size_(size) {}

MemoryReservation::ScopedMapping::~ScopedMapping() {
  if (reservation_ == nullptr) {
    return;
  }
  UnmapAndLogIfError(reservation_, reservation_offset_, size_);
}

MemoryReservation::ScopedMapping::ScopedMapping(ScopedMapping&& other) noexcept
    : reservation_(other.reservation_),
      reservation_offset_(other.reservation_offset_),
      size_(other.size_) {
  other.reservation_ = nullptr;
}

MemoryReservation::ScopedMapping& MemoryReservation::ScopedMapping::operator=(
    ScopedMapping&& other) noexcept {
  if (this != &other) {
    if (reservation_ != nullptr) {
      UnmapAndLogIfError(reservation_, reservation_offset_, size_);
    }
    reservation_ = other.reservation_;
    reservation_offset_ = other.reservation_offset_;
    size_ = other.size_;
    other.reservation_ = nullptr;
  }
  return *this;
}

DeviceAddressBase MemoryReservation::ScopedMapping::mapped_address() const {
  return reservation_->address().GetByteSlice(reservation_offset_, size_);
}

// MemoryReservation::MapTo

absl::StatusOr<MemoryReservation::ScopedMapping> MemoryReservation::MapTo(
    size_t reservation_offset, size_t allocation_offset, size_t size,
    MemoryAllocation& allocation) {
  RETURN_IF_ERROR(Map(reservation_offset, allocation_offset, size, allocation));

  auto cleanup = absl::MakeCleanup([&] {
    absl::Status unmap_status = UnMap(reservation_offset, size);
    if (!unmap_status.ok()) {
      LOG(ERROR) << "MapTo: failed to unmap after failure: "
                 << unmap_status.message();
    }
  });

  RETURN_IF_ERROR(SetAccess(reservation_offset, size));

  std::move(cleanup).Cancel();
  return ScopedMapping(this, reservation_offset, size);
}

absl::StatusOr<MemoryReservation::ScopedMapping> MemoryReservation::MapTo(
    absl::Span<const MappingDescriptor> mappings) {
  if (mappings.empty()) {
    return absl::InvalidArgumentError("MapTo: mappings must not be empty");
  }

  size_t start_offset = mappings[0].reservation_offset;
  size_t expected_offset = start_offset;
  for (const MappingDescriptor& desc : mappings) {
    if (desc.reservation_offset != expected_offset) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "MapTo: mappings are not contiguous. Expected reservation_offset=%zu "
          "but got %zu",
          expected_offset, desc.reservation_offset));
    }
    expected_offset += desc.size;
  }

  size_t total_size = 0;

  auto cleanup = absl::MakeCleanup([&] {
    if (total_size > 0) {
      absl::Status unmap_status = UnMap(start_offset, total_size);
      if (!unmap_status.ok()) {
        LOG(ERROR) << "MapTo: failed to unmap after failure: "
                   << unmap_status.message();
      }
    }
  });

  for (const MappingDescriptor& desc : mappings) {
    RETURN_IF_ERROR(Map(desc.reservation_offset, desc.allocation_offset,
                        desc.size, *desc.allocation));
    total_size += desc.size;
  }

  RETURN_IF_ERROR(SetAccess(start_offset, total_size));

  std::move(cleanup).Cancel();
  return ScopedMapping(this, start_offset, total_size);
}

// ScopedMapping::Remap

absl::StatusOr<size_t>
MemoryReservation::ScopedMapping::ValidateRemapDescriptors(
    absl::Span<const MemoryReservation::RemappingDescriptor> mappings,
    size_t existing_reservation_offset, size_t existing_size) {
  const size_t start_offset = mappings[0].reservation_offset;
  size_t expected_offset = start_offset;
  for (const MemoryReservation::RemappingDescriptor& desc : mappings) {
    if (desc.reservation_offset != expected_offset) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Remap: mappings are not contiguous. Expected "
                          "reservation_offset=%zu but got %zu",
                          expected_offset, desc.reservation_offset));
    }
    if (desc.allocation == nullptr) {
      return absl::InvalidArgumentError("Remap: allocation must not be null");
    }
    expected_offset += desc.size;
  }
  const size_t total_size = expected_offset - start_offset;
  if (start_offset != existing_reservation_offset ||
      total_size != existing_size) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Remap: mappings must cover the existing mapping range [%zu, %zu), got "
        "[%zu, %zu)",
        existing_reservation_offset,
        existing_reservation_offset + existing_size, start_offset,
        start_offset + total_size));
  }
  return total_size;
}

absl::StatusOr<int> MemoryReservation::ScopedMapping::UnmapChangedRuns(
    MemoryReservation* reservation,
    absl::Span<const MemoryReservation::RemappingDescriptor> mappings,
    std::vector<bool>& slice_mapped) {
  // Unmap the slices whose backing physical handle changed, coalescing
  // contiguous runs into a single UnMap. A single hipMemUnmap can release a
  // range spanning several separately-mapped slices (the ScopedMapping
  // destructor relies on exactly this to unmap the whole reservation in one
  // call), so unmapping per contiguous run instead of per slice cuts the number
  // of driver round-trips on the hot path. Unchanged slices keep their mapping,
  // which is the whole point: it avoids the page-table churn of remapping every
  // slice on every step.
  int changed_count = 0;
  for (size_t k = 0; k < mappings.size();) {
    if (!mappings[k].remap_required) {
      ++k;
      continue;
    }
    const size_t run_start = mappings[k].reservation_offset;
    size_t run_end = run_start;
    size_t j = k;
    while (j < mappings.size() && mappings[j].remap_required) {
      run_end = mappings[j].reservation_offset + mappings[j].size;
      ++changed_count;
      ++j;
    }
    RETURN_IF_ERROR(reservation->UnMap(run_start, run_end - run_start));
    for (size_t m = k; m < j; ++m) {
      slice_mapped[m] = false;
    }
    k = j;
  }
  return changed_count;
}

absl::Status MemoryReservation::ScopedMapping::MapChangedSlices(
    MemoryReservation* reservation,
    absl::Span<const MemoryReservation::RemappingDescriptor> mappings,
    std::vector<bool>& slice_mapped) {
  // Map each changed slice to its new physical handle. Map must be per slice
  // because each slice has its own backing allocation and offset.
  for (size_t k = 0; k < mappings.size(); ++k) {
    if (!mappings[k].remap_required) {
      continue;
    }
    const MemoryReservation::RemappingDescriptor& dk = mappings[k];
    RETURN_IF_ERROR(reservation->Map(
        dk.reservation_offset, dk.allocation_offset, dk.size, *dk.allocation));
    slice_mapped[k] = true;
  }
  return absl::OkStatus();
}

absl::StatusOr<MemoryReservation::ScopedMapping>
MemoryReservation::ScopedMapping::Remap(
    absl::Span<const MemoryReservation::RemappingDescriptor> mappings) && {
  // Checks.
  if (reservation_ == nullptr) {
    return absl::FailedPreconditionError("Remap: mapping is empty");
  }
  if (mappings.empty()) {
    return absl::InvalidArgumentError("Remap: mappings must not be empty");
  }
  ASSIGN_OR_RETURN(
      const size_t total_size,
      ValidateRemapDescriptors(mappings, reservation_offset_, size_));

  MemoryReservation* reservation = reservation_;
  const size_t start_offset = mappings[0].reservation_offset;

  // Track per-slice mapped state for cleanup on failure. Every slice starts
  // mapped because this ScopedMapping owns the full range. Changed slices are
  // temporarily unmapped before being mapped to their new allocation.
  std::vector<bool> slice_mapped(mappings.size(), true);

  // Detach the prior ScopedMapping without invoking its destructor; the
  // per-slice unmaps below replace the full-range unmap it would do.
  reservation_ = nullptr;

  // On failure, unmap every slice that is still mapped so the reservation
  // is left in a clean (fully unmapped) state rather than partially mapped
  // with no RAII owner.
  auto cleanup = absl::MakeCleanup([&] {
    for (size_t k = 0; k < mappings.size(); ++k) {
      if (slice_mapped[k]) {
        absl::Status s = reservation->UnMap(mappings[k].reservation_offset,
                                            mappings[k].size);
        if (!s.ok()) {
          LOG(ERROR) << "Remap: cleanup failed to unmap slice at offset "
                     << mappings[k].reservation_offset << ": " << s.message();
        }
      }
    }
  });

  // Phase 1: unmap the slices whose backing physical handle changed.
  const absl::Time t_unmap0 = absl::Now();
  ASSIGN_OR_RETURN(const int changed_count,
                   UnmapChangedRuns(reservation, mappings, slice_mapped));
  const bool any_remapped = changed_count > 0;

  // Phase 2: map each changed slice to its new physical handle.
  const absl::Time t_map0 = absl::Now();
  RETURN_IF_ERROR(MapChangedSlices(reservation, mappings, slice_mapped));

  // Grant device access once over the FULL reservation range. On ROCm,
  // hipMemSetAccess for peer devices rejects any partial range (sub-range or
  // even a prefix that starts at the reservation base) with
  // HIP_ERROR_InvalidValue -- empirically only the full [start_offset,
  // total_size) range is accepted, matching MapTo's known-good behavior. Skip
  // it entirely when no slice changed so the steady state stays free of driver
  // calls.
  const absl::Time t_map1 = absl::Now();
  if (any_remapped) {
    RETURN_IF_ERROR(reservation->SetAccess(start_offset, total_size));
  }
  const absl::Time t_sa1 = absl::Now();
  VLOG(2) << "Remap timing: slices=" << mappings.size()
          << " changed=" << changed_count << " range=" << total_size
          << " unmap_us=" << absl::ToInt64Microseconds(t_map0 - t_unmap0)
          << " map_us=" << absl::ToInt64Microseconds(t_map1 - t_map0)
          << " setaccess_us=" << absl::ToInt64Microseconds(t_sa1 - t_map1);

  std::move(cleanup).Cancel();
  return ScopedMapping(reservation, start_offset, total_size);
}

}  // namespace stream_executor
