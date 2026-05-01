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

#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/memory_allocation.h"
#include "tsl/platform/errors.h"

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
  TF_RETURN_IF_ERROR(
      Map(reservation_offset, allocation_offset, size, allocation));

  auto cleanup = absl::MakeCleanup([&] {
    absl::Status unmap_status = UnMap(reservation_offset, size);
    if (!unmap_status.ok()) {
      LOG(ERROR) << "MapTo: failed to unmap after failure: "
                 << unmap_status.message();
    }
  });

  TF_RETURN_IF_ERROR(SetAccess(reservation_offset, size));

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
    TF_RETURN_IF_ERROR(Map(desc.reservation_offset, desc.allocation_offset,
                           desc.size, *desc.allocation));
    total_size += desc.size;
  }

  TF_RETURN_IF_ERROR(SetAccess(start_offset, total_size));

  std::move(cleanup).Cancel();
  return ScopedMapping(this, start_offset, total_size);
}

}  // namespace stream_executor
