/* Copyright 2019 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_INTEGRATIONS_DEVICE_MEM_ALLOCATOR_H_
#define XLA_STREAM_EXECUTOR_INTEGRATIONS_DEVICE_MEM_ALLOCATOR_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_reservation.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/framework/device_id.h"
#include "tsl/profiler/lib/traceme.h"

namespace stream_executor {

// Suballocator for StreamExecutor-based device memory.
class DeviceMemAllocator : public tsl::SubAllocator {
 public:
  // 'platform_device_id' refers to the ID of the device within
  // the process and must reference a valid ID in the process.
  // Note: stream_exec cannot be null.
  DeviceMemAllocator(StreamExecutor* stream_exec,
                     tsl::PlatformDeviceId device_id,
                     const std::vector<Visitor>& alloc_visitors = {},
                     const std::vector<Visitor>& free_visitors = {})
      : SubAllocator(alloc_visitors, free_visitors),
        stream_exec_(stream_exec),
        device_id_(device_id) {
    CHECK(stream_exec_ != nullptr);
  }

  ~DeviceMemAllocator() override {
    if (mapped_bytes_ != 0) {
      // Qualified: virtual dispatch is gone during destruction anyway.
      DeviceMemAllocator::Free(reservation_->address().opaque(), mapped_bytes_);
    }
  }

  // Optional mode for a growing BFC arena. Reserve capacity bytes of VA, then
  // back a contiguous prefix on demand in Alloc. Existing mappings never move.
  // Call before any Alloc and before passing this suballocator to BFCAllocator,
  // whose constructor caches SupportsCoalescing(). Returns Unimplemented on
  // unsupported backends.
  // Calls to this suballocator must be externally serialized (as in BFC).
  absl::Status ReserveMemory(size_t capacity) {
    if (capacity == 0 || allocations_started_ || reservation_ != nullptr) {
      return absl::InvalidArgumentError(
          "ReserveMemory requires a positive capacity and an unused allocator");
    }
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<MemoryReservation> reservation,
                     stream_exec_->CreateMemoryReservation(capacity));
    if (reservation->address().opaque() == nullptr ||
        reservation->granularity() == 0) {
      return absl::InternalError(
          "Device VA reservation requires a non-null address and positive "
          "granularity");
    }
    if (reservation->address().size() < capacity) {
      return absl::InternalError(
          "Device VA reservation is smaller than capacity");
    }
    reservation_ = std::move(reservation);
    capacity_ = capacity;
    return absl::OkStatus();
  }

  void* Alloc(size_t alignment, size_t num_bytes,
              size_t* bytes_received) override {
    tsl::profiler::TraceMe traceme("DeviceMemAllocator::Alloc");

    if (num_bytes > 0) {
      allocations_started_ = true;
    }
    if (reservation_ != nullptr) {
      return AllocReserved(num_bytes, bytes_received);
    }

    void* ptr = nullptr;
    *bytes_received = num_bytes;
    if (num_bytes > 0) {
      // Propagate the allocator-reported size so BFC can use any backend
      // padding that is part of the addressable allocation.
      DeviceAddressBase result = stream_exec_->Allocate(num_bytes);
      ptr = result.opaque();
      *bytes_received = result.size();
      VisitAlloc(ptr, device_id_.value(), *bytes_received);
    }
    return ptr;
  }

  void Free(void* ptr, size_t num_bytes) override {
    tsl::profiler::TraceMe traceme("DeviceMemAllocator::Free");

    if (ptr != nullptr) {
      if (reservation_ != nullptr) {
        // BFC frees the coalesced region at teardown, or returns a just-added
        // suffix if its size is unsuitable. Unmap each original mapping before
        // releasing its physical handle, and pair visitors with original Alloc
        // calls rather than the coalesced BFC region.
        const uintptr_t base =
            reinterpret_cast<uintptr_t>(reservation_->address().opaque());
        const uintptr_t start = reinterpret_cast<uintptr_t>(ptr);
        CHECK_GE(start, base);
        CHECK_LE(start - base, mapped_bytes_);
        CHECK_EQ(num_bytes, mapped_bytes_ - (start - base));
        if (start == base) {
          // Whole-range return: BFC tearing down, rejecting the initial
          // region, or garbage-collecting the region. Buffers in this range
          // may have been handed out, and cuMemUnmap (unlike cuMemFree)
          // invalidates mappings immediately, so drain the device first.
          // SynchronizeAllActivity bypasses CudaDeviceAllocator's graph
          // capture guard; that is acceptable only because none of these
          // callers runs while a capture using this arena is active.
          //
          // A suffix return (start > base) hands back backing that Extend
          // mapped moments ago and never published, so no device work can
          // reference it. It must not synchronize: Extend can run during
          // stream capture, where a context synchronize would fail and
          // invalidate the capture.
          if (!stream_exec_->SynchronizeAllActivity()) {
            LOG(WARNING) << "Failed to synchronize device "
                         << device_id_.value()
                         << " before unmapping its BFC arena; in-flight work "
                            "may still reference it.";
          }
        }
        while (num_bytes > 0) {
          const DeviceAddressBase address =
              regions_.back().mapping.mapped_address();
          CHECK_GE(num_bytes, address.size());
          VisitFree(address.opaque(), device_id_.value(), address.size());
          num_bytes -= address.size();
          mapped_bytes_ -= address.size();
          regions_.pop_back();
        }
        return;
      }
      VisitFree(ptr, device_id_.value(), num_bytes);
      DeviceAddressBase device_ptr(ptr);
      stream_exec_->Deallocate(&device_ptr);
    }
  }

  bool SupportsCoalescing() const override { return reservation_ != nullptr; }

  // Mapping granularity of the reserved arena; BFC sizes extensions and
  // backpedal retries in multiples of it. Also used to round the initial
  // backing size when reserving a growing arena.
  size_t GetAllocationGranularity() const override {
    return reservation_ != nullptr ? reservation_->granularity() : 1;
  }

  tsl::AllocatorMemoryType GetMemoryType() const override {
    return tsl::AllocatorMemoryType::kDevice;
  }

 private:
  void* AllocReserved(size_t num_bytes, size_t* bytes_received) {
    *bytes_received = 0;
    const size_t available_bytes = capacity_ - mapped_bytes_;
    if (num_bytes == 0 || num_bytes > available_bytes) {
      return nullptr;
    }

    const size_t granularity = reservation_->granularity();
    const size_t remainder = num_bytes % granularity;
    if (remainder != 0) {
      const size_t padding = granularity - remainder;
      // Check before adding so rounding cannot overflow or exceed capacity.
      if (padding > available_bytes - num_bytes) {
        return nullptr;
      }
      num_bytes += padding;
    }

    auto allocation = stream_exec_->CreatePhysicalMemoryAllocation(num_bytes);
    if (!allocation.ok()) {
      VLOG(2) << "Physical device allocation failed: " << allocation.status();
      return nullptr;
    }
    const size_t size = (*allocation)->address().size();
    if (size < num_bytes || size > available_bytes) {
      return nullptr;
    }
    auto mapping = reservation_->MapTo(mapped_bytes_, 0, size, **allocation);
    if (!mapping.ok()) {
      VLOG(2) << "Mapping device memory failed: " << mapping.status();
      return nullptr;
    }
    // MapTo also sets local/peer access, and unmaps this slice on access
    // failure. Publish only after success; a failed extension leaves the prefix
    // intact.
    void* ptr = mapping->mapped_address().opaque();
    regions_.push_back({std::move(*allocation), std::move(*mapping)});
    mapped_bytes_ += size;
    *bytes_received = size;
    VisitAlloc(ptr, device_id_.value(), size);
    return ptr;
  }

  StreamExecutor* stream_exec_;  // not owned, non-null
  const tsl::PlatformDeviceId device_id_;
  bool allocations_started_ = false;

  // Declaration order keeps the reservation and physical backing alive until
  // after their mappings are destroyed. The reserved suffix has no backing.
  std::unique_ptr<MemoryReservation> reservation_;
  struct Region {
    std::unique_ptr<MemoryAllocation> allocation;
    MemoryReservation::ScopedMapping mapping;
  };
  std::vector<Region> regions_;
  size_t capacity_ = 0;
  size_t mapped_bytes_ = 0;

  DeviceMemAllocator(const DeviceMemAllocator&) = delete;
  void operator=(const DeviceMemAllocator&) = delete;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_INTEGRATIONS_DEVICE_MEM_ALLOCATOR_H_
