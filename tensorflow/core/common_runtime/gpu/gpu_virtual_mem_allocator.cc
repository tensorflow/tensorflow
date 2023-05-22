/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/gpu/gpu_virtual_mem_allocator.h"

#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor.h"
#include "tensorflow/tsl/platform/numbers.h"
#include "tensorflow/tsl/profiler/lib/traceme.h"

#if CUDA_VERSION >= 10020

namespace tensorflow {
namespace {

using ::stream_executor::gpu::GpuContext;
using ::stream_executor::gpu::GpuDeviceHandle;
using ::stream_executor::gpu::GpuDevicePtr;
using ::stream_executor::gpu::GpuDriver;
using ::tsl::Status;
using ::tsl::StatusOr;

// Rounds value up to the specified power of two alignment.
size_t AlignUp(size_t value, size_t alignment) {
  DCHECK_EQ(alignment & (alignment - 1), 0)
      << "Alignment must be a power of two; alignment=" << alignment;
  return (value + alignment - 1) & ~(alignment - 1);
}

StatusOr<bool> SupportsVirtualAddressManagement(GpuDeviceHandle device) {
  return GpuDriver::GetDeviceAttribute(
      CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, device);
}

Status CheckVirtualAddressManagementSupport(GpuDeviceHandle device,
                                            tsl::PlatformDeviceId gpu_id) {
  TF_ASSIGN_OR_RETURN(bool supports_virtual_address_management,
                      SupportsVirtualAddressManagement(device));
  if (!supports_virtual_address_management) {
    return tsl::errors::Internal(absl::StrFormat(
        "GPU %d does not support virtual memory address management.",
        gpu_id.value()));
  }
  return {};
}

}  // namespace

/* static */ tsl::StatusOr<std::unique_ptr<GpuVirtualMemAllocator>>
GpuVirtualMemAllocator::Create(
    const std::vector<Visitor>& alloc_visitors,
    const std::vector<Visitor>& free_visitors, GpuContext& gpu_context,
    tsl::PlatformDeviceId gpu_id, size_t virtual_address_space_size,
    const std::vector<tsl::PlatformDeviceId>& peer_gpu_ids) {
  tsl::profiler::TraceMe traceme("GpuVirtualMemAllocator::Create");

  std::vector<GpuDeviceHandle> access_gpu_handles;
  access_gpu_handles.reserve(peer_gpu_ids.size() + 1);

  GpuDeviceHandle gpu_handle;
  TF_RETURN_IF_ERROR(GpuDriver::GetDevice(gpu_id.value(), &gpu_handle));
  TF_RETURN_IF_ERROR(CheckVirtualAddressManagementSupport(gpu_handle, gpu_id));

  access_gpu_handles.push_back(gpu_handle);
  for (const auto& peer_id : peer_gpu_ids) {
    GpuDeviceHandle peer_handle;
    TF_RETURN_IF_ERROR(GpuDriver::GetDevice(peer_id.value(), &peer_handle));
    TF_ASSIGN_OR_RETURN(bool supports_virtual_address_management,
                        SupportsVirtualAddressManagement(peer_handle));
    if (GpuDriver::CanEnablePeerAccess(gpu_handle, peer_handle) &&
        supports_virtual_address_management) {
      access_gpu_handles.push_back(peer_handle);
    }
  }

  // Find the min granularity for all devices that have access to this memory;
  // that is, the maximum min granularity among all devices.
  size_t max_granularity = 1;
  for (const auto device_handle : access_gpu_handles) {
    TF_ASSIGN_OR_RETURN(size_t granularity,
                        GpuDriver::GetMinAllocationGranularity(device_handle));
    max_granularity = std::max(max_granularity, granularity);
  }

  // Create the virtual memory reservation. Must be aligned to system page size,
  // and larger than the CUDA min granularity. Empirically, the granularity
  // check is sufficient as the granularity is some multiple of the page size.
  // TODO(imintz): Create OS agnostic page size utility for completeness.
  TF_ASSIGN_OR_RETURN(
      GpuDriver::VmemSpan vmem,
      GpuDriver::ReserveVirtualMemory(
          &gpu_context, AlignUp(virtual_address_space_size, max_granularity)));
  VLOG(1) << "Reserved GPU virtual memory at " << vmem.base << " of size "
          << tsl::strings::HumanReadableNumBytes(vmem.size_bytes) << " bytes";

  return std::unique_ptr<GpuVirtualMemAllocator>(new GpuVirtualMemAllocator(
      alloc_visitors, free_visitors, gpu_context, gpu_id,
      std::move(access_gpu_handles), vmem, max_granularity));
}

GpuVirtualMemAllocator::GpuVirtualMemAllocator(
    const std::vector<Visitor>& alloc_visitors,
    const std::vector<Visitor>& free_visitors, GpuContext& gpu_context,
    tsl::PlatformDeviceId gpu_id,
    const std::vector<GpuDeviceHandle> access_gpu_handles,
    GpuDriver::VmemSpan vmem, size_t granularity)
    : SubAllocator(alloc_visitors, free_visitors),
      gpu_context_(gpu_context),
      gpu_id_(gpu_id),
      access_gpu_handles_(access_gpu_handles),
      vmem_(vmem),
      granularity_(granularity) {}

GpuVirtualMemAllocator::~GpuVirtualMemAllocator() {
  for (const auto mapping : mappings_) {
    GpuDriver::UnmapMemory(&gpu_context_, mapping.va, mapping.physical.bytes);
    GpuDriver::ReleaseMemoryHandle(&gpu_context_, std::move(mapping.physical));
  }
  GpuDriver::FreeVirtualMemory(&gpu_context_, vmem_);
}

void* GpuVirtualMemAllocator::Alloc(size_t alignment, size_t num_bytes,
                                    size_t* bytes_received) {
  tsl::profiler::TraceMe traceme("GpuVirtualMemAllocator::Alloc");

  if (num_bytes == 0) return nullptr;
  size_t padded_bytes = (num_bytes + granularity_ - 1) & ~(granularity_ - 1);

  GpuDevicePtr next_va = vmem_.base + next_alloc_offset_;

  // TODO(imintz): Attempt to extend the vmem allocation by reserving additional
  // virtual memory at the specific address at the end of the initial vmem
  // reservation.
  if (next_va + padded_bytes > vmem_.base + vmem_.size_bytes) {
    LOG(ERROR) << "OOM in GPU virtual memory allocator when attempting to "
                  "allocate {request: "
               << tsl::strings::HumanReadableNumBytes(num_bytes)
               << ", aligned: " << padded_bytes << "} bytes.";
    return nullptr;
  }

  // Create physical memory backing allocation.
  auto maybe_handle =
      GpuDriver::CreateMemoryHandle(&gpu_context_, padded_bytes);
  if (!maybe_handle.ok()) {
    LOG(ERROR) << maybe_handle.status();
    return nullptr;
  }
  GpuDriver::GenericMemoryHandle handle = std::move(maybe_handle).value();

  // Map VAs for this physical memory.
  auto status =
      GpuDriver::MapMemory(&gpu_context_, next_va, handle, access_gpu_handles_);
  if (!status.ok()) {
    LOG(ERROR) << status;
    GpuDriver::ReleaseMemoryHandle(&gpu_context_, std::move(handle));
    return nullptr;
  }
  next_alloc_offset_ += handle.bytes;
  mappings_.push_back({next_va, std::move(handle)});
  VisitAlloc(reinterpret_cast<void*>(next_va), gpu_id_.value(), padded_bytes);
  *bytes_received = padded_bytes;
  return reinterpret_cast<void*>(next_va);
}

void GpuVirtualMemAllocator::Free(void* ptr, size_t num_bytes) {
  tsl::profiler::TraceMe traceme("GpuVirtualMemAllocator::Free");

  if (ptr == nullptr) return;

  auto mapping_it =
      std::lower_bound(mappings_.begin(), mappings_.end(), ptr,
                       [](const Mapping& mapping, const void* ptr) {
                         return reinterpret_cast<const void*>(mapping.va) < ptr;
                       });
  if (mapping_it == mappings_.end() ||
      (reinterpret_cast<void*>(mapping_it->va) != ptr)) {
    LOG(ERROR) << "Could not find GPU vmem mapping for address at "
               << reinterpret_cast<uintptr_t>(ptr);
    return;
  }

  int num_mappings_to_free = 0;
  int total_bytes = 0;
  for (auto it = mapping_it; it != mappings_.end() && total_bytes < num_bytes;
       ++it) {
    ++num_mappings_to_free;
    total_bytes += it->physical.bytes;
  }
  if (total_bytes != num_bytes) {
    LOG(ERROR) << "Invalid size requested for freeing GPU vmem mapping. Got "
               << tsl::strings::HumanReadableNumBytes(num_bytes)
               << " but expected "
               << tsl::strings::HumanReadableNumBytes(
                      mapping_it->physical.bytes);
    return;
  }

  VLOG(1) << "Freeing " << num_mappings_to_free << " mappings for a total of "
          << total_bytes << " bytes";
  for (auto it = mapping_it; it < mapping_it + num_mappings_to_free; ++it) {
    GpuDriver::UnmapMemory(&gpu_context_, it->va, it->physical.bytes);
    GpuDriver::ReleaseMemoryHandle(&gpu_context_, std::move(it->physical));
  }

  // Move back the next_alloc_offset_ if this free was at the end.
  if (mapping_it + num_mappings_to_free == mappings_.end()) {
    next_alloc_offset_ = mapping_it->va - vmem_.base;
  }

  mappings_.erase(mapping_it, mapping_it + num_mappings_to_free);
  VisitFree(ptr, gpu_id_.value(), num_bytes);
}

}  // namespace tensorflow

#endif
