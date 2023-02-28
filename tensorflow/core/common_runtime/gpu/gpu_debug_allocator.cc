/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/gpu/gpu_debug_allocator.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "tensorflow/compiler/xla/stream_executor/device_id_utils.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_init.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor.h"
#include "tensorflow/tsl/framework/device_id.h"
#include "tensorflow/tsl/platform/logging.h"
#include "tensorflow/tsl/platform/status.h"

#define MASK_WORDS 2
#define MASK_BYTES (MASK_WORDS * sizeof(int64_t))

namespace tensorflow {
namespace {

int64_t* NewMask(int64_t word) {
  int64_t* m = new int64_t[MASK_WORDS];
  for (int i = 0; i < MASK_WORDS; ++i) {
    m[i] = word;
  }
  return m;
}

int64_t* before_mask = NewMask(0xabababababababab);
int64_t* after_mask = NewMask(0xcdcdcdcdcdcdcdcd);

bool CheckMask(se::StreamExecutor* exec, void* ptr, int64_t* mask) {
  se::DeviceMemory<int64_t> gpu_ptr{se::DeviceMemoryBase{ptr, MASK_BYTES}};
  int64_t tmp[MASK_WORDS];

  tsl::Status result = exec->SynchronousMemcpyD2H(gpu_ptr, MASK_BYTES, tmp);
  if (!result.ok()) {
    LOG(FATAL) << "Could not copy debug mask, " << result;
  }

  bool ok = true;
  for (int i = 0; i < MASK_WORDS; ++i) {
    ok &= (mask[i] == tmp[i]);
    if (!ok) {
      LOG(ERROR) << "i=" << i
                 << " mask=" << reinterpret_cast<const void*>(mask[i])
                 << " field=" << reinterpret_cast<const void*>(tmp[i]);
    }
  }

  return ok;
}

void InitMask(se::StreamExecutor* exec, void* ptr, int64_t* mask) {
  se::DeviceMemory<int64_t> gpu_ptr{se::DeviceMemoryBase{ptr, MASK_BYTES}};
  tsl::Status result = exec->SynchronousMemcpyH2D(mask, MASK_BYTES, &gpu_ptr);
  if (!result.ok()) {
    LOG(FATAL) << "Could not copy debug mask, " << result;
  }
}

}  // namespace

// -----------------------------------------------------------------------------
// GPUDebugAllocator
// -----------------------------------------------------------------------------
GPUDebugAllocator::GPUDebugAllocator(Allocator* allocator,
                                     tsl::PlatformDeviceId platform_device_id,
                                     tsl::int32 stream_id)
    : base_allocator_(allocator) {
  stream_exec_ = se::DeviceIdUtil::ExecutorForPlatformDeviceId(
                     se::GPUMachineManager(), platform_device_id, stream_id)
                     .value();
}

GPUDebugAllocator::~GPUDebugAllocator() { delete base_allocator_; }

void* GPUDebugAllocator::AllocateRaw(size_t alignment, size_t num_bytes) {
  num_bytes += (2 * MASK_BYTES);
  void* allocated_ptr = base_allocator_->AllocateRaw(alignment, num_bytes);
  if (allocated_ptr == nullptr) return allocated_ptr;

  // Return the pointer after the header
  void* rv = static_cast<char*>(allocated_ptr) + MASK_BYTES;

  // Write the header at allocated_ptr
  InitMask(stream_exec_, allocated_ptr, before_mask);

  // Write the footer at the end.
  size_t req_size = base_allocator_->RequestedSize(allocated_ptr);
  InitMask(stream_exec_,
           static_cast<char*>(allocated_ptr) + req_size - MASK_BYTES,
           after_mask);
  return rv;
}
void GPUDebugAllocator::DeallocateRaw(void* ptr) {
  if (ptr != nullptr) {
    CHECK(CheckHeader(ptr)) << "before_mask has been overwritten";
    CHECK(CheckFooter(ptr)) << "after_mask has been overwritten";

    // Backtrack to the beginning of the header.
    ptr = static_cast<void*>(static_cast<char*>(ptr) - MASK_BYTES);
  }
  // Deallocate the memory
  base_allocator_->DeallocateRaw(ptr);
}

bool GPUDebugAllocator::TracksAllocationSizes() const { return true; }

size_t GPUDebugAllocator::RequestedSize(const void* ptr) const {
  auto req_size = base_allocator_->RequestedSize(static_cast<const char*>(ptr) -
                                                 MASK_BYTES);
  return req_size - 2 * MASK_BYTES;
}

size_t GPUDebugAllocator::AllocatedSize(const void* ptr) const {
  return base_allocator_->AllocatedSize(static_cast<const char*>(ptr) -
                                        MASK_BYTES);
}

int64_t GPUDebugAllocator::AllocationId(const void* ptr) const {
  return base_allocator_->AllocationId(static_cast<const char*>(ptr) -
                                       MASK_BYTES);
}

std::optional<tsl::AllocatorStats> GPUDebugAllocator::GetStats() {
  return base_allocator_->GetStats();
}

bool GPUDebugAllocator::ClearStats() { return base_allocator_->ClearStats(); }

bool GPUDebugAllocator::CheckHeader(void* ptr) {
  return CheckMask(stream_exec_, static_cast<char*>(ptr) - MASK_BYTES,
                   before_mask);
}

bool GPUDebugAllocator::CheckFooter(void* ptr) {
  char* original_ptr = static_cast<char*>(ptr) - MASK_BYTES;
  size_t req_size = base_allocator_->RequestedSize(original_ptr);
  return CheckMask(stream_exec_, original_ptr + req_size - MASK_BYTES,
                   after_mask);
}

// -----------------------------------------------------------------------------
// GPUNanResetAllocator
// -----------------------------------------------------------------------------
GPUNanResetAllocator::GPUNanResetAllocator(
    Allocator* allocator, tsl::PlatformDeviceId platform_device_id,
    tsl::int32 stream_id)
    : base_allocator_(allocator) {
  stream_exec_ = se::DeviceIdUtil::ExecutorForPlatformDeviceId(
                     se::GPUMachineManager(), platform_device_id, stream_id)
                     .value();
}

GPUNanResetAllocator::~GPUNanResetAllocator() { delete base_allocator_; }

void* GPUNanResetAllocator::AllocateRaw(size_t alignment, size_t num_bytes) {
  void* allocated_ptr = base_allocator_->AllocateRaw(alignment, num_bytes);
  if (allocated_ptr == nullptr) return allocated_ptr;

  // Initialize the buffer to Nans
  size_t req_size = base_allocator_->RequestedSize(allocated_ptr);
  std::vector<float> nans((req_size + sizeof(float) - 1) / sizeof(float),
                          std::nanf(""));
  se::DeviceMemory<float> nan_ptr{
      se::DeviceMemoryBase{static_cast<float*>(allocated_ptr), req_size}};

  tsl::Status result =
      stream_exec_->SynchronousMemcpyH2D(&nans[0], req_size, &nan_ptr);
  if (!result.ok()) {
    LOG(ERROR) << "Could not initialize to NaNs, " << result;
  }

  return allocated_ptr;
}
void GPUNanResetAllocator::DeallocateRaw(void* ptr) {
  if (ptr != nullptr) {
    // Reset the buffer to Nans
    size_t req_size = base_allocator_->RequestedSize(ptr);
    std::vector<float> nans((req_size + sizeof(float) - 1) / sizeof(float),
                            std::nanf(""));
    se::DeviceMemory<float> nan_ptr{
        se::DeviceMemoryBase{static_cast<float*>(ptr), req_size}};
    tsl::Status result =
        stream_exec_->SynchronousMemcpyH2D(&nans[0], req_size, &nan_ptr);
    if (!result.ok()) {
      LOG(ERROR) << "Could not initialize to NaNs, " << result;
    }
  }

  // Deallocate the memory
  base_allocator_->DeallocateRaw(ptr);
}

size_t GPUNanResetAllocator::RequestedSize(const void* ptr) const {
  return base_allocator_->RequestedSize(ptr);
}

size_t GPUNanResetAllocator::AllocatedSize(const void* ptr) const {
  return base_allocator_->AllocatedSize(ptr);
}

std::optional<tsl::AllocatorStats> GPUNanResetAllocator::GetStats() {
  return base_allocator_->GetStats();
}

bool GPUNanResetAllocator::ClearStats() {
  return base_allocator_->ClearStats();
}

}  // namespace tensorflow
