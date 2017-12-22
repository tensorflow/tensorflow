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
#include <vector>

#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id_utils.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/platform/stream_executor.h"

#define MASK_WORDS 2
#define MASK_BYTES (MASK_WORDS * sizeof(int64))

namespace tensorflow {
namespace {

int64* NewMask(int64 word) {
  int64* m = new int64[MASK_WORDS];
  for (int i = 0; i < MASK_WORDS; ++i) {
    m[i] = word;
  }
  return m;
}

int64* before_mask = NewMask(0xabababababababab);
int64* after_mask = NewMask(0xcdcdcdcdcdcdcdcd);

bool CheckMask(perftools::gputools::StreamExecutor* exec, void* ptr,
               int64* mask) {
  gpu::DeviceMemory<int64> gpu_ptr{gpu::DeviceMemoryBase{ptr, MASK_BYTES}};
  int64 tmp[MASK_WORDS];

  if (!exec->SynchronousMemcpy(&tmp, gpu_ptr, MASK_BYTES)) {
    LOG(FATAL) << "Could not copy debug mask";
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

void InitMask(perftools::gputools::StreamExecutor* exec, void* ptr,
              int64* mask) {
  gpu::DeviceMemory<int64> gpu_ptr{gpu::DeviceMemoryBase{ptr, MASK_BYTES}};
  if (!exec->SynchronousMemcpy(&gpu_ptr, mask, MASK_BYTES)) {
    LOG(FATAL) << "Could not copy debug mask";
  }
}

}  // namespace

// -----------------------------------------------------------------------------
// GPUDebugAllocator
// -----------------------------------------------------------------------------
GPUDebugAllocator::GPUDebugAllocator(VisitableAllocator* allocator,
                                     CudaGpuId cuda_gpu_id)
    : base_allocator_(allocator) {
  stream_exec_ = GpuIdUtil::ExecutorForCudaGpuId(cuda_gpu_id).ValueOrDie();
}

GPUDebugAllocator::~GPUDebugAllocator() { delete base_allocator_; }

void* GPUDebugAllocator::AllocateRaw(size_t alignment, size_t num_bytes) {
  num_bytes += (2 * MASK_BYTES);

  void* allocated_ptr = base_allocator_->AllocateRaw(alignment, num_bytes);

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
  CHECK(CheckHeader(ptr)) << "before_mask has been overwritten";
  CHECK(CheckFooter(ptr)) << "after_mask has been overwritten";

  // Backtrack to the beginning of the header.
  ptr = static_cast<void*>(static_cast<char*>(ptr) - MASK_BYTES);
  // Deallocate the memory
  base_allocator_->DeallocateRaw(ptr);
}

void GPUDebugAllocator::AddAllocVisitor(Visitor visitor) {
  return base_allocator_->AddAllocVisitor(visitor);
}

void GPUDebugAllocator::AddFreeVisitor(Visitor visitor) {
  return base_allocator_->AddFreeVisitor(visitor);
}

bool GPUDebugAllocator::TracksAllocationSizes() { return true; }

size_t GPUDebugAllocator::RequestedSize(void* ptr) {
  auto req_size =
      base_allocator_->RequestedSize(static_cast<char*>(ptr) - MASK_BYTES);
  return req_size - 2 * MASK_BYTES;
}

size_t GPUDebugAllocator::AllocatedSize(void* ptr) {
  return base_allocator_->AllocatedSize(static_cast<char*>(ptr) - MASK_BYTES);
}

int64 GPUDebugAllocator::AllocationId(void* ptr) {
  return base_allocator_->AllocationId(static_cast<char*>(ptr) - MASK_BYTES);
}

void GPUDebugAllocator::GetStats(AllocatorStats* stats) {
  base_allocator_->GetStats(stats);
}

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
GPUNanResetAllocator::GPUNanResetAllocator(VisitableAllocator* allocator,
                                           CudaGpuId cuda_gpu_id)
    : base_allocator_(allocator) {
  stream_exec_ = GpuIdUtil::ExecutorForCudaGpuId(cuda_gpu_id).ValueOrDie();
}

GPUNanResetAllocator::~GPUNanResetAllocator() { delete base_allocator_; }

void* GPUNanResetAllocator::AllocateRaw(size_t alignment, size_t num_bytes) {
  void* allocated_ptr = base_allocator_->AllocateRaw(alignment, num_bytes);

  // Initialize the buffer to Nans
  size_t req_size = base_allocator_->RequestedSize(allocated_ptr);
  std::vector<float> nans(req_size / sizeof(float), std::nanf(""));
  gpu::DeviceMemory<float> nan_ptr{
      gpu::DeviceMemoryBase{static_cast<float*>(allocated_ptr), req_size}};

  if (!stream_exec_->SynchronousMemcpy(&nan_ptr, &nans[0], req_size)) {
    LOG(ERROR) << "Could not initialize to NaNs";
  }

  return allocated_ptr;
}
void GPUNanResetAllocator::DeallocateRaw(void* ptr) {
  // Reset the buffer to Nans
  size_t req_size = base_allocator_->RequestedSize(ptr);
  std::vector<float> nans(req_size / sizeof(float), std::nanf(""));
  gpu::DeviceMemory<float> nan_ptr{
      gpu::DeviceMemoryBase{static_cast<float*>(ptr), req_size}};
  if (!stream_exec_->SynchronousMemcpy(&nan_ptr, &nans[0], req_size)) {
    LOG(ERROR) << "Could not initialize to NaNs";
  }

  // Deallocate the memory
  base_allocator_->DeallocateRaw(ptr);
}

void GPUNanResetAllocator::AddAllocVisitor(Visitor visitor) {
  return base_allocator_->AddAllocVisitor(visitor);
}

void GPUNanResetAllocator::AddFreeVisitor(Visitor visitor) {
  return base_allocator_->AddFreeVisitor(visitor);
}

size_t GPUNanResetAllocator::RequestedSize(void* ptr) {
  return base_allocator_->RequestedSize(ptr);
}

size_t GPUNanResetAllocator::AllocatedSize(void* ptr) {
  return base_allocator_->AllocatedSize(ptr);
}

void GPUNanResetAllocator::GetStats(AllocatorStats* stats) {
  base_allocator_->GetStats(stats);
}

}  // namespace tensorflow
