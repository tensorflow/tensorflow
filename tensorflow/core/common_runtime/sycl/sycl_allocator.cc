/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifdef TENSORFLOW_USE_SYCL

#include "tensorflow/core/common_runtime/sycl/sycl_allocator.h"

namespace tensorflow {

SYCLAllocator::SYCLAllocator(Eigen::QueueInterface* queue)
    : sycl_device_(new Eigen::SyclDevice(queue)) {
  cl::sycl::queue& sycl_queue = sycl_device_->sycl_queue();
  const cl::sycl::device& device = sycl_queue.get_device();
  stats_.bytes_limit =
      device.get_info<cl::sycl::info::device::max_mem_alloc_size>();
}

SYCLAllocator::~SYCLAllocator() {
  if (sycl_device_) {
    delete sycl_device_;
  }
}

string SYCLAllocator::Name() { return "device:SYCL"; }

void* SYCLAllocator::AllocateRaw(size_t alignment, size_t num_bytes) {
  mutex_lock lock(mu_);
  assert(sycl_device_);
  if (num_bytes == 0) {
    // Cannot allocate no bytes in SYCL, so instead allocate a single byte
    num_bytes = 1;
  }
  auto p = sycl_device_->allocate(num_bytes);
  const auto& allocated_buffer = sycl_device_->get_sycl_buffer(p);
  const std::size_t bytes_allocated = allocated_buffer.get_range().size();

  ++stats_.num_allocs;
  stats_.bytes_in_use += bytes_allocated;
  stats_.max_bytes_in_use =
      std::max<int64>(stats_.max_bytes_in_use, stats_.bytes_in_use);
  stats_.max_alloc_size =
      std::max<int64>(stats_.max_alloc_size, bytes_allocated);

  return p;
}

void SYCLAllocator::DeallocateRaw(void* ptr) {
  mutex_lock lock(mu_);
  if (sycl_device_) {
    const auto& buffer_to_delete = sycl_device_->get_sycl_buffer(ptr);
    const std::size_t dealloc_size = buffer_to_delete.get_range().size();
    stats_.bytes_in_use -= dealloc_size;
    sycl_device_->deallocate(ptr);
  }
}

void SYCLAllocator::GetStats(AllocatorStats* stats) {
  mutex_lock lock(mu_);
  *stats = stats_;
}

void SYCLAllocator::ClearStats() override {
  mutex_lock l(mu_);
  stats_.num_allocs = 0;
  stats_.max_bytes_in_use = stats_.bytes_in_use;
  stats_.max_alloc_size = 0;
}

size_t SYCLAllocator::RequestedSize(const void* ptr) const {
  mutex_lock lock(mu_);
  if (!sycl_device_) {
    return 0;
  }
  const auto& buffer = sycl_device_->get_sycl_buffer(ptr);
  return buffer.get_size();
}

}  // namespace tensorflow

#endif  // TENSORFLOW_USE_SYCL
