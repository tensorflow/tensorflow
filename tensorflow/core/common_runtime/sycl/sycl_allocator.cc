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

SYCLAllocator::~SYCLAllocator() {
  if(sycl_device_) {
    delete sycl_device_;
  }
}

string SYCLAllocator::Name() { return "device:SYCL"; }

void *SYCLAllocator::AllocateRaw(size_t alignment, size_t num_bytes) {
  assert(sycl_device_);
  if (num_bytes == 0) {
    return sycl_device_->allocate(1);
  }
  auto p = sycl_device_->allocate(num_bytes);
  return p;
}

void SYCLAllocator::DeallocateRaw(void *ptr) {
  if (sycl_device_) {
    sycl_device_->deallocate(ptr);
  }
}

}  // namespace tensorflow

#endif  // TENSORFLOW_USE_SYCL
