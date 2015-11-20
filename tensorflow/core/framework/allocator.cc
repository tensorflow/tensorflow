/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/port.h"

namespace tensorflow {

Allocator::~Allocator() {}

class CPUAllocator : public Allocator {
 public:
  ~CPUAllocator() override {}

  string Name() override { return "cpu"; }
  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
    return port::aligned_malloc(num_bytes, alignment);
  }

  void DeallocateRaw(void* ptr) override { port::aligned_free(ptr); }
};

Allocator* cpu_allocator() {
  static CPUAllocator* cpu_alloc = new CPUAllocator;
  return cpu_alloc;
}

}  // namespace tensorflow
