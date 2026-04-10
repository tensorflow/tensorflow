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

#include "xla/pjrt/extensions/host_allocator/host_memory_allocator/host_memory_allocator_interface_impl.h"

#include <cstddef>
#include <memory>

#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/extensions/host_allocator/host_memory_allocator/host_memory_allocator_extension.h"
#include "xla/pjrt/host_memory_allocator.h"

namespace pjrt {

namespace {

class HostMemoryAllocatorInterfaceImpl : public xla::HostMemoryAllocator {
 public:
  HostMemoryAllocatorInterfaceImpl(
      PJRT_Client* c_client,
      const PJRT_HostMemoryAllocator_Extension* extension,
      const PJRT_Api* c_api)
      : c_client_(c_client), extension_(extension), c_api_(c_api) {}

  OwnedPtr Allocate(size_t size, const AllocateOptions& options) override {
    PJRT_HostMemoryAllocator_Allocate_Args args{};
    args.struct_size = PJRT_HostMemoryAllocator_Allocate_Args_STRUCT_SIZE;
    args.client = c_client_;
    args.size = size;
    args.numa_node = options.numa_node;
    pjrt::LogFatalIfPjrtError(extension_->allocate(&args), c_api_);
    xla::HostMemoryAllocator::Deleter deleter = {args.deleter,
                                                 args.deleter_arg};
    return xla::HostMemoryAllocator::OwnedPtr(args.ptr, deleter);
  }

 private:
  PJRT_Client* c_client_;
  const PJRT_HostMemoryAllocator_Extension* extension_;
  const PJRT_Api* c_api_;
};

}  // namespace

std::unique_ptr<xla::HostMemoryAllocator> CreateHostMemoryAllocatorWrapper(
    PJRT_Client* c_client, const PJRT_HostMemoryAllocator_Extension* extension,
    const PJRT_Api* c_api) {
  return std::make_unique<HostMemoryAllocatorInterfaceImpl>(c_client, extension,
                                                            c_api);
}

}  // namespace pjrt
